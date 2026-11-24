"""Per-layer, per-stage timing of one inference step, so the bottlenecks can be found (Alpha,
2026-11-23: "record how much time is spent on each individual layer/calculation per layer of the
model ... later on I can analyze and see where the major bottlenecks are to optimize").

    cd python
    python -m bench.layer_timing --arms baseline,lut,hf --device cuda
    python -m bench.layer_timing --arms baseline --prefill 64,224,1024 --decode 64,224,1024 --repeats 30

Our arms (baseline, lut): every stage function the wiring calls through ModelOps (embedding, each
layer's norm, qkv_proj, rope, attn_scores, softmax, attn_values, out_proj, ffn_up, ffn_act,
ffn_gate, ffn_down, then final_norm and lm_head; base.STAGES names) is wrapped here, after the
host has resolved it, so baseline/host.py and LUT/host.py are not touched. On cuda each call gets a
CUDA event before and after it and nothing is synchronized between calls; the one synchronize is
at the end of the step. On cpu the calls are synchronous and time.perf_counter() brackets them.
A stage called several times per layer (norm twice, qkv_proj three times, rope twice, ffn_up twice
in a gated feed-forward) is summed over its calls; `calls` says how many.

Hugging Face arm (hf): coarser, labelled so. Forward hooks on transformers' modules: per layer the
two RMSNorms, q/k/v/o projections, gate/up/down projections and the SiLU module, plus the whole
attention and MLP modules. attn_core = attention module minus its four projections (rotary, scores,
softmax and PV inside one SDPA call); ffn_gate = MLP module minus its children (the elementwise
product). rope_tables is HF's rotary module, called once per step.

What one repeat measures (the same steps as bench/sweep.py's speed benchmark, B = 1):
    prefill T    the T-token prompt into an empty KV cache, then the argmax of the last position
    decode T     the step that makes the T-th token: one token against a cache holding T - 1 tokens
                 (rewound to T - 1 before every repeat), so it attends to T keys (bench/opcount.py
                 uses the same definition); argmax. (Runs before 2026-11-24 held T tokens in the cache.)
Each repeat runs the step twice, uninstrumented and instrumented, alternating, so the overhead of
the timing itself is visible (step_ms_uninstrumented vs step_ms_instrumented). Stage times are
medians over the timed repeats of the per-step sums; share_of_step = stage median / instrumented
step median. "unattributed" = step minus the stages (split/merge heads, KV append, residual adds,
the rotary table, the mask, argmax, and on cuda the GPU's idle gaps between launches).

How to read the times. At this model size on a GPU, almost every stage is limited by kernel launch
and Python overhead, not by arithmetic: a stage's event time is roughly how long the CPU takes to
issue its kernels, so time tracks the number of kernels more than the work. Each row therefore
carries the modeled cost per token from bench/opcount.py (multiply-adds, nonlinear evaluations,
bytes moved). A context beyond the model's training context (256 for stories15M) is timing only.

Writes one run per arm (results/runs/<id>_layer_timing_<model>_<arm>/) and appends long-format
rows to results/layer_timing.csv (LAYER_COLUMNS), which bench/excel.py turns into the workbook's
per-layer timing sheet. Emulation speed in PyTorch says nothing about FPGA speed (agent.md 2.6).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from baseline import host as base
from bench import opcount, resources, sweep
from bench.runlog import Run
from common import precision
from reference import local_cache

LAYER_TIMING_CSV = "layer_timing.csv"
LAYER_COLUMNS = (
    "timestamp_utc", "run_id", "git_commit", "flow", "model", "arm", "variant", "device", "platform", "dtype",
    "phase", "context", "tokens_in_step", "beyond_trained_context", "layer", "stage", "calls", "ms_median",
    "ms_p10", "ms_p90", "share_of_step", "step_ms_instrumented", "step_ms_uninstrumented",
    "instrumentation_overhead_pct", "mul_adds_per_token", "nonlinear_per_token", "bytes_per_token", "repeats",
    "warmup", "method", "note",
)
UNATTRIBUTED = "unattributed"
HF_ATTN_CORE = "attn_core"
HF_ATTN_CORE_STAGES = ("rope", "attn_scores", "softmax", "attn_values")


# ----------------------------------------------------------------------------------------------
# Timestamps: CUDA events (no synchronize between them) or perf_counter on the CPU
# ----------------------------------------------------------------------------------------------

class Recorder:
    """Marks points in a step and keeps (layer, stage, start, end) for every timed call. On cuda the
    marks are events from a pool reused every step, so creating events costs nothing per call."""

    def __init__(self, device: str) -> None:
        self.cuda = device.startswith("cuda")
        self.device = device
        self.pool: list[Any] = []
        self.used = 0
        self.calls: list[tuple[Any, str, Any, Any]] = []
        self.stack: dict[int, list[Any]] = defaultdict(list)    # hooks: module id -> open start marks

    def mark(self) -> Any:
        if not self.cuda:
            return time.perf_counter()
        if self.used == len(self.pool):
            self.pool.append(torch.cuda.Event(enable_timing=True))
        event = self.pool[self.used]
        self.used += 1
        event.record()
        return event

    def begin_step(self) -> None:
        self.used = 0
        self.calls.clear()
        self.stack.clear()

    def ms(self, start: Any, end: Any) -> float:
        return start.elapsed_time(end) if self.cuda else (end - start) * 1000.0

    def finish_step(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)


def wrap(rec: Recorder, fn: Callable, layer: Any, stage: str) -> Callable:
    """fn with a mark before and after every call, recorded as (layer, stage)."""
    def timed(*args: Any, **kwargs: Any) -> Any:
        start = rec.mark()
        out = fn(*args, **kwargs)
        rec.calls.append((layer, stage, start, rec.mark()))
        return out
    timed.__wrapped__ = fn
    return timed


def instrument_ops(ops: base.ModelOps, rec: Recorder) -> base.ModelOps:
    """A copy of ops (base.ModelOps, from base_ops or lut_ops) with every stage function wrapped."""
    return base.ModelOps(
        embedding=wrap(rec, ops.embedding, "model", "embedding"),
        layers=[{stage: wrap(rec, fn, i, stage) for stage, fn in layer.items()} for i, layer in enumerate(ops.layers)],
        final_norm=wrap(rec, ops.final_norm, "model", "final_norm"),
        lm_head=wrap(rec, ops.lm_head, "model", "lm_head"),
    )


def hf_hook_targets(model: Any) -> list[tuple[Any, Any, str]]:
    """(module, layer, label) for transformers' Llama. Labels starting with "_" are whole modules,
    used only to derive attn_core and ffn_gate and the unattributed time."""
    inner = model.model
    targets: list[tuple[Any, Any, str]] = [(inner.embed_tokens, "model", "embedding"),
                                           (inner.rotary_emb, "model", "rope_tables")]
    for i, layer in enumerate(inner.layers):
        attn, mlp = layer.self_attn, layer.mlp
        targets += [(layer.input_layernorm, i, "norm"), (attn, i, "_attention"),
                    (attn.q_proj, i, "qkv_proj"), (attn.k_proj, i, "qkv_proj"), (attn.v_proj, i, "qkv_proj"),
                    (attn.o_proj, i, "out_proj"), (layer.post_attention_layernorm, i, "norm"), (mlp, i, "_mlp"),
                    (mlp.gate_proj, i, "ffn_up"), (mlp.up_proj, i, "ffn_up"), (mlp.act_fn, i, "ffn_act"),
                    (mlp.down_proj, i, "ffn_down")]
    targets += [(inner.norm, "model", "final_norm"), (model.lm_head, "model", "lm_head")]
    return targets


def add_hooks(rec: Recorder, targets: list[tuple[Any, Any, str]]) -> list[Any]:
    handles = []
    for module, layer, label in targets:
        def pre(mod: Any, _inputs: Any) -> None:
            rec.stack[id(mod)].append(rec.mark())

        def post(mod: Any, _inputs: Any, _output: Any, layer: Any = layer, label: str = label) -> None:
            rec.calls.append((layer, label, rec.stack[id(mod)].pop(), rec.mark()))
        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    return handles


# ----------------------------------------------------------------------------------------------
# One arm, one phase and context
# ----------------------------------------------------------------------------------------------

def per_step_sums(rec: Recorder, arm: str) -> tuple[dict[tuple[Any, str], float], dict[tuple[Any, str], int], float]:
    """This step's {(layer, stage): ms summed over calls}, {(layer, stage): calls} and the attributed
    total (top-level entries only, so nested HF modules are not counted twice)."""
    sums: dict[tuple[Any, str], float] = defaultdict(float)
    calls: dict[tuple[Any, str], int] = defaultdict(int)
    for layer, stage, start, end in rec.calls:
        sums[(layer, stage)] += rec.ms(start, end)
        calls[(layer, stage)] += 1
    if arm != "hf":
        return sums, calls, sum(sums.values())
    top = sum(v for (layer, stage), v in sums.items()
              if stage in ("embedding", "rope_tables", "norm", "_attention", "_mlp", "final_norm", "lm_head"))
    for (layer, stage) in [k for k in list(sums) if k[1] == "_attention"]:
        sums[(layer, HF_ATTN_CORE)] = sums[(layer, stage)] - sums[(layer, "qkv_proj")] - sums[(layer, "out_proj")]
        calls[(layer, HF_ATTN_CORE)] = 1
    for (layer, stage) in [k for k in list(sums) if k[1] == "_mlp"]:
        sums[(layer, "ffn_gate")] = sums[(layer, stage)] - sums[(layer, "ffn_up")] - sums[(layer, "ffn_act")] \
            - sums[(layer, "ffn_down")]
        calls[(layer, "ffn_gate")] = 1
    for key in [k for k in list(sums) if k[1].startswith("_")]:
        del sums[key], calls[key]
    return sums, calls, top


def measure(arm: str, plain: Any, timed: Any, rec: Recorder, step_rec: Recorder, phase: str, context: int,
            vocab: int, seed: int, device: str, warmup: int, repeats: int,
            hooks: tuple[list, Callable] | None) -> dict[str, Any]:
    """Alternate uninstrumented and instrumented steps; returns per-repeat step times and stage sums."""
    cached = context if phase == "prefill" else context - 1      # decode T: T - 1 cached, the new token is the T-th
    ids = sweep.make_prompt(vocab, 1, cached, seed, device)
    for runner in {id(plain): plain, id(timed): timed}.values():
        runner.start(1, context + 1)
    tokens = {}
    if phase == "decode":
        for key, runner in (("plain", plain), ("timed", timed)):
            if hooks is not None and key == "timed":
                tokens[key] = tokens["plain"]
                continue
            tokens[key] = runner.prefill(ids)
    plain_ms, timed_ms = [], []
    stage_ms: dict[tuple[Any, str], list[float]] = defaultdict(list)
    stage_calls: dict[tuple[Any, str], int] = {}
    unattributed: list[float] = []

    def one(runner: Any, key: str, recorder: Recorder) -> float:
        if phase == "decode":
            runner.rewind(cached)
        recorder.begin_step()
        step_rec.begin_step()
        start = step_rec.mark()
        if phase == "prefill":
            runner.prefill(ids)
        else:
            runner.decode(tokens[key])
        end = step_rec.mark()
        step_rec.finish_step()
        return step_rec.ms(start, end)

    for r in range(warmup + repeats):
        if hooks is not None:
            handles = []
        t_plain = one(plain, "plain", rec)
        if hooks is not None:
            targets, attach = hooks
            handles = attach(rec, targets)
        t_timed = one(timed, "timed", rec)
        if hooks is not None:
            for handle in handles:
                handle.remove()
        if r < warmup:
            continue
        plain_ms.append(t_plain)
        timed_ms.append(t_timed)
        sums, calls, attributed = per_step_sums(rec, arm)
        for key, value in sums.items():
            stage_ms[key].append(value)
        stage_calls.update(calls)
        unattributed.append(t_timed - attributed)
    for runner in {id(plain): plain, id(timed): timed}.values():
        runner.stop()
    stage_ms[("model", UNATTRIBUTED)] = unattributed
    stage_calls[("model", UNATTRIBUTED)] = 0
    return {"plain_ms": plain_ms, "timed_ms": timed_ms, "stage_ms": stage_ms, "stage_calls": stage_calls}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def modeled(cfg: base.TransformerConfig, phase: str, context: int, stage: str) -> dict[str, float | str]:
    costs = opcount.stage_costs(cfg, phase, context)
    if stage == HF_ATTN_CORE:
        parts = [costs[s] for s in HF_ATTN_CORE_STAGES if s in costs]
        n = parts[0].tokens
        return {"mul_adds_per_token": sum(p.mul_adds for p in parts) / n,
                "nonlinear_per_token": sum(p.nonlinear for p in parts) / n,
                "bytes_per_token": sum(p.bytes for p in parts) / n}
    if stage in costs:
        return costs[stage].per_token()
    return {"mul_adds_per_token": "", "nonlinear_per_token": "", "bytes_per_token": ""}


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append to results/layer_timing.csv, skipping run ids already in it."""
    existing: set[str] = set()
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(LAYER_COLUMNS):
                raise SystemExit(f"{path} has other columns than this script writes; move it aside")
            existing = {line["run_id"] for line in reader}
    new = [row for row in rows if row["run_id"] not in existing]
    fresh = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LAYER_COLUMNS), extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerows(new)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stories15m")
    parser.add_argument("--arms", default="baseline,lut,hf")
    parser.add_argument("--lut-preset", default="all_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefill", default="64,128,256", help="prompt lengths T for prefill")
    parser.add_argument("--decode", default="64,128,256", help="contexts T for a decode step (T keys, T - 1 cached)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=sweep.REPO_ROOT / "results")
    args = parser.parse_args(argv)

    main_start = time.perf_counter()        # wall-clock durations are recorded in each run's summary (unit s)
    fp = precision.full_fp32()
    settings = sweep.Settings(results_dir=args.results_dir, seed=args.seed)
    source = sweep.resolve_model(args.model, local_cache.CHECKPOINT_DIR)
    load_start = time.perf_counter()
    loaded = sweep.load_model(source, args.device, settings)
    load_s = time.perf_counter() - load_start
    cfg = loaded.cfg
    trained = source.eval_context or cfg.context_length
    points = [("prefill", t) for t in _ints(args.prefill)] + [("decode", t) for t in _ints(args.decode)]
    platform = "gpu" if args.device.startswith("cuda") else "cpu"
    gpu_settings = resources.gpu_settings(args.device)
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        row = sweep.Row(line=0, enabled=True, name=f"layer_timing_{arm}", arm=arm, model=args.model,
                        lut_preset=args.lut_preset if arm == "lut" else None, device=args.device, batch=1,
                        context_lengths=[t for _, t in points], decode_tokens=1, repeats=args.repeats)
        variant = args.lut_preset if arm == "lut" else ("hf" if arm == "hf" else "baseline")
        method = ("forward hooks on transformers modules, CUDA events, coarser (attn_core and ffn_gate derived)"
                  if arm == "hf" else "CUDA events around each resolved stage function, one sync per step")
        if platform == "cpu":
            method = method.replace("CUDA events", "perf_counter")
        rec, step_rec = sweep_recorders(args.device)
        plain = sweep.make_runner(row, loaded)
        hooks = None
        if arm == "hf":
            timed = plain
            hooks = (hf_hook_targets(loaded.hf_model), add_hooks)
        else:
            timed = sweep.OurRunner(plain.host, plain.params, instrument_ops(plain.ops, rec), cfg)
        run = Run(f"layer_timing_{args.model}_{arm}", results_dir=args.results_dir,
                  devices={"prefill": args.device, "decode": args.device},
                  configs={"model": source.key, "weights": source.load_name, "arm": arm, "variant": variant,
                           "lut_preset": row.lut_preset, "dtype": sweep.row_dtype(row, loaded), "precision": fp,
                           "points": points, "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
                           "trained_context": trained, "method": method, "model_config": vars(cfg),
                           "decode_context": "T keys: T - 1 cached plus the new token"})
        commit = (run.git.get("commit") or "")[:10] + ("+dirty" if run.git.get("dirty") else "")
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows: list[dict[str, Any]] = []
        summary: dict[str, Any] = {"arm": arm, "variant": variant, "method": method, "points": {}}
        sampler = resources.ResourceSampler(args.device).start()      # GPU and host use over the whole arm
        for phase, context in points:
            beyond = context > trained
            label = f"{phase}_T{context}"
            try:
                with torch.inference_mode():
                    result = measure(arm, plain, timed, rec, step_rec, phase, context, loaded.vocab_size,
                                     args.seed, args.device, args.warmup, args.repeats, hooks)
            except Exception as error:      # an arm that refuses a context is skipped and recorded
                summary["points"][label] = {"refused": f"{type(error).__name__}: {error}"[:300]}
                print(f"{arm} {label}: refused ({type(error).__name__}: {error})", flush=True)
                continue
            step_plain = statistics.median(result["plain_ms"])
            step_timed = statistics.median(result["timed_ms"])
            overhead = 100.0 * (step_timed - step_plain) / step_plain
            run.log(kind="layer_timing", phase=phase, context=context, plain_ms=result["plain_ms"],
                    timed_ms=result["timed_ms"],
                    stages={f"{layer}:{stage}": ms for (layer, stage), ms in result["stage_ms"].items()})
            n_tokens = context if phase == "prefill" else 1
            point_rows = []
            for (layer, stage), values in result["stage_ms"].items():
                median = statistics.median(values)
                note = ""
                if stage == HF_ATTN_CORE:
                    note = "derived: attention module minus q/k/v/o (rope, scores, softmax, PV in one SDPA call)"
                elif arm == "hf" and stage == "ffn_gate":
                    note = "derived: MLP module minus its children (the elementwise product)"
                elif stage == "rope_tables":
                    note = "HF rotary module, once per step (our arms: inside unattributed)"
                elif stage == UNATTRIBUTED:
                    note = "step minus stages: heads split/merge, KV append, residual adds, rope table, mask, argmax, launch gaps"
                if beyond:
                    note = ("beyond trained context (256), timing only; " + note).rstrip("; ")
                point_rows.append({
                    "timestamp_utc": stamp, "run_id": run.run_id, "git_commit": commit, "flow": "pytorch",
                    "model": source.key, "arm": arm, "variant": variant, "device": args.device, "platform": platform,
                    "dtype": sweep.row_dtype(row, loaded), "phase": phase, "context": context,
                    "tokens_in_step": n_tokens, "beyond_trained_context": "yes" if beyond else "no",
                    "layer": layer, "stage": stage, "calls": result["stage_calls"].get((layer, stage), 0),
                    "ms_median": round(median, 6), "ms_p10": round(percentile(values, 0.1), 6),
                    "ms_p90": round(percentile(values, 0.9), 6), "share_of_step": round(median / step_timed, 6),
                    "step_ms_instrumented": round(step_timed, 6), "step_ms_uninstrumented": round(step_plain, 6),
                    "instrumentation_overhead_pct": round(overhead, 2),
                    **{k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in modeled(cfg, phase, context, stage).items()},
                    "repeats": args.repeats, "warmup": args.warmup, "method": method, "note": note})
            point_rows.sort(key=lambda r: -r["ms_median"])
            rows += point_rows
            top = [(f"{r['layer']}:{r['stage']}", r["ms_median"], r["share_of_step"]) for r in point_rows[:10]]
            summary["points"][label] = {"step_ms_uninstrumented": step_plain, "step_ms_instrumented": step_timed,
                                        "overhead_pct": overhead, "beyond_trained_context": beyond, "top10": top}
            print(f"{arm} {label}: step {step_plain:.3f} ms uninstrumented, {step_timed:.3f} ms instrumented "
                  f"(+{overhead:.1f}%); top: " + ", ".join(f"{n} {ms:.3f}" for n, ms, _ in top[:4]), flush=True)
        sampler.stop()
        summary["resources"] = sampler.summary()
        summary["gpu_settings"] = {k: v for k, v in gpu_settings.items() if k != "raw_q"}
        if gpu_settings.get("raw_q"):
            (run.dir / "gpu_settings.txt").write_text(gpu_settings["raw_q"], encoding="utf-8")
        summary["load_s"] = load_s              # the model is loaded once, before the first arm
        summary["main_wall_s_so_far"] = time.perf_counter() - main_start
        run.finish(summary)
        append_rows(Path(args.results_dir) / LAYER_TIMING_CSV, rows)
        print(f"{arm}: run {run.run_id}, {len(rows)} rows -> {Path(args.results_dir) / LAYER_TIMING_CSV}", flush=True)
    return 0


def sweep_recorders(device: str) -> tuple[Recorder, Recorder]:
    """One recorder for the stage marks, one for the step's own start and end marks."""
    return Recorder(device), Recorder(device)


if __name__ == "__main__":
    raise SystemExit(main())
