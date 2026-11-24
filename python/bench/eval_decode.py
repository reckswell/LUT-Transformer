"""GPU (or CPU) inference by the FPGA kernel's protocol: every evaluation window decoded one token
per step on a KV cache, the same windows, steps and scored tokens as the HLS inference records
(2026-11-24, Alpha: GPU and FPGA inference must run the same tokens; coordinator: same decode-step
protocol, so the Paper table compares like with like).

    cd python
    python -m bench.eval_decode --max-eval-chars 187000      # 50,211 scored tokens, 392 windows, 99,868 steps
    python -m bench.eval_decode --max-eval-chars 1200000     # 321,526 scored tokens, 2,511 windows, 640,296 steps

Windows are bench/sweep.py measure_perplexity's: [b, b + W) for b = 0, S, 2S, ... (W = 256, S = 128
for stories15M) until one reaches the end; token t is scored once, in the first window that reaches
it. Per window, the HLS host (and this script) starts from an empty cache and runs decode steps for
positions 0 .. L-2 (L = window length): step p feeds token b + p and returns the logits that predict
token b + p + 1. Steps before the first scored position only fill the cache; their logits are
discarded. So a full window is 255 steps, and the step count is sum(L - 1) over the windows (99,868
on the slice, matching the HLS record's host.decode_steps).

Per arm (hf: transformers' own cache, logits_to_keep=1; baseline and lut: the host's decode_step on
a preallocated cache of W positions):
  1. a warm-up: windows decoded back to back for at least --warmup-seconds and until the power
     readings settle (bench/resources.py warm_up_until_steady);
  2. one timed pass over every window inside bench/resources.py ResourceSampler (nvidia-smi every
     100 ms). Per window: CUDA sync, decode steps (scored logits copied into a GPU buffer), CUDA sync;
     that interval is the window's decode time. Scoring (float64 logsumexp, top-5) runs after the
     window's timed interval, so it is outside the decode time but inside the power window.
Reported: decode steps, total decode seconds, ms per step (total / steps, the FPGA estimate's unit),
the median window's ms per step, mean power over the pass after its first second, energy per token
and per step (mean power x decode seconds / tokens or steps), and accuracy with a token-by-token
check against the FP32 reference files of bench/reference_nll.py (decode and window-at-once forward
round differently, so the NLLs agree to ~1e-5, not bit for bit). The windows and step counts are
checked against every HLS inference record that covers the same token set.
One run per invocation (results/runs/<id>_eval_decode_<model>_chars<limit>, per-token results in
decode_<arm>.pt), long rows in results/extra_results.csv (phase inference) for the workbooks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from tqdm import tqdm

from bench import resources, sweep
from bench.runlog import Run
from bench.telemetry import PowerMonitor
from common import precision
from reference import local_cache

EXTRA_COLUMNS = ["timestamp_utc", "run_id", "flow", "variant", "arm", "model", "platform", "phase", "context",
                 "layer", "stage", "metric", "value", "unit", "git_commit", "note"]
REFERENCE_DIR = sweep.REPO_ROOT.parent / "LUT-Transformer-local" / "benchmarking" / "pytorch" / "reference"
HLS_INFERENCE = sweep.REPO_ROOT / "results" / "hls" / "inference"


def windows(n: int, window: int, stride: int) -> Iterator[tuple[int, int, int]]:
    """(begin, end, first) per window, measure_perplexity's order: the window covers tokens
    [begin, end) and scores tokens first .. end-1 (each token after the first exactly once)."""
    prev_end, begin = 0, 0
    while True:
        end = min(begin + window, n)
        yield begin, end, max(prev_end, begin + 1)
        if end == n:
            return
        prev_end, begin = end, begin + stride


class Decoder:
    """One token per call on a KV cache: step(token (1, 1)) -> logits (1, V) predicting the next token."""

    def __init__(self, arm: str, loaded: sweep.Loaded, device: str, window: int, lut_preset: str) -> None:
        row = sweep.Row(line=0, enabled=True, name=f"eval_decode_{arm}", arm=arm, model="", device=device, batch=1,
                        lut_preset=lut_preset if arm == "lut" else None, context_lengths=[], decode_tokens=0,
                        repeats=0, benchmarks=[], datasets=[])
        self.runner = sweep.make_runner(row, loaded)
        self.dtype = sweep.row_dtype(row, loaded)
        self.hf = arm == "hf"
        if not self.hf:
            self.runner.start(1, window)              # preallocated (1, H, W, Dh) per layer, reused per window

    def reset(self) -> None:
        if self.hf:
            self.runner.past = None                   # transformers builds a fresh DynamicCache
        else:
            self.runner.cache.length = 0

    def step(self, token: Tensor) -> Tensor:
        r = self.runner
        if self.hf:
            out = r.model(input_ids=token, past_key_values=r.past, use_cache=True, **r.extra)
            r.past = out.past_key_values
            return out.logits[:, -1]
        return r.host.decode_step(r.params, token, r.cache, r.ops, r.cfg)


def decode_window(dec: Decoder, ids: Tensor, begin: int, end: int, first: int, buf: Tensor) -> int:
    """Steps for positions 0 .. end-begin-2; the logits of every step whose next token is scored
    (begin + p + 1 >= first) go to buf in order. Returns the number of rows written."""
    dec.reset()
    k = 0
    for p in range(end - begin - 1):
        z = dec.step(ids[begin + p:begin + p + 1].view(1, 1))
        if begin + p + 1 >= first:
            buf[k] = z[0]
            k += 1
    return k


def host_load() -> dict[str, Any]:
    """Load averages and every other process above 20 % of a core (ps), to flag CPU contention."""
    try:
        text = subprocess.run(["ps", "-eo", "pid,pcpu,etime,comm", "--sort=-pcpu", "--no-headers"],
                              capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        text = ""
    busy = []
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and float(parts[1]) >= 20.0 and int(parts[0]) != os.getpid():
            busy.append({"pid": int(parts[0]), "cpu_pct": float(parts[1]), "elapsed": parts[2], "command": parts[3]})
    return {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "loadavg_1_5_15": [round(x, 2) for x in os.getloadavg()], "busy_processes": busy}


def hls_records(scored: int) -> list[dict[str, Any]]:
    """The HLS inference records (whole token set, no window or step limit) that scored `scored` tokens."""
    out = []
    for path in sorted(HLS_INFERENCE.glob("*/summary.json")):
        try:
            s = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        host, sel = s.get("host") or {}, s.get("selection") or {}
        if host.get("scored_tokens") == scored and sel.get("max_windows", -1) in (-1, None) \
                and sel.get("max_steps", -1) in (-1, None) and not sel.get("first_window"):
            out.append({"run_id": s.get("run_id", path.parent.name), "windows": host.get("windows"),
                        "decode_steps": host.get("decode_steps")})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stories15m")
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--max-eval-chars", type=int, default=187_000)
    parser.add_argument("--arms", default="hf,baseline,lut")
    parser.add_argument("--lut-preset", default="all_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--max-windows", type=int, default=0, help="0: every window (a smaller number is a test)")
    parser.add_argument("--results-dir", type=Path, default=sweep.REPO_ROOT / "results")
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    precision.full_fp32()
    idle = {}
    if args.device.startswith("cuda") and args.idle_seconds > 0:       # before any CUDA context exists
        monitor = PowerMonitor(target="cuda", enabled=True, note="idle, before the model is loaded")
        monitor.start()
        time.sleep(args.idle_seconds)
        idle = {s: d.get("average_power_watts") for s, d in monitor.stop().get("sources", {}).items()}
    settings = sweep.Settings(max_eval_chars=args.max_eval_chars, results_dir=args.results_dir)
    source = sweep.resolve_model(args.model, local_cache.CHECKPOINT_DIR)
    text = sweep.eval_text(args.dataset, settings)
    tokens, _ = sweep.eval_tokens(source, args.dataset, settings)
    loaded = sweep.load_model(source, args.device, settings)
    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "hf" not in arm_list:                 # keep only what the arms use, so the peak memory is theirs alone
        loaded.hf_model = None
    elif arm_list == ["hf"]:
        loaded.params.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loaded_only_arm = len({"hf" if a == "hf" else "ours" for a in arm_list}) == 1
    window, stride = sweep.eval_window(source, loaded.cfg)
    spans = list(windows(tokens.numel(), window, stride))
    if args.max_windows:
        spans = spans[:args.max_windows]
    steps = sum(e - b - 1 for b, e, _ in spans)
    scored = sum(e - f for _, e, f in spans)
    complete = len(spans) == sweep.count_windows(tokens.numel(), window, stride)
    hls = hls_records(scored) if complete else []
    hls_check = [{**h, "windows_match": h["windows"] == len(spans), "steps_match": h["decode_steps"] == steps}
                 for h in hls]
    settings_gpu = resources.gpu_settings(args.device)
    run = Run(f"eval_decode_{args.model}_chars{args.max_eval_chars}", results_dir=args.results_dir,
              devices={"inference": args.device},
              configs={"model": source.key, "weights": source.load_name, "dataset": args.dataset,
                       "char_limit": args.max_eval_chars, "dataset_sha256": text.file_sha256, "window": window,
                       "stride": stride, "arms": args.arms, "lut_preset": args.lut_preset, "protocol": "decode-step",
                       "warmup_seconds": args.warmup_seconds, "max_windows": args.max_windows,
                       "loaded_only_arm": loaded_only_arm,
                       "model_config": vars(loaded.cfg)})
    summary: dict[str, Any] = {"protocol": "decode-step", "windows": len(spans), "decode_steps": steps,
                               "scored_tokens": scored, "token_ids": tokens.numel(), "complete": complete,
                               "eval_chars": text.chars, "hls_records": hls_check, "idle_power_w": idle,
                               "gpu_settings": {k: v for k, v in settings_gpu.items() if k != "raw_q"}}
    print(f"{len(spans):,} windows, {steps:,} decode steps, {scored:,} scored tokens; HLS records on the same set: "
          f"{[(h['run_id'], h['windows_match'], h['steps_match']) for h in hls_check] or 'none'}", flush=True)
    sync = (lambda: torch.cuda.synchronize(args.device)) if args.device.startswith("cuda") else (lambda: None)
    ids = tokens.to(args.device)
    buf = torch.empty(window, loaded.cfg.vocab_size, device=args.device, dtype=torch.float32)
    stem = f"{source.key.split('/')[-1]}_{args.dataset}_chars{args.max_eval_chars}_fp32"
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        dec = Decoder(arm, loaded, args.device, window, args.lut_preset)
        load_before = host_load()
        n = tokens.numel()
        nll = torch.full((n - 1,), float("nan"), dtype=torch.float64)
        top1 = torch.zeros(n - 1, dtype=torch.bool)
        top5 = torch.zeros(n - 1, dtype=torch.bool)
        with torch.no_grad():
            cycle = iter(range(10**9))
            warm = resources.warm_up_until_steady(
                lambda: decode_window(dec, ids, *spans[next(cycle) % min(8, len(spans))], buf), sync,
                min_s=args.warmup_seconds, device=args.device)
            window_s = []
            with resources.ResourceSampler(args.device) as sampler:
                pass_start = time.perf_counter()
                for b, e, f in tqdm(spans, desc=f"{arm} decode", unit="win", mininterval=30, leave=False):
                    sync()
                    t0 = time.perf_counter()
                    k = decode_window(dec, ids, b, e, f, buf)
                    sync()
                    window_s.append(time.perf_counter() - t0)
                    z = buf[:k].double()
                    targets = ids[f:e]
                    nll[f - 1:e - 1] = (torch.logsumexp(z, dim=-1) - z.gather(-1, targets[:, None])[:, 0]).cpu()
                    best5 = z.topk(5, dim=-1).indices
                    top1[f - 1:e - 1] = (best5[:, 0] == targets).cpu()
                    top5[f - 1:e - 1] = (best5 == targets[:, None]).any(dim=-1).cpu()
                pass_wall = time.perf_counter() - pass_start
        res = sampler.summary()
        mask = ~torch.isnan(nll)
        decode_s = sum(window_s)
        power = res.get("power_w_mean_steady") or res.get("power_draw_mean")
        per_window_ms = [1000.0 * s / (e - b - 1) for s, (b, e, _) in zip(window_s, spans)]
        total = float(nll[mask].sum())
        out: dict[str, Any] = {
            "scored_tokens": int(mask.sum()), "windows": len(spans), "decode_steps": steps,
            "decode_s": decode_s, "pass_wall_s": pass_wall, "ms_per_step": 1000.0 * decode_s / steps,
            "ms_per_step_median_window": statistics.median(per_window_ms),
            "ms_per_step_p10_window": sorted(per_window_ms)[len(per_window_ms) // 10],
            "ms_per_step_p90_window": sorted(per_window_ms)[(9 * len(per_window_ms)) // 10],
            "tokens_per_s": int(mask.sum()) / decode_s,
            "power_w_mean_steady": res.get("power_w_mean_steady"), "power_w_mean": res.get("power_draw_mean"),
            "power_w_max": res.get("power_draw_max"),
            "energy_per_token_mj": power * decode_s / int(mask.sum()) * 1000.0 if power else None,
            "energy_per_step_mj": power * decode_s / steps * 1000.0 if power else None,
            "perplexity": math.exp(total / int(mask.sum())), "nll_total": total,
            "top1_accuracy": float(top1[mask].double().mean()), "top5_accuracy": float(top5[mask].double().mean()),
            "warmup": warm, "resources": res, "dtype": dec.dtype,
            "host_load": {"before": load_before, "after": host_load()}}
        refs = {}
        for name in dict.fromkeys(["hf" if arm == "hf" else "baseline", "baseline"]):
            path = REFERENCE_DIR / f"{stem}_{name}.pt"
            if not path.is_file():
                continue
            ref = torch.load(path, weights_only=True)
            refs[name] = {
                "file": path.name, "token_ids_match": bool(torch.equal(ref["token_ids"], tokens)),
                "window_begin_match": bool(torch.equal(ref["window_begin"][mask], torch.cat(
                    [torch.full((e - f,), b, dtype=torch.int32) for b, e, f in spans]))),
                "nll_abs_diff_max": float((nll[mask] - ref["nll"][mask]).abs().max()),
                "nll_abs_diff_mean": float((nll[mask] - ref["nll"][mask]).abs().mean()),
                "top1_flag_disagreements": int((top1[mask] != ref["top1_correct"][mask]).sum()),
                "top5_flag_disagreements": int((top5[mask] != ref["top5_correct"][mask]).sum()),
                "reference_perplexity_same_tokens": math.exp(float(ref["nll"][mask].sum()) / int(mask.sum())),
                "reference_top1_same_tokens": float(ref["top1_correct"][mask].double().mean())}
        out["reference_check"] = refs
        summary[arm] = out
        torch.save({"token_ids": tokens, "nll": nll, "top1_correct": top1, "top5_correct": top5,
                    "window_s": torch.tensor(window_s, dtype=torch.float64)}, run.dir / f"decode_{arm}.pt")
        run.log(arm=arm, **{k: v for k, v in out.items() if k != "resources"})
        own = refs.get("hf" if arm == "hf" else "baseline", {})
        watts = f"{power:.1f} W, {out['energy_per_token_mj']:.3f} mJ/token" if power else "no power reading"
        print(f"{arm}: {out['scored_tokens']:,} tokens, {steps:,} steps in {decode_s:.1f} s = {out['ms_per_step']:.4f} ms/step "
              f"(median window {out['ms_per_step_median_window']:.4f}); {watts}; ppl {out['perplexity']:.6f}, "
              f"top-1 {out['top1_accuracy']:.4%}, top-5 {out['top5_accuracy']:.4%}; vs reference: max |dNLL| "
              f"{own.get('nll_abs_diff_max', float('nan')):.3g}, top-1/top-5 flag disagreements "
              f"{own.get('top1_flag_disagreements')}/{own.get('top5_flag_disagreements')}", flush=True)
        del dec
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if settings_gpu.get("raw_q"):
        (run.dir / "gpu_settings.txt").write_text(settings_gpu["raw_q"], encoding="utf-8")
    run.finish(summary)

    commit = (run.git.get("commit") or "")[:10] + ("+dirty" if run.git.get("dirty") else "")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        o = summary[arm]
        own = o["reference_check"].get("hf" if arm == "hf" else "baseline", {})
        note = (f"decode-step protocol (the FPGA kernel's): {o['scored_tokens']:,} scored tokens, {o['windows']:,} windows, "
                f"{o['decode_steps']:,} decode steps, one token per step on a KV cache; power = mean over the timed pass "
                f"after a {o['warmup']['warmup_s']:.1f} s warm-up")
        metrics = [("eval_scored_tokens", o["scored_tokens"], "tokens"), ("eval_windows", o["windows"], "windows"),
                   ("decode_steps", o["decode_steps"], "steps"), ("decode_s_total", o["decode_s"], "s"),
                   ("decode_ms_per_step", o["ms_per_step"], "ms"),
                   ("decode_ms_per_step_median_window", o["ms_per_step_median_window"], "ms"),
                   ("eval_tokens_per_s", o["tokens_per_s"], "tokens/s"),
                   ("power_w_mean_steady", o["power_w_mean_steady"], "W"), ("power_w_max", o["power_w_max"], "W"),
                   ("energy_per_step_mj", o["energy_per_step_mj"], "mJ/step"),
                   ("energy_per_scored_token_mj", o["energy_per_token_mj"], "mJ per scored token (secondary)"),
                   ("perplexity", o["perplexity"], ""), ("top1_accuracy", o["top1_accuracy"], "fraction"),
                   ("top5_accuracy", o["top5_accuracy"], "fraction"),
                   ("ref_nll_abs_diff_max", own.get("nll_abs_diff_max"), "nats"),
                   ("ref_top1_flag_disagreements", own.get("top1_flag_disagreements"), "tokens"),
                   ("ref_top5_flag_disagreements", own.get("top5_flag_disagreements"), "tokens")]
        for metric, value, unit in metrics:
            if value is not None:
                rows.append({"timestamp_utc": stamp, "run_id": run.run_id, "flow": "pytorch",
                             "variant": args.lut_preset if arm == "lut" else arm, "arm": arm, "model": source.key,
                             "platform": "gpu" if args.device.startswith("cuda") else "cpu", "phase": "inference",
                             "context": "", "layer": "", "stage": "whole model", "metric": metric,
                             "value": round(value, 6) if isinstance(value, float) else value, "unit": unit,
                             "git_commit": commit, "note": note})
    path = Path(args.results_dir) / "extra_results.csv"
    fresh = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTRA_COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)
    print(f"run {run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
