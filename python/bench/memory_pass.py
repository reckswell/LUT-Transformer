"""Memory-only pass: how much memory each arm uses, what it is, and where the LUT tables live
(2026-11-24, Alpha: memory usage and LUT location for every PyTorch setup; bench/memory.py).

    cd python
    python -m bench.memory_pass --max-eval-chars 1200000            # hf, baseline, lut + the training setups

Per arm, on a fresh load holding only that arm's weights (so other arms' copies don't inflate it):
  inventory   weights by place, the LUT storage inventory of the arm's ModelOps (bench/memory.py
              lut_inventory: every table tensor with its stage, shape, dtype, bytes and place, or
              "none (all base ops)"), and the harness buffers (evaluation token ids, the logits buffer).
  decode      --decode-windows windows decoded one token per step on the KV cache (bench/eval_decode.py's
              protocol). Every full window has the same shapes, so the peak over a few windows is the
              peak of the whole token set; the KV cache is measured at the end of a window, when it is fullest.
  window-forward  one pass over every window of the token set, each window at once (bench/sweep.py
              measure_perplexity, as bench/eval_power.py runs it).
  copies      torch.profiler over one decode window and over four forward windows: host-to-device
              copies per step / per window (0 means tables, weights and indices all stay on the GPU).
Training setups (unless --no-train), as the newest train_throughput run: ours (baseline host.train_step,
our AdamW state) and hf_fp32 (LlamaForCausalLM + autograd + torch AdamW), batch 64 x 256 random tokens,
one step to create the optimizer state, then --train-steps measured steps.
Each phase records bench/memory.py PhaseMemory (torch peak allocated / reserved, steady allocated,
this process's GPU memory from nvidia-smi, host RSS, pinned host memory) and the breakdown. One run
(results/runs/<id>_memory_<model>_chars<limit>), long rows in results/extra_results.csv (phase memory).
"""

from __future__ import annotations

import argparse
import csv
import gc
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from baseline import host as base
from bench import eval_decode, memory, sweep
from bench.runlog import Run
from common import precision
from reference import local_cache

EXTRA_COLUMNS = eval_decode.EXTRA_COLUMNS
MiB = memory.MiB


def device_memory_used_bytes() -> int | None:
    """nvidia-smi memory.used for the whole device (every process, the desktop included)."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        text = subprocess.run([tool, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout
        return int(float(text.splitlines()[0])) * MiB
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def fresh(source: sweep.ModelSource, device: str, settings: sweep.Settings, keep: str) -> sweep.Loaded:
    """A new load keeping only what one arm needs: keep = "hf" (the library model) or "ours" (our params)."""
    loaded = sweep.load_model(source, device, settings)
    if keep == "hf":
        loaded.params.clear()
    else:
        loaded.hf_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loaded


def gpu_bytes(tensors: Any) -> int:
    return memory.bytes_by_place(list(tensors)).get("GPU VRAM", 0)


def flush() -> None:
    """Free what the caller has just deleted (Python objects first, then the CUDA cache)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stories15m")
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--max-eval-chars", type=int, default=1_200_000)
    parser.add_argument("--arms", default="hf,baseline,lut")
    parser.add_argument("--lut-preset", default="all_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-windows", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--train-batch", type=int, default=64)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=sweep.REPO_ROOT / "results")
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    precision.full_fp32()
    settings = sweep.Settings(max_eval_chars=args.max_eval_chars, results_dir=args.results_dir)
    source = sweep.resolve_model(args.model, local_cache.CHECKPOINT_DIR)
    text = sweep.eval_text(args.dataset, settings)
    tokens, _ = sweep.eval_tokens(source, args.dataset, settings)
    device_idle = device_memory_used_bytes()
    probe = fresh(source, args.device, settings, "ours")
    window, stride = sweep.eval_window(source, probe.cfg)
    cfg = probe.cfg
    del probe
    flush()
    spans = list(eval_decode.windows(tokens.numel(), window, stride))
    run = Run(f"memory_{args.model}_chars{args.max_eval_chars}", results_dir=args.results_dir,
              devices={"inference": args.device, "train": args.device},
              configs={"model": source.key, "weights": source.load_name, "dataset": args.dataset,
                       "char_limit": args.max_eval_chars, "window": window, "stride": stride, "arms": args.arms,
                       "lut_preset": args.lut_preset, "decode_windows": args.decode_windows,
                       "train_steps": args.train_steps, "train_batch": args.train_batch, "model_config": vars(cfg)})
    summary: dict[str, Any] = {"scored_tokens": tokens.numel() - 1, "windows": len(spans),
                               "device_memory_used_before_load_bytes": device_idle, "arms": {}, "training": {}}
    from baseline import inference as base_inference

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        base_inference.clear_rope_tables()            # each arm builds (and is charged for) its own rope table
        loaded = fresh(source, args.device, settings, "hf" if arm == "hf" else "ours")
        dec = eval_decode.Decoder(arm, loaded, args.device, window, args.lut_preset)
        r = dec.runner
        if arm == "hf":
            weights = list(r.model.named_parameters()) + list(r.model.named_buffers())
            inventory = {"rows": [], "totals": {"lut_tables": 0, "lut_bytes_gpu_vram": 0, "lut_bytes_host_dram": 0,
                                                "lut_bytes_disk_mmap": 0, "lut_bytes_pinned": 0},
                         "lut_stages": 0, "label": "none (library model, no LUT ops)"}
        else:
            weights = list(r.params.items())
            inventory = memory.lut_inventory(r.ops)
        ids = tokens.to(args.device)
        buf = torch.empty(window, cfg.vocab_size, device=args.device, dtype=torch.float32)
        harness = gpu_bytes([("ids", ids), ("logits buffer", buf)])
        w_place = memory.bytes_by_place(weights)
        lut_gpu = inventory["totals"]["lut_bytes_gpu_vram"]
        out: dict[str, Any] = {"weights_bytes_by_place": w_place, "weights_tensors": len(weights),
                               "lut_inventory": inventory, "harness_bytes": harness, "dtype": dec.dtype, "phases": {}}
        with torch.no_grad():
            eval_decode.decode_window(dec, ids, *spans[1], buf)              # warm: cuBLAS handles, kernels
            with memory.PhaseMemory(args.device) as m:
                for b, e, f in spans[:args.decode_windows]:
                    eval_decode.decode_window(dec, ids, b, e, f, buf)
                    cache_state = r.past if arm == "hf" else r.cache        # fullest at the end of a window
                    kv = gpu_bytes(memory.walk_tensors(cache_state))
            out["kv_cache_bytes"] = kv
            if arm != "hf":                               # the rope table built during decode (a ROM, not a LUT op)
                inventory = memory.with_precomputed(inventory, memory.precomputed_tables())
                out["lut_inventory"] = inventory
            b = memory.breakdown(m.result, weights_gpu=w_place.get("GPU VRAM", 0), lut_gpu=lut_gpu, kv_cache_gpu=kv)
            b["harness_bytes"] = harness
            if b["activations_workspace_bytes"] is not None:
                b["activations_workspace_bytes"] = max(0, b["activations_workspace_bytes"] - harness)
            out["phases"]["decode"] = {**m.result, **b, "windows": args.decode_windows,
                                       "steps": sum(e - b0 - 1 for b0, e, _ in spans[:args.decode_windows])}
            dec.reset()
            if arm != "hf":
                r.cache.length = 0
            sweep.measure_perplexity(r.forward, tokens[:600], window=window, stride=stride, device=args.device)
            with memory.PhaseMemory(args.device) as m:
                stats = sweep.measure_perplexity(r.forward, tokens, window=window, stride=stride, device=args.device)
            kv_fwd = gpu_bytes(memory.walk_tensors(r.cache)) if arm != "hf" else 0   # allocated, unused by forward
            b = memory.breakdown(m.result, weights_gpu=w_place.get("GPU VRAM", 0), lut_gpu=lut_gpu, kv_cache_gpu=kv_fwd)
            b["harness_bytes"] = harness
            if b["activations_workspace_bytes"] is not None:
                b["activations_workspace_bytes"] = max(0, b["activations_workspace_bytes"] - harness)
            out["phases"]["window_forward"] = {**m.result, **b, "windows": len(spans), "perplexity": stats["perplexity"]}
            steps = spans[1][1] - spans[1][0] - 1
            copies = memory.htod_copies(lambda: eval_decode.decode_window(dec, ids, *spans[1], buf))
            copies["per_step_copies"] = copies["per_call_copies"] / steps if copies.get("per_call_copies") is not None else None
            out["htod_decode_window"] = {**copies, "steps": steps}
            fwd = memory.htod_copies(lambda: sweep.measure_perplexity(r.forward, tokens[:640], window=window,
                                                                     stride=stride, device=args.device))
            fwd["windows"] = sweep.count_windows(640, window, stride)
            fwd["per_window_copies"] = fwd["htod_copies"] / fwd["windows"] if fwd.get("htod_copies") is not None else None
            out["htod_window_forward"] = fwd
        summary["arms"][arm] = out
        d, f = out["phases"]["decode"], out["phases"]["window_forward"]
        print(f"{arm}: weights {sum(w_place.values()) / MiB:.1f} MiB ({', '.join(w_place)}); LUT: {inventory['label']}; "
              f"KV cache {kv / MiB:.2f} MiB; decode peak allocated {(d.get('torch_peak_allocated_bytes') or 0) / MiB:.1f} MiB, "
              f"process GPU {(d.get('process_gpu_peak_bytes') or 0) / MiB:.0f} MiB; window-forward peak allocated "
              f"{(f.get('torch_peak_allocated_bytes') or 0) / MiB:.1f} MiB; HtoD copies per decode step "
              f"{out['htod_decode_window'].get('per_step_copies')}, per forward window {fwd.get('per_window_copies')}",
              flush=True)
        del dec, loaded, ids, buf, r, weights, cache_state
        flush()

    if not args.no_train:
        gen = torch.Generator(device="cpu").manual_seed(0)
        batch = torch.randint(0, cfg.vocab_size, (args.train_batch, window + 1), generator=gen).to(args.device)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        # ours: the baseline host's hand-written forward/backward and our AdamW (its state in `state`)
        loaded = fresh(source, args.device, settings, "ours")
        params, ops, state = loaded.params, base.base_ops(cfg, "train"), {}
        with torch.no_grad():
            base.train_step(params, (inp, tgt), ops, state, cfg, lr=1e-4)        # creates the optimizer state
            with memory.PhaseMemory(args.device) as m:
                for _ in range(args.train_steps):
                    base.train_step(params, (inp, tgt), ops, state, cfg, lr=1e-4)
        w = gpu_bytes(params.items())
        opt = gpu_bytes(memory.walk_tensors(state))
        b = memory.breakdown(m.result, weights_gpu=w, optimizer_gpu=opt)
        b["harness_bytes"] = gpu_bytes([("batch", batch)])
        if b["activations_workspace_bytes"] is not None:
            b["activations_workspace_bytes"] = max(0, b["activations_workspace_bytes"] - b["harness_bytes"])
        summary["training"]["ours"] = {**m.result, **b, "batch": args.train_batch, "T": window, "steps": args.train_steps,
                                       "note": "gradients are made and used inside each train_step, so they sit in "
                                               "activations/workspace", "lut_label": "none (all base ops)"}
        del params, state, loaded
        flush()
        # hf_fp32: the library model with autograd and torch AdamW
        loaded = fresh(source, args.device, settings, "hf")
        model = loaded.hf_model.to(args.device).train()
        adamw = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)

        def hf_step() -> None:
            logits = model(input_ids=inp).logits
            loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            adamw.step()

        hf_step()
        adamw.zero_grad(set_to_none=True)
        with memory.PhaseMemory(args.device) as m:
            for _ in range(args.train_steps):
                hf_step()
                grads = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
                adamw.zero_grad(set_to_none=True)
        w = gpu_bytes(list(model.named_parameters()) + list(model.named_buffers()))
        opt = gpu_bytes(memory.walk_tensors(adamw.state))
        b = memory.breakdown(m.result, weights_gpu=w, optimizer_gpu=opt, grads_gpu=grads)
        b["harness_bytes"] = gpu_bytes([("batch", batch)])
        if b["activations_workspace_bytes"] is not None:
            b["activations_workspace_bytes"] = max(0, b["activations_workspace_bytes"] - b["harness_bytes"])
        summary["training"]["hf_fp32"] = {**m.result, **b, "batch": args.train_batch, "T": window, "steps": args.train_steps,
                                          "lut_label": "none (library model, no LUT ops)"}
        del model, adamw, loaded
        flush()
        for key, t in summary["training"].items():
            print(f"train {key}: peak allocated {(t.get('torch_peak_allocated_bytes') or 0) / MiB:.0f} MiB (weights "
                  f"{t['weights_bytes'] / MiB:.1f}, optimizer {t['optimizer_state_bytes'] / MiB:.1f}, gradients "
                  f"{t['gradient_bytes'] / MiB:.1f}, activations/workspace {(t['activations_workspace_bytes'] or 0) / MiB:.0f}); "
                  f"process GPU {(t.get('process_gpu_peak_bytes') or 0) / MiB:.0f} MiB", flush=True)
    summary["device_memory_used_after_bytes"] = device_memory_used_bytes()
    run.finish(summary)

    commit = (run.git.get("commit") or "")[:10] + ("+dirty" if run.git.get("dirty") else "")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    platform = "gpu" if args.device.startswith("cuda") else "cpu"
    rows: list[dict[str, Any]] = []

    def add(arm: str, phase: str, stage: str, metric: str, value: Any, unit: str, note: str = "") -> None:
        if value is None:
            return
        rows.append({"timestamp_utc": stamp, "run_id": run.run_id, "flow": "pytorch",
                     "variant": args.lut_preset if arm == "lut" else arm, "arm": arm, "model": source.key,
                     "platform": platform, "phase": phase, "context": "", "layer": "", "stage": stage, "metric": metric,
                     "value": round(value, 4) if isinstance(value, float) else value, "unit": unit, "git_commit": commit,
                     "note": note})

    fields = [("torch_peak_allocated_bytes", "torch_peak_allocated_mib"), ("torch_peak_reserved_bytes", "torch_peak_reserved_mib"),
              ("torch_allocated_steady_bytes", "torch_allocated_steady_mib"), ("process_gpu_peak_bytes", "process_gpu_peak_mib"),
              ("host_rss_peak_bytes", "host_rss_peak_mib"), ("host_rss_steady_bytes", "host_rss_steady_mib"),
              ("pinned_host_bytes", "pinned_host_mib"), ("weights_bytes", "weights_mib"), ("lut_table_bytes", "lut_gpu_mib"),
              ("kv_cache_bytes", "kv_cache_mib"), ("optimizer_state_bytes", "optimizer_state_mib"),
              ("gradient_bytes", "gradient_mib"), ("activations_workspace_bytes", "activations_workspace_mib"),
              ("context_and_allocator_bytes", "context_and_allocator_mib"), ("harness_bytes", "harness_mib")]
    for arm, out in summary["arms"].items():
        for phase_name, phase in out["phases"].items():
            for key, metric in fields:
                v = phase.get(key)
                add(arm, "memory", f"inference {phase_name.replace('_', '-')}", metric, v / MiB if v is not None else None, "MiB")
        t = out["lut_inventory"]["totals"]
        add(arm, "memory", "LUT storage", "lut_tables", t["lut_tables"], "tables", out["lut_inventory"]["label"])
        for key in ("lut_bytes_gpu_vram", "lut_bytes_host_dram", "lut_bytes_disk_mmap", "lut_bytes_pinned"):
            add(arm, "memory", "LUT storage", key, t[key], "bytes", out["lut_inventory"]["label"])
        add(arm, "memory", "copies", "htod_copies_per_decode_step", out["htod_decode_window"].get("per_step_copies"), "copies")
        add(arm, "memory", "copies", "htod_copies_per_forward_window", out["htod_window_forward"].get("per_window_copies"),
            "copies", "the window's input token ids, copied from the CPU by the evaluation loop")
    for key, t in summary["training"].items():
        for name, metric in fields:
            v = t.get(name)
            add("baseline" if key == "ours" else key, "memory", "training step", metric, v / MiB if v is not None else None,
                "MiB", f"{key}: batch {t['batch']} x {t['T']}")
    path = Path(args.results_dir) / "extra_results.csv"
    fresh_file = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTRA_COLUMNS)
        if fresh_file:
            writer.writeheader()
        writer.writerows(rows)
    print(f"run {run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
