"""Inference on a fixed token set with a steady power reading: the same tokens the FPGA kernel ran
(2026-11-24, Alpha: GPU and FPGA inference must run the same number of tokens).

    cd python
    python -m bench.eval_power --max-eval-chars 187000          # the 50,211-token screening slice, all arms

The perplexity pass is bench/sweep.py measure_perplexity (windows of 256 tokens every 128, every
token after the first scored once), so accuracy and NLL equal the sweep's on the same text. One
pass over the slice takes about 1 s on the GPU, shorter than NVIDIA's ~1 s power averaging, so a
single pass gives no trustworthy power. Per arm:
  1. one pass for accuracy (top-1, top-5, NLL, perplexity);
  2. a power warm-up: passes until at least --warmup-seconds and until the power readings settle
     (bench/resources.py warm_up_until_steady);
  3. passes back to back for at least --min-seconds while bench/resources.py ResourceSampler samples
     power, utilization, memory and clocks.
Reported: the median pass time (the time to run the whole token set once), the steady mean power
over the repeated passes, energy per token (steady power x median pass time / scored tokens), and
whether every pass gave the same NLL. Idle power is sampled before the CUDA context exists.
One run per invocation (results/runs/<id>_eval_power_<model>_chars<limit>), long rows in
results/extra_results.csv (phase inference) for the workbooks.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from bench import resources, sweep
from bench.runlog import Run
from bench.telemetry import PowerMonitor
from common import precision
from reference import local_cache

EXTRA_COLUMNS = ["timestamp_utc", "run_id", "flow", "variant", "arm", "model", "platform", "phase", "context",
                 "layer", "stage", "metric", "value", "unit", "git_commit", "note"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stories15m")
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--max-eval-chars", type=int, default=187_000)
    parser.add_argument("--arms", default="hf,baseline,lut")
    parser.add_argument("--lut-preset", default="all_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--min-seconds", type=float, default=8.0)
    parser.add_argument("--min-passes", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
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
    window, stride = sweep.eval_window(source, loaded.cfg)
    settings_gpu = resources.gpu_settings(args.device)
    run = Run(f"eval_power_{args.model}_chars{args.max_eval_chars}", results_dir=args.results_dir,
              devices={"inference": args.device},
              configs={"model": source.key, "weights": source.load_name, "dataset": args.dataset,
                       "char_limit": args.max_eval_chars, "dataset_sha256": text.file_sha256, "window": window,
                       "stride": stride, "arms": args.arms, "lut_preset": args.lut_preset,
                       "warmup_seconds": args.warmup_seconds, "min_seconds": args.min_seconds,
                       "min_passes": args.min_passes, "protocol": "window-forward",
                       "model_config": vars(loaded.cfg)})
    summary: dict[str, Any] = {"scored_tokens": tokens.numel() - 1, "eval_chars": text.chars, "idle_power_w": idle,
                               "gpu_settings": {k: v for k, v in settings_gpu.items() if k != "raw_q"}}
    sync = (lambda: torch.cuda.synchronize(args.device)) if args.device.startswith("cuda") else (lambda: None)
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        row = sweep.Row(line=0, enabled=True, name=f"eval_power_{arm}", arm=arm, model=args.model,
                        lut_preset=args.lut_preset if arm == "lut" else None, device=args.device, batch=1,
                        context_lengths=[], decode_tokens=0, repeats=0, benchmarks=["perplexity"], datasets=[args.dataset])
        runner = sweep.make_runner(row, loaded)

        def one_pass() -> dict[str, Any]:
            with torch.no_grad():
                return sweep.measure_perplexity(runner.forward, tokens, window=window, stride=stride, device=args.device)

        first = one_pass()                                   # accuracy, and a first warm pass
        warm = resources.warm_up_until_steady(one_pass, sync, min_s=args.warmup_seconds, device=args.device)
        passes, nlls = [], []
        with resources.ResourceSampler(args.device) as sampler:
            deadline = time.perf_counter() + args.min_seconds
            while time.perf_counter() < deadline or len(passes) < args.min_passes:
                t0 = time.perf_counter()
                stats = one_pass()
                sync()
                passes.append(time.perf_counter() - t0)
                nlls.append(stats["nll_total"])
        res = sampler.summary()
        median = statistics.median(passes)
        power = res.get("power_w_mean_steady") or res.get("power_draw_mean")
        n = first["scored_tokens"]
        out = {"perplexity": first["perplexity"], "nll_total": first["nll_total"], "top1_accuracy": first["top1_accuracy"],
               "top5_accuracy": first["top5_accuracy"], "scored_tokens": n, "passes": len(passes),
               "pass_s_median": median, "pass_s_min": min(passes), "pass_s_max": max(passes),
               "tokens_per_s": n / median, "power_w_mean_steady": res.get("power_w_mean_steady"),
               "power_w_mean": res.get("power_draw_mean"), "power_w_max": res.get("power_draw_max"),
               "energy_per_token_mj": power * median / n * 1000.0 if power else None,
               "nll_identical_across_passes": all(x == first["nll_total"] for x in nlls),
               "warmup": warm, "resources": res, "dtype": sweep.row_dtype(row, loaded)}
        summary[arm] = out
        run.log(arm=arm, **{k: v for k, v in out.items() if k != "resources"})
        watts = f"{power:.1f} W steady, {out['energy_per_token_mj']:.3f} mJ/token" if power else "no power reading"
        print(f"{arm}: {n:,} tokens, top-1 {out['top1_accuracy']:.4%}, ppl {out['perplexity']:.6f}; {len(passes)} passes, "
              f"median {median:.3f} s ({out['tokens_per_s']:,.0f} tok/s); {watts}; "
              f"same NLL every pass: {out['nll_identical_across_passes']}", flush=True)
    if settings_gpu.get("raw_q"):
        (run.dir / "gpu_settings.txt").write_text(settings_gpu["raw_q"], encoding="utf-8")
    run.finish(summary)

    commit = (run.git.get("commit") or "")[:10] + ("+dirty" if run.git.get("dirty") else "")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        o = summary[arm]
        variant = args.lut_preset if arm == "lut" else arm
        note = (f"{o['scored_tokens']:,} scored tokens (TinyStories V2 valid, {text.chars:,} chars); {o['passes']} passes "
                f"after a {o['warmup']['warmup_s']:.1f} s warm-up; power = steady mean over the passes")
        for metric, value, unit in (("eval_scored_tokens", o["scored_tokens"], "tokens"), ("perplexity", o["perplexity"], ""),
                                    ("top1_accuracy", o["top1_accuracy"], "fraction"), ("top5_accuracy", o["top5_accuracy"], "fraction"),
                                    ("eval_pass_s_median", o["pass_s_median"], "s"), ("eval_passes", o["passes"], "passes"),
                                    ("eval_tokens_per_s", o["tokens_per_s"], "tokens/s"),
                                    ("power_w_mean_steady", o["power_w_mean_steady"], "W"), ("power_w_max", o["power_w_max"], "W"),
                                    ("energy_per_scored_token_mj", o["energy_per_token_mj"], "mJ per scored token")):
            if value is not None:
                rows.append({"timestamp_utc": stamp, "run_id": run.run_id, "flow": "pytorch", "variant": variant, "arm": arm,
                             "model": source.key, "platform": "gpu" if args.device.startswith("cuda") else "cpu",
                             "phase": "inference", "context": "", "layer": "", "stage": "whole model", "metric": metric,
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
