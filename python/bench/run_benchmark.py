"""Benchmark a workload for latency, throughput, power and energy, and log it as a run.

Today only the `matmul` workload exists: a dense matrix multiply used to check that timing,
power telemetry and run logging work on this machine before the model is built.

    cd python
    python -m bench.run_benchmark --workload matmul --device cuda --size 4096 --power
    python -m bench.run_benchmark --workload matmul --device cpu --size 1024 --power

Prefill and decode of the models themselves (Hugging Face, baseline and LUT arms, one or many at
once) are benchmarked by bench/sweep.py, from the list in configs/sweep/models.csv.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

from bench.runlog import Run
from bench.telemetry import PowerMonitor
from bench.timing import time_fn
from common import precision


def _matmul_workload(device: str, size: int, dtype_name: str) -> tuple[Callable[[], Any], dict[str, Any]]:
    """C = A @ B with A, B of shape (size, size). Returns the callable and its work counts."""
    import torch

    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(size, size, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    b = torch.randn(size, size, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)

    def run() -> Any:
        return a @ b

    work = {"flops_per_call": 2 * size**3, "macs_per_call": size**3}
    return run, work


def _power_window(
    fn: Callable[[], Any], device: str, target: str, seconds: float, warmup_s: float = 2.0,
    settle_tolerance: float = 0.03, max_warmup_s: float = 10.0,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Warm up, then call fn() back to back for at least `seconds` while sampling power.
    Returns (report, calls, warm-up record).

    The warm-up runs fn() unsampled for at least warmup_s seconds and, on a GPU, until three power
    readings in a row agree within settle_tolerance (at most max_warmup_s): bench/resources.py
    warm_up_until_steady. Without it the window started cold: GPU power climbed from 57 W to 330 W
    over the first 0.8 s and the reported average (329.5 W) was 8.6% below the steady 360.3 W (run
    20261119T060430Z_matmul_cuda_4096_float32, agent/STATE.md Next item 6). nvidia-smi's power.draw
    is itself a ~1 s average, which adds to the lag."""
    from bench.resources import warm_up_until_steady
    from bench.timing import _synchronize

    warm = warm_up_until_steady(fn, lambda: _synchronize(device), min_s=warmup_s, device=device,
                                tolerance=settle_tolerance, max_s=max_warmup_s) if warmup_s > 0 else \
        {"warmup_s": 0.0, "warmup_calls": 0, "settled": False, "readings_w": [], "tolerance": settle_tolerance,
         "min_s": 0.0, "max_s": max_warmup_s}
    monitor = PowerMonitor(target=target, enabled=True, note=f"workload on {device}")
    monitor.start()
    calls = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        fn()
        calls += 1
        if calls % 8 == 0:
            _synchronize(device)
    _synchronize(device)
    return monitor.stop(), calls, warm


def _idle_power(target: str, seconds: float) -> dict[str, Any]:
    """Sample power for `seconds` with nothing running. Called before the workload is built, so
    no CUDA context exists yet and the GPU is in its idle power state. (Measured after the timed
    loop, the GPU was still in its high-power state: 56.6 W "idle" against 17.6 to 19.9 W from
    nvidia-smi just before, run 20261119T051445Z_matmul_cuda_4096_float32.)"""
    monitor = PowerMonitor(target=target, enabled=True, note="idle baseline, before the workload")
    monitor.start()
    time.sleep(seconds)
    return monitor.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", choices=["matmul", "model"], default="matmul")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--size", type=int, default=2048, help="matrix size for the matmul workload")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                        help="float32 (default) is the base metric; the half-precision choices exist only "
                        "for this hardware smoke test and never reach the model")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--power", action="store_true", help="also measure power and energy")
    parser.add_argument("--power-seconds", type=float, default=10.0, help="length of the power window")
    parser.add_argument("--power-warmup-seconds", type=float, default=2.0,
                        help="unsampled run before the power window, at least this long (0: none)")
    parser.add_argument("--power-settle-tolerance", type=float, default=0.03,
                        help="keep warming up until 3 power readings agree within this fraction (0: fixed warm-up)")
    parser.add_argument("--power-max-warmup-seconds", type=float, default=10.0, help="upper bound on the warm-up")
    parser.add_argument(
        "--power-target",
        choices=["cpu", "cuda", "xrt", "all"],
        default=None,
        help="which power sources to sample (default: cpu for CPU runs, cuda for GPU runs). "
        "'all' also queries the U50 through XRT, which is slow and lowers the GPU sampling rate.",
    )
    parser.add_argument("--idle-seconds", type=float, default=3.0,
                        help="idle power baseline length, measured before the workload starts (0 to skip). "
                        "Leave a gap after a previous GPU run so the GPU has dropped back to idle.")
    parser.add_argument("--name", default=None, help="run name (default: workload_device)")
    args = parser.parse_args(argv)

    if args.workload == "model":
        raise SystemExit("The model workload is added in window 1.")
    precision.full_fp32()   # no TF32: float32 means full float32; recorded in meta.json by Run below

    name = args.name or f"{args.workload}_{args.device.replace(':', '')}_{args.size}_{args.dtype}"
    run = Run(
        name,
        configs={"workload": args.workload, "size": args.size, "dtype": args.dtype},
        devices={"workload": args.device},
    )
    summary: dict[str, Any] = {}
    target = args.power_target or ("cpu" if args.device == "cpu" else "cuda")

    # Idle first, before any tensor or CUDA context exists (see _idle_power).
    if args.power and args.idle_seconds > 0:
        idle = _idle_power(target, args.idle_seconds)
        run.log(kind="idle_power", report=idle)
        for source, data in idle.get("sources", {}).items():
            if data.get("average_power_watts") is not None:
                summary[f"idle_power_w.{source}"] = data["average_power_watts"]

    fn, work = _matmul_workload(args.device, args.size, args.dtype)
    stats = time_fn(fn, device=args.device, warmup=args.warmup, repeats=args.repeats)
    summary.update({
        "latency_ms_median": stats.median_ms,
        "latency_ms_p10": stats.p10_ms,
        "latency_ms_p90": stats.p90_ms,
        "throughput_gflops": work["flops_per_call"] / (stats.median_ms / 1000.0) / 1e9,
    })
    run.log(kind="timing", **stats.to_dict(include_samples=True))

    if args.power:
        report, calls, warm = _power_window(fn, args.device, target, args.power_seconds, args.power_warmup_seconds,
                                            args.power_settle_tolerance, args.power_max_warmup_seconds)
        run.save_power(report)
        run.log(kind="power_warmup", **warm)
        summary["power_window_calls"] = calls
        summary["power_warmup_s"] = warm["warmup_s"]
        summary["power_warmup_calls"] = warm["warmup_calls"]
        summary["power_warmup_settled"] = warm["settled"]
        for source, data in report.get("sources", {}).items():
            if data.get("energy_joules") is not None:
                summary[f"avg_power_w.{source}"] = data["average_power_watts"]
                summary[f"energy_per_call_mj.{source}"] = data["energy_joules"] / calls * 1000.0
            else:
                summary[f"power_unavailable.{source}"] = data.get("reason")

    run_dir = run.finish(summary)
    print(f"run: {run.run_id}")
    for key, value in summary.items():
        print(f"  {key}: {value:.4g}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
