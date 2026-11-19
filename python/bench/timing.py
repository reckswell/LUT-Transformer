"""Wall-clock timing for CPU and GPU workloads.

time_fn() runs a function a few times untimed (warm-up), then times it `repeats` times and
returns summary statistics. For CUDA it synchronizes the device before starting and after
each call, so the time covers the GPU work rather than just the kernel launch.

torch is imported lazily so this module also works without it (CPU-only timing).
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class TimingStats:
    """Summary of repeated timings, all in milliseconds."""

    repeats: int
    warmup: int
    device: str
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p10_ms: float
    p90_ms: float
    samples_ms: list[float]

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_samples:
            data.pop("samples_ms")
        return data


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already sorted list (fraction in [0, 1])."""
    if not sorted_values:
        return math.nan
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(samples_ms: list[float], warmup: int, device: str) -> TimingStats:
    """Turn raw per-call times (ms) into TimingStats."""
    ordered = sorted(samples_ms)
    return TimingStats(
        repeats=len(samples_ms),
        warmup=warmup,
        device=device,
        median_ms=statistics.median(ordered),
        mean_ms=statistics.fmean(ordered),
        std_ms=statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        min_ms=ordered[0],
        max_ms=ordered[-1],
        p10_ms=_percentile(ordered, 0.10),
        p90_ms=_percentile(ordered, 0.90),
        samples_ms=list(samples_ms),
    )


def _synchronize(device: str) -> None:
    """Wait for queued GPU work to finish so wall-clock time covers it."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def time_fn(
    fn: Callable[[], Any],
    *,
    device: str = "cpu",
    warmup: int = 3,
    repeats: int = 10,
    after_call: Callable[[], Any] | None = None,
) -> TimingStats:
    """Time fn() on `device` ("cpu" or "cuda[:n]").

    Runs fn() `warmup` times untimed (fills caches, triggers lazy CUDA init and kernel
    selection), then `repeats` timed calls with a device synchronize around each one.

    after_call(), if given, runs after every warm-up and timed call, after that call's stop
    timestamp and before the next call's start, so whatever it does (a progress bar update,
    bench/progress.py) is never inside a timed region.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    for _ in range(warmup):
        fn()
        if after_call is not None:
            _synchronize(device)
            after_call()
    _synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        fn()
        _synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        if after_call is not None:
            after_call()
    return summarize(samples_ms, warmup, device)


class Stopwatch:
    """Context manager for timing one block, such as a whole training run.

        with Stopwatch("cuda") as sw:
            train(...)
        print(sw.elapsed_s)
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.elapsed_s = math.nan
        self._start = 0.0

    def __enter__(self) -> "Stopwatch":
        _synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        _synchronize(self.device)
        self.elapsed_s = time.perf_counter() - self._start
