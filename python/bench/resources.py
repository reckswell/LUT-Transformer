"""GPU and host resource use during a benchmark phase, the GPU's power settings, and a power
warm-up that waits for steady state (2026-11-23: Alpha asked for power consumption "based on nvidia
power settings" and other resource utilization; agent/STATE.md Next item 6, the power ramp).

    from bench import resources
    settings = resources.gpu_settings()                   # power limits, persistence, P-state, clocks
    with resources.ResourceSampler("cuda") as sampler:    # samples every 100 ms in a thread
        ...workload...
    stats = sampler.summary(tokens=n)                     # means, maxima, energy, CPU time, peak RSS

GPU, from `nvidia-smi --query-gpu ... -lms 100` running beside the workload (nothing to install):
    power.draw            NVIDIA's board power, averaged by the driver over about 1 s ("Average
                          Power Draw" in nvidia-smi -q); power.draw.instant is the instantaneous one
    utilization.gpu       % of the sample period a kernel was running
    utilization.memory    % of the sample period device memory was read or written
    memory.used           MiB on the device (all processes, including the CUDA context)
    clocks.sm, clocks.mem MHz; temperature.gpu in C; pstate P0..P12
    clocks_event_reasons.active   bit mask of why clocks are held below maximum (decoded below;
                          "gpu_idle" is set whenever no kernel runs, so it is listed apart)
Host, from the process itself: user and system CPU seconds (resource.getrusage), the share of the
machine's CPU time that was busy over the window (/proc/stat), and the peak resident memory of this
process during the window (VmRSS sampled with the GPU readings).
torch: peak allocated and reserved CUDA memory in the window (reset at its start).

Sources keep their scopes: GPU board power is never added to CPU package power (agent.md 6).
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

GPU_FIELDS = ("timestamp", "power.draw", "power.draw.instant", "utilization.gpu", "utilization.memory",
              "memory.used", "clocks.sm", "clocks.mem", "temperature.gpu", "pstate", "clocks_event_reasons.active")
NUMERIC = ("power.draw", "power.draw.instant", "utilization.gpu", "utilization.memory", "memory.used", "clocks.sm",
           "clocks.mem", "temperature.gpu")
SETTINGS_FIELDS = ("name", "driver_version", "persistence_mode", "power.limit", "power.default_limit",
                   "power.min_limit", "power.max_limit", "enforced.power.limit", "clocks.max.sm", "clocks.max.memory",
                   "compute_mode", "pstate")
# clocks_event_reasons bits (nvidia-smi -q "Clocks Event Reasons"; NVML nvmlClocksEventReason*)
EVENT_BITS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
              0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
              0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting", 0x200: "board_limit",
              0x400: "reliability"}


def decode_event_reasons(mask: int) -> list[str]:
    names = [name for bit, name in EVENT_BITS.items() if mask & bit]
    rest = mask & ~sum(EVENT_BITS)
    return names + ([f"0x{rest:x}"] if rest else [])


def _gpu_index(device: str) -> str:
    return device.split(":")[1] if ":" in device else "0"


def gpu_settings(device: str = "cuda") -> dict[str, Any]:
    """The GPU's power and clock settings once: power limits (current, default, enforced, min, max),
    persistence mode, compute mode, P-state, maximum clocks; plus the raw `nvidia-smi -q -d
    POWER,CLOCK,PERFORMANCE` text. {} with a reason when nvidia-smi isn't there."""
    tool = shutil.which("nvidia-smi")
    if not tool or not device.startswith("cuda"):
        return {"available": False, "reason": "no nvidia-smi or not a cuda device"}
    index = _gpu_index(device)
    out: dict[str, Any] = {"available": True, "queried_utc": datetime.now(timezone.utc).isoformat()}
    try:
        line = subprocess.run([tool, "-i", index, f"--query-gpu={','.join(SETTINGS_FIELDS)}",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        values = [v.strip() for v in line.strip().split(",")]
        if len(values) == len(SETTINGS_FIELDS):
            out.update(dict(zip(SETTINGS_FIELDS, values)))
        out["raw_q"] = subprocess.run([tool, "-i", index, "-q", "-d", "POWER,CLOCK,PERFORMANCE"], capture_output=True,
                                      text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        out.update(available=False, reason=str(error))
    return out


def _cpu_times() -> tuple[int, int] | None:
    """(busy, total) jiffies of the whole machine from /proc/stat."""
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            parts = [int(x) for x in handle.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)       # idle + iowait
    return sum(parts) - idle, sum(parts)


def _rss_mib() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        return None
    return None


class ResourceSampler:
    """Samples the GPU (nvidia-smi streaming every interval_ms) and this process's RSS in a thread
    while a phase runs; summary() gives the window's means and maxima. Usable as a context manager."""

    def __init__(self, device: str, interval_ms: int = 100) -> None:
        self.device, self.interval_ms = device, interval_ms
        self.rows: list[dict[str, Any]] = []
        self.rss: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> "ResourceSampler":
        self.wall0 = time.perf_counter()
        self.usage0 = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu0 = _cpu_times()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.torch_cuda = False
        if self.device.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.reset_peak_memory_stats(self.device)
                    self.torch_cuda = True
            except Exception:                    # resource accounting must never break a benchmark
                pass
            tool = shutil.which("nvidia-smi")
            if tool:
                self._proc = subprocess.Popen([tool, "-i", _gpu_index(self.device), f"--query-gpu={','.join(GPU_FIELDS)}",
                                               "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
                                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                self._threads.append(threading.Thread(target=self._read_gpu, daemon=True))
        self._threads.append(threading.Thread(target=self._sample_rss, daemon=True))
        for thread in self._threads:
            thread.start()
        return self

    def _read_gpu(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            values = [v.strip() for v in line.split(",")]
            if len(values) != len(GPU_FIELDS):
                continue
            row: dict[str, Any] = {"t": time.perf_counter() - self.wall0}
            for field, value in zip(GPU_FIELDS, values):
                if field in NUMERIC:
                    try:
                        row[field] = float(value)
                    except ValueError:
                        row[field] = None
                elif field == "clocks_event_reasons.active":
                    try:
                        row[field] = int(value, 16)
                    except ValueError:
                        row[field] = None
                else:
                    row[field] = value
            self.rows.append(row)

    def _sample_rss(self) -> None:
        while not self._stop.is_set():
            value = _rss_mib()
            if value is not None:
                self.rss.append(value)
            self._stop.wait(self.interval_ms / 1000.0)

    def stop(self) -> "ResourceSampler":
        self.wall = time.perf_counter() - self.wall0
        self.usage1 = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu1 = _cpu_times()
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        for thread in self._threads:
            thread.join(timeout=2)
        self.torch_peak = {}
        if self.torch_cuda:
            try:
                import torch

                self.torch_peak = {"torch_peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
                                   "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20}
            except Exception:
                pass
        return self

    def __enter__(self) -> "ResourceSampler":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def summary(self, tokens: int | None = None, steady_after_s: float = 1.0) -> dict[str, Any]:
        """Means and maxima over the window. Power: mean and max of power.draw, and the energy as
        mean power x window (J) and per token (mJ). power.draw lags by the driver's ~1 s averaging,
        so power_w_mean_steady leaves out the first steady_after_s seconds of the window."""
        out: dict[str, Any] = {"window_s": self.wall, "gpu_samples": len(self.rows), "interval_ms": self.interval_ms,
                               "started_utc": self.started_utc}
        for field in NUMERIC:
            vals = [r[field] for r in self.rows if r.get(field) is not None]
            key = field.replace(".", "_")
            if vals:
                out[f"{key}_mean"] = sum(vals) / len(vals)
                out[f"{key}_max"] = max(vals)
        steady = [r["power.draw"] for r in self.rows if r.get("power.draw") is not None and r["t"] >= steady_after_s]
        if steady:
            out["power_w_mean_steady"] = sum(steady) / len(steady)
            out["power_steady_samples"] = len(steady)
        if "power_draw_mean" in out:
            out["energy_j"] = out["power_draw_mean"] * self.wall
            if tokens:
                out["energy_per_token_mj"] = out["energy_j"] / tokens * 1000.0
        masks = [r["clocks_event_reasons.active"] for r in self.rows if r.get("clocks_event_reasons.active") is not None]
        seen = sorted({name for m in masks for name in decode_event_reasons(m)})
        out["clock_event_reasons_seen"] = ";".join(seen)
        out["throttled"] = any(name not in ("gpu_idle", "reliability") for name in seen)
        out["pstates_seen"] = ";".join(sorted({str(r.get("pstate")) for r in self.rows if r.get("pstate")}))
        u0, u1 = self.usage0, self.usage1
        out["process_cpu_user_s"] = u1.ru_utime - u0.ru_utime
        out["process_cpu_system_s"] = u1.ru_stime - u0.ru_stime
        out["process_cpu_cores_mean"] = (out["process_cpu_user_s"] + out["process_cpu_system_s"]) / self.wall \
            if self.wall > 0 else None
        if self.cpu0 and self.cpu1 and self.cpu1[1] > self.cpu0[1]:
            out["host_cpu_busy_pct"] = 100.0 * (self.cpu1[0] - self.cpu0[0]) / (self.cpu1[1] - self.cpu0[1])
        if self.rss:
            out["process_rss_peak_mib"] = max(self.rss)
        out["process_rss_lifetime_peak_mib"] = u1.ru_maxrss / 1024.0
        out.update(self.torch_peak)
        return out


def warm_up_until_steady(fn: Callable[[], Any], sync: Callable[[], None], *, min_s: float, device: str,
                         tolerance: float = 0.03, max_s: float = 10.0, window: int = 3) -> dict[str, Any]:
    """Run fn() back to back (sync() after each) for at least min_s seconds, then, on a GPU, keep
    going until the last `window` power readings (nvidia-smi power.draw, every ~0.25 s) agree within
    `tolerance` of their mean, or max_s is reached. Returns what happened, for the run record:
    {"warmup_s", "warmup_calls", "settled", "readings_w"}. tolerance 0 means a fixed min_s warm-up."""
    tool = shutil.which("nvidia-smi") if device.startswith("cuda") else None
    readings: list[float] = []
    calls, start = 0, time.perf_counter()
    next_reading = start
    settled = tolerance <= 0 or tool is None
    while True:
        fn()
        sync()
        calls += 1
        now = time.perf_counter()
        if tool and tolerance > 0 and now >= next_reading:
            next_reading = now + 0.25
            try:
                text = subprocess.run([tool, "-i", _gpu_index(device), "--query-gpu=power.draw",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                readings.append(float(text.strip().splitlines()[0]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            last = readings[-window:]
            if len(last) == window:
                mean = sum(last) / window
                settled = mean > 0 and max(abs(x - mean) for x in last) <= tolerance * mean
        elapsed = now - start
        if elapsed >= max_s or (elapsed >= min_s and settled):
            break
    return {"warmup_s": time.perf_counter() - start, "warmup_calls": calls, "settled": settled,
            "readings_w": readings[-10:], "tolerance": tolerance, "min_s": min_s, "max_s": max_s}
