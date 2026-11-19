"""CPU (RAPL), NVIDIA GPU (nvidia-smi) and Alveo (XRT) power telemetry.

Copied unchanged from Alpha's FPGA-co project (scripts/telemetry.py) so both projects
measure power the same way. Usage:

    monitor = PowerMonitor(target="all", enabled=True)   # "cpu", "cuda", "xrt" or "all"
    monitor.start()
    ...workload...
    report = monitor.stop()   # per-source energy (J) and average/peak power (W)

Sources with different scopes (CPU package, GPU board, FPGA board) are reported
separately and never summed into one number.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _read_float(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _find_rapl_energy_files() -> list[Path]:
    roots = (Path("/sys/class/powercap"), Path("/sys/devices/virtual/powercap"))
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("energy_uj"):
            if "rapl" not in str(path.parent).lower():
                continue
            resolved = path.resolve()
            if resolved not in seen and _read_float(path) is not None:
                seen.add(resolved)
                paths.append(path)

    # Package counters include nested core/DRAM domains on many systems.
    # Keep only the shallowest readable counter within each hierarchy.
    top_level: list[Path] = []
    selected_domains: list[Path] = []
    for path in sorted(paths, key=lambda candidate: len(candidate.resolve().parts)):
        domain = path.resolve().parent
        if any(parent == domain or parent in domain.parents for parent in selected_domains):
            continue
        top_level.append(path)
        selected_domains.append(domain)
    return top_level


def _rapl_counter_delta_joules(
    path: Path, starts: dict[Path, float], ends: dict[Path, float]
) -> float | None:
    start = starts.get(path)
    end = ends.get(path)
    if start is None or end is None:
        return None
    delta = end - start
    if delta < 0:
        maximum = _read_float(path.parent / "max_energy_range_uj")
        if maximum is not None:
            delta += maximum
    return delta / 1_000_000.0 if delta >= 0 else None


def _rapl_source_name(path: Path) -> str:
    return f"cpu_package_rapl:{path.parent.name}"


def _find_xrt_tool() -> str | None:
    for name in ("xrt-smi", "xbutil"):
        tool = shutil.which(name)
        if tool:
            return tool
    for candidate in (Path("/opt/xilinx/xrt/bin/xrt-smi"), Path("/opt/xilinx/xrt/bin/xbutil")):
        if candidate.is_file():
            return str(candidate)
    return None


def _query_nvidia_power(tool: str) -> list[float]:
    try:
        result = subprocess.run(
            [tool, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    values: list[float] = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return values


def _query_xrt_power(tool: str, device: str | None) -> list[float]:
    command = [tool, "examine", "--report", "electrical"]
    if device:
        command.extend(["--device", device])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    text = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        return []
    patterns = (
        r'"power_consumption_watts"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        r"^\s*Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*Watts",
        r"\bpower_consumption_watts\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return [float(match.group(1))]
    return []


def _integrate_samples(samples: list[tuple[float, float]], start: float, end: float) -> float | None:
    samples = [(t, w) for t, w in samples if start <= t <= end]
    if len(samples) < 2:
        return None
    if samples[-1][0] <= samples[0][0]:
        return None
    energy = 0.0
    for (time_a, power_a), (time_b, power_b) in zip(samples, samples[1:]):
        energy += max(0.0, time_b - time_a) * (power_a + power_b) / 2.0
    return energy


class PowerMonitor:
    """Sample available CPU, NVIDIA, or XRT power telemetry around a run."""

    def __init__(
        self,
        target: str,
        enabled: bool = False,
        note: str | None = None,
        interval_seconds: float = 0.1,
        xrt_device: str | None = None,
    ) -> None:
        self.target = target
        self.enabled = enabled
        self.note = note
        self.interval_seconds = interval_seconds
        self._start_time = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: dict[str, list[tuple[float, float]]] = {}
        self._sample_timestamps_ns: dict[str, list[int]] = {}
        self._source_status: dict[str, dict[str, Any]] = {}
        self._rapl_paths: list[Path] = []
        self._rapl_start: dict[Path, float] = {}
        self._nvidia_tool: str | None = None
        self._xrt_tool: str | None = None
        self._xrt_device = xrt_device or os.environ.get("XRT_DEVICE_INDEX")

    def start(self) -> None:
        if not self.enabled:
            self._start_time = time.perf_counter()
            return

        self._stop_event.clear()
        self._thread = None
        self._samples = {}
        self._sample_timestamps_ns = {}
        self._source_status = {}
        self._rapl_paths = []
        self._rapl_start = {}
        self._nvidia_tool = None
        self._xrt_tool = None
        poll_devices = False

        # CPU RAPL is included only for explicit CPU or all-source requests.
        if self.target in ("cpu", "all"):
            self._rapl_paths = _find_rapl_energy_files()
            if self._rapl_paths:
                for path in self._rapl_paths:
                    source = _rapl_source_name(path)
                    self._source_status[source] = {
                        "status": "sampling",
                        "scope": f"CPU package RAPL counter {path.parent.name}",
                        "reason": None,
                    }
            else:
                self._source_status["cpu_package_rapl"] = {
                    "status": "unavailable",
                    "scope": "CPU package RAPL energy",
                    "reason": "no readable RAPL energy counter under /sys/class/powercap/intel-rapl* "
                    "(the kernel uses this name for AMD CPUs too; energy_uj is root-only by default)",
                }

        if self.target in ("cuda", "all"):
            self._nvidia_tool = shutil.which("nvidia-smi")
            probe_values = (
                _query_nvidia_power(self._nvidia_tool) if self._nvidia_tool else []
            )
            if probe_values:
                for index, _ in enumerate(probe_values):
                    source = f"nvidia_gpu:{index}"
                    self._source_status[source] = {
                        "status": "sampling",
                        "scope": f"NVIDIA GPU {index} power",
                    }
                poll_devices = True
            else:
                self._source_status["nvidia_gpu"] = {
                    "status": "unavailable",
                    "scope": "NVIDIA GPU power",
                    "reason": "nvidia-smi is missing or no NVIDIA driver/GPU is available",
                }

        if self.target in ("xrt", "all"):
            emulation = os.environ.get("XCL_EMULATION_MODE")
            if emulation:
                self._source_status["xrt_alveo"] = {
                    "status": "unavailable",
                    "scope": "XRT-reported Alveo board power",
                    "reason": f"suppressed during XCL_EMULATION_MODE={emulation}",
                    "device": self._xrt_device or "default",
                }
            else:
                self._xrt_tool = _find_xrt_tool()
                probe_values = (
                    _query_xrt_power(self._xrt_tool, self._xrt_device)
                    if self._xrt_tool
                    else []
                )
                if probe_values:
                    self._source_status["xrt_alveo"] = {
                        "status": "sampling",
                        "scope": "XRT-reported Alveo board power",
                        "device": self._xrt_device or "default",
                    }
                    poll_devices = True
                else:
                    self._source_status["xrt_alveo"] = {
                        "status": "unavailable",
                        "scope": "XRT-reported Alveo board power",
                        "reason": "xrt-smi/xbutil could not return an electrical power reading",
                        "device": self._xrt_device or "default",
                    }

        # Probes above only establish availability; none are workload samples.
        self._start_time = time.perf_counter()
        self._rapl_start = {
            path: value
            for path in self._rapl_paths
            if (value := _read_float(path)) is not None
        }

        self._sample_active_devices()
        if poll_devices:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def _sample_active_devices(self) -> None:
        if any(source.startswith("nvidia_gpu:") for source in self._source_status):
            values = _query_nvidia_power(self._nvidia_tool) if self._nvidia_tool else []
            if values:
                self._record_samples("nvidia_gpu", values)
        if self._xrt_tool and self._source_status.get("xrt_alveo", {}).get("status") == "sampling":
            values = _query_xrt_power(self._xrt_tool, self._xrt_device)
            if values:
                self._record_samples("xrt_alveo", values)

    def _record_samples(self, source: str, values: list[float]) -> None:
        if not values:
            return
        if source == "nvidia_gpu" or len(values) > 1:
            keyed_values = [(f"{source}:{index}", value) for index, value in enumerate(values)]
        else:
            keyed_values = [(source, values[0])]

        for source_key, watts in keyed_values:
            # Capture both clocks after the device query returns. The monotonic
            # clock supports interval math; Unix nanoseconds identify acquisition.
            acquired_monotonic = time.perf_counter()
            acquired_unix_ns = time.time_ns()
            self._samples.setdefault(source_key, []).append((acquired_monotonic, watts))
            self._sample_timestamps_ns.setdefault(source_key, []).append(acquired_unix_ns)
            if source_key.startswith("nvidia_gpu:"):
                index = source_key.split(":", 1)[1]
                self._source_status.setdefault(
                    source_key,
                    {"status": "sampling", "scope": f"NVIDIA GPU {index} power"},
                )
            else:
                self._source_status.setdefault(
                    source_key,
                    {"status": "sampling", "scope": "XRT-reported Alveo board power"},
                )

    def _poll(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            if self._nvidia_tool:
                values = _query_nvidia_power(self._nvidia_tool)
                if values:
                    self._record_samples("nvidia_gpu", values)
            if self._xrt_tool and not os.environ.get("XCL_EMULATION_MODE"):
                values = _query_xrt_power(self._xrt_tool, self._xrt_device)
                if values:
                    self._record_samples("xrt_alveo", values)

    def stop(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "scope": self.note,
                "sources": {},
            }

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        self._sample_active_devices()
        end_time = time.perf_counter()

        rapl_end = {
            path: value
            for path in self._rapl_paths
            if (value := _read_float(path)) is not None
        }
        duration = max(0.0, end_time - self._start_time)
        source_reports: dict[str, dict[str, Any]] = {}

        for path in self._rapl_paths:
            source = _rapl_source_name(path)
            energy = _rapl_counter_delta_joules(path, self._rapl_start, rapl_end)
            report = dict(self._source_status.get(source, {}))
            if energy is None:
                report.update({
                    "status": "unavailable",
                    "energy_joules": None,
                    "average_power_watts": None,
                    "reason": "RAPL counter could not be read",
                })
            else:
                report.update({
                    "status": "ok",
                    "energy_joules": energy,
                    "average_power_watts": energy / duration if duration else None,
                    "reason": None,
                })
            source_reports[source] = report

        if not self._rapl_paths and "cpu_package_rapl" in self._source_status:
            source_reports["cpu_package_rapl"] = dict(self._source_status["cpu_package_rapl"])

        all_sample_sources = {
            source
            for source in (set(self._source_status) | set(self._samples))
            if source != "cpu_package_rapl" and not source.startswith("cpu_package_rapl:")
        }
        for source in sorted(all_sample_sources):
            samples = self._samples.get(source, [])
            timestamps = self._sample_timestamps_ns.get(source, [])
            in_window: list[tuple[float, float, int | None]] = []
            for index, (sample_time, watts) in enumerate(samples):
                if self._start_time <= sample_time <= end_time:
                    timestamp = timestamps[index] if index < len(timestamps) else None
                    in_window.append((sample_time, watts, timestamp))

            pairs = [(sample_time, watts) for sample_time, watts, _ in in_window]
            observed_span = pairs[-1][0] - pairs[0][0] if len(pairs) >= 2 else 0.0
            report = dict(self._source_status.get(source, {}))
            sample_rows = [
                {
                    "time_seconds": sample_time - self._start_time,
                    "timestamp_unix_ns": timestamp,
                    "watts": watts,
                }
                for sample_time, watts, timestamp in in_window
            ]
            common_fields = {
                "sample_count": len(in_window),
                "sample_interval_seconds": self.interval_seconds,
                "observed_interval_seconds": observed_span,
                "samples": sample_rows,
                "energy_scope": "trapezoidal integration between in-window samples only; no boundary extrapolation",
            }

            if len(in_window) < 2:
                if report.get("status") != "unavailable":
                    report["reason"] = "fewer than two in-window samples; sampled power and energy unavailable"
                report.update({
                    **common_fields,
                    "status": "unavailable",
                    "average_power_watts": None,
                    "peak_power_watts": None,
                    "energy_joules": None,
                })
            else:
                energy = _integrate_samples(pairs, self._start_time, end_time)
                if energy is None or observed_span <= 0.0:
                    report.update({
                        **common_fields,
                        "status": "unavailable",
                        "average_power_watts": None,
                        "peak_power_watts": None,
                        "energy_joules": None,
                        "reason": "in-window samples do not span a positive interval",
                    })
                else:
                    powers = [watts for _, watts in pairs]
                    report.update({
                        **common_fields,
                        "status": "ok",
                        "average_power_watts": energy / observed_span,
                        "peak_power_watts": max(powers),
                        "energy_joules": energy,
                        "reason": None,
                    })
            source_reports[source] = report

        measured_sources = [
            report for report in source_reports.values() if report.get("status") == "ok"
        ]
        if len(measured_sources) == 1:
            sole_source = measured_sources[0]
            total_energy = sole_source.get("energy_joules")
            average_power = sole_source.get("average_power_watts")
            peak_power = sole_source.get("peak_power_watts")
            aggregation_status = "single_source"
        elif measured_sources:
            # CPU package, GPU, and board power have distinct scopes; never add
            # them into one unlabeled total.
            total_energy = average_power = peak_power = None
            aggregation_status = "multiple_sources_not_aggregated"
        else:
            total_energy = average_power = peak_power = None
            aggregation_status = "no_measurement"

        return {
            "enabled": True,
            "status": "ok" if measured_sources else "unavailable",
            "scope": self.note,
            "measurement_window_us": duration * 1_000_000.0,
            "aggregation_status": aggregation_status,
            "total_energy_joules": total_energy,
            "average_power_watts": average_power,
            "peak_power_watts": peak_power,
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": sum(
                report.get("sample_count", 0) for report in source_reports.values()
            ),
            "sources": source_reports,
        }
