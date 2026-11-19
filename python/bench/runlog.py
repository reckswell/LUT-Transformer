"""Run directories and metadata, so every number traces back to its code, config and hardware.

    run = Run("baseline_fp32_gpu", configs={"model": ..., "variant": ..., "hw": ...},
              devices={"train": "cuda", "prefill": "cuda", "decode": "cpu"})
    run.log(step=10, loss=3.2, tokens_per_s=51000)
    run.save_power(monitor.stop())
    run.finish({"val_loss": 2.9, "train_time_s": 812.4})

Creates results/runs/<UTC time>_<name>/ with meta.json, metrics.jsonl, power.json and
summary.json, and appends one row to results/summary.csv. See results/README.md.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]   # python/bench/runlog.py -> repository root


def _run_cmd(args: list[str]) -> str | None:
    """Run a command and return stripped stdout, or None if it fails or isn't installed."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, cwd=REPO_ROOT, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_info() -> dict[str, Any]:
    """Current commit, branch, and whether tracked files have uncommitted changes. Untracked files
    (such as earlier runs' result folders) don't count, so one run doesn't mark the next dirty."""
    status = _run_cmd(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": _run_cmd(["git", "rev-parse", "HEAD"]),
        "branch": _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def _cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return platform.processor() or None
    match = re.search(r"^model name\s*:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else (platform.processor() or None)


def _gpu_info() -> list[dict[str, str]]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,power.limit",
        "--format=csv,noheader",
    ])
    gpus = []
    for line in (out or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            gpus.append(dict(zip(("index", "name", "driver", "memory_total", "power_limit"), parts)))
    return gpus


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
        # How float32 math runs (common/precision.py): full float32 unless TF32 is allowed.
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if torch.cuda.is_available():
        info["cudnn"] = torch.backends.cudnn.version()
        info["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def system_info() -> dict[str, Any]:
    """Host, OS, CPU, GPU, driver, Python and PyTorch versions."""
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpus": _gpu_info(),
        "python": sys.version.split()[0],
        "torch": _torch_info(),
    }


def _jsonable(value: Any) -> Any:
    """Fallback for json.dump: tensors and numpy scalars become plain numbers or lists."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), inner, out)
    else:
        out[prefix] = value


class Run:
    """One experiment run: a directory of metadata, metrics, power and a summary."""

    def __init__(
        self,
        name: str,
        *,
        configs: dict[str, Any] | None = None,
        devices: dict[str, str] | None = None,
        notes: str | None = None,
        results_dir: Path | str | None = None,
    ) -> None:
        self.name = name
        self.devices = devices or {}
        self.results_dir = Path(results_dir) if results_dir else REPO_ROOT / "results"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
        self.run_id = f"{stamp}_{safe_name}"
        self.dir = self.results_dir / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.git = git_info()
        self.meta = {
            "run_id": self.run_id,
            "name": name,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "git": self.git,
            "devices": self.devices,
            "configs": configs or {},
            "notes": notes,
            "system": system_info(),
        }
        self._write_json("meta.json", self.meta)

    def _write_json(self, filename: str, data: Any) -> None:
        with open(self.dir / filename, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=_jsonable)

    def log(self, **metrics: Any) -> None:
        """Append one record (for example one training step) to metrics.jsonl."""
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), **metrics}
        with open(self.dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_jsonable) + "\n")

    def save_power(self, report: dict[str, Any], window: str | None = None) -> None:
        """Save a PowerMonitor.stop() report as power.json. Raw samples go to a separate, gitignored
        file (power_samples.json). A run with several power windows names each one: window
        "decode_T512" writes power_decode_T512.json and power_samples_decode_T512.json."""
        suffix = f"_{window}" if window else ""
        samples = {
            source: data.get("samples", [])
            for source, data in report.get("sources", {}).items()
        }
        trimmed = json.loads(json.dumps(report, default=_jsonable))
        for data in trimmed.get("sources", {}).values():
            data.pop("samples", None)
        self._write_json(f"power{suffix}.json", trimmed)
        if any(samples.values()):
            self._write_json(f"power_samples{suffix}.json", samples)

    def finish(self, summary: dict[str, Any]) -> Path:
        """Write summary.json and append a row to results/summary.csv. Returns the run directory."""
        self.meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_json("meta.json", self.meta)
        self._write_json("summary.json", summary)

        row: dict[str, Any] = {
            "run_id": self.run_id,
            "name": self.name,
            "git_commit": (self.git.get("commit") or "")[:10],
            "git_dirty": self.git.get("dirty"),
            "devices": json.dumps(self.devices, sort_keys=True),
        }
        _flatten("", summary, row)
        self._append_summary_row(row)
        return self.dir

    def _append_summary_row(self, row: dict[str, Any]) -> None:
        path = self.results_dir / "summary.csv"
        rows: list[dict[str, Any]] = []
        fields: list[str] = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
        for key in row:
            if key not in fields:
                fields.append(key)
        rows.append(row)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
