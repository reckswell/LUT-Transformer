"""Where a PyTorch run's memory goes, and where the LUT tables live (2026-11-24, Alpha: "take note
of the memory usage ... as well as where these LUTs are stored, whether in DRAM, in GPU RAM, etc.").

Three pieces, used by bench/memory_pass.py and the workbooks:

1. Placement of a tensor (place_of): its device (cuda:N = GPU VRAM, cpu = host DRAM), whether host
   memory is pinned (page-locked, what a DMA copy to the GPU reads from) or pageable, and whether
   its storage is memory-mapped from a file on disk (torch.load(..., mmap=True)).

2. LUT storage inventory (lut_inventory): every stage slot of a ModelOps (embedding, each layer's
   stages, final_norm, lm_head). A slot is a LUT op when it is a functools.partial carrying an
   `arch` keyword that is a LUT/stages.py LUTArch (how LUT/ops_registry.py builds them). The tables
   a LUT op uses are found by walking what the slot holds (partial arguments and keywords, closure
   cells, dicts, lists, dataclass and object attributes) for tensors. Each table gets a row: stage,
   layer, op, architecture and table shape (entries x out_bits), tensor name, shape, dtype, bytes,
   place, pinned, memory-mapped. A LUT op with no tensor yet (LUT/inference_lut.build_tables is
   still pending) gets a row saying so, with the planned size from its LUTArch. With every stage on
   BASE the inventory says "none (all base ops)" instead of staying blank.

3. Peak memory of one measured phase (PhaseMemory): torch.cuda peak allocated and peak reserved
   (reset at the start of the phase), allocated after the phase (steady state), this process's own
   GPU memory from nvidia-smi --query-compute-apps (CUDA context + allocator pool + everything else
   the process holds; the device-wide memory.used also counts the desktop and other programs), host
   RSS (peak sampled every interval and after the phase), and pinned host memory held by PyTorch's
   caching host allocator. breakdown() splits the peak: weights, LUT tables, KV cache, optimizer
   state and gradients when training, activations and workspace (peak allocated minus the others),
   and CUDA context plus allocator overhead (process GPU memory minus peak reserved).

Plus htod_copies(): a torch.profiler pass that counts host-to-device copies per call, to show
whether anything (a table, an index, a mask) is copied to the GPU during inference.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import resource
import shutil
import subprocess
import threading
import time
import types
from typing import Any, Callable, Iterator

import torch
from torch import Tensor

MiB = 2 ** 20


# ----------------------------------------------------------------------------------------------
# 1. Placement
# ----------------------------------------------------------------------------------------------

def tensor_bytes(t: Tensor) -> int:
    """numel x element size: the bytes the tensor's elements occupy (a view counts its own elements)."""
    return t.numel() * t.element_size()


def mapped_file(address: int) -> str | None:
    """The file mapped at a host address (Linux /proc/self/maps), or None for anonymous memory."""
    if not address:
        return None
    try:
        with open("/proc/self/maps") as handle:
            for line in handle:
                parts = line.split(None, 5)
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                if lo <= address < hi:
                    path = parts[5].strip() if len(parts) == 6 else ""
                    return path if path.startswith("/") and not path.startswith("/dev/") else None
    except (OSError, ValueError, IndexError):
        pass
    return None


def place_of(t: Tensor) -> dict[str, Any]:
    """{"device", "place", "pinned", "mmap_file"}: place is "GPU VRAM" for cuda tensors and "host DRAM"
    for cpu ones (or "disk (memory-mapped)" when the storage is a mapped file, as with
    torch.load(..., mmap=True))."""
    if t.is_cuda:
        return {"device": str(t.device), "place": "GPU VRAM", "pinned": False, "mmap_file": None}
    pinned = False
    try:
        pinned = bool(t.is_pinned())
    except RuntimeError:                               # no CUDA driver: nothing can be pinned
        pass
    mmap_file = mapped_file(t.untyped_storage().data_ptr())
    return {"device": str(t.device), "place": "disk (memory-mapped)" if mmap_file else "host DRAM",
            "pinned": pinned, "mmap_file": mmap_file}


def walk_tensors(obj: Any, path: str = "", depth: int = 6, seen: set[int] | None = None) -> Iterator[tuple[str, Tensor]]:
    """(path, tensor) for every tensor reachable from obj through partial arguments and keywords,
    closure cells, dicts, lists, tuples, sets, dataclass fields and plain object attributes. Each
    object is visited once; modules, classes and builtins are not entered."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth < 0:
        return
    seen.add(id(obj))
    if isinstance(obj, Tensor):
        yield path or "tensor", obj
        return
    if obj is None or isinstance(obj, (str, bytes, int, float, bool, complex, type, types.ModuleType,
                                       types.BuiltinFunctionType, torch.dtype, torch.device)):
        return
    if isinstance(obj, functools.partial):
        yield from walk_tensors(obj.func, f"{path}.func", depth - 1, seen)
        for i, a in enumerate(obj.args):
            yield from walk_tensors(a, f"{path}.args[{i}]", depth - 1, seen)
        for k, v in obj.keywords.items():
            yield from walk_tensors(v, f"{path}.{k}" if path else k, depth - 1, seen)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_tensors(v, f"{path}[{k!r}]", depth - 1, seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            yield from walk_tensors(v, f"{path}[{i}]", depth - 1, seen)
        return
    if isinstance(obj, (types.FunctionType, types.MethodType)):
        fn = obj.__func__ if isinstance(obj, types.MethodType) else obj
        if isinstance(obj, types.MethodType):
            yield from walk_tensors(obj.__self__, f"{path}.__self__", depth - 1, seen)
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
            try:
                yield from walk_tensors(cell.cell_contents, f"{path}.<closure {name}>", depth - 1, seen)
            except ValueError:                         # an empty cell
                pass
        return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            yield from walk_tensors(getattr(obj, f.name, None), f"{path}.{f.name}", depth - 1, seen)
        return
    attrs = getattr(obj, "__dict__", None)
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            yield from walk_tensors(v, f"{path}.{k}", depth - 1, seen)


def bytes_by_place(tensors: Iterator[tuple[str, Tensor]] | list[tuple[str, Tensor]]) -> dict[str, int]:
    """Total bytes per place (GPU VRAM, host DRAM, disk), each storage counted once."""
    out: dict[str, int] = {}
    storages: set[tuple[str, int]] = set()
    for _, t in tensors:
        key = (str(t.device), t.untyped_storage().data_ptr())
        if key in storages and t.untyped_storage().data_ptr() != 0:
            continue
        storages.add(key)
        place = place_of(t)["place"]
        out[place] = out.get(place, 0) + tensor_bytes(t)
    return out


# ----------------------------------------------------------------------------------------------
# 2. LUT storage inventory
# ----------------------------------------------------------------------------------------------

def _slots(ops: Any) -> Iterator[tuple[str, int | None, Any]]:
    """(stage, layer or None, slot) for every stage slot of a baseline/host.py ModelOps."""
    yield "embedding", None, ops.embedding
    for i, layer in enumerate(ops.layers):
        for stage, slot in layer.items():
            yield stage, i, slot
    yield "final_norm", None, ops.final_norm
    yield "lm_head", None, ops.lm_head


def _arch_of(slot: Any) -> Any:
    """The LUTArch of a LUT op (a partial with an `arch` keyword), else None. A train-mode slot is a
    (forward, backward) pair: its forward decides."""
    fn = slot[0] if isinstance(slot, tuple) and slot else slot
    while isinstance(fn, functools.partial):
        arch = fn.keywords.get("arch")
        if arch is not None and all(hasattr(arch, a) for a in ("inputs_per_table", "bits_per_input", "out_bits")):
            return arch
        fn = fn.func
    return None


def _op_name(slot: Any) -> str:
    fn = slot[0] if isinstance(slot, tuple) and slot else slot
    while isinstance(fn, functools.partial):
        fn = fn.func
    return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', repr(fn)))}".lstrip(".")


def lut_inventory(ops: Any) -> dict[str, Any]:
    """{"rows": [...], "totals": {...}, "lut_stages": n, "label": "..."}: one row per LUT table
    tensor (or per LUT op whose tables are not built yet). totals: bytes in GPU VRAM, host DRAM and
    on disk (memory-mapped), table count, pinned bytes. label: a short cell for the Paper table."""
    rows: list[dict[str, Any]] = []
    lut_stages = 0
    tensors: list[tuple[str, Tensor]] = []
    for stage, layer, slot in _slots(ops):
        arch = _arch_of(slot)
        if arch is None:
            continue
        lut_stages += 1
        common = {"kind": "LUT op table", "stage": stage, "layer": layer, "op": _op_name(slot), "arch": getattr(arch, "name", ""),
                  "inputs_per_table": arch.inputs_per_table, "bits_per_input": arch.bits_per_input,
                  "out_bits": arch.out_bits, "entries_per_table": 2 ** (arch.inputs_per_table * arch.bits_per_input),
                  "planned_table_bits": 2 ** (arch.inputs_per_table * arch.bits_per_input) * arch.out_bits}
        found = [(p, t) for p, t in walk_tensors(slot) if not p.startswith(".func")]
        if not found:
            rows.append({**common, "name": None, "shape": None, "dtype": None, "bytes": None, "device": None,
                         "place": "not materialized (build_tables pending)", "pinned": None, "mmap_file": None})
            continue
        for path, t in found:
            tensors.append((path, t))
            rows.append({**common, "name": path, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                         "bytes": tensor_bytes(t), **place_of(t)})
    by_place = bytes_by_place(tensors)
    totals = {"lut_tables": len(tensors), "lut_bytes_gpu_vram": by_place.get("GPU VRAM", 0),
              "lut_bytes_host_dram": by_place.get("host DRAM", 0), "lut_bytes_disk_mmap": by_place.get("disk (memory-mapped)", 0),
              "lut_bytes_pinned": sum(tensor_bytes(t) for _, t in tensors if place_of(t)["pinned"])}
    if not lut_stages:
        label = "none (all base ops)"
    elif not tensors:
        label = f"{lut_stages} LUT ops, tables not built yet"
    else:
        parts = [f"{place}, {nbytes / 1e6:.3g} MB" for place, nbytes in sorted(by_place.items(), key=lambda kv: -kv[1])]
        label = "; ".join(parts)
    return {"rows": rows, "totals": totals, "lut_stages": lut_stages, "label": label}


def precomputed_tables() -> list[dict[str, Any]]:
    """Constant tables the from-scratch code builds once and reads at run time without being a LUT
    op (today: baseline/inference.py rope_tables' cos/sin, a ROM in hardware), in lut_inventory's row
    format with kind = "precomputed table (not a LUT op)"."""
    try:
        from baseline import inference
    except ImportError:
        return []
    rows = []
    for (device, dtype, head_dim, theta), tensors in list(getattr(inference, "_ROPE_TABLES", {}).items()):
        for name, t in zip(("cos", "sin"), tensors):
            rows.append({"kind": "precomputed table (not a LUT op)", "stage": "rope", "layer": None,
                         "op": "baseline.inference.rope_tables", "arch": "", "inputs_per_table": None,
                         "bits_per_input": None, "out_bits": t.element_size() * 8, "entries_per_table": t.numel(),
                         "planned_table_bits": None,
                         "name": f"rope {name}: positions 0..{t.shape[0] - 1} x {t.shape[1]} pairs (head_dim {head_dim}, theta {theta:g})",
                         "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "bytes": tensor_bytes(t),
                         **place_of(t)})
    return rows


def with_precomputed(inventory: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """lut_inventory plus the precomputed tables: rows appended, their bytes per place in totals
    (precomputed_bytes_*), and the label naming them when no stage runs a LUT op."""
    if not rows:
        return inventory
    out = {**inventory, "rows": list(inventory["rows"]) + rows, "totals": dict(inventory["totals"])}
    by_place: dict[str, int] = {}
    for r in rows:
        by_place[r["place"]] = by_place.get(r["place"], 0) + (r["bytes"] or 0)
    out["totals"]["precomputed_bytes_gpu_vram"] = by_place.get("GPU VRAM", 0)
    out["totals"]["precomputed_bytes_host_dram"] = by_place.get("host DRAM", 0)
    where = "; ".join(f"{p}, {b / 1e6:.3g} MB" for p, b in sorted(by_place.items(), key=lambda kv: -kv[1]))
    if inventory["lut_stages"] == 0:
        out["label"] = f"no LUT ops; rope cos/sin table: {where}"
    else:
        out["label"] = f"{inventory['label']}; rope cos/sin table: {where}"
    return out


# ----------------------------------------------------------------------------------------------
# 3. Peak memory of a phase
# ----------------------------------------------------------------------------------------------

def _rss_bytes() -> int | None:
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def process_gpu_bytes(pid: int | None = None) -> int | None:
    """This process's GPU memory as nvidia-smi reports it (compute apps): CUDA context, the caching
    allocator's pool and anything else the process holds. None without nvidia-smi."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    pid = os.getpid() if pid is None else pid
    try:
        text = subprocess.run([tool, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    total = 0
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                total += int(float(parts[1])) * MiB
            except ValueError:
                pass
    return total


def pinned_host_bytes() -> int | None:
    """Pinned host memory currently held by PyTorch's caching host allocator (None if not exposed)."""
    stats = getattr(torch.cuda, "host_memory_stats", None)
    if stats is None or not torch.cuda.is_available():
        return None
    try:
        s = stats()
    except RuntimeError:
        return None
    for key in ("allocated_bytes.current", "allocated_bytes.all.current", "reserved_bytes.current"):
        if key in s:
            return int(s[key])
    return None


class PhaseMemory:
    """with PhaseMemory(device) as m: run the phase. Afterwards m.result has the peaks (bytes)."""

    def __init__(self, device: str, interval_s: float = 0.25, poll_gpu: bool = True) -> None:
        self.device, self.interval_s, self.poll_gpu = device, interval_s, poll_gpu
        self.cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.result: dict[str, Any] = {}

    def _poll(self) -> None:
        while not self._stop.is_set():
            rss = _rss_bytes()
            if rss:
                self.rss.append(rss)
            if self.cuda and self.poll_gpu:
                g = process_gpu_bytes()
                if g:
                    self.gpu.append(g)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PhaseMemory":
        self.rss: list[int] = []
        self.gpu: list[int] = []
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.allocated_before = torch.cuda.memory_allocated(self.device)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self.t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self._stop.set()
        self._thread.join(timeout=10)
        r: dict[str, Any] = {"seconds": time.perf_counter() - self.t0}
        rss_now = _rss_bytes()
        r["host_rss_peak_bytes"] = max(self.rss + ([rss_now] if rss_now else [])) if (self.rss or rss_now) else None
        r["host_rss_steady_bytes"] = rss_now
        r["host_rss_lifetime_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        r["pinned_host_bytes"] = pinned_host_bytes()
        if self.cuda:
            r["torch_allocated_before_bytes"] = self.allocated_before
            r["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            r["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            r["torch_allocated_steady_bytes"] = torch.cuda.memory_allocated(self.device)
            now = process_gpu_bytes()
            samples = self.gpu + ([now] if now else [])
            r["process_gpu_peak_bytes"] = max(samples) if samples else None
            r["process_gpu_samples"] = len(samples)
        self.result = r


def breakdown(phase: dict[str, Any], *, weights_gpu: int = 0, lut_gpu: int = 0, kv_cache_gpu: int = 0,
              optimizer_gpu: int = 0, grads_gpu: int = 0) -> dict[str, Any]:
    """The phase's GPU peak split into parts (bytes). activations_workspace = peak allocated minus
    the resident parts (weights, LUT tables, KV cache, optimizer state, gradients), floored at 0;
    context_and_allocator = process GPU memory minus peak reserved (the CUDA context, cuBLAS
    workspaces outside the allocator, and the allocator's own overhead)."""
    peak = phase.get("torch_peak_allocated_bytes")
    out = {"weights_bytes": weights_gpu, "lut_table_bytes": lut_gpu, "kv_cache_bytes": kv_cache_gpu,
           "optimizer_state_bytes": optimizer_gpu, "gradient_bytes": grads_gpu}
    resident = weights_gpu + lut_gpu + kv_cache_gpu + optimizer_gpu + grads_gpu
    out["activations_workspace_bytes"] = max(0, peak - resident) if peak is not None else None
    reserved, proc = phase.get("torch_peak_reserved_bytes"), phase.get("process_gpu_peak_bytes")
    out["allocator_cached_bytes"] = (reserved - peak) if reserved is not None and peak is not None else None
    out["context_and_allocator_bytes"] = (proc - reserved) if proc is not None and reserved is not None else None
    return out


# ----------------------------------------------------------------------------------------------
# Host-to-device copies during a call
# ----------------------------------------------------------------------------------------------

def htod_copies(fn: Callable[[], Any], calls: int = 1) -> dict[str, Any]:
    """Run fn() `calls` times under torch.profiler (CPU + CUDA) and count the host-to-device memcpy
    kernels: {"htod_copies", "htod_bytes", "per_call_copies", "per_call_bytes", "pinned_sources",
    "pageable_sources"}. Zero means everything the call needs already lives on the GPU."""
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        return {"htod_copies": None, "note": "no CUDA"}
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    copies, nbytes, pinned, pageable = 0, 0, 0, 0
    for event in prof.events():
        name = event.name or ""
        if "Memcpy HtoD" not in name and "memcpy HtoD" not in name.lower():
            continue
        if getattr(event, "device_type", None) is not None and "CUDA" not in str(event.device_type):
            continue
        copies += 1
        pinned += "Pinned" in name
        pageable += "Pageable" in name
        extra = getattr(event, "extra_fields", None) or {}
        nbytes += int(extra.get("bytes", 0) or 0)
    return {"htod_copies": copies, "htod_bytes": nbytes or None, "per_call_copies": copies / calls,
            "per_call_bytes": (nbytes / calls) if nbytes else None, "pinned_sources": pinned,
            "pageable_sources": pageable, "calls": calls}
