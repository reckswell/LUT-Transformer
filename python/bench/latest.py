"""The latest-only view of the benchmarks: the newest valid result of each kind, for Alpha's second
Google Sheet and results/benchmarks-latest.xlsx (2026-11-24). No history, no per-token data.

    cd python
    python -m bench.latest              # write results/benchmarks-latest.xlsx (and print what it picked)

Built from the same sources as bench/excel.py (results/, read only for results/hls/). Tabs, in order:
    Paper table             the minimum for a research-paper table: Training and Inference sections,
                            one small table per platform (GPU: time, accuracy, power; FPGA: resources,
                            accuracy, latency, frequency, power), a footnote per table, source run ids
    Latest summary          per arm the newest base-metrics run (full span and the screening slice),
                            the newest training run per setup, the newest HLS inference record per
                            variant and platform, the newest HLS kernel build per kernel and variant
    Per-stage (latest)      per-stage timing from the newest per-stage run of each arm, in forward
                            order (embedding ... lm_head), with the caveat row and the bar charts
    PyTorch vs HLS (latest) the newest Final comparison row per variant, stage and kernel
    Time budget (latest)    the newest duration of each task (PyTorch and HLS)
Selection rule: the newest run of each kind is shown; if another heavy job overlapped it
("contended", bench/excel.py mark_contention) it is marked, and the newest clean run of that kind is
shown on the next row. Every row names its source run id, so it can be found in the full sheet.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import excel
from bench.excel import Column, Sheet

LATEST_PATH = excel.RESULTS_DIR / "benchmarks-latest.xlsx"
PAPER = "Paper table"
SUMMARY = "Latest summary"
PER_STAGE = "Per-stage (latest)"
COMPARISON = "PyTorch vs HLS (latest)"
TIME = "Time budget (latest)"
try:
    from zoneinfo import ZoneInfo
    NEW_YORK = ZoneInfo("America/New_York")
except Exception:                       # no tz database: show UTC only
    NEW_YORK = None


def when(ts: Any) -> tuple[str, str]:
    """(America/New_York, UTC) strings for a run timestamp (ISO 8601 or 20261124T012255Z)."""
    moment = excel._parse_time(ts)
    if moment is None:
        return "", ""
    utc = moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    local = moment.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M:%S %Z") if NEW_YORK else ""
    return local, utc


def run_time(run_id: str, results: excel.Results) -> str:
    """A sortable UTC timestamp for a run id: its meta.json start, else the id's own stamp."""
    meta = results.runs.get(run_id, {}).get("meta", {})
    moment = excel._parse_time(meta.get("started_utc")) or excel._parse_time(run_id[:16])
    return moment.astimezone(timezone.utc).isoformat() if moment else ""


def pick(candidates: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """[(row, status)] for one kind of result: the newest; if it is contended, also the newest clean
    one. Each candidate has "ts" (sortable) and "contended" ("" or a description)."""
    if not candidates:
        return []
    newest = max(candidates, key=lambda c: (c["ts"], c.get("run_id", "")))
    if not newest.get("contended"):
        return [(newest, "newest")]
    out = [(newest, f"newest, contended ({newest['contended'][:80]})")]
    clean = [c for c in candidates if not c.get("contended")]
    if clean:
        out.append((max(clean, key=lambda c: (c["ts"], c.get("run_id", ""))), "newest clean"))
    return out


SUMMARY_COLUMNS = [
    ("section", "text", "general"), ("arm / variant / kernel", "text", "general"), ("status", "text", "general"),
    ("run_id", "text", "general"), ("date (New York)", "text", "general"), ("date (UTC)", "text", "general"),
    ("perplexity", "num", "0.000000"), ("top-1", "num", "pct"), ("top-5", "num", "pct"),
    ("tokens/s", "num", "#,##0.0"), ("decode ms/step", "num", "0.000"), ("avg GPU power (W)", "num", "0.0"),
    ("peak GPU power (W)", "num", "0.0"), ("energy per decode step (mJ)", "num", "0.000"), ("GPU util (%)", "num", "0.0"),
    ("torch peak allocated (MiB)", "num", "0.0"), ("process GPU peak (MiB)", "num", "0.0"),
    ("device memory used (MiB, all processes)", "num", "int"), ("host RSS peak (MiB)", "num", "0.0"),
    ("LUT storage", "text", "general"), ("wall (s)", "num", "0.00"), ("C model ms/step", "num", "0.00"),
    ("FPGA estimate ms/step", "num", "0.00"), ("latency (us)", "num", "0.0000"), ("II", "text", "general"),
    ("clock (MHz)", "num", "0.0"), ("LUT", "int", "int"), ("FF", "int", "int"), ("DSP", "int", "int"),
    ("BRAM", "int", "int"), ("threads", "text", "general"), ("note", "text", "general")]


def _sweep_runs(results: excel.Results) -> dict[str, dict[str, Any]]:
    """run id -> {"line": first long line, "speed": {T: {metric: v}}, "ppl": {dataset: {..}}, "res": {phase: {..}}}."""
    out: dict[str, dict[str, Any]] = {}
    for line in results.sweep:
        r = out.setdefault(line["run_id"], {"line": line, "speed": {}, "ppl": {}, "res": {}})
        if line["benchmark"] == "speed":
            r["speed"].setdefault(line["context_tokens"], {})[line["metric"]] = excel.num(line["value"])
        elif line["benchmark"] == "perplexity":
            r["ppl"].setdefault(line["dataset"], {})[line["metric"]] = excel.num(line["value"])
        elif line["benchmark"] == "resources":
            r["res"].setdefault(line["dataset"], {})[line["metric"]] = excel.num(line["value"])
    return out


def summary_sheet(results: excel.Results, timing: list[dict[str, Any]]) -> Sheet:
    contended = {r["run_id"]: r["contended"] for r in timing
                 if r.get("component") in ("row total", "run total (meta start to finish)")}
    flag = {rid: ("" if c == "uncontended" else c) for rid, c in contended.items()}
    rows: list[dict[str, Any]] = []

    def add(section: str, name: str, cand: dict[str, Any], status: str, **values: Any) -> None:
        local, utc = when(cand["ts"])
        rows.append({"section": section, "arm / variant / kernel": name, "status": status, "run_id": cand["run_id"],
                     "date (New York)": local, "date (UTC)": utc, **values})

    sweeps = _sweep_runs(results)
    for arm in ("hf", "baseline", "lut"):
        for span, title in (("full", "PyTorch base metrics (stories15M, GPU, full span)"),
                            ("slice", "PyTorch screening slice (187k chars)")):
            cands = []
            for rid, r in sweeps.items():
                if r["line"]["arm"] != arm or not r["ppl"] or not r["line"]["device"].startswith("cuda"):
                    continue
                dataset = next(iter(r["ppl"]))
                limit = excel.char_limit_of(results, rid, dataset)
                is_full = bool(limit) and int(limit) == 1_200_000
                if (span == "full") != is_full or (span == "full" and not r["speed"]):
                    continue
                cands.append({"run_id": rid, "ts": run_time(rid, results), "contended": flag.get(rid, ""), "r": r,
                              "dataset": dataset})
            for cand, status in pick(cands):
                r, ppl = cand["r"], cand["r"]["ppl"][cand["dataset"]]
                speed = r["speed"].get("224", {})
                res = r["res"].get("speed", {})
                summary = results.runs.get(cand["run_id"], {}).get("summary", {})
                name = excel.variant_of(arm, r["line"].get("lut_preset", "")) + (f" ({arm})" if arm == "lut" else "")
                add(title, name, cand, status, **{
                    "perplexity": ppl.get("perplexity"), "top-1": ppl.get("top1_accuracy"),
                    "top-5": ppl.get("top5_accuracy"),
                    "tokens/s": speed.get("decode_tokens_per_s") if span == "full" else None,
                    "decode ms/step": speed.get("decode_ms_per_token_median") if span == "full" else None,
                    "avg GPU power (W)": speed.get("decode_power_w") if span == "full" else
                    r["res"].get(cand["dataset"], {}).get("power_w_mean_steady"),
                    "peak GPU power (W)": res.get("power_draw_max") if span == "full" else
                    r["res"].get(cand["dataset"], {}).get("power_draw_max"),
                    "energy per decode step (mJ)": speed.get("decode_energy_per_token_mj") if span == "full" else None,
                    "GPU util (%)": (res if span == "full" else r["res"].get(cand["dataset"], {})).get("utilization_gpu_mean"),
                    "device memory used (MiB, all processes)":
                        (res if span == "full" else r["res"].get(cand["dataset"], {})).get("memory_used_max"),
                    "torch peak allocated (MiB)":
                        (res if span == "full" else r["res"].get(cand["dataset"], {})).get("torch_peak_allocated_mib"),
                    "host RSS peak (MiB)": (res if span == "full" else r["res"].get(cand["dataset"], {})).get("process_rss_peak_mib"),
                    "wall (s)": summary.get("row_s"),
                    "note": ("decode at T=224, batch 1 (one generated token per step); power from the decode power window"
                             if span == "full" else f"{ppl.get('eval_tokens')} scored tokens; power over the perplexity eval"
                             + (f"; {r['res'][cand['dataset']]['energy_per_token_mj']:.3f} mJ per scored token (window-forward)"
                                if (r["res"].get(cand["dataset"]) or {}).get("energy_per_token_mj") else ""))
                    + ("" if ppl.get("top1_accuracy") is not None else "; accuracy not measured in this run")})
    # training: newest run per setup
    train_runs = [rid for rid in results.runs if "_train_throughput_" in rid]
    for setup, arm, variant in (("ours", "baseline", "baseline"), ("hf_fp32", "hf", "hf"), ("hf_bf16", "hf", "hf_bf16_tf32")):
        cands = [{"run_id": rid, "ts": run_time(rid, results), "contended": flag.get(rid, ""),
                  "s": results.runs[rid]["summary"].get(setup) or {}} for rid in train_runs
                 if (results.runs[rid]["summary"].get(setup) or {}).get("tok_s")]
        for cand, status in pick(cands):
            s, pw = cand["s"], cand["s"].get("power") or {}
            add("PyTorch training (stories15M, GPU, batch 64 x 256)", f"{variant} ({setup})", cand, status, **{
                "tokens/s": s.get("tok_s"), "avg GPU power (W)": pw.get("power_w_mean_steady"),
                "peak GPU power (W)": pw.get("power_draw_max"),
                "GPU util (%)": pw.get("utilization_gpu_mean"), "device memory used (MiB, all processes)": pw.get("memory_used_max"),
                "torch peak allocated (MiB)": pw.get("torch_peak_allocated_mib"), "host RSS peak (MiB)": pw.get("process_rss_peak_mib"),
                "wall (s)": (s.get("ms_step") or 0) / 1000.0 or None,
                "note": "wall = one training step" + (f"; {pw['energy_per_token_mj']:.3f} mJ per training token"
                                                      if pw.get("energy_per_token_mj") else "")
                        + ("" if pw else "; power not measured in this run")})
    # PyTorch inference on the FPGA's token sets (bench/eval_decode.py, bench/eval_power.py): newest per arm, set, protocol
    evals: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for (platform, n, arm), cands in _pytorch_inference(results, flag).items():
        for c in cands:
            evals.setdefault((arm, n, c["protocol"]), []).append(c)
    for (arm, n, protocol), cands in sorted(evals.items(), key=lambda kv: (-kv[0][1], kv[0][2], kv[0][0])):
        for cand, status in pick(cands):
            info = (results.runs[cand["run_id"]]["summary"] or {}).get(arm) or {}
            res = info.get("resources") or {}
            add(f"PyTorch inference on the FPGA's tokens ({n:,} tokens, {protocol})", MODEL_NAMES.get(arm, arm), cand, status, **{
                "perplexity": info.get("perplexity"), "top-1": info.get("top1_accuracy"), "top-5": info.get("top5_accuracy"),
                "tokens/s": info.get("tokens_per_s"), "decode ms/step": info.get("ms_per_step"),
                "avg GPU power (W)": cand["power"], "peak GPU power (W)": info.get("power_w_max"),
                "energy per decode step (mJ)": info.get("energy_per_step_mj"), "GPU util (%)": res.get("utilization_gpu_mean"),
                "torch peak allocated (MiB)": res.get("torch_peak_allocated_mib"),
                "device memory used (MiB, all processes)": res.get("memory_used_max"),
                "host RSS peak (MiB)": res.get("process_rss_peak_mib"), "wall (s)": cand["time_s"],
                "note": ("wall = all decode steps, one token per step on a KV cache" if protocol == "decode-step" else
                         "wall = one pass, each 256-token window at once (median of repeated passes), no decode steps"
                         + (f"; {info['energy_per_token_mj']:.3f} mJ per scored token" if info.get("energy_per_token_mj") else ""))
                        + "; memory includes both models loaded in the process"})
    # averages over the clean repeated runs (bench/excel.py average_rows; every run is in the full sheet's Averages tab)
    avg: dict[tuple, dict[str, dict[str, Any]]] = {}
    for a in excel.average_rows(results, timing):
        if a["n (clean)"]:
            avg.setdefault((a["group"], a["tokens"] or 0, a["arm / setup"], a["code version"]), {})[a["metric"]] = a
    for (group, tokens, who, code), metrics in sorted(avg.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2], kv[0][3])):
        def m(name: str, key: str = "mean") -> Any:
            return (metrics.get(name) or {}).get(key)
        n = max(x["n (clean)"] for x in metrics.values())
        if n < 2:
            continue                                 # a single run is already shown above
        clean_ids = sorted({rid for x in metrics.values() for rid in str(x["clean run ids"]).split("; ") if rid})
        busy_n = max(x["contended runs (not averaged)"] for x in metrics.values())
        time_metric = next((k for k in ("time for all decode steps", "time per pass (median of passes)") if k in metrics), None)
        stds = [f"{label} ±{m(k, 'std'):.3g}" for k, label in ((time_metric, "time (s)"), ("ms per decode step", "ms/step"),
                ("avg GPU power", "power (W)"), ("energy per decode step", "energy/step (mJ)"),
                ("ms per training step", "ms/step"), ("decode step (uninstrumented)", "step (ms)"))
                if k and m(k, "std") is not None]
        title = f"PyTorch averages over clean runs: {group}" + (f" ({tokens:,} tokens)" if tokens else "") + f", {code}"
        cand = {"run_id": f"{n} clean runs (Averages tab)", "ts": max(run_time(r, results) for r in clean_ids) if clean_ids else "",
                "contended": ""}
        add(title, MODEL_NAMES.get(who, who), cand, "average", **{
            "top-1": m("top-1 accuracy"), "tokens/s": m("tokens per second"),
            "decode ms/step": m("ms per decode step") or m("decode step (uninstrumented)"),
            "avg GPU power (W)": m("avg GPU power"), "energy per decode step (mJ)": m("energy per decode step"),
            "torch peak allocated (MiB)": m("peak GPU memory, PyTorch allocations"),
            "wall (s)": m(time_metric) if time_metric else ((m("ms per training step") or 0) / 1000.0 or None),
            "note": f"mean over {n} clean runs ({busy_n} contended left out); std: " + ", ".join(stds)
                    + ("; " + (metrics.get("top-1 accuracy") or {}).get("note", "") if metrics.get("top-1 accuracy") else "")})
    # memory pass: one arm at a time, with the split of the peak and where the LUT tables live
    mem_runs = [rid for rid in results.runs if (results.runs[rid]["meta"].get("name") or "").startswith("memory_")]
    if mem_runs:
        rid = max(mem_runs, key=lambda r: run_time(r, results))
        cand = {"run_id": rid, "ts": run_time(rid, results), "contended": flag.get(rid, "")}
        for m in excel.memory_rows(results):
            if m["run_id"] != rid:
                continue
            parts = [f"{label} {m[key]:.1f}" for key, label in (("weights (MiB)", "weights"), ("KV cache (MiB)", "KV cache"),
                     ("LUT tables on GPU (MiB)", "LUT"), ("optimizer state (MiB)", "optimizer"), ("gradients (MiB)", "gradients"),
                     ("activations / workspace (MiB)", "activations/workspace"),
                     ("CUDA context + overhead (MiB)", "CUDA context")) if m.get(key)]
            add("PyTorch memory (memory pass, one arm loaded at a time)", f"{m['variant']} ({m['arm']}), {m['phase']}", cand,
                "contended" if cand["contended"] else "newest", **{
                    "torch peak allocated (MiB)": m.get("torch peak allocated (MiB)"),
                    "process GPU peak (MiB)": m.get("process GPU peak (MiB)"), "host RSS peak (MiB)": m.get("host RSS peak (MiB)"),
                    "LUT storage": m.get("LUT storage"),
                    "note": f"{m['shape / tokens']}; MiB: " + ", ".join(parts)})
    # HLS inference: newest per variant and platform
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in excel.hls_inference_rows(results):
        if row.get("kind") != "inference":
            continue
        ts = excel._parse_time(row.get("timestamp"))
        groups.setdefault((row["variant"], row["platform"]), []).append(
            {"run_id": row["run_id"], "ts": ts.isoformat() if ts else "", "contended": "", "row": row})
    for (variant, platform), cands in sorted(groups.items()):
        for cand, status in pick(cands):
            row = cand["row"]
            add("HLS inference (slice tokens)", f"{row['kernel']} {variant}, {platform}", cand, status, **{
                "perplexity": row["perplexity"], "top-1": row["top1"], "top-5": row["top5"], "wall (s)": row["wall (s)"],
                "C model ms/step": row["C model ms per decode step"], "FPGA estimate ms/step": row["FPGA estimate ms per step"],
                "threads": str(row.get("threads") or ""),
                "note": f"{row['scored_tokens']:,} scored tokens ({row['note'].split(';')[0]}); {row['match']}"})
    # HLS kernel builds: newest per kernel and variant
    builds: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in results.hls:
        view = excel.hls_view(record)
        if view.get("latency_cycles_max") is None and view.get("LUT") is None:
            continue                     # a check-only run (csim, a sync test): no build result to show
        ts = excel._parse_time(view.get("timestamp"))
        builds.setdefault((view["kernel"], view["variant"]), []).append(
            {"run_id": view["run_id"], "ts": ts.isoformat() if ts else "", "contended": "", "view": view})
    for (kernel, variant), cands in sorted(builds.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        for cand, status in pick(cands):
            v = cand["view"]
            add("HLS kernel build (newest per kernel and variant)", f"{kernel} {variant}", cand, status, **{
                "latency (us)": v.get("latency_us"), "II": str(v.get("ii") or ""), "clock (MHz)": v.get("clock_mhz"),
                "LUT": v.get("LUT"), "FF": v.get("FF"), "DSP": v.get("DSP"), "BRAM": v.get("BRAM"),
                "wall (s)": v.get("wall_s"), "threads": str(v.get("threads") or "unknown"),
                "note": f"{v.get('hls_variant') or ''}; {v.get('platform') or ''}; stages {v.get('stage') or '-'}; "
                        f"latency {v.get('latency_cycles_max')} cycles ({v.get('latency_source')}); clock "
                        f"{v.get('clock_source')}; profile {v.get('profile') or '-'}".strip("; ")})
    newest = max((r["date (UTC)"] for r in rows if r["date (UTC)"]), default="")
    newest_local = next((r["date (New York)"] for r in rows if r["date (UTC)"] == newest), "")
    preamble = [f"Results as of {newest_local} ({newest}). Only the newest valid result of each kind; every row "
                f"names its source run id (look it up in the full sheet). 'contended' = another heavy job "
                f"overlapped the run; the newest clean run is then shown below it."]
    sheet = Sheet(SUMMARY, [Column(h, k, f) for h, k, f in SUMMARY_COLUMNS], preamble=preamble)
    for r in rows:
        sheet.lines.append([excel.cell_value(r.get(h), k) if k != "text" else r.get(h) for h, k, _ in SUMMARY_COLUMNS])
    return sheet


# ----------------------------------------------------------------------------------------------
# Paper table: the minimum for a research-paper table (Alpha, 2026-11-24)
# ----------------------------------------------------------------------------------------------

DASH = "\u2014"
GPU_COLUMNS = ["Model", "Dataset", "Platform", "Time elapsed (s)", "Accuracy, top-1 (%)", "Avg power (W)",
               "Peak GPU memory, PyTorch allocations (MB)", "LUT storage"]
NO_LUT = {"hf": "none (library model, no LUT ops)", "baseline": "none (baseline, no LUT ops)"}
FPGA_COLUMNS = ["Model", "Dataset", "Platform", "LUT", "FF", "DSP", "BRAM", "URAM", "Accuracy, top-1 (%)",
                "Latency (ms/step)", "Frequency (MHz)", "Power (W)"]
MODEL_NAMES = {"hf": "stories15M (HF FP32)", "baseline": "stories15M baseline", "lut": "stories15M LUT"}


def _fmt(value: Any, digits: int) -> str:
    if value is None or value == "":
        return DASH
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(fraction: Any) -> str:
    return DASH if fraction is None else _fmt(float(fraction) * 100.0, 2)


def _gpu_name(results: excel.Results, run_id: str) -> str:
    gpus = (results.runs.get(run_id, {}).get("meta", {}).get("system") or {}).get("gpus") or []
    name = (gpus[0].get("name") if gpus else "") or ""
    return f"GPU ({name.replace('NVIDIA GeForce ', '').replace('NVIDIA ', '') or 'unknown'})"


def _cpu_name(results: excel.Results, run_id: str) -> str:
    cpu = (results.runs.get(run_id, {}).get("meta", {}).get("system") or {}).get("cpu") or "unknown"
    return f"CPU ({cpu.replace(' 8-Core Processor', '').replace('AMD ', '')})"


def _resource(count: Any, pct: Any) -> str:
    if count is None:
        return DASH
    text = f"{int(count):,}"
    return text + (f" ({float(pct):.1f}%)" if pct is not None else "")


def _clean_first(cands: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The newest clean candidate, else the newest (contended) one."""
    if not cands:
        return None
    clean = [c for c in cands if not c.get("contended")]
    return max(clean or cands, key=lambda c: (c["ts"], c.get("run_id", "")))


def paper_rows(results: excel.Results, timing: list[dict[str, Any]]) -> dict[str, Any]:
    """{"sections": [(section, [(subtable, columns, rows, footnote)])], "sources": [run ids], "as_of": ts}."""
    contended = {r["run_id"]: ("" if r["contended"] == "uncontended" else r["contended"]) for r in timing
                 if r.get("component") in ("row total", "run total (meta start to finish)")}
    sources: list[str] = []
    stamps: list[str] = []
    mem = _memory_pass(results)
    mem_used: set[str] = set()
    # Training: GPU (the newest clean training-throughput run)
    busy_train = excel.contended_runs(results, timing)
    train_cands = [{"run_id": rid, "ts": run_time(rid, results), "contended": contended.get(rid, "") or busy_train.get(rid, "")}
                   for rid, run in results.runs.items() if "_train_throughput_" in rid
                   and (run["summary"].get("ours") or {}).get("tok_s")]
    train = _clean_first(train_cands)
    newest_code = excel.code_version(results, train["run_id"]) if train else ""
    train_clean = [c for c in train_cands if not c["contended"] and excel.code_version(results, c["run_id"]) == newest_code]
    training = []
    if train:
        summary = results.runs[train["run_id"]]["summary"]
        steps = (results.runs[train["run_id"]]["meta"].get("configs") or {}).get("timed_steps") or 10
        rows = []
        train_n: dict[str, int] = {}
        for setup, name in (("ours", "stories15M baseline (FP32)"), ("hf_fp32", "stories15M (HF FP32)")):
            s = summary.get(setup) or {}
            if not s.get("tok_s"):
                continue
            power = s.get("power") or {}
            tm = mem["training"].get(setup) or {}
            peak = tm.get("torch_peak_allocated_bytes") or ((power.get("torch_peak_allocated_mib") or 0) * 2 ** 20) or None
            if tm:
                mem_used.add(mem["run_id"])
            t_cell = _fmt((s.get("ms_step") or 0) * steps / 1000.0, 1)
            p_cell = _fmt(power.get("power_w_mean_steady") or power.get("power_draw_mean"), 1)
            if len(train_clean) >= 2:
                runs = [results.runs[c["run_id"]]["summary"].get(setup) or {} for c in train_clean]
                t_cell = _mean_std([(r.get("ms_step") or 0) * steps / 1000.0 for r in runs if r.get("ms_step")], 2)
                p_cell = _mean_std([(r.get("power") or {}).get("power_w_mean_steady") for r in runs], 1)
                train_n[setup] = len(train_clean)
            rows.append([name, f"Random tokens, {s.get('batch', 64)} x {s.get('T', 256)} per step",
                         _gpu_name(results, train["run_id"]), t_cell, DASH, p_cell, _mb(peak),
                         tm.get("lut_label") or NO_LUT["baseline" if setup == "ours" else "hf"]])
        if train_n:
            sources += [c["run_id"] for c in train_clean]
        if rows:
            ours, hf = summary.get("ours") or {}, summary.get("hf_fp32") or {}
            note = (f"Time: {steps} timed training steps (throughput {ours.get('tok_s', 0):,.0f} tokens/s ours, "
                    f"{hf.get('tok_s', 0):,.0f} HF). No full training run yet, so no accuracy. Power: mean board "
                    f"power over 5 s of steps. " + _memory_note(mem, training=True)
                    + (f" Time and power: mean ± std over n = {max(train_n.values())} clean runs." if train_n else ""))
            training.append(("Training: GPU", GPU_COLUMNS, rows, note))
            sources.append(train["run_id"])
            stamps.append(train["ts"])
    # Inference on the same tokens on every platform (Alpha, 2026-11-24): GPU and CPU rows come from
    # bench/eval_decode.py (decode-step protocol, the FPGA kernel's own: one token per step on a KV
    # cache) or else bench/eval_power.py (window-forward: each 256-token window at once, repeated
    # passes), FPGA rows from the HLS inference records that ran every window. Only the largest token
    # set that the GPU and the FPGA both have is shown; a larger set waiting on one platform gets a line.
    busy = excel.contended_runs(results, timing)          # HLS overlap, or HLS processes at the overnight runner's start
    cpu_gpu = _pytorch_inference(results, {**contended, **busy})
    fpga = _fpga_inference(results)
    fpga_shape = {int((i["summary"].get("host") or {}).get("scored_tokens") or 0):
                  {"decode_steps": (i["summary"].get("host") or {}).get("decode_steps")}
                  for i in results.hls_inference if i["summary"] and (i["summary"].get("host") or {}).get("decode_steps")}
    gpu_sets = {n for (platform, n, _) in cpu_gpu if platform == "gpu"}
    common = sorted(gpu_sets & set(fpga))
    inference: list[Any] = []
    title = "Inference"
    if common:
        n = common[-1]
        title = f"Inference ({n:,} tokens, the same tokens on GPU and FPGA)"
        for platform in ("gpu", "cpu"):
            picks = {arm: _best_protocol(cpu_gpu.get((platform, n, arm), [])) for arm in ("hf", "baseline", "lut")}
            picks = {arm: c for arm, c in picks.items() if c}
            if not picks:
                continue
            rows = []
            shared = False
            counts: dict[str, int] = {}
            accuracy_differs = []
            for arm, c in picks.items():
                place = _gpu_name(results, c["run_id"]) if platform == "gpu" else _cpu_name(results, c["run_id"])
                same = [x for x in cpu_gpu.get((platform, n, arm), []) if x["protocol"] == c["protocol"]]
                newest_code = excel.code_version(results, c["run_id"], arm)
                same = [x for x in same if excel.code_version(results, x["run_id"], arm) == newest_code]
                clean = [x for x in same if not x["contended"]]
                if len({x["top1"] for x in same if x["top1"] is not None}) > 1:
                    accuracy_differs.append(arm)
                am = mem["arms"].get(arm) or {}
                one_arm = [x["torch_peak_mib"] * 2 ** 20 for x in clean if x.get("loaded_only_arm") and x.get("torch_peak_mib")]
                peak = am.get("decode" if c["protocol"] == "decode-step" else "window_forward")
                if len(clean) >= 2:                      # repeated clean runs: mean +- std
                    counts[arm] = len(clean)
                    t_cell = _mean_std([x["time_s"] for x in clean], 1)
                    p_cell = _mean_std([x["power"] for x in clean], 1)
                    if peak is not None:                 # the memory pass: the model alone, without the scoring harness
                        m_cell = _mb(peak)
                        mem_used.add(mem["run_id"])
                    elif len(one_arm) >= 2:
                        m_cell = _mean_std([v / 1e6 for v in one_arm], 1)
                    else:
                        m_cell = DASH
                    for x in clean:
                        sources.append(x["run_id"])
                        stamps.append(x["ts"])
                else:
                    t = c["time_s"]
                    t_cell, p_cell = _fmt(t, 2 if t is not None and float(t) < 10 else 1), _fmt(c["power"], 1)
                    if peak is not None:
                        mem_used.add(mem["run_id"])
                    elif c.get("torch_peak_mib") is not None:
                        peak, shared = c["torch_peak_mib"] * 2 ** 20, not c.get("loaded_only_arm")
                    m_cell = _mb(peak)
                    sources.append(c["run_id"])
                    stamps.append(c["ts"])
                rows.append([MODEL_NAMES[arm], f"TinyStories V2 valid, {n:,} tokens", place, t_cell, _pct(c["top1"]), p_cell,
                             m_cell if platform == "gpu" else DASH, am.get("label") or NO_LUT.get(arm, DASH)])
            note = _pytorch_note(n, picks, platform) + (" " + _memory_note(mem, shared=shared) if platform == "gpu" else "")
            if counts:
                names = {"hf": "HF", "baseline": "baseline", "lut": "LUT"}
                note += (" Time, power and memory: mean ± std over n clean runs ("
                         + ", ".join(f"{names[a]} n={k}" for a, k in counts.items()) + ")"
                         + ("; accuracy identical in every run." if not accuracy_differs else
                            f"; ACCURACY DIFFERS BETWEEN RUNS for {', '.join(names[a] for a in accuracy_differs)}."))
            single = [a for a in picks if a not in counts]
            contended_single = [a for a in single if picks[a]["contended"]]
            if contended_single:
                note += (" Contended run only (an HLS job overlapped it) for "
                         + ", ".join({"hf": "HF", "baseline": "baseline", "lut": "LUT"}[a] for a in contended_single) + ".")
            inference.append((f"Inference: {platform.upper()}", GPU_COLUMNS, rows, note))
        rows, ids, notes, ts = fpga[n]
        inference.append(("Inference: FPGA", FPGA_COLUMNS, rows, notes))
        shape = fpga_shape.get(n) or {}
        if shape.get("decode_steps"):
            inference.append(f"One decode step = one token through the model with a KV cache; with 256-token windows at "
                             f"stride 128, the {n:,} scored tokens take {shape['decode_steps']:,} steps.")
        sources += ids
        stamps += ts
    for m in sorted((gpu_sets | set(fpga)) - set(common[-1:])):
        if common and m < common[-1]:
            continue
        waiting = "the FPGA run" if m not in fpga else "the GPU run"
        inference.append(f"{m:,}-token comparison running: waiting on {waiting}.")
    if any(isinstance(t, str) for t in inference) and not common:
        inference.insert(0, "No token set has both GPU and FPGA results yet.")
    sources += sorted(mem_used)
    stamps += [run_time(rid, results) for rid in mem_used]
    return {"sections": [("Training", training), (title, inference)], "sources": sources,
            "as_of": max(stamps) if stamps else ""}


def _mean_std(values: list[Any], digits: int) -> str:
    """"mean ± std" of the numbers given (sample std), or a dash."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return DASH
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:,.{digits}f} ± {std:,.{digits}f}"


def _mb(nbytes: Any) -> str:
    return DASH if not nbytes else f"{float(nbytes) / 1e6:,.1f}"


def _memory_pass(results: excel.Results) -> dict[str, Any]:
    """The newest bench/memory_pass.py run: {"run_id", "arms": {arm: {"decode", "window_forward" (peak
    allocated bytes), "label" (LUT storage), "overhead" (bytes)}}, "training": {setup: {...}}}."""
    runs = [rid for rid in results.runs if (results.runs[rid]["meta"].get("name") or "").startswith("memory_")]
    if not runs:
        return {"run_id": "", "arms": {}, "training": {}}
    rid = max(runs, key=lambda r: run_time(r, results))
    summary = results.runs[rid]["summary"] or {}
    arms = {}
    for arm, out in (summary.get("arms") or {}).items():
        phases = out.get("phases") or {}
        overhead = [(p.get("context_and_allocator_bytes") or 0) + (p.get("allocator_cached_bytes") or 0)
                    for p in phases.values() if p.get("context_and_allocator_bytes") is not None]
        arms[arm] = {"process": (phases.get("decode") or {}).get("process_gpu_peak_bytes"),
                     "decode": (phases.get("decode") or {}).get("torch_peak_allocated_bytes"),
                     "window_forward": (phases.get("window_forward") or {}).get("torch_peak_allocated_bytes"),
                     "label": (out.get("lut_inventory") or {}).get("label"),
                     "overhead": max(overhead) if overhead else None}
    return {"run_id": rid, "arms": arms, "training": summary.get("training") or {}}


def _memory_note(mem: dict[str, Any], *, training: bool = False, shared: bool = False) -> str:
    """The footnote on what "Peak GPU memory, PyTorch allocations" includes, with the per-process
    total from nvidia-smi (never the device-wide figure, which counts the desktop and other programs)."""
    what = ("weights, optimizer state, gradients and activations" if training else
            "weights, KV cache, activations and workspace, LUT tables, and the evaluation buffers")
    alone = ("both models loaded in one process" if shared else "only that model loaded")
    note = f"Peak GPU memory, PyTorch allocations: the peak of PyTorch's allocated GPU memory with {alone} ({what}); MB = 10^6 bytes."
    if training:
        totals = [t.get("process_gpu_peak_bytes") for t in mem["training"].values() if t.get("process_gpu_peak_bytes")]
    else:
        totals = [a.get("process") for a in mem["arms"].values() if a.get("process")]
    if totals:
        lo, hi = min(totals) / 2 ** 20, max(totals) / 2 ** 20
        span = f"about {round((lo + hi) / 2, -1):,.0f} MiB" if hi - lo < 0.05 * hi else f"{lo:,.0f} to {hi:,.0f} MiB"
        note += f" Per-process total including the CUDA context: {span} (nvidia-smi)."
    return note

def _pytorch_inference(results: excel.Results, contended: dict[str, str]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    """(platform, scored tokens, arm) -> candidate results of bench/eval_decode.py (decode-step, only
    runs over every window) and bench/eval_power.py (window-forward)."""
    out: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for rid, run in results.runs.items():
        meta, summary = run["meta"], run["summary"] or {}
        name = meta.get("name", "")
        decode = name.startswith("eval_decode")
        if not (decode or name.startswith("eval_power")) or (decode and not summary.get("complete")):
            continue
        device = (meta.get("devices") or {}).get("inference", "")
        for arm in ("hf", "baseline", "lut"):
            info = summary.get(arm)
            if not isinstance(info, dict) or not info.get("scored_tokens"):
                continue
            out.setdefault(("gpu" if device.startswith("cuda") else "cpu", int(info["scored_tokens"]), arm), []).append({
                "run_id": rid, "ts": run_time(rid, results), "contended": contended.get(rid, ""),
                "protocol": "decode-step" if decode else "window-forward",
                "time_s": info.get("decode_s") if decode else info.get("pass_s_median"),
                "ms_per_step": info.get("ms_per_step"), "top1": info.get("top1_accuracy"),
                "power": info.get("power_w_mean_steady") or info.get("power_w_mean"),
                "windows": summary.get("windows") or (meta.get("configs") or {}).get("windows"),
                "steps": summary.get("decode_steps"), "passes": info.get("passes"),
                "torch_peak_mib": (info.get("resources") or {}).get("torch_peak_allocated_mib"),
                "loaded_only_arm": bool((meta.get("configs") or {}).get("loaded_only_arm")),
                "warmup_s": (info.get("warmup") or {}).get("warmup_s"),
                "reference": (info.get("reference_check") or {}).get("hf" if arm == "hf" else "baseline") or {}})
    return out


def _best_protocol(cands: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The decode-step result if there is one (newest clean first), else the window-forward one."""
    return _clean_first([c for c in cands if c["protocol"] == "decode-step"]) or _clean_first(cands)


def _pytorch_note(n: int, picks: dict[str, dict[str, Any]], platform: str) -> str:
    """The footnote for the GPU or CPU rows: protocol, what the time is, what the power is."""
    by = {}
    for arm, c in picks.items():
        by.setdefault(c["protocol"], []).append((arm, c))
    parts = []
    names = {"hf": "HF", "baseline": "baseline", "lut": "LUT"}
    for protocol, items in by.items():
        who = "" if len(by) == 1 else " (" + ", ".join(names[a] for a, _ in items) + ")"
        c = items[0][1]
        if protocol == "decode-step":
            ms = ", ".join(f"{names[a]} {x['ms_per_step']:.3f}" for a, x in items if x.get("ms_per_step"))
            flags = [x["reference"] for _, x in items if x["reference"]]
            check = ""
            if flags:
                worst = max(f.get("nll_abs_diff_max") or 0.0 for f in flags)
                dis = sum((f.get("top1_flag_disagreements") or 0) for f in flags)
                check = (f" Accuracy checked token by token against the FP32 reference: max |dNLL| {worst:.1g}, "
                         f"{dis} top-1 flag disagreement{'s' if dis != 1 else ''}.")
            parts.append(f"{platform.upper()} and FPGA both decode the same {n:,} tokens one step at a time{who} "
                         f"({c['windows']:,} windows of 256 tokens every 128, {c['steps']:,} decode steps, KV cache, "
                         f"batch 1). Time: wall time of all {c['steps']:,} steps, one pass (ms per step: {ms}). "
                         + ("Power: mean board power (nvidia-smi) over the timed pass, after a warm-up."
                            if platform == "gpu" else "Power: not measured.") + check)
        else:
            parts.append(f"Window-forward protocol{who}: time is one pass over the {n:,} tokens with each 256-token window "
                         f"run at once (median of {c.get('passes') or 'several'} passes, batch 1); "
                         + ("power is the mean board power (nvidia-smi) over the repeated passes after a warm-up; "
                            if platform == "gpu" else "") +
                         "every pass gave identical accuracy. The FPGA decodes the same tokens one step at a time, so "
                         "the times are not like for like; GPU decode-step timing pending.")
    return " ".join(parts)


def _fpga_inference(results: excel.Results) -> dict[int, tuple[list[list[str]], list[str], str, list[str]]]:
    """scored tokens -> (FPGA rows, source run ids, footnote, timestamps): per variant, the newest
    HLS inference record that ran every window of that token set (a board run first), with the build
    it cites for resources, clock and power."""
    builds = {}
    for item in results.hls_all:
        s = item["summary"]
        builds[(item["ledger"] or {}).get("run_id") or s.get("run_id") or item["dir"]] = s
    records: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for item in results.hls_inference:
        s = item["summary"]
        host, sel = (s or {}).get("host") or {}, (s or {}).get("selection") or {}
        if not s or host.get("windows") is None or not host.get("scored_tokens"):
            continue
        whole = (sel.get("max_windows", -1) in (-1, None) and sel.get("max_steps", -1) in (-1, None)
                 and not sel.get("first_window")) if sel else host.get("windows") == 392
        if not whole:
            continue
        variant = s.get("variant") or ("baseline" if "baseline" in ((s.get("timing") or {}).get("fpga_estimate") or {})
                                       .get("source_run_id", "") else "")
        board = str(s.get("backend") or "").lower() in ("hw", "board", "fpga")
        ts = excel._parse_time(s.get("finished_utc") or s.get("started_utc"))
        records.setdefault(int(host["scored_tokens"]), {}).setdefault(variant, []).append(
            {"run_id": s.get("run_id") or item["dir"], "ts": ts.isoformat() if ts else "", "contended": "", "s": s,
             "board": board})
    matched_ids = {r["run_id"] for r in excel.hls_inference_rows(results) if r.get("kind") == "inference"
                   and str(r.get("match", "")).startswith("match")}
    out = {}
    for n, by_variant in records.items():
        rows, ids, notes, stamps = [], [], set(), []
        for variant, cands in sorted(by_variant.items()):
            best = _clean_first([c for c in cands if c["board"]] or cands)
            s = best["s"]
            host, est = s.get("host") or {}, (s.get("timing") or {}).get("fpga_estimate") or {}
            build = builds.get(est.get("source_run_id", ""), {})
            impl = (build.get("results") or {}).get("impl") or {}
            synth = (build.get("results") or {}).get("synth") or {}
            res, pct = impl.get("resources") or synth.get("resources") or {}, impl.get("utilization_pct") or {}
            power = (build.get("results") or {}).get("power") or {}
            if best["board"]:
                latency, freq, watts = host.get("mean_ms_per_step"), s.get("clock_mhz") or impl.get("post_route_fmax_mhz"), \
                    s.get("power_w") or power.get("total_w")
                notes.add("Measured on the board" + ("" if s.get("power_w") else "; power from Vivado's estimate") + ".")
            else:
                latency = est.get("ms_per_step")
                freq = impl.get("post_route_fmax_mhz") or (1000.0 / synth["target_clock_ns"] if synth.get("target_clock_ns") else None)
                watts = power.get("total_w")
                source = "place and route" if impl else "synthesis"
                notes.add(f"Accuracy from C simulation of the HLS kernel; resources and frequency from Vivado {source}, "
                          f"latency per decode step from HLS synthesis cycles at that clock, power from Vivado's vectorless "
                          f"estimate (total" + (f"; dynamic {power.get('dynamic_w'):.1f} W" if power.get("dynamic_w") else "")
                          + "). No board run yet.")
            part = build.get("part") or s.get("part") or ""
            board_name = "Alveo U50" if part.startswith("xcu50") else (part or "FPGA")
            rows.append([f"stories15M {variant}".strip() + " (HLS kernel)", f"TinyStories V2 valid, {n:,} tokens",
                         f"FPGA ({board_name}, {part})" if part and board_name != part else f"FPGA ({board_name})",
                         _resource(res.get("LUT"), pct.get("LUT")), _resource(res.get("FF"), pct.get("FF")),
                         _resource(res.get("DSP"), pct.get("DSP")), _resource(res.get("BRAM", res.get("BRAM_18K")), pct.get("BRAM")),
                         _resource(res.get("URAM"), pct.get("URAM")), _pct(host.get("top1")), _fmt(latency, 2),
                         _fmt(freq, 0), _fmt(watts, 1)])
            if best["run_id"] in matched_ids:
                notes.add(f"On the same {n:,} tokens the GPU FP32 model gives the identical top-1, token for token "
                          f"(PyTorch vs HLS tab).")
            ids += [best["run_id"]] + ([est["source_run_id"]] if est.get("source_run_id") else [])
            stamps.append(best["ts"])
        out[n] = (rows, ids, " ".join(sorted(notes)), stamps)
    return out


def paper_sheet(results: excel.Results, timing: list[dict[str, Any]]) -> Sheet:
    """Training, then Inference, separated by a blank row; in each, one small table per platform that
    has data (no empty tables, nothing invented: a missing value is a dash); a footnote line under
    each table; the source run ids and the results' date at the bottom."""
    data = paper_rows(results, timing)
    lines: list[list[Any]] = []
    bold: list[int] = []
    width = max(len(GPU_COLUMNS), len(FPGA_COLUMNS))
    first = True
    for section, tables in data["sections"]:
        if not tables:
            continue
        if not first:
            lines.append([])
        first = False
        bold.append(len(lines))
        lines.append([section])
        for k, table in enumerate(tables):
            if isinstance(table, str):            # a line of its own (a token set still waiting on a platform)
                lines.append([table])
                continue
            label, columns, rows, note = table
            if k:
                lines.append([])
            bold.append(len(lines))
            lines.append([label])
            bold.append(len(lines))
            lines.append(list(columns))
            lines.extend(rows)
            if note:
                lines.append([note])
    if lines:
        lines.append([])
    local, utc = when(data["as_of"])
    lines.append(["Source runs: " + ", ".join(dict.fromkeys(data["sources"])) if data["sources"] else
                  "Source runs: none yet"])
    lines.append([f"Results as of {local} ({utc})." if utc else "No results yet."])
    sheet = Sheet(PAPER, [Column(f"col{i + 1}") for i in range(width)], freeform=True, bold_rows=bold)
    sheet.lines = lines
    return sheet


def per_stage_sheet(results: excel.Results) -> Sheet:
    """bench/excel.py's per-stage sheet: already the newest per-stage run of each arm, in forward order."""
    sheet = excel.per_stage_sheet(results)
    sheet.name = PER_STAGE
    return sheet


def comparison_sheet(results: excel.Results) -> Sheet:
    """The newest Final comparison row for each (variant, stage, kernel, platform)."""
    full = excel.comparison_sheet(results)
    headers = [c.header for c in full.columns]
    i_var, i_stage, i_kernel, i_run, i_note = (headers.index(h) for h in
                                               ("variant", "stage", "HLS kernel", "HLS run_id", "note"))
    stamps: dict[str, str] = {}
    for item in results.hls_all:
        s = item["summary"]
        rid = (item["ledger"] or {}).get("run_id") or s.get("run_id") or item["dir"]
        ts = excel._parse_time(s.get("finished_utc") or s.get("started_utc"))
        stamps[rid] = ts.isoformat() if ts else ""
    for row in excel.hls_inference_rows(results):
        ts = excel._parse_time(row.get("timestamp"))
        stamps[row["run_id"]] = ts.isoformat() if ts else ""
    i_lat = headers.index("HLS latency (us)")
    newest: dict[tuple, list[Any]] = {}
    for line in full.lines:
        accuracy_row = str(line[i_stage]).startswith("whole model")
        if not accuracy_row and line[i_lat] is None:
            continue                     # an HLS record without a latency (a check-only run): nothing to compare
        platform = str(line[i_note] or "").split(";")[0] if accuracy_row else ""
        key = (line[i_var], line[i_stage], line[i_kernel], platform)
        ts = stamps.get(str(line[i_run]), "")
        if key not in newest or (ts, str(line[i_run])) > (stamps.get(str(newest[key][i_run]), ""), str(newest[key][i_run])):
            newest[key] = line
    keep = [i for i, h in enumerate(headers) if "(secondary)" not in h]   # per scored token stays in the full sheet
    sheet = Sheet(COMPARISON, [full.columns[i] for i in keep] + [Column("date (UTC)")])
    for key, line in newest.items():
        sheet.lines.append([line[i] for i in keep] + [when(stamps.get(str(line[i_run]), ""))[1]])
    return sheet


TIME_COLUMNS = [("flow", "text", "general"), ("task", "text", "general"), ("kernel / model", "text", "general"),
                ("variant / arm", "text", "general"), ("platform / device", "text", "general"),
                ("seconds", "num", "0.000"), ("status", "text", "general"), ("threads", "text", "general"),
                ("condition", "text", "general"), ("basis", "text", "general"), ("run_id", "text", "general"),
                ("date (UTC)", "text", "general")]


def time_sheet(results: excel.Results, timing: list[dict[str, Any]]) -> Sheet:
    """The newest duration of each task; a contended newest is marked, with the newest clean one below."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in timing:
        if r["flow"] == "pytorch" and r["component"] in ("warm-up repeats", "timed repeats", "tokenize"):
            continue
        key = (r["flow"], r["activity"], r.get("model") or "", r.get("variant") or r.get("arm") or "", r.get("platform") or "")
        ts = excel._parse_time(r.get("start") or r.get("timestamp"))
        groups.setdefault(key, []).append({"run_id": r["run_id"], "ts": ts.isoformat() if ts else "",
                                           "contended": "" if r.get("contended") in ("uncontended", None) else
                                           r.get("contended", ""), "r": r})
    sheet = Sheet(TIME, [Column(h, k, f) for h, k, f in TIME_COLUMNS])
    order = {"pytorch": 0, "hls": 1}
    for key in sorted(groups, key=lambda k: (order.get(k[0], 9), k[2], k[1], k[3], k[4])):
        for cand, status in pick(groups[key]):
            r = cand["r"]
            sheet.lines.append([r["flow"], r["activity"], r.get("model"), key[3], r.get("platform"), r["seconds"], status,
                                str(r.get("threads") or ""), r.get("condition") or "", r.get("basis"), r["run_id"],
                                when(cand["ts"])[1]])
    return sheet


def build(results_dir: Path | None = None) -> list[Sheet]:
    results = excel.load(results_dir or excel.RESULTS_DIR)
    timing = excel.timing_rows(results)
    excel.mark_contention(results, timing)
    return [paper_sheet(results, timing), summary_sheet(results, timing), per_stage_sheet(results),
            comparison_sheet(results), time_sheet(results, timing)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=excel.RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: results-dir/benchmarks-latest.xlsx")
    parser.add_argument("--csv-dir", type=Path, default=None, help="also write each tab as a CSV here")
    args = parser.parse_args(argv)
    sheets = build(args.results_dir)
    out = args.out or (LATEST_PATH if args.results_dir == excel.RESULTS_DIR else args.results_dir / LATEST_PATH.name)
    data = excel.xlsx_bytes(sheets)
    excel.write_atomically(out, data)
    print(f"{out}: {len(data):,} bytes; " + ", ".join(f"{s.name} {len(s.lines)}" for s in sheets))
    if args.csv_dir:
        import re

        args.csv_dir.mkdir(parents=True, exist_ok=True)
        for sheet in sheets:
            excel.write_sheet_csv(sheet, args.csv_dir / (re.sub(r"[^A-Za-z0-9]+", "_", sheet.name).strip("_").lower() + ".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
