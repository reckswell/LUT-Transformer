"""Check the HLS kernel's inference records token by token against the FP32 GPU reference files
(2026-11-24). Independent of the pipeline's own comparison in each record's summary.json.

    cd python
    python -m bench.check_hls_inference            # every results/hls/inference/* record with scores.csv.gz

For each record (read only): its scores.csv.gz has one line per scored token (j, window, pos, target,
nll, argmax, rank, top1, top5); j is the index of the scored token, the same as the reference files'
index (token j + 1 of the slice). The reference is ../LUT-Transformer-local/benchmarking/pytorch/
reference/<model>_tinystories_chars<limit>_fp32_baseline.pt (the baseline arm, FP32 on the GPU,
bench/reference_nll.py). Compared: the target tokens, the top-1 and top-5 correct flags, and the
per-token NLL (max and mean absolute difference). Results are appended to results/extra_results.csv
(flow = hls, the record's run id, metric names starting with check_), skipping run ids already
checked, so bench/excel.py shows them next to the pipeline's numbers.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
INFERENCE = RESULTS / "hls" / "inference"
REFERENCE_DIR = REPO.parent / "LUT-Transformer-local" / "benchmarking" / "pytorch" / "reference"
EXTRA = RESULTS / "extra_results.csv"
EXTRA_COLUMNS = ["timestamp_utc", "run_id", "flow", "variant", "arm", "model", "platform", "phase", "context",
                 "layer", "stage", "metric", "value", "unit", "git_commit", "note"]


CHECK_METRICS = ("check_scored_tokens", "check_targets_match", "check_top1_flag_agreement", "check_top5_flag_agreement",
                 "check_top1_flag_disagreements", "check_top5_flag_disagreements", "check_nll_abs_diff_max",
                 "check_nll_abs_diff_mean", "check_reference_top1_same_tokens", "check_reference_top5_same_tokens",
                 "check_reference_nll_sum_same_tokens")


def reference_for(dataset: str) -> Path | None:
    match = re.search(r"chars(\d+)", dataset or "")
    if not match:
        return None
    hits = sorted(REFERENCE_DIR.glob(f"*_tinystories_chars{match.group(1)}_fp32_baseline.pt"))
    return hits[0] if hits else None


def reference_of(summary: dict) -> Path | None:
    """The reference file the record's input vectors were made from (its dataset manifest's
    source_reference), else the one named by the dataset's chars<limit>."""
    manifest = Path(summary.get("dataset_dir") or "") / "manifest.json"
    if summary.get("dataset_dir") and manifest.is_file():
        source = json.loads(manifest.read_text()).get("source_reference")
        if source and Path(source).is_file():
            return Path(source)
    return reference_for(summary.get("dataset", ""))


def check(record_dir: Path) -> dict | None:
    summary = json.loads((record_dir / "summary.json").read_text())
    ref_path = reference_of(summary)
    if ref_path is None or not (record_dir / "scores.csv.gz").is_file():
        return None
    ref = torch.load(ref_path, weights_only=True)
    tokens = ref["token_ids"]
    rows = []
    with gzip.open(record_dir / "scores.csv.gz", "rt") as handle:
        for line in csv.DictReader(handle):
            rows.append((int(line["j"]), int(line["target"]), float(line["nll"]), int(line["top1"]), int(line["top5"])))
    j = torch.tensor([r[0] for r in rows])
    target = torch.tensor([r[1] for r in rows])
    nll = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    top1 = torch.tensor([bool(r[3]) for r in rows])
    top5 = torch.tensor([bool(r[4]) for r in rows])
    return {
        "scored_tokens": len(rows),
        "targets_match": bool(torch.equal(target, tokens[j + 1])),
        "top1_flags_match": int((top1 == ref["top1_correct"][j]).sum()),
        "top5_flags_match": int((top5 == ref["top5_correct"][j]).sum()),
        "nll_abs_diff_max": float((nll - ref["nll"][j]).abs().max()),
        "nll_abs_diff_mean": float((nll - ref["nll"][j]).abs().mean()),
        "top1_accuracy": float(top1.double().mean()), "top5_accuracy": float(top5.double().mean()),
        "reference_top1_accuracy_same_tokens": float(ref["top1_correct"][j].double().mean()),
        "reference_top5_accuracy_same_tokens": float(ref["top5_correct"][j].double().mean()),
        "reference_nll_sum_same_tokens": float(ref["nll"][j].sum()),
        "reference": ref_path.name,
    }


def main() -> int:
    torch.set_num_threads(2)
    done: set[tuple[str, str]] = set()          # (run id, metric) already recorded
    if EXTRA.is_file():
        with open(EXTRA, newline="", encoding="utf-8") as handle:
            done = {(r["run_id"], r["metric"]) for r in csv.DictReader(handle) if r["metric"].startswith("check_")}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for record_dir in sorted(p for p in INFERENCE.glob("*") if (p / "summary.json").is_file()):
        run_id = json.loads((record_dir / "summary.json").read_text()).get("run_id") or record_dir.name
        if all((run_id, m) in done for m in CHECK_METRICS):
            continue
        try:
            result = check(record_dir)
        except Exception as error:            # a record still being written: try again next time
            print(f"{record_dir.name}: not checked ({type(error).__name__}: {error})")
            continue
        if result is None:
            print(f"{record_dir.name}: no per-token scores or no matching reference; skipped")
            continue
        n = result["scored_tokens"]
        note = (f"independent check vs {result['reference']} (FP32 GPU baseline) by bench/check_hls_inference.py; "
                f"targets match: {result['targets_match']}")
        for metric, value, unit in (
                ("check_scored_tokens", n, "tokens"), ("check_targets_match", "yes" if result["targets_match"] else "no", ""),
                ("check_top1_flag_agreement", result["top1_flags_match"] / n, "fraction"),
                ("check_top5_flag_agreement", result["top5_flags_match"] / n, "fraction"),
                ("check_top1_flag_disagreements", n - result["top1_flags_match"], "tokens"),
                ("check_top5_flag_disagreements", n - result["top5_flags_match"], "tokens"),
                ("check_nll_abs_diff_max", result["nll_abs_diff_max"], "nats"),
                ("check_nll_abs_diff_mean", result["nll_abs_diff_mean"], "nats"),
                ("check_reference_top1_same_tokens", result["reference_top1_accuracy_same_tokens"], "fraction"),
                ("check_reference_top5_same_tokens", result["reference_top5_accuracy_same_tokens"], "fraction"),
                ("check_reference_nll_sum_same_tokens", result["reference_nll_sum_same_tokens"], "nats")):
            if (run_id, metric) in done:
                continue
            out.append({"timestamp_utc": stamp, "run_id": run_id, "flow": "hls", "variant": "baseline", "arm": "hls",
                        "model": "stories15m", "platform": "C model", "phase": "accuracy", "context": "",
                        "layer": "", "stage": "whole model", "metric": metric, "value": value, "unit": unit,
                        "git_commit": "", "note": note})
        print(f"{record_dir.name}: {n} tokens, targets match {result['targets_match']}, top-1 flags "
              f"{result['top1_flags_match']}/{n}, top-5 flags {result['top5_flags_match']}/{n}, max |dNLL| "
              f"{result['nll_abs_diff_max']:.3g}")
    if out:
        fresh = not EXTRA.is_file()
        with open(EXTRA, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXTRA_COLUMNS)
            if fresh:
                writer.writeheader()
            writer.writerows(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
