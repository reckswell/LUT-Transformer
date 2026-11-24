"""FP32 per-token negative log-likelihood on a fixed text slice, saved with the token ids, for each arm
(hf, baseline, lut). These files are the reference the quantization screen compares against
(2026-11-23, coordinator's brief for the training/benchmarking thread).

    cd python
    python -m bench.reference_nll --max-eval-chars 187000 --out-dir ../../LUT-Transformer-local/benchmarking/pytorch/reference

What is scored is exactly what bench/sweep.py's perplexity scores: the dataset's evaluation text
(common/data.py eval_text, cut to --max-eval-chars characters at a space), tokenized with the model's
own tokenizer, read in windows of W tokens (the model's training context, 256 for stories15M) that
start every S = W/2 tokens. Token t (t >= 1) is scored once, in the first window that reaches it,
with context tokens[begin_t : t]:

    nll[t-1] = logsumexp(z) - z[token t]      z = float64 copy of the logits at position t-1

so nll.sum() is the sweep's nll_total (up to the order of the float64 additions; checked below).

One file per arm, <out-dir>/<stem>_<arm>.pt, a dict of plain tensors (load with
torch.load(path, weights_only=True)):
    token_ids     int64 (N,)        the evaluation text's token ids (BOS first for Llama)
    window_begin  int32 (N-1,)      begin_t: the start of the window that scored token t
    nll           float64 (N-1,)    nll[t-1] for t = 1 .. N-1, in nats
    top1_correct  bool (N-1,)       the true token t was the argmax at position t-1
    top5_correct  bool (N-1,)       it was among the 5 largest logits (accuracy = mean of the flags)
plus <stem>_manifest.json: the dataset file's sha256, character limit, token counts, window and
stride, each arm's nll sum, perplexity, file sha256, and the run id.

Each arm is also run twice and the two nll tensors compared, so the files are known to be
reproducible on this device (needed if a later sweep recomputes FP32 on the fly, see the manifest's
kl_note).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from bench import sweep
from bench.runlog import Run
from common import data, precision
from reference import local_cache

KL_NOTE = (
    "KL(P_fp32 || Q_config) needs the full next-token distribution of both models at every scored position. "
    "Storing FP32 log-probs for the slice would take N x V x 4 bytes (50,211 x 32,000 x 4 = 6.4 GB in float32, "
    "3.2 GB in float16), so the screen recomputes them: for each window [b, b+W) of token_ids, run the FP32 "
    "baseline arm (cheapest exact reference) and the config under test on the same window, take the positions "
    "that window scores (window_begin == b), compute log_softmax of both in float64 and accumulate "
    "sum_v p (log p - log q) per position. The FP32 forward's own nll for the target tokens must equal this file's "
    "nll exactly (same device, same code): a free integrity check per window. Cost: one extra FP32 forward per "
    "window, about the baseline slice time (0.9 s on the RTX 5080), and it can be shared by several configs "
    "evaluated in the same pass over the windows. A cheaper approximation, if needed: store the FP32 top-k "
    "log-probs plus logsumexp per position (k = 64: 50,211 x 64 x 8 bytes = 26 MB) and bound the tail mass.")


def score_tokens(forward: Any, tokens: Tensor, *, window: int, stride: int,
                 device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(nll float64 (N-1,), window_begin int32 (N-1,), top1 bool (N-1,), top5 bool (N-1,)) with the
    sweep's windowing (measure_perplexity): top1 = the true next token is the argmax, top5 = it is in
    the 5 largest logits."""
    n = tokens.numel()
    nll = torch.empty(n - 1, dtype=torch.float64)
    begins = torch.empty(n - 1, dtype=torch.int32)
    top1 = torch.empty(n - 1, dtype=torch.bool)
    top5 = torch.empty(n - 1, dtype=torch.bool)
    prev_end, begin = 0, 0
    while True:
        end = min(begin + window, n)
        logits = forward(tokens[begin:end][None].to(device))
        first = max(prev_end, begin + 1)
        targets = tokens[first:end].to(device)
        z = logits[0, first - 1 - begin:end - 1 - begin].double()
        nll[first - 1:end - 1] = (torch.logsumexp(z, dim=-1) - z.gather(-1, targets[:, None])[:, 0]).cpu()
        begins[first - 1:end - 1] = begin
        best5 = z.topk(5, dim=-1).indices
        top1[first - 1:end - 1] = (best5[:, 0] == targets).cpu()
        top5[first - 1:end - 1] = (best5 == targets[:, None]).any(dim=-1).cpu()
        del logits, z
        if end == n:
            break
        prev_end, begin = end, begin + stride
    return nll, begins, top1, top5


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stories15m")
    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--max-eval-chars", type=int, default=187_000)
    parser.add_argument("--arms", default="hf,baseline,lut")
    parser.add_argument("--lut-preset", default="all_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=sweep.REPO_ROOT / "results")
    args = parser.parse_args(argv)

    main_start = time.perf_counter()        # wall-clock durations are recorded in the summary (unit s)
    precision.full_fp32()
    settings = sweep.Settings(max_eval_chars=args.max_eval_chars, results_dir=args.results_dir)
    source = sweep.resolve_model(args.model, local_cache.CHECKPOINT_DIR)
    text = sweep.eval_text(args.dataset, settings)
    tokens, _ = sweep.eval_tokens(source, args.dataset, settings)
    load_start = time.perf_counter()
    loaded = sweep.load_model(source, args.device, settings)
    load_s = time.perf_counter() - load_start
    window, stride = sweep.eval_window(source, loaded.cfg)
    stem = f"{source.key.split('/')[-1]}_{args.dataset}_chars{args.max_eval_chars}_fp32"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run = Run(f"reference_nll_{args.model}_{args.dataset}_chars{args.max_eval_chars}", results_dir=args.results_dir,
              devices={"forward": args.device},
              configs={"model": source.key, "weights": source.load_name, "dataset": args.dataset,
                       "char_limit": args.max_eval_chars, "dataset_sha256": text.file_sha256, "window": window,
                       "stride": stride, "arms": args.arms, "lut_preset": args.lut_preset,
                       "out_dir": str(args.out_dir.resolve())})
    manifest: dict[str, Any] = {
        "run_id": run.run_id, "model": source.key, "weights": source.load_name, "dataset": args.dataset,
        "dataset_file": str(text.file), "dataset_sha256": text.file_sha256, "char_limit": args.max_eval_chars,
        "eval_chars": text.chars, "eval_bytes": text.utf8_bytes, "eval_words": text.words,
        "token_ids": tokens.numel(), "scored_tokens": tokens.numel() - 1, "window": window, "stride": stride,
        "windows": sweep.count_windows(tokens.numel(), window, stride), "dtype": "float32 (TF32 off), loss in float64",
        "device": args.device, "kl_note": KL_NOTE, "arms": {}}
    summary: dict[str, Any] = {"token_ids": tokens.numel(), "scored_tokens": tokens.numel() - 1}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        row = sweep.Row(line=0, enabled=True, name=f"ref_{arm}", arm=arm, model=args.model,
                        lut_preset=args.lut_preset if arm == "lut" else None, device=args.device, batch=1,
                        context_lengths=[], decode_tokens=0, repeats=0, benchmarks=["perplexity"],
                        datasets=[args.dataset])
        runner = sweep.make_runner(row, loaded)
        arm_start = time.perf_counter()
        with torch.no_grad():
            nll, begins, top1, top5 = score_tokens(runner.forward, tokens, window=window, stride=stride,
                                                   device=args.device)
            again = score_tokens(runner.forward, tokens, window=window, stride=stride, device=args.device)[0]
        reproducible = bool(torch.equal(nll, again))
        arm_s = time.perf_counter() - arm_start      # both passes over the slice
        path = args.out_dir / f"{stem}_{arm}.pt"
        partial = path.with_name(f".{path.name}.partial")   # atomic: a watcher never sees a half-written file
        torch.save({"token_ids": tokens.clone(), "window_begin": begins, "nll": nll, "top1_correct": top1,
                    "top5_correct": top5}, partial)
        os.replace(partial, path)
        total = float(nll.sum())
        info = {"file": str(path.resolve()), "file_sha256": file_sha256(path), "nll_sum": total,
                "nll_per_token": total / nll.numel(), "perplexity": math.exp(total / nll.numel()),
                "bits_per_byte": total / (math.log(2) * text.utf8_bytes), "reproducible_run_to_run": reproducible,
                "dtype": sweep.row_dtype(row, loaded), "lut_preset": row.lut_preset or None,
                "two_passes_s": arm_s, "top1_accuracy": float(top1.double().mean()),
                "top5_accuracy": float(top5.double().mean())}
        manifest["arms"][arm] = info
        summary[arm] = {k: v for k, v in info.items() if k not in ("file",)}
        run.log(arm=arm, **summary[arm])
        print(f"{arm}: nll sum {total:.6f} over {nll.numel():,} tokens, perplexity {info['perplexity']:.6f}, "
              f"reproducible {reproducible} -> {path}", flush=True)
    names = list(manifest["arms"])
    if "baseline" in names and "lut" in names:
        a = torch.load(manifest["arms"]["baseline"]["file"], weights_only=True)["nll"]
        b = torch.load(manifest["arms"]["lut"]["file"], weights_only=True)["nll"]
        manifest["lut_equals_baseline"] = summary["lut_equals_baseline"] = bool(torch.equal(a, b))
    if "baseline" in names and "hf" in names:
        a = torch.load(manifest["arms"]["baseline"]["file"], weights_only=True)["nll"]
        b = torch.load(manifest["arms"]["hf"]["file"], weights_only=True)["nll"]
        manifest["hf_vs_baseline_max_abs_nll_diff"] = summary["hf_vs_baseline_max_abs_nll_diff"] = \
            float((a - b).abs().max())
    manifest_path = args.out_dir / f"{stem}_manifest.json"
    partial = manifest_path.with_name(f".{manifest_path.name}.partial")    # written last, atomically
    partial.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(partial, manifest_path)
    summary["manifest"] = str(manifest_path.resolve())
    summary["load_s"] = load_s
    summary["main_wall_s"] = time.perf_counter() - main_start   # main() only: python start and imports excluded
    run.finish(summary)
    print(f"manifest {manifest_path}; run {run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
