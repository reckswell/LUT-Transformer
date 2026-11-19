"""Evaluation harness: runs any arm (Hugging Face, baseline, LUT) on the same tokens and weights.

Files:
    harness.py  run_arm (one entry point for every arm), Weights, ArmResult, compare

Usage, from python/:

    from eval.harness import Weights, run_arm, compare
    w = Weights.from_hf(hf_model)                 # or Weights(cfg, params) for the core
    hf, ours = run_arm("hf", w, ids), run_arm("baseline", w, ids)
    print(compare(ours, hf), ours.perplexity, ours.timing["median_ms"])
"""
