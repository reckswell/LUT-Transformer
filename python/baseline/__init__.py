"""Baseline transformer: every function written from base math, no LUTs.

Files:
    host.py            model definition, stage list and entry points (start reading here)
    train.py           training functions: forward/backward pairs, loss, optimizer, training loop
    inference.py       inference functions: forward ops, KV cache, sampling
    reference_torch.py ground-truth model from torch.nn, for correctness and speed comparison

Run from python/:  python -m baseline.host info
"""
