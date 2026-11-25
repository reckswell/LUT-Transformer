"""Baseline transformer: every function written from base math, no LUTs.

Files:
    host.py            model definition, stage list and entry points (start reading here)
    train.py           training functions: forward/backward pairs, loss, optimizer, training loop
    inference.py       inference functions: forward ops, KV cache, sampling

The torch.nn ground-truth model is reference/torch_model.py, outside this folder, so everything
here is from-scratch code.

Run from python/:  python -m baseline.host info
"""
