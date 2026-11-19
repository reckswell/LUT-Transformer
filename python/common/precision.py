"""Floating-point precision for every run: plain IEEE float32 (Alpha, 2026-11-19).

Until quantization is designed (decision D4), every value the model computes on is float32:
parameters, activations, position codes, the KV cache and the logits. That is the base metric the
LUT and quantized variants will be compared against. Token ids are integer indices into the token
table (int64), not values, so they are never converted to float or quantized.

On NVIDIA GPUs, PyTorch can run float32 matrix multiplies and convolutions in TF32, which keeps
float32's exponent range but rounds each input to a 10-bit mantissa (float32 has 23 bits). That
would make the "FP32" baseline quietly less precise on the GPU than on the CPU, so every run path
(the hosts' command lines, eval/harness.py, bench/run_benchmark.py) calls full_fp32() first, and
the flags are recorded in each run's metadata. Nothing in the model uses autocast or half precision
(tests/test_fp32_defaults.py checks the source).

The dtype itself is set where each model is defined: DTYPE in baseline/host.py, and DTYPE in the
PARAMETERS block of LUT/host.py.
"""

from __future__ import annotations

from typing import Any

import torch


def full_fp32() -> dict[str, Any]:
    """Turn TF32 off for float32 matmuls and convolutions (CPU and GPU), so float32 means full
    float32 everywhere. Returns precision_info() afterwards, for the run's metadata."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return precision_info()


def precision_info() -> dict[str, Any]:
    """The settings that decide how float32 math runs right now."""
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
