"""Replaceable nonlinear functions inside a stage: the softmax exp, the norms' reciprocal square
root and the feed-forward activation (SiLU, GELU or ReLU). Stage.alt picks one by name per stage,
for example

    SOFTMAX = Stage(alt={"exp": "exp"})          # LUT/host.py
    FFN_ACT = Stage(alt={"act": "silu"}, layers=(0,))

A stage without alt runs the baseline function itself (baseline/inference.py) and nothing here.
A stage with alt runs the stage function below with the named function plugged in.

To try a new approximation (a table for exp over a fixed-point range, a piecewise-linear SiLU,
...), write it below as a function of one tensor, from base operations (no torch.nn), and add its
name to FUNCTIONS under its slot. Each entry is elementwise, takes and returns a tensor of the
same shape and dtype, and must handle what the stage feeds it (softmax gives exp the scores minus
the row max, so every input is <= 0, and -inf for masked keys, which must give 0).

The functions already listed are the baseline math, so a stage set to them computes what the
baseline computes:
    exp    torch.exp, bit-identical to the baseline softmax
    rsqrt  1 / sqrt(v); the norm then multiplies by it, as hardware with an rsqrt table does,
           where the baseline divides by sqrt. The two differ by at most a rounding step.
    act    the baseline relu, gelu and silu, bit-identical
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch
from torch import Tensor

from baseline import inference


def exp(x: Tensor) -> Tensor:
    """e^x, the baseline's exp (torch.exp). exp(-inf) = 0."""
    return torch.exp(x)


def rsqrt(v: Tensor) -> Tensor:
    """1 / sqrt(v) for v > 0, the quantity a norm's rsqrt table holds."""
    return 1.0 / torch.sqrt(v)


# Every replaceable function, by slot, then by name. Add an approximation here to make it
# available to Stage.alt.
FUNCTIONS: dict[str, dict[str, Callable[[Tensor], Tensor]]] = {
    "exp": {"exp": exp},
    "rsqrt": {"rsqrt": rsqrt},
    "act": {"relu": inference.relu, "gelu": inference.gelu, "gelu_exact": inference.gelu_exact, "silu": inference.silu},
}

# Which slot each stage offers.
STAGE_SLOTS: dict[str, str] = {"softmax": "exp", "norm": "rsqrt", "final_norm": "rsqrt", "ffn_act": "act"}


def lookup(slot: str, choice: Any) -> Callable[[Tensor], Tensor]:
    """The function a Stage.alt entry names: a name in FUNCTIONS[slot], or a function itself (for
    a quick experiment or a test)."""
    if callable(choice):
        return choice
    if choice not in FUNCTIONS.get(slot, {}):
        raise ValueError(f"no {slot!r} function named {choice!r} in LUT/nonlinear.py; "
                         f"known: {sorted(FUNCTIONS.get(slot, {}))}")
    return FUNCTIONS[slot][choice]


def name_of(choice: Any) -> str:
    return choice if isinstance(choice, str) else getattr(choice, "__name__", repr(choice))


# ----------------------------------------------------------------------------------------------
# The stage functions with a slot (same signatures as baseline/inference.py, plus the slot)
# ----------------------------------------------------------------------------------------------

def softmax(scores: Tensor, dim: int = -1, *, exp: Callable[[Tensor], Tensor]) -> Tensor:
    """inference.softmax with its exp replaced: p = exp(s - max s) / sum exp(s - max s)."""
    m = scores.amax(dim=dim, keepdim=True)                    # row max
    e = exp(scores - m)                                       # every input <= 0; masked -> exp(-inf) = 0
    return e / e.sum(dim=dim, keepdim=True)


def rmsnorm(x: Tensor, gamma: Tensor, eps: float = 1e-6, *, rsqrt: Callable[[Tensor], Tensor]) -> Tensor:
    """inference.rmsnorm with the reciprocal square root replaced: gamma * x * rsqrt(mean(x^2) + eps)."""
    d = x.shape[-1]
    mean_sq = (x * x).sum(dim=-1, keepdim=True) / d           # (..., 1)
    return gamma * (x * rsqrt(mean_sq + eps))


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5, *,
              rsqrt: Callable[[Tensor], Tensor]) -> Tensor:
    """inference.layernorm with the reciprocal square root replaced."""
    d = x.shape[-1]
    mu = x.sum(dim=-1, keepdim=True) / d
    centered = x - mu
    var = (centered * centered).sum(dim=-1, keepdim=True) / d
    return gamma * (centered * rsqrt(var + eps)) + beta


def stage_function(stage: str, norm: str, alt: dict[str, Any]) -> Callable[..., Tensor]:
    """The inference function for `stage` with the functions alt names plugged in. norm is the
    model's norm ("rmsnorm" or "layernorm")."""
    slot = STAGE_SLOTS[stage]
    fn = lookup(slot, alt[slot])
    if stage == "softmax":
        return functools.partial(softmax, exp=fn)
    if stage in ("norm", "final_norm"):
        return functools.partial(rmsnorm if norm == "rmsnorm" else layernorm, rsqrt=fn)
    return fn                                                  # ffn_act: the activation itself
