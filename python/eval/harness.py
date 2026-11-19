"""Run any implementation ("arm") on the same token ids and weights, so the arms can be compared
as controls (decision D9).

Arms:
    "hf"        the Hugging Face model, exactly as `transformers` ships it (reference/hf_gpt2.py,
                reference/hf_llama.py)
    "baseline"  our from-scratch model (baseline/host.py)
    "lut"       our LUT model (LUT/host.py) with a set of toggles: "parameters" (its PARAMETERS
                block, read when the arm runs) or a PRESETS name such as "all_base"

Controls (tests/test_eval_harness.py runs them on small models):
    1. hf vs baseline, same weights: is our from-scratch model right on a real model?
    2. lut "all_base" vs baseline: does the LUT wiring by itself change anything? (It must not.)
    3. lut with one stage set in PARAMETERS vs lut "all_base": the effect of that one stage alone.
       Runs once a LUT architecture exists (D5) or quantization does (D4).

Per arm, run_arm returns the logits, the next-token loss and perplexity on the given tokens, the
wall-clock time of one forward pass (bench/timing.py: warm-up, then repeats, median and spread),
and for the lut arm the toggle table from LUT/ops_registry.describe (what runs in each stage, the
table shape and table bits). Cautions:
  - The timing is PyTorch on a CPU or GPU. For LUT and integer stages it is emulation speed and
    says nothing about FPGA speed (agent.md 2.6); compare LUT variants on modeled hardware cost.
  - The modeled cost is only table storage for now. Lookups, adds and multiplies per stage come
    with the LUT design (D5) and quantization (D4).
  - The loss is computed here with plain torch (log-sum-exp in float64), not with any arm's own
    loss function, so every arm is measured with the same stick.
  - Nothing here writes a run directory yet. The command that evaluates a dataset and logs a run
    under results/runs/ comes with the dataset and tokenizer (D2).
  - Precision: every arm runs in float32 by default, the base metric (Alpha, 2026-11-19), with TF32
    turned off (common/precision.py). Each result records the dtype and the precision flags.
    Tests that check the math pass float64 weights on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from baseline import host as base
from bench.progress import progress_bar
from bench.timing import time_fn
from common import precision
from LUT import host as lut_host
from LUT import ops_registry

ARMS = ("hf", "baseline", "lut")


@dataclass
class Weights:
    """One set of weights that every arm runs.

    cfg, params  our config and parameter dict (host.init_params names and layouts), used by the
                 baseline and lut arms
    hf_model     the Hugging Face model the params were exported from, used by the hf arm. None
                 for models Hugging Face has no version of, such as the core.
    """

    cfg: base.TransformerConfig
    params: dict[str, Tensor]
    hf_model: Any | None = None

    @classmethod
    def from_hf(cls, hf_model: Any, *, dtype: torch.dtype = base.DTYPE) -> "Weights":
        """Export a Hugging Face model's weights into our layout (GPT-2 or Llama, through
        reference/hf_profiles.py), keeping the model for the hf arm. Both are put in `dtype`
        (float32 by default). The Hugging Face model is converted in place (torch's Module.to), so
        every arm runs at the same precision."""
        from reference import hf_profiles   # needs transformers only when an hf model is given

        hf_model = hf_model.to(dtype).eval()
        cfg, params = hf_profiles.export(hf_model)
        return cls(cfg, params, hf_model)

    @property
    def dtype(self) -> torch.dtype:
        """The one float dtype every parameter (and the Hugging Face model, if any) is in."""
        dtypes = {t.dtype for t in self.params.values()}
        if self.hf_model is not None:
            dtypes |= {p.dtype for p in self.hf_model.parameters()}
        if len(dtypes) != 1:
            raise ValueError(f"weights mix dtypes {sorted(map(str, dtypes))}; every arm must run at one precision")
        return dtypes.pop()


@dataclass
class ArmResult:
    """What one arm produced on one batch of tokens.

    logits        (B, T, V), every position
    loss          mean next-token cross-entropy in nats: position t predicts token t+1
    perplexity    exp(loss)
    timing        bench.timing.TimingStats.to_dict() for one forward pass (None if repeats = 0)
    modeled_cost  lut arm only: one row per stage from ops_registry.describe
    """

    arm: str
    preset: str | None
    device: str
    dtype: str                    # the weights' and activations' dtype, "torch.float32" by default
    precision: dict[str, Any]     # common/precision.precision_info() when the arm ran (TF32 flags)
    logits: Tensor
    loss: float
    perplexity: float
    timing: dict[str, Any] | None
    modeled_cost: list[dict[str, str]] | None


def next_token_loss(logits: Tensor, token_ids: Tensor) -> float:
    """Mean cross-entropy of logits[:, t] predicting token_ids[:, t+1], over all B x (T-1)
    positions, in float64: loss = mean(logsumexp(z) - z[target])."""
    z = logits[:, :-1].double()
    targets = token_ids[:, 1:]
    log_probs = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return float(-log_probs.gather(-1, targets[..., None]).mean())


def _forward_fn(arm: str, weights: Weights, token_ids: Tensor, preset: str):
    """A no-argument function that runs one forward pass of the arm and returns logits (B, T, V)."""
    cfg, params = weights.cfg, weights.params
    if arm == "hf":
        if weights.hf_model is None:
            raise ValueError("the hf arm needs Weights.hf_model (use Weights.from_hf); the core has no Hugging Face version")

        def run() -> Tensor:
            with torch.no_grad():
                return weights.hf_model(token_ids).logits
        return run
    if arm == "baseline":
        ops = base.base_ops(cfg, "inference")
        return lambda: base.forward(params, token_ids, ops, cfg)
    if arm == "lut":
        ops = lut_host.lut_ops(cfg, "inference", lut_host.preset_toggles(preset))
        return lambda: lut_host.forward(params, token_ids, ops, cfg)
    raise ValueError(f"unknown arm {arm!r}; arms are {ARMS}")


def run_arm(arm: str, weights: Weights, token_ids: Tensor, *, preset: str = "parameters",
            repeats: int = 5, warmup: int = 2, progress: bool | None = None) -> ArmResult:
    """Run one arm on token_ids (B, T), which must be on the same device as the weights.

    preset applies to the lut arm only: "parameters" reads LUT/host.py's PARAMETERS block at this
    moment (so an override like lut_host.SOFTMAX = Stage(...) is picked up); other names are
    LUT/host.py PRESETS. repeats = 0 skips the timing.

    progress: a tqdm bar with the time remaining over the warm-up and timed calls (bench/progress.py).
    None shows it in a terminal and hides it in pytest and logs; True or False forces it. The bar is
    updated between calls, never inside a timed one.

    token_ids are integer indices (int64) into the token table, not values; float token ids are
    refused. TF32 is turned off first (common/precision.full_fp32), so float32 means full float32.
    """
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token_ids must be int64 indices, got {token_ids.dtype}")
    fp = precision.full_fp32()
    dtype = weights.dtype
    run = _forward_fn(arm, weights, token_ids, preset)
    logits = run()
    loss = next_token_loss(logits, token_ids)
    device = str(token_ids.device)
    timing = None
    if repeats:
        label = f"{arm}" + (f" {preset}" if arm == "lut" else "") + f" forward B={token_ids.shape[0]} T={token_ids.shape[1]}"
        with progress_bar(total=warmup + repeats, desc=label, unit="call", show=progress, leave=False) as bar:
            timing = time_fn(run, device=device, warmup=warmup, repeats=repeats,
                             after_call=lambda: bar.update(1)).to_dict()
    cost = ops_registry.describe(lut_host.preset_toggles(preset), weights.cfg) if arm == "lut" else None
    return ArmResult(arm=arm, preset=preset if arm == "lut" else None, device=device, dtype=str(dtype),
                     precision=fp, logits=logits,
                     loss=loss, perplexity=float(torch.tensor(loss).exp()), timing=timing, modeled_cost=cost)


def compare(a: ArmResult, b: ArmResult) -> dict[str, float]:
    """How far arm a is from arm b on the same tokens: the largest absolute logit difference, the
    fraction of positions where both pick the same top-1 token, and the perplexity ratio a / b."""
    return {
        "max_abs_logit_diff": float((a.logits.double() - b.logits.double()).abs().max()),
        "top1_agreement": float((a.logits.argmax(-1) == b.logits.argmax(-1)).double().mean()),
        "perplexity_ratio": a.perplexity / b.perplexity,
    }
