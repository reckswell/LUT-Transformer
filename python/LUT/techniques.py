"""The precision techniques the LUT model's slots call, one function each, for Alpha to write
(decision D4). Each function's docstring is its contract: what it receives, what it returns and
what hardware it stands for.

Which slot calls which function (the slots are the fields of LUT/stages.py's Stage and KVStore):

    slot                         function(s) here                    when it runs
    Stage.quant, Stage.quant_b   input_scale, quantize_input         every call of the stage
    QuantSpec.acc_bits           accumulator                         every call, on the stage's output
    Stage.weight                 prepare_weight                      once per weight (then cached)
    Stage.smooth                 smoothing_factors                   once per site (then cached)
    KVStore.k, KVStore.v         kv_encode, kv_decode                every key / value written to the cache
    Stage.alt                    LUT/nonlinear.py (not here)         every call of the stage

A slot left at its default never calls its function: the stage runs the baseline function itself
(LUT/ops_registry.py), so the defaults are bit-identical to the baseline and exactly as fast. Each
function here is called only once a stage line in LUT/host.py sets its slot.

Status: every function below is marked @pending("D4") and raises NotImplementedError, because the
quantization technique is Alpha's. LUT/ops_registry.py refuses a stage line that needs a pending
function when the model is built, before a run starts. To fill one in, write its body and delete
its @pending line. The decisions D4 has already fixed (agent/PLAN.md, 2026-11-19): INT8
post-training quantization, symmetric per-channel weights, per-tensor activations,
round-half-to-even, saturating, 32-bit accumulators. common/quant.py has int_range and
accumulator_bits, and stubs for choose_scale / quantize / dequantize / fake_quantize.

Tests replace these functions with stand-ins (tests/test_lut_slots.py) to check that every slot
reaches the right tensor; the stand-ins are not a quantization design.

Every function receives `site`, which names where it is called, so a technique can treat stages,
layers or operands differently and so errors can say where they happened.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from torch import Tensor

from common.quant import QuantSpec


class Site(NamedTuple):
    """Where in the model a technique runs.

    stage    a stage name from baseline/host.py STAGES, or "kv_cache"
    layer    decoder layer counted from 0, or None for embedding, final_norm and lm_head
    operand  which input: "x" (the only input), "q" / "k" (attn_scores), "p" / "v" (attn_values),
             "a" / "b" (ffn_gate), "s" (softmax), "w" (a weight, or a norm's gamma), "table"
             (the token table), "k" / "v" (kv_cache)
    """

    stage: str
    layer: int | None
    operand: str

    def key(self) -> str:
        """"qkv_proj[0].x", "lm_head.x", "kv_cache[3].k": the name used in calibration files."""
        where = self.stage if self.layer is None else f"{self.stage}[{self.layer}]"
        return f"{where}.{self.operand}"


def pending(decision: str) -> Callable[[Callable], Callable]:
    """Mark a technique as not written yet. LUT/ops_registry.py checks the mark when a model is
    built, so a stage line that needs it fails at start-up with the decision's name."""
    def mark(fn: Callable) -> Callable:
        fn.pending = decision
        return fn
    return mark


def require(fn: Callable, setting: str) -> None:
    """Raise NotImplementedError if fn is still marked @pending (called at start-up)."""
    decision = getattr(fn, "pending", None)
    if decision is not None:
        raise NotImplementedError(
            f"{setting} needs LUT/techniques.{fn.__name__}, which is pending Alpha (decision {decision})")


# ----------------------------------------------------------------------------------------------
# Run-time inputs: Stage.quant and Stage.quant_b
# ----------------------------------------------------------------------------------------------

@pending("D4")
def input_scale(x: Tensor, spec: QuantSpec, site: Site, ranges: Any) -> Tensor:
    """The scale (and so the clipping range) for quantizing x, one stage input.

    spec.scale_mode "static": from the calibrated ranges of this site (`ranges`, a
    LUT/observer.py Ranges recorded on calibration text, never on evaluation text); spec.percentile
    None uses the largest |x| seen, a number uses that percentile of |x| (percentile vs max
    calibration). Called once; the stage keeps the result.
    spec.scale_mode "dynamic": from x itself, on every call (ranges is None).
    spec.scale_mode "none": no scale (1, or 2^-frac_bits for fixed point); called once.
    Shape follows spec.granularity: () per tensor, one per channel (last axis, or per head and
    channel for q, k, v), one per token (row).
    For kind "float" the scale maps the range onto the format's largest finite value (448 for E4M3).
    """
    raise NotImplementedError("techniques.input_scale")


@pending("D4")
def quantize_input(x: Tensor, spec: QuantSpec, scale: Tensor, site: Site) -> Tensor:
    """x as the stage's quantized input: a tensor of x's shape and dtype holding only values the
    format can represent (q * scale, with q rounded per spec.rounding and saturated or wrapped
    per spec.saturate), so the stage function that follows computes on the grid. For an integer
    format whose products and sums fit float32's 24-bit mantissa, the float matmul that follows is
    then the exact integer result times the scales."""
    raise NotImplementedError("techniques.quantize_input")


@pending("D4")
def accumulator(y: Tensor, spec: QuantSpec, site: Site, inputs: tuple[Tensor, ...]) -> Tensor:
    """The stage's output y as an accumulator of spec.acc_bits bits would leave it: wrapped or
    saturated (per spec.saturate) where the integer sum overflows. `inputs` are the quantized
    inputs the stage received, from which the integer sum's scale follows. Modeling the integer
    sum exactly is the integer stage function's job (with D4); this is where its width applies."""
    raise NotImplementedError("techniques.accumulator")


# ----------------------------------------------------------------------------------------------
# Weights, prepared once: Stage.weight and Stage.smooth
# ----------------------------------------------------------------------------------------------

@pending("D4")
def prepare_weight(w: Tensor, spec: Any, site: Site) -> Tensor:
    """The weight the stage multiplies by, prepared offline: w (D_out, D_in) for a linear stage,
    the token table (V, D) for embedding, gamma (D,) for a norm. For a QuantSpec, the values the
    format represents (fake-quantized, same shape and dtype), for example INT8 per output
    channel, or INT4 in groups of 32 with a scale_bits multiplier per group. A different `spec`
    type (a codebook, say) is dispatched here too. Called once per weight; the stage keeps the
    result and leaves the original parameter untouched."""
    raise NotImplementedError("techniques.prepare_weight")


@pending("D4")
def smoothing_factors(act_absmax: Tensor, weight_absmax: Tensor, alpha: float, site: Site) -> Tensor:
    """SmoothQuant's per-input-channel factors s (D_in,), with x_j / s_j and W[:, j] * s_j, which
    leaves x @ W^T unchanged while moving range from the activations to the weights.
    act_absmax (D_in,): the largest |x_j| seen in calibration at this site. weight_absmax
    (D_in,): the largest |W[:, j]| over every weight that shares this input (wq, wk and wv for
    qkv_proj; w1 and w3 for ffn_up). The published form is s_j = act_j^alpha / weight_j^(1-alpha)."""
    raise NotImplementedError("techniques.smoothing_factors")


# ----------------------------------------------------------------------------------------------
# KV cache storage: KVStore.k and KVStore.v
# ----------------------------------------------------------------------------------------------

@pending("D4")
def kv_encode(x: Tensor, spec: QuantSpec, site: Site, ranges: Any) -> Any:
    """What the cache stores for new keys or values x (B, H, T, Dh): for example integer codes and
    their scales. Called once per token, when it is written; ranges are this site's calibrated
    ranges for a static spec, None for a dynamic one."""
    raise NotImplementedError("techniques.kv_encode")


@pending("D4")
def kv_decode(stored: Any, spec: QuantSpec, site: Site) -> Tensor:
    """The keys or values (B, H, T, Dh) in DTYPE that attention reads back from what kv_encode
    stored. The model runs decode(encode(x)) when x is written, which gives every later read the
    same values as decoding on each read would."""
    raise NotImplementedError("techniques.kv_decode")
