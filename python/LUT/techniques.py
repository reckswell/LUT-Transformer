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

Status (2026-11-24): written from decision D4 (agent/PLAN.md): INT8 post-training quantization,
symmetric per-channel weights, per-tensor activations, round-half-to-even, saturating, 32-bit
accumulators, with the formats the benchmarking copies use (python/precision, outside this repository)
so the presets reproduce them: INT4 weights in groups of 32 with a 4-bit multiplier per group on a
per-row scale, FP8 E4M3 per tensor, FP16. A function can be put back to "pending" with
@pending("D4"); LUT/ops_registry.py then refuses any stage line that needs it when the model is
built.

Values are "stored number x scale": quantize_input, prepare_weight and kv_encode return a
Quantized (codes, scale). The integer kernels below (integer_stage) are the bit-exact integer
reference: when every input of a linear layer, QK^T, PV or the GLU gate is an integer format whose
scale is constant along the summed axis, the stage multiplies the integer codes, sums them exactly,
applies the accumulator width to the integer sums, and only then scales back:
    y = float(acc) x (s_x x s_w[n])  (+ bias)
which is what the HLS kernels compute with ap_int types. Any other combination (a float weight, a
per-input-channel activation scale, FP8, a LUT or alt function) runs the stage's own float
function on the dequantized values (fake quantization), and an accumulator width there is refused.

Rounding to whole numbers is torch.round (half to even) unless spec.rounding says otherwise; every
conversion saturates unless spec.saturate is False (then integers wrap like two's complement).
Scales are float32, as in the benchmarking copies, whatever DTYPE is; an all-zero range gets the
floor TINY so nothing divides by zero.

Tests replace these functions with stand-ins (tests/test_lut_slots.py) to check that every slot
reaches the right tensor; the stand-ins are not a quantization design.

Every function receives `site`, which names where it is called, so a technique can treat stages,
layers or operands differently and so errors can say where they happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import torch
from torch import Tensor

from common.quant import QuantSpec, int_range


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
# Stored values
# ----------------------------------------------------------------------------------------------

TINY = 1e-12                       # floor for a range, so an all-zero tensor doesn't divide by zero
FLOAT_FORMATS: dict[tuple[int, int], tuple[torch.dtype, float]] = {
    (8, 4): (torch.float8_e4m3fn, 448.0),          # FP8 E4M3 (no infinity; 448 is the largest value)
    (8, 5): (torch.float8_e5m2, 57344.0),          # FP8 E5M2
    (16, 5): (torch.float16, 65504.0),             # FP16
    (16, 8): (torch.bfloat16, 3.3895313892515355e38),   # BF16
}


@dataclass
class Quantized:
    """A tensor as stored: value = codes x scale.

    codes   whole numbers (kind "int") or values of the small float format (kind "float"), held in
            a float tensor of the original dtype (exact: |codes| < 2^24)
    scale   float32, broadcastable to codes: () per tensor, (N, 1) per output channel, (..., 1) per
            token, (C,) or (H, 1, Dh) per input channel
    spec    the QuantSpec it follows
    """

    codes: Tensor
    scale: Tensor
    spec: QuantSpec

    def dequantize(self) -> Tensor:
        return self.codes * self.scale.to(self.codes.dtype)

    @property
    def shape(self) -> torch.Size:
        return self.codes.shape


def _qmax(spec: QuantSpec) -> float:
    """The largest magnitude the format stores: 127 for signed INT8, 255 for unsigned, 448 for E4M3."""
    if spec.kind == "float":
        return _float_format(spec)[1]
    lo, hi = int_range(spec.bits, spec.signed)
    return float(hi)


def _float_format(spec: QuantSpec) -> tuple[torch.dtype, float]:
    fmt = FLOAT_FORMATS.get((spec.bits, spec.exp_bits))
    if fmt is None:
        raise NotImplementedError(f"no rounding written for the float format {spec.label()} "
                                  f"(written: {sorted(FLOAT_FORMATS)} as (bits, exp_bits))")
    return fmt


def round_to_int(v: Tensor, spec: QuantSpec) -> Tensor:
    """v -> whole numbers in the format's range, as spec.rounding and spec.saturate say."""
    if spec.rounding == "nearest_even":
        q = torch.round(v)
    elif spec.rounding == "nearest_away":
        q = torch.sign(v) * torch.floor(v.abs() + 0.5)
    elif spec.rounding == "floor":
        q = torch.floor(v)
    else:                                                # stochastic: floor(v + u), u ~ U[0, 1)
        q = torch.floor(v + torch.rand_like(v))
    lo, hi = int_range(spec.bits, spec.signed)
    if spec.saturate:
        return torch.clamp(q, lo, hi)
    return torch.remainder(q - lo, float(hi - lo + 1)) + lo          # two's complement wrap-around


def round_to_float(v: Tensor, spec: QuantSpec) -> Tensor:
    """v -> the nearest value of the small float format (ties to even), saturating at its largest
    value (an unclamped cast of 500 to E4M3 gives NaN)."""
    if spec.rounding != "nearest_even":
        raise NotImplementedError(f"{spec.label()}: float formats round to nearest even only")
    dtype, fmax = _float_format(spec)
    if spec.saturate:
        v = torch.clamp(v, -fmax, fmax)
    return v.to(dtype).to(v.dtype)


def _to_codes(v: Tensor, spec: QuantSpec) -> Tensor:
    return round_to_float(v, spec) if spec.kind == "float" else round_to_int(v, spec)


def _check_symmetric(spec: QuantSpec, site: Site) -> None:
    if not spec.symmetric:
        raise NotImplementedError(f"{site.key()}: asymmetric formats (a zero point) are not written; D4 is symmetric")


def _scale_from_range(amax: Any, spec: QuantSpec) -> Tensor:
    """float32 scale = max(range, TINY) / qmax. A Python float range is divided in double and
    rounded once to float32, as the benchmarking copies do; a tensor range is divided in float32."""
    if isinstance(amax, Tensor):
        return torch.clamp(amax.to(torch.float32), min=TINY) / _qmax(spec)
    return torch.tensor(max(float(amax), TINY) / _qmax(spec), dtype=torch.float32)


# ----------------------------------------------------------------------------------------------
# Run-time inputs: Stage.quant and Stage.quant_b
# ----------------------------------------------------------------------------------------------

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
    _check_symmetric(spec, site)
    if spec.scale_mode == "none":
        return torch.tensor(2.0 ** -(spec.frac_bits or 0) if spec.kind == "int" else 1.0, dtype=torch.float32,
                            device=x.device)
    g = spec.granularity
    if spec.scale_mode == "dynamic":
        a = x.detach().abs().to(torch.float32)
        if g == "per_tensor":
            amax = a.amax()
        elif g == "per_token":
            amax = a.amax(dim=-1, keepdim=True)                      # one per row
        elif g == "per_channel" and a.dim() == 4:              # q, k, v (B, H, T, Dh): per head and column
            amax = a.amax(dim=(0, 2)).unsqueeze(1)                   # (H, 1, Dh)
        elif g == "per_channel":                                     # per column of the last axis
            amax = a.reshape(-1, a.shape[-1]).amax(dim=0)
        else:
            raise NotImplementedError(f"{site.key()}: per_group scales are for weights (prepare_weight)")
        return _scale_from_range(amax, spec)
    # static
    if g == "per_tensor":
        amax = ranges.absmax if spec.percentile is None else ranges.percentile(spec.percentile)
        return _scale_from_range(amax, spec).to(x.device)
    if g == "per_channel":
        if spec.percentile is not None or ranges.channel_absmax is None:
            raise ValueError(f"{site.key()}: a static per-channel scale needs per-channel max |x| from calibration "
                             "(and no percentile)")
        ch = ranges.channel_absmax.to(torch.float32)
        if ch.dim() == 2:                                            # (H, Dh) for q, k, v
            ch = ch.reshape(ch.shape[0], 1, ch.shape[1])
        return _scale_from_range(ch, spec).to(x.device)
    raise ValueError(f"{site.key()}: a static scale is per_tensor or per_channel; per_token scales are dynamic")


def quantize_input(x: Tensor, spec: QuantSpec, scale: Tensor, site: Site) -> Quantized:
    """x as the stage's quantized input: Quantized(codes, scale) with codes = round(x / scale) per
    spec.rounding, saturated or wrapped per spec.saturate (x is multiplied by the float32
    reciprocal of the scale, as hardware multiplies by a constant). Its dequantize() is the value
    the format represents, which the float path computes on; the integer kernels use the codes."""
    inv = torch.tensor(1.0, dtype=torch.float32, device=scale.device) / scale
    return Quantized(_to_codes(x * inv.to(x.dtype), spec), scale, spec)


def accumulator(y: Tensor, spec: QuantSpec, site: Site, inputs: tuple[Any, ...]) -> Tensor:
    """The stage's output y as an accumulator of spec.acc_bits bits would leave it: wrapped or
    saturated (per spec.saturate) where the integer sum overflows. `inputs` are the quantized
    inputs the stage received, from which the integer sum's scale follows. Modeling the integer
    sum exactly is the integer stage function's job (with D4); this is where its width applies.

    y must be the integer sums from integer_stage (every input a Quantized integer format); the
    stage then scales them back. A float output can't be wrapped meaningfully, so it is refused."""
    if not inputs or not all(isinstance(i, Quantized) and i.spec.kind == "int" for i in inputs):
        raise ValueError(f"{site.key()}: acc_bits needs every input of the stage in an integer format with a scale "
                         "that is constant along the summed axis (the integer kernel); this stage runs in float")
    lo, hi = int_range(spec.acc_bits, True)
    acc = y.to(torch.float64)                            # whole numbers, exact in float64
    out = torch.clamp(acc, lo, hi) if spec.saturate else torch.remainder(acc - lo, float(hi - lo + 1)) + lo
    # Back to y's dtype exactly: a float32 y holds sums below 2^24 (_matmul_exact), and wrapping or
    # clamping never makes them larger unless acc_bits <= 25, which still fits.
    return out.to(y.dtype)


# ----------------------------------------------------------------------------------------------
# Integer kernels: the bit-exact reference for linear layers, QK^T, PV and the GLU gate
# ----------------------------------------------------------------------------------------------

INTEGER_STAGES = ("qkv_proj", "out_proj", "ffn_up", "ffn_down", "lm_head", "attn_scores", "attn_values", "ffn_gate")


def integer_kernel(stage: str, quant: QuantSpec | None, quant_b: QuantSpec | None, weight: Any) -> bool:
    """Whether the stage can run as integer arithmetic for these specs: every input an integer
    format whose scale doesn't vary along the summed axis (decided from the specs, when the model
    is built)."""
    def ints(*specs: Any) -> bool:
        return all(isinstance(sp, QuantSpec) and sp.kind == "int" and sp.symmetric for sp in specs)

    if stage not in INTEGER_STAGES:
        return False
    if stage in ("qkv_proj", "out_proj", "ffn_up", "ffn_down", "lm_head"):
        if not ints(quant, weight) or quant.granularity == "per_channel":      # x's scale must not vary along K
            return False
        return weight.granularity in ("per_tensor", "per_channel") and weight.axis in (None, 0) or \
            (weight.granularity == "per_group" and weight.scale_bits is not None)
    if not ints(quant, quant_b):
        return False
    if stage == "attn_scores":                    # summed over Dh: per tensor or per token (row) only
        return quant.granularity in ("per_tensor", "per_token") and quant_b.granularity in ("per_tensor", "per_token")
    if stage == "attn_values":                    # summed over keys: P per tensor/row, V per tensor/channel
        return quant.granularity in ("per_tensor", "per_token") and quant_b.granularity in ("per_tensor", "per_channel")
    return quant.granularity != "per_group" and quant_b.granularity != "per_group"          # ffn_gate: elementwise


def _matmul_exact(a: Tensor, b: Tensor, bound: float) -> Tensor:
    """a @ b of whole numbers, exactly: in the inputs' dtype when every partial sum stays below
    2^24 (float32 adds whole numbers exactly up to there) or the dtype is float64, else in float64."""
    if bound < 2 ** 24 or a.dtype == torch.float64:
        return a @ b
    return a.to(torch.float64) @ b.to(torch.float64)


def _code_max(q: Quantized) -> float:
    lo, hi = int_range(q.spec.bits, q.spec.signed)
    m = float(max(-lo, hi))
    if q.spec.granularity == "per_group" and q.spec.scale_bits is not None:
        m *= float(1 << q.spec.scale_bits)                    # INT4 codes folded with their group multiplier
    return m


def integer_stage(stage: str, args: list[Any], kwargs: dict[str, Any], accumulate: Callable[[Tensor], Tensor]) -> Tensor:
    """Run a stage on Quantized integer inputs: exact integer sums of the codes, then
    accumulate(acc) (the accumulator width), then one multiply by the combined float32 scale.

        linear       acc = q_x @ q_w^T,  y = acc x (s_x x s_w[n]) + b
        attn_scores  acc = q_q @ q_k^T,  s = acc x (s_q x s_k) / sqrt(Dh), masked to -inf
        attn_values  acc = q_p @ q_v,    out = acc x (s_p x s_v)
        ffn_gate     acc = q_a x q_b (elementwise), y = acc x (s_a x s_b)
    The float32 multiply by the scale is the only rounding after the integer sums."""
    if stage in ("qkv_proj", "out_proj", "ffn_up", "ffn_down", "lm_head"):
        x, w = args[0], args[1]
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        dtype = x.codes.dtype
        k = x.shape[-1]
        acc = accumulate(_matmul_exact(x.codes, w.codes.to(dtype).transpose(0, 1), k * _code_max(x) * _code_max(w)))
        s_w = w.scale.reshape(-1) if w.scale.dim() else w.scale              # (N,) or ()
        out_scale = x.scale * s_w                                           # per token: (..., 1) x (N,)
        y = acc.to(dtype) * out_scale.to(dtype)
        return y + bias if bias is not None else y
    a, b = args[0], args[1]
    dtype = a.codes.dtype
    if stage == "attn_scores":
        mask = args[2] if len(args) > 2 else kwargs.get("mask")
        dh = a.shape[-1]
        acc = accumulate(_matmul_exact(a.codes, b.codes.to(dtype).transpose(-2, -1), dh * _code_max(a) * _code_max(b)))
        s_b = b.scale.transpose(-2, -1) if b.scale.dim() >= 2 else b.scale
        s = acc.to(dtype) * (a.scale * s_b).to(dtype) / math.sqrt(dh)
        if mask is not None:
            s = torch.where(mask, s, torch.full_like(s, float("-inf")))
        return s
    if stage == "attn_values":
        n = a.shape[-1]
        acc = accumulate(_matmul_exact(a.codes, b.codes.to(dtype), n * _code_max(a) * _code_max(b)))
        s_b = b.scale.reshape(*b.scale.shape[:-3], b.scale.shape[-3], 1, b.scale.shape[-1]) if b.scale.dim() == 3 else b.scale
        return acc.to(dtype) * (a.scale * s_b).to(dtype)
    # ffn_gate
    acc = accumulate(a.codes * b.codes.to(dtype))
    return acc.to(dtype) * (a.scale * b.scale).to(dtype)


# ----------------------------------------------------------------------------------------------
# Weights, prepared once: Stage.weight and Stage.smooth
# ----------------------------------------------------------------------------------------------

def prepare_weight(w: Tensor, spec: Any, site: Site) -> Quantized:
    """The weight the stage multiplies by, prepared offline: w (D_out, D_in) for a linear stage,
    the token table (V, D) for embedding, gamma (D,) for a norm. For a QuantSpec, the values the
    format represents (fake-quantized, same shape and dtype), for example INT8 per output
    channel, or INT4 in groups of 32 with a scale_bits multiplier per group. A different `spec`
    type (a codebook, say) is dispatched here too. Called once per weight; the stage keeps the
    result and leaves the original parameter untouched.

    Returns Quantized(codes, scale). Scales are from the weight itself (scale_mode "static" or
    "dynamic" mean the same for a weight):
        per_tensor            s = max |w| / qmax
        per_channel, axis 0   s[n] = max_k |w[n, k]| / qmax, scale (N, 1): one per output row (and
                              one per token row for the table)
        per_group, size G     along the last axis. scale_bits None: a float32 scale per group,
                              s_g = max |w| over the group / qmax. scale_bits m: an m-bit
                              multiplier r per group on a per-row scale (the benchmarking copies'
                              INT4): s_row = max_g s_g, r = clamp(ceil(2^m s_g / s_row), 1, 2^m),
                              q = round(w / (r s_row / 2^m)), codes = q r, scale = s_row / 2^m. So
                              the codes stay small integers (q r in [-128, 112] for INT4) and one
                              integer matmul plus one per-row scale computes the layer.
        scale_mode "none"     FP16 / BF16, or fixed point: codes are the values (/ 2^-frac_bits)
    The weight is divided by its scale (w / s), as the benchmarking copies do."""
    if not isinstance(spec, QuantSpec):
        raise NotImplementedError(f"{site.key()}: no preparation written for a {type(spec).__name__}")
    _check_symmetric(spec, site)
    w32 = w.detach().to(torch.float32)
    if spec.scale_mode == "none":
        scale = torch.tensor(2.0 ** -(spec.frac_bits or 0) if spec.kind == "int" else 1.0, dtype=torch.float32,
                             device=w.device)
        return Quantized(_to_codes(w32 / scale, spec).to(w.dtype), scale, spec)
    qmax = _qmax(spec)
    g = spec.granularity
    if g == "per_tensor" or w.dim() == 1 and g != "per_group":
        s = torch.clamp(w32.abs().amax(), min=TINY) / qmax
        return Quantized(_to_codes(w32 / s, spec).to(w.dtype), s, spec)
    if g == "per_channel":
        if spec.axis not in (None, 0):
            raise NotImplementedError(f"{site.key()}: per_channel weights are per output row (axis 0)")
        s = torch.clamp(w32.abs().amax(dim=1), min=TINY) / qmax
        return Quantized(_to_codes(w32 / s[:, None], spec).to(w.dtype), s[:, None], spec)
    if g == "per_group":
        n, k = w32.shape
        size = spec.group_size
        if k % size:
            raise ValueError(f"{site.key()}: {k} columns are not a multiple of the group size {size}")
        wg = w32.reshape(n, k // size, size)
        s_g = torch.clamp(wg.abs().amax(dim=-1), min=TINY) / qmax                      # (N, G)
        if spec.scale_bits is None:
            codes = _to_codes(wg / s_g[..., None], spec)
            return Quantized(codes.reshape(n, k).to(w.dtype), s_g.repeat_interleave(size, dim=1), spec)
        if spec.kind != "int":
            raise NotImplementedError(f"{site.key()}: scale_bits multipliers are written for integer formats")
        steps = float(1 << spec.scale_bits)
        s_row = s_g.amax(dim=-1)                                                        # (N,)
        r = torch.clamp(torch.ceil(steps * s_g / s_row[:, None]), 1.0, steps)            # (N, G) whole numbers
        kernel_scale = s_row / steps
        q = _to_codes(wg / (r * kernel_scale[:, None])[..., None], spec)
        return Quantized((q * r[..., None]).reshape(n, k).to(w.dtype), kernel_scale[:, None], spec)
    raise ValueError(f"{site.key()}: per_token is a run-time granularity, not a weight's")


def smoothing_factors(act_absmax: Tensor, weight_absmax: Tensor, alpha: float, site: Site) -> Tensor:
    """SmoothQuant's per-input-channel factors s (D_in,), with x_j / s_j and W[:, j] * s_j, which
    leaves x @ W^T unchanged while moving range from the activations to the weights.
    act_absmax (D_in,): the largest |x_j| seen in calibration at this site. weight_absmax
    (D_in,): the largest |W[:, j]| over every weight that shares this input (wq, wk and wv for
    qkv_proj; w1 and w3 for ffn_up). The published form is s_j = act_j^alpha / weight_j^(1-alpha).

    Written as SmoothQuant's reference code does: both ranges floored at 1e-5, and s floored at
    1e-5, so a dead channel doesn't divide by zero."""
    act = torch.clamp(act_absmax.to(torch.float32), min=1e-5)
    wmax = torch.clamp(weight_absmax.to(torch.float32), min=1e-5)
    return torch.clamp(act.pow(alpha) / wmax.pow(1.0 - alpha), min=1e-5)


# ----------------------------------------------------------------------------------------------
# KV cache storage: KVStore.k and KVStore.v
# ----------------------------------------------------------------------------------------------

def kv_encode(x: Tensor, spec: QuantSpec, site: Site, ranges: Any) -> Quantized:
    """What the cache stores for new keys or values x (B, H, T, Dh): for example integer codes and
    their scales. Called once per token, when it is written; ranges are this site's calibrated
    ranges for a static spec, None for a dynamic one.

    Scales as input_scale gives them: static per tensor or per channel (per head and column,
    (H, 1, Dh)) from calibration; dynamic per tensor, or per token (one scale per head and
    position, over Dh: (B, H, T, 1), what KV-cache quantization papers call per-token)."""
    return quantize_input(x, spec, input_scale(x, spec, site, ranges), site)


def kv_decode(stored: Quantized, spec: QuantSpec, site: Site) -> Tensor:
    """The keys or values (B, H, T, Dh) in DTYPE that attention reads back from what kv_encode
    stored. The model runs decode(encode(x)) when x is written, which gives every later read the
    same values as decoding on each read would."""
    return stored.dequantize()
