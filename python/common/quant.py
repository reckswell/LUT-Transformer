"""Fixed-point and integer quantization, shared by the training and inference variants.

The goal is to model hardware arithmetic exactly in PyTorch: values are stored as integers of a
stated bit width, with an explicit scale, rounding mode and saturation, and products are
accumulated in integers of a stated width. A later HLS port uses ap_int / ap_fixed with the same
parameters, so the PyTorch numbers predict the hardware numbers bit for bit.

Alpha's direction: the quantization technique stays fairly fixed while LUT architectures vary.
Which technique (bit widths, symmetric or asymmetric, per-tensor or per-channel, rounding) is a
window 2 decision; QuantSpec records the choice so every run logs it.

QuantSpec only describes a format. The LUT model's slots (LUT/stages.py) say where a QuantSpec
applies (a stage's run-time inputs, its weights, the KV cache), and LUT/techniques.py holds the
functions that carry it out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor

Rounding = Literal["nearest_even", "nearest_away", "floor", "stochastic"]
Granularity = Literal["per_tensor", "per_channel", "per_token", "per_group"]
Kind = Literal["int", "float"]
ScaleMode = Literal["static", "dynamic", "none"]


@dataclass(frozen=True)
class QuantSpec:
    """How one tensor is quantized. Logged with every run.

    bits: total width.  signed: two's complement if True.  symmetric: zero point fixed at 0.
    granularity / axis / group_size: which elements share one scale.  rounding: how real values
    map to integers.  saturate: clamp to the integer range (True) or wrap like raw hardware.

    Added 2026-11-24 for the LUT model's slots (every default keeps the meaning above, and the
    fields above keep their names and defaults, because run records log them):
      kind         "int": an integer of `bits` bits.  "float": a small float of `bits` bits with
                   exp_bits exponent bits and bits - 1 - exp_bits mantissa bits (FP8 E4M3: bits=8,
                   exp_bits=4; FP16: bits=16, exp_bits=5; BF16: bits=16, exp_bits=8).
      exp_bits     exponent width, kind "float" only.
      granularity  adds "per_token": one scale per row (each token's vector), for activations
                   and the KV cache.
      scale_mode   "static": a fixed scale, from calibration for activations and the KV cache
                   (LUT/observer.py) or from the tensor itself for weights.  "dynamic": taken
                   from each tensor at run time.  "none": no scale; the format holds the value
                   itself (FP16 or BF16, or fixed point with frac_bits fraction bits).
      frac_bits    scale_mode "none" with kind "int": fixed point, value = q * 2^-frac_bits
                   (ap_fixed<bits, bits - frac_bits> in HLS). None means 0.
      percentile   static activation scales only: None maps the largest |x| seen in calibration
                   to the top of the range; for example 99.99 maps that percentile of |x|, and
                   larger values saturate.
      scale_bits   width of each stored scale.  None: a float32 scale.  An integer: the scale is
                   itself stored in that many bits, for example per_group with scale_bits=4 is a
                   4-bit multiplier per group on top of a per-row float scale.
      acc_bits     width of the accumulator of the multiply-accumulate this input feeds (None:
                   wide enough never to overflow). On a two-operand stage, set it on the first
                   operand's spec.
    """

    bits: int = 8
    signed: bool = True
    symmetric: bool = True
    granularity: Granularity = "per_tensor"
    axis: int | None = None
    group_size: int | None = None
    rounding: Rounding = "nearest_even"
    saturate: bool = True
    kind: Kind = "int"
    exp_bits: int | None = None
    scale_mode: ScaleMode = "static"
    percentile: float | None = None
    scale_bits: int | None = None
    acc_bits: int | None = None
    frac_bits: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        """Raise ValueError if the fields contradict each other (checked when a model is built)."""
        if self.bits < 1:
            raise ValueError(f"bits must be at least 1, got {self.bits}")
        if self.kind not in ("int", "float"):
            raise ValueError(f"kind must be int or float, got {self.kind!r}")
        if self.kind == "float":
            if self.exp_bits is None or not 1 <= self.exp_bits <= self.bits - 2:
                raise ValueError(f"a float format needs 1 <= exp_bits <= bits - 2, got exp_bits={self.exp_bits}, bits={self.bits}")
        elif self.exp_bits is not None:
            raise ValueError("exp_bits is for kind='float' only")
        if self.granularity not in ("per_tensor", "per_channel", "per_token", "per_group"):
            raise ValueError(f"unknown granularity {self.granularity!r}")
        if self.granularity == "per_group" and (self.group_size is None or self.group_size < 1):
            raise ValueError("per_group needs group_size >= 1")
        if self.scale_mode not in ("static", "dynamic", "none"):
            raise ValueError(f"scale_mode must be static, dynamic or none, got {self.scale_mode!r}")
        if self.frac_bits is not None and (self.kind != "int" or self.scale_mode != "none"):
            raise ValueError("frac_bits is for fixed point: kind='int' with scale_mode='none'")
        if self.percentile is not None and not 0.0 < self.percentile <= 100.0:
            raise ValueError(f"percentile must be in (0, 100], got {self.percentile}")
        if self.scale_bits is not None and self.scale_bits < 1:
            raise ValueError("scale_bits must be at least 1")
        if self.acc_bits is not None and self.acc_bits < 2:
            raise ValueError("acc_bits must be at least 2")
        if self.rounding not in ("nearest_even", "nearest_away", "floor", "stochastic"):
            raise ValueError(f"unknown rounding {self.rounding!r}")

    def label(self) -> str:
        """A short name for tables and logs, for example "int8/tensor/static/p99.99/acc32" or
        "e4m3/tensor"."""
        if self.kind == "float":
            mant = self.bits - 1 - (self.exp_bits or 0)
            name = {(16, 5): "fp16", (16, 8): "bf16"}.get((self.bits, self.exp_bits), f"e{self.exp_bits}m{mant}")
        else:
            name = f"{'int' if self.signed else 'uint'}{self.bits}"
        parts = [name, self.granularity.removeprefix("per_")]
        if self.granularity == "per_group":
            parts[-1] = f"g{self.group_size}"
        if self.scale_bits is not None:
            parts.append(f"s{self.scale_bits}")
        if self.scale_mode == "none":                       # no scale: fp16, or fixed point int16/q8
            parts = [name] if self.kind == "float" else [name, f"q{self.frac_bits or 0}"]
        if self.scale_mode == "dynamic":
            parts.append("dynamic")
        if self.percentile is not None:
            parts.append(f"p{self.percentile:g}")
        if not self.symmetric:
            parts.append("asym")
        if self.rounding != "nearest_even":
            parts.append(self.rounding)
        if not self.saturate:
            parts.append("wrap")
        if self.acc_bits is not None:
            parts.append(f"acc{self.acc_bits}")
        return "/".join(parts)


def int_range(bits: int, signed: bool) -> tuple[int, int]:
    """Smallest and largest representable integer: signed b bits -> [-2^(b-1), 2^(b-1) - 1],
    unsigned -> [0, 2^b - 1]."""
    if bits < 1:
        raise ValueError("bits must be at least 1")
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def accumulator_bits(in_bits_a: int, in_bits_b: int, n_terms: int) -> int:
    """Bits needed to sum n_terms products of a-bit and b-bit signed integers without overflow:
    a + b + ceil(log2(n_terms)). For example 8 x 8 bit, 256 terms -> 24 bits."""
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1")
    return in_bits_a + in_bits_b + (n_terms - 1).bit_length()


def choose_scale(x: Tensor, spec: QuantSpec) -> tuple[Tensor, Tensor]:
    """Pick (scale, zero_point) so x's observed range maps onto the integer range of spec.

    Symmetric: scale = max|x| / q_max, zero_point = 0. Asymmetric: scale = (max - min) /
    (q_max - q_min), zero_point = round(q_min - min / scale). Shapes follow spec.granularity.
    DECISION (window 2): min/max vs percentile calibration.
    """
    raise NotImplementedError("quant.choose_scale is written in window 2")


def quantize(x: Tensor, scale: Tensor, zero_point: Tensor, spec: QuantSpec) -> Tensor:
    """q = clamp(round(x / scale) + zero_point, q_min, q_max), returned as an integer tensor
    (int8/int16/int32 as wide as spec.bits needs). Rounding follows spec.rounding exactly."""
    raise NotImplementedError("quant.quantize is written in window 2")


def dequantize(q: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
    """x_hat = (q - zero_point) * scale, as a float tensor."""
    raise NotImplementedError("quant.dequantize is written in window 2")


def fake_quantize(x: Tensor, scale: Tensor, zero_point: Tensor, spec: QuantSpec) -> Tensor:
    """dequantize(quantize(x)): a float tensor holding only representable values. For training,
    the backward pass treats this as identity inside the range (straight-through estimator)."""
    raise NotImplementedError("quant.fake_quantize is written in window 2")
