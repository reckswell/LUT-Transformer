"""Fixed-point and integer quantization, shared by the training and inference variants.

The goal is to model hardware arithmetic exactly in PyTorch: values are stored as integers of a
stated bit width, with an explicit scale, rounding mode and saturation, and products are
accumulated in integers of a stated width. A later HLS port uses ap_int / ap_fixed with the same
parameters, so the PyTorch numbers predict the hardware numbers bit for bit.

Alpha's direction: the quantization technique stays fairly fixed while LUT architectures vary.
Which technique (bit widths, symmetric or asymmetric, per-tensor or per-channel, rounding) is a
window 2 decision; QuantSpec records the choice so every run logs it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor

Rounding = Literal["nearest_even", "nearest_away", "floor", "stochastic"]
Granularity = Literal["per_tensor", "per_channel", "per_group"]


@dataclass(frozen=True)
class QuantSpec:
    """How one tensor is quantized. Logged with every run.

    bits: integer width.  signed: two's complement if True.  symmetric: zero point fixed at 0.
    granularity / axis / group_size: which elements share one scale.  rounding: how real values
    map to integers.  saturate: clamp to the integer range (True) or wrap like raw hardware.
    """

    bits: int = 8
    signed: bool = True
    symmetric: bool = True
    granularity: Granularity = "per_tensor"
    axis: int | None = None
    group_size: int | None = None
    rounding: Rounding = "nearest_even"
    saturate: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


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
