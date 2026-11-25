"""What a toggle can say: LUTArch (a LUT architecture and its table shape), Stage (what one stage
does), BASE (the baseline function, float inputs, every layer), KVStore (how the KV cache stores
keys and values) and KV_BASE (as they are).

These live in their own file so that LUT/host.py can open with its PARAMETERS block, the way a
SystemVerilog module opens with its parameter list, with nothing defined above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.quant import QuantSpec
from LUT import inference_lut


@dataclass
class LUTArch:
    """One LUT architecture and the shape of its tables.

    name              which architecture: a key of LUT/inference_lut.ARCHITECTURES
    inputs_per_table  k, how many values together index one table
    bits_per_input    b, the bit width of each of those values (address bits = k * b)
    out_bits          the width of each table entry
    acc_bits          the width of the integer accumulator that sums table outputs (None: no sum)
    options           anything specific to this architecture, passed to it unchanged

    One table has 2^(k*b) entries of out_bits bits, so 2^(k*b) * out_bits bits of storage.
    For example k = 2, b = 4, out_bits = 16 gives 256 entries and 4,096 bits.
    """

    name: str
    inputs_per_table: int
    bits_per_input: int
    out_bits: int
    acc_bits: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def address_bits(self) -> int:
        return self.inputs_per_table * self.bits_per_input

    def entries(self) -> int:
        return inference_lut.lut_entries(self.inputs_per_table, self.bits_per_input)

    def table_bits(self) -> int:
        return inference_lut.lut_size_bits(self.inputs_per_table, self.bits_per_input, self.out_bits)


@dataclass
class Stage:
    """What one stage does: a set of slots, each of which defaults to the baseline. Every field is
    independent, so each stage (and each technique within it) can be set on its own. A stage with
    every slot at its default runs the baseline function itself, untouched, so it is bit-identical
    to the baseline and exactly as fast (tests/test_lut_slots.py).

    lut      None runs the baseline function. A LUTArch runs that LUT architecture instead (D5).
    quant    None keeps float inputs. A QuantSpec quantizes this stage's first (or only) run-time
             input before the stage runs, with or without a LUT, so quantization and LUTs can sit
             at different stages (decision D4): x of a linear stage, norm, ffn_act or rope; q of
             attn_scores; p of attn_values; a = act(W1 h) of ffn_gate; the scores of softmax.
    quant_b  the second run-time input of a two-operand stage, set separately from quant: k of
             attn_scores, v of attn_values, b = W3 h of ffn_gate. None keeps it float.
    weight   offline weight preparation: a QuantSpec (or another setting LUT/techniques.py's
             prepare_weight understands, such as a codebook later) applied once to the weights
             this stage multiplies by: wq/wk/wv, wo, w1/w3, w2, the output layer, the token table,
             or a norm's gamma. Each stage prepares its own copy, so a tied output layer and the
             token lookup can be prepared differently. None uses the weights as they are.
    smooth   SmoothQuant-style smoothing of a linear stage's input, migration strength alpha in
             [0, 1] (None: off). Per input channel j, x_j / s_j and W[:, j] * s_j, with s from
             LUT/techniques.py's smoothing_factors and the calibrated ranges. In hardware s folds
             into the preceding norm's gamma (qkv_proj, ffn_up, lm_head), into wv's rows
             (out_proj) or into w3's rows (ffn_down of a gated feed-forward), so it costs nothing at
             run time; the model checks that such a fold exists.
    alt      replacement nonlinear functions inside the stage, {slot: name in LUT/nonlinear.py}:
             {"exp": ...} in softmax, {"rsqrt": ...} in norm and final_norm, {"act": ...} in
             ffn_act (the whole elementwise activation, such as SiLU). None runs the baseline math.
    layers   the decoder layers the choice applies to, counted from 0 (None: all layers). Other
             layers run the baseline. Ignored for embedding, final_norm and lm_head.

    A stage line in LUT/host.py may also be a list of Stages, applied in order, so different layers
    can get different settings: [Stage(quant=Q8), Stage(quant=Q4, layers=(2,))] puts every layer at
    8 bits except layer 2 at 4 bits (a later entry wins for the layers it names).
    """

    lut: LUTArch | None = None
    quant: QuantSpec | None = None
    layers: tuple[int, ...] | None = None
    quant_b: QuantSpec | None = None
    weight: Any = None
    smooth: float | None = None
    alt: dict[str, Any] | None = None

    def is_base(self) -> bool:
        """True when every slot is at its default (layers aside): the stage runs the baseline."""
        return (self.lut is None and self.quant is None and self.quant_b is None and self.weight is None
                and self.smooth is None and not self.alt)


BASE = Stage()   # the baseline function, float inputs, every layer


@dataclass
class KVStore:
    """How the KV cache stores keys and values: what each attention layer reads back is
    decode(encode(x)), written once when the token is appended (LUT/techniques.py's kv_encode and
    kv_decode). The defaults store them as they are, exactly like the baseline.

    k, v           a QuantSpec for cached keys / values (None: stored as they are, in DTYPE)
    k_before_rope  store keys before the rotary rotation and rotate every cached key by its own
                   position on each read (rotary-position models only). False stores rotated
                   keys, as the baseline does. With k and v None this changes the order of the work,
                   not the values, so the logits stay bit-identical.
    layers         the decoder layers this applies to (None: all); other layers store as they are.
    """

    k: QuantSpec | None = None
    v: QuantSpec | None = None
    k_before_rope: bool = False
    layers: tuple[int, ...] | None = None

    def is_base(self) -> bool:
        return self.k is None and self.v is None and not self.k_before_rope


KV_BASE = KVStore()   # keys and values stored as they are, rotated keys, every layer
