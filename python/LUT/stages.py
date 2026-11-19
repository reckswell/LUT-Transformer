"""What a toggle can say: LUTArch (a LUT architecture and its table shape), Stage (what one stage
does) and BASE (the baseline function, float inputs, every layer).

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
    """What one stage does. Every field is independent, so each stage can be set on its own.

    lut     None runs the baseline function. A LUTArch runs that LUT architecture instead.
    quant   None keeps float inputs. A QuantSpec quantizes this stage's inputs first, with or
            without a LUT, so quantization and LUTs can sit at different stages (decision D4).
    layers  the decoder layers the choice applies to, counted from 0 (None: all layers). Other
            layers run the baseline. Ignored for embedding, final_norm and lm_head.
    """

    lut: LUTArch | None = None
    quant: QuantSpec | None = None
    layers: tuple[int, ...] | None = None
    # Room for later: a non-LUT alternative implementation of a stage (for example another
    # softmax approximation, or a restructured kernel) would be one more field here, such as
    # `alt: str | None` naming a function that LUT/ops_registry.py swaps in the same way as `lut`.
    # Not added until there is a first alternative to point it at.


BASE = Stage()   # the baseline function, float inputs, every layer
