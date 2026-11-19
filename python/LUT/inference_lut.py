"""Inference functions, LUT versions. Same signatures as baseline/inference.py, plus the
architecture to use and its tables.

The toggles in LUT/host.py decide which stages run these instead of the baseline functions, and
with which architecture; LUT/ops_registry.py binds that choice as `arch=` (a LUT/host.py LUTArch).
Each function also takes the `tables` built ahead of time by build_tables(), because in hardware
the tables are filled before inference starts.

Modeling rules: inputs and outputs are integer or fixed-point tensors with
explicit bit widths; tables are real tensors whose size can be reported; lookups, adds and
multiplies are counted; nothing silently falls back to float math.

Status: the LUT design is Alpha's (decision D5), so ARCHITECTURES is empty and the functions are
placeholders with the interfaces the rest of the code expects. Do not invent the scheme; implement
what Alpha specifies, add its name to ARCHITECTURES, and dispatch on arch.name inside each function
it covers. lut_size_bits() and LUTCost already work, because the table-size arithmetic holds for
any scheme.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from torch import Tensor

if TYPE_CHECKING:
    from LUT.host import LUTArch

# LUT architectures the toggles may name: name -> one-line description of what it computes and
# which stages it covers. Empty until the LUT design (decision D5) is implemented.
ARCHITECTURES: dict[str, str] = {}


def _pending(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"inference_lut.{name}: the LUT design is pending Alpha (decision D5)"
    )


# ----------------------------------------------------------------------------------------------
# Table-size and cost accounting (works now)
# ----------------------------------------------------------------------------------------------

def lut_entries(num_inputs: int, bits_per_input: int) -> int:
    """A table indexed by num_inputs values of bits_per_input bits each has 2^(num_inputs * bits) entries."""
    if num_inputs < 1 or bits_per_input < 1:
        raise ValueError("num_inputs and bits_per_input must be at least 1")
    return 1 << (num_inputs * bits_per_input)


def lut_size_bits(num_inputs: int, bits_per_input: int, out_bits: int) -> int:
    """Total storage of one table: entries x output width. Example: 2 inputs x 4 bits -> 256
    entries; at 16-bit outputs that is 4096 bits. The exponential growth in num_inputs * bits is
    the reason multi-input, multi-bit LUTs explode."""
    if out_bits < 1:
        raise ValueError("out_bits must be at least 1")
    return lut_entries(num_inputs, bits_per_input) * out_bits


@dataclass
class LUTCost:
    """Modeled hardware cost of one LUT-based operation, reported next to measured CPU/GPU time.

    PyTorch emulation speed says nothing about FPGA speed, so these counts are what the LUT and
    non-LUT variants are compared on until the HLS port exists.
    """

    tables: int = 0
    table_bits: int = 0
    lookups: int = 0
    adds: int = 0
    multiplies: int = 0
    notes: list[str] = field(default_factory=list)

    def add_table(self, num_inputs: int, bits_per_input: int, out_bits: int, count: int = 1) -> None:
        self.tables += count
        self.table_bits += count * lut_size_bits(num_inputs, bits_per_input, out_bits)

    def __iadd__(self, other: "LUTCost") -> "LUTCost":
        self.tables += other.tables
        self.table_bits += other.table_bits
        self.lookups += other.lookups
        self.adds += other.adds
        self.multiplies += other.multiplies
        self.notes.extend(other.notes)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------------------------
# Table construction (pending the design, D5)
# ----------------------------------------------------------------------------------------------

def build_tables(params: dict[str, Tensor], toggles: dict[str, Any]) -> dict[str, Any]:
    """Precompute every table the toggles (LUT/host.py) call for, from the trained parameters (for
    weight layers) or from the function alone (for activations, exp). Returns tables keyed by
    stage name, and by layer for stages inside the layers, each with its LUTCost."""
    raise _pending("build_tables")


# ----------------------------------------------------------------------------------------------
# LUT versions of baseline/inference.py functions (pending the design, D5)
# ----------------------------------------------------------------------------------------------

def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.linear: y = x @ W^T + b computed by table lookups and additions
    instead of multiplies. Must match inference.linear within the quantization error the scheme
    predicts, and bit-exactly match a slow integer reference."""
    raise _pending("linear")


def attention_scores(q: Tensor, k: Tensor, mask: Tensor | None = None, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.attention_scores (QK^T). Both operands are produced at run time,
    so a scheme here cannot precompute products with fixed weights."""
    raise _pending("attention_scores")


def softmax(scores: Tensor, dim: int = -1, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.softmax, for example a table for exp() over a quantized range."""
    raise _pending("softmax")


def attention_values(probs: Tensor, v: Tensor, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.attention_values (PV)."""
    raise _pending("attention_values")


def relu(x: Tensor, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.relu (usually not worth a table; listed so every stage can toggle)."""
    raise _pending("relu")


def gelu(x: Tensor, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.gelu: a table over the quantized input range."""
    raise _pending("gelu")


def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.layernorm, for example a table for 1 / sqrt(var + eps)."""
    raise _pending("layernorm")


def rmsnorm(x: Tensor, gamma: Tensor, eps: float = 1e-6, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """LUT version of inference.rmsnorm."""
    raise _pending("rmsnorm")


def embedding(token_ids: Tensor, table: Tensor, *, arch: LUTArch, tables: Any = None) -> Tensor:
    """Embedding is already a lookup; this version reads a quantized table."""
    raise _pending("embedding")
