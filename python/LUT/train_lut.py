"""Training functions, LUT versions. Same signatures as baseline/train.py (<op>_forward / <op>_backward),
plus the architecture (`arch=`, a LUT/host.py LUTArch) on each forward.

Open question for Alpha (window 2), set by TRAIN_WITH_LUTS in LUT/host.py:
does training ever see the LUTs?
  - No (post-training): train in float with baseline/train.py, then build tables for inference_lut.py.
    This file stays empty.
  - Yes (LUT- or quantization-aware training): the forward pass uses the LUT/quantized math and
    the backward pass needs a surrogate gradient, usually the straight-through estimator (treat
    the rounding or lookup as identity inside its range). LogicNets and PolyLUT-style methods
    train this way.

Until that's decided, every function here raises. They are only used when TRAIN_WITH_LUTS is True
and a toggle gives the stage a LUT architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import Tensor

if TYPE_CHECKING:
    from LUT.host import LUTArch

Cache = dict[str, Any]


def _pending(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"train_lut.{name}: LUT-aware training is pending Alpha's decision"
    )


def linear_forward(x: Tensor, weight: Tensor, bias: Tensor | None = None, *, arch: LUTArch, tables: Any = None) -> tuple[Tensor, Cache]:
    """LUT/quantized forward of a linear layer, saving what the surrogate backward needs."""
    raise _pending("linear_forward")


def linear_backward(dy: Tensor, cache: Cache) -> tuple[Tensor, Tensor, Tensor | None]:
    """Surrogate gradients (dx, dW, db) for the LUT linear layer, for example straight-through."""
    raise _pending("linear_backward")


def softmax_forward(scores: Tensor, dim: int = -1, *, arch: LUTArch, tables: Any = None) -> tuple[Tensor, Cache]:
    """LUT/quantized softmax forward."""
    raise _pending("softmax_forward")


def softmax_backward(dp: Tensor, cache: Cache) -> Tensor:
    """Surrogate gradient for the LUT softmax."""
    raise _pending("softmax_backward")


def attention_scores_forward(q: Tensor, k: Tensor, mask: Tensor | None = None, *, arch: LUTArch, tables: Any = None) -> tuple[Tensor, Cache]:
    """LUT/quantized QK^T forward."""
    raise _pending("attention_scores_forward")


def attention_scores_backward(ds: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Surrogate gradients (dq, dk) for the LUT QK^T."""
    raise _pending("attention_scores_backward")


def attention_values_forward(probs: Tensor, v: Tensor, *, arch: LUTArch, tables: Any = None) -> tuple[Tensor, Cache]:
    """LUT/quantized PV forward."""
    raise _pending("attention_values_forward")


def attention_values_backward(d_out: Tensor, cache: Cache) -> tuple[Tensor, Tensor]:
    """Surrogate gradients (d_probs, dv) for the LUT PV."""
    raise _pending("attention_values_backward")
