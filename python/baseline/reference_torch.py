"""Ground-truth transformer built from standard PyTorch modules (torch.nn, autograd).

Two jobs:
  1. Correctness: given the same weights and inputs, the from-scratch model in baseline/host.py must
     produce the same logits, loss and gradients as this one (python/tests/test_model_against_reference.py).
  2. Speed baseline: how fast library PyTorch runs the same model on the same hardware, so the
     cost of the from-scratch and LUT versions is measured against a fair reference.

It must mirror TransformerConfig exactly (norm type and placement, activation, positional
encoding, tied embeddings, bias), and use the same parameter names as host.init_params so
copy_params_* can move weights between the two.

Status: written in window 1, right after the architecture decisions, before the
from-scratch model, so the from-scratch functions have something to be tested against.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class ReferenceTransformer(nn.Module):
    """Decoder-only transformer from nn.Linear, nn.LayerNorm (or RMSNorm), nn.Embedding and
    torch.nn.functional.scaled_dot_product_attention with a causal mask."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        raise NotImplementedError("reference_torch.ReferenceTransformer is written in window 1")

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids (B, T) -> logits (B, T, V)."""
        raise NotImplementedError


def copy_params_from_scratch(model: ReferenceTransformer, params: dict[str, Tensor]) -> None:
    """Load host.init_params-style tensors into the reference model (same names, same layouts)."""
    raise NotImplementedError("written in window 1")


def copy_params_to_scratch(model: ReferenceTransformer) -> dict[str, Tensor]:
    """Export the reference model's weights as a host.init_params-style dict."""
    raise NotImplementedError("written in window 1")


@torch.no_grad()
def reference_logits(model: ReferenceTransformer, token_ids: Tensor) -> Tensor:
    """Convenience wrapper for tests and benchmarks."""
    return model(token_ids)
