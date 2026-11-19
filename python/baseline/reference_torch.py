"""Ground-truth transformer built from standard PyTorch modules (torch.nn, autograd).

Two jobs:
  1. Correctness: given the same weights and inputs, the from-scratch model in baseline/host.py must
     produce the same logits, loss and gradients as this one (python/tests/test_model_against_reference.py).
  2. Speed baseline: how fast library PyTorch runs the same model on the same hardware, so the
     cost of the from-scratch and LUT versions is measured against a fair reference.

It mirrors TransformerConfig exactly (norm type and placement, activation, positional encoding,
tied embeddings, bias) and uses the same parameter names as host.init_params, so copy_params_*
can move weights between the two. It deliberately shares no code with baseline/inference.py:
everything here comes from torch.nn, torch.nn.functional and torch's own kernels, so a mistake
in a from-scratch function cannot hide in both models at once.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _norm(cfg: Any) -> nn.Module:
    """nn.LayerNorm (eps 1e-5, with a shift) or nn.RMSNorm (eps 1e-6, no shift), as cfg.norm says."""
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model, eps=1e-5)
    return nn.RMSNorm(cfg.d_model, eps=1e-6)


def _sinusoidal_table(length: int, d_model: int) -> Tensor:
    """The fixed sine/cosine position table, written the way PyTorch's own tutorial writes it
    (a different form from baseline/inference.sinusoidal_positions, on purpose)."""
    position = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float64) * (-math.log(10000.0) / d_model))
    table = torch.zeros(length, d_model, dtype=torch.float64)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
    return table   # float64; forward() casts it to the activations' dtype


class Attention(nn.Module):
    """Multi-head causal self-attention from four nn.Linear layers and torch's fused
    scaled_dot_product_attention kernel."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = (proj(x).view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
                   for proj in (self.wq, self.wk, self.wv))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(b, t, d))


class FeedForward(nn.Module):
    """w2(act(w1(x))): two nn.Linear layers with nn.ReLU or nn.GELU (tanh approximation) between."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.act = nn.GELU(approximate="tanh") if cfg.activation == "gelu" else nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)))


class Block(nn.Module):
    """One decoder layer: attention and feed-forward, each with a residual connection, and a norm
    before (pre-norm) or after (post-norm) each residual branch."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.pre_norm = cfg.norm_placement == "pre"
        self.norm1, self.attn = _norm(cfg), Attention(cfg)
        self.norm2, self.ffn = _norm(cfg), FeedForward(cfg)

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_norm:
            x = x + self.attn(self.norm1(x))
            return x + self.ffn(self.norm2(x))
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class ReferenceTransformer(nn.Module):
    """Decoder-only transformer from nn.Linear, nn.LayerNorm (or nn.RMSNorm), nn.Embedding and
    torch.nn.functional.scaled_dot_product_attention with a causal mask."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        if cfg.vocab_size is None:
            raise ValueError("cfg.vocab_size must be set (it comes from the tokenizer)")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.positional_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        elif cfg.positional_encoding == "sinusoidal":
            self.register_buffer("pos_table", _sinusoidal_table(cfg.context_length, cfg.d_model), persistent=False)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = _norm(cfg)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids (B, T) -> logits (B, T, V)."""
        t = token_ids.shape[1]
        x = self.tok_emb(token_ids)
        if self.cfg.positional_encoding == "learned":
            x = x + self.pos_emb(torch.arange(t, device=token_ids.device))
        elif self.cfg.positional_encoding == "sinusoidal":
            x = x + self.pos_table[:t].to(x.dtype)
        # positional_encoding == "none": no position code; order comes from the causal mask only.
        for block in self.layers:
            x = block(x)
        x = self.norm_f(x)
        if self.lm_head is None:
            return F.linear(x, self.tok_emb.weight)         # tied: logits = x @ tok_emb^T
        return self.lm_head(x)

    def scratch_params(self) -> dict[str, nn.Parameter]:
        """This model's parameters under host.init_params names: the one mapping both copy
        functions use, so the names can't drift apart."""
        named: dict[str, nn.Parameter] = {"tok_emb": self.tok_emb.weight}
        if self.cfg.positional_encoding == "learned":
            named["pos_emb"] = self.pos_emb.weight
        for i, block in enumerate(self.layers):
            p = f"layers.{i}"
            named.update(_norm_params(f"{p}.norm1", block.norm1))
            for name in ("wq", "wk", "wv", "wo"):
                linear = getattr(block.attn, name)
                named[f"{p}.attn.{name}"] = linear.weight
                if linear.bias is not None:
                    named[f"{p}.attn.b{name[1]}"] = linear.bias
            named.update(_norm_params(f"{p}.norm2", block.norm2))
            for name in ("w1", "w2"):
                linear = getattr(block.ffn, name)
                named[f"{p}.ffn.{name}"] = linear.weight
                if linear.bias is not None:
                    named[f"{p}.ffn.b{name[1]}"] = linear.bias
        named.update(_norm_params("norm_f", self.norm_f))
        if self.lm_head is not None:
            named["lm_head"] = self.lm_head.weight
        return named


def _norm_params(prefix: str, norm: nn.Module) -> dict[str, nn.Parameter]:
    named = {f"{prefix}.gamma": norm.weight}
    if getattr(norm, "bias", None) is not None:
        named[f"{prefix}.beta"] = norm.bias
    return named


@torch.no_grad()
def copy_params_from_scratch(model: ReferenceTransformer, params: dict[str, Tensor]) -> None:
    """Load host.init_params-style tensors into the reference model (same names, same layouts).
    Raises if a name is missing on either side, so a silent partial copy can't happen."""
    named = model.scratch_params()
    if set(named) != set(params):
        raise KeyError(f"parameter names differ: only in reference {sorted(set(named) - set(params))}, "
                       f"only in params {sorted(set(params) - set(named))}")
    for name, param in named.items():
        param.copy_(params[name])


@torch.no_grad()
def copy_params_to_scratch(model: ReferenceTransformer) -> dict[str, Tensor]:
    """Export the reference model's weights as a host.init_params-style dict (detached copies)."""
    return {name: param.detach().clone() for name, param in model.scratch_params().items()}


@torch.no_grad()
def reference_logits(model: ReferenceTransformer, token_ids: Tensor) -> Tensor:
    """Convenience wrapper for tests and benchmarks."""
    return model(token_ids)
