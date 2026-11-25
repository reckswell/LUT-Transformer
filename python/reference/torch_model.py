"""Ground-truth transformer built from standard PyTorch modules (torch.nn, autograd).

A control, not part of our model: it lives in reference/ with the other library-built models, so
baseline/ and LUT/ hold only from-scratch code. (Formerly baseline/reference_torch.py.)

Two jobs:
  1. Correctness: given the same weights and inputs, the from-scratch model in baseline/host.py must
     produce the same logits, loss and gradients as this one (python/tests/test_model_against_reference.py).
  2. Speed baseline: how fast library PyTorch runs the same model on the same hardware, so the
     cost of the from-scratch and LUT versions is measured against a fair reference.

It mirrors TransformerConfig exactly (norm type, eps and placement, activation, plain or gated
feed-forward, positional encoding including rotary, tied embeddings, bias, and the options for
other families: grouped-query attention, head width, separate biases, parallel blocks, partial
rotary, exact GELU, embedding scale, output-layer bias) and uses the same
parameter names as host.init_params, so copy_params_*
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
    """nn.LayerNorm (with a shift) or nn.RMSNorm (no shift), as cfg.norm says, with the config's eps
    (cfg.norm_epsilon: norm_eps if set, else 1e-5 for LayerNorm and 1e-6 for RMSNorm)."""
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model, eps=cfg.norm_epsilon)
    return nn.RMSNorm(cfg.d_model, eps=cfg.norm_epsilon)


def _sinusoidal_table(length: int, d_model: int) -> Tensor:
    """The fixed sine/cosine position table, written the way PyTorch's own tutorial writes it
    (a different form from baseline/inference.sinusoidal_positions, on purpose)."""
    position = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float64) * (-math.log(10000.0) / d_model))
    table = torch.zeros(length, d_model, dtype=torch.float64)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
    return table   # float64; forward() casts it to the activations' dtype


def _rotary_cos_sin(length: int, head_dim: int, theta: float, like: Tensor) -> tuple[Tensor, Tensor]:
    """Rotary cos and sin for positions 0 .. length-1, each (length, Dh), written the way Hugging
    Face's LlamaRotaryEmbedding writes it: inverse frequencies and angles in float32 (the constants
    Llama checkpoints were trained with), the angles repeated over both halves of the head, then
    cast to the activations' dtype. (baseline/inference.rope_tables builds the same constants its
    own way, Dh/2 wide.)"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)       # (length, Dh/2)
    emb = torch.cat((freqs, freqs), dim=-1)                                         # (length, Dh)
    return emb.cos().to(like.device, like.dtype), emb.sin().to(like.device, like.dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """[x1, x2] -> [-x2, x1] over the last axis (Hugging Face's rotate_half)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """Multi-head causal self-attention from four nn.Linear layers and torch's fused
    scaled_dot_product_attention kernel, with rotary positions on q and k if the config asks
    (on the first rotary_dim columns of each head when that is set). With grouped-query attention
    the key/value heads are repeated with repeat_interleave before the kernel."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.n_heads, self.n_kv, self.dh = cfg.n_heads, cfg.kv_heads, cfg.head_dim
        self.rope_theta = cfg.rope_theta if cfg.positional_encoding == "rope" else None
        self.rotary = cfg.rotary_columns
        self.window = cfg.attention_window
        qkv_bias, out_bias = cfg.has_bias("qkv"), cfg.has_bias("out")
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.dh, bias=qkv_bias)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.dh, bias=qkv_bias)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.dh, bias=qkv_bias)
        self.wo = nn.Linear(self.n_heads * self.dh, cfg.d_model, bias=out_bias)

    def _rotary(self, z: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        r = self.rotary
        rotated = z[..., :r] * cos + _rotate_half(z[..., :r]) * sin
        return rotated if r == z.shape[-1] else torch.cat((rotated, z[..., r:]), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.dh).transpose(1, 2)
        k, v = (proj(x).view(b, t, self.n_kv, self.dh).transpose(1, 2) for proj in (self.wk, self.wv))
        if self.rope_theta is not None:
            cos, sin = _rotary_cos_sin(t, self.rotary, self.rope_theta, q)
            q, k = self._rotary(q, cos, sin), self._rotary(k, cos, sin)
        if self.n_kv != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        if self.window is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:                                            # sliding window: keys j with i - window < j <= i
            pos = torch.arange(t, device=x.device)
            gap = pos[:, None] - pos[None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=(gap >= 0) & (gap < self.window))
        return self.wo(out.transpose(1, 2).reshape(b, t, self.n_heads * self.dh))


class FeedForward(nn.Module):
    """w2(act(w1(x))) from nn.Linear layers and nn.ReLU, nn.GELU (tanh approximation or exact)
    or nn.SiLU. Gated (cfg.ffn = glu): w2(act(w1(x)) * w3(x)), which with SiLU is Llama's SwiGLU."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        bias = cfg.has_bias("ffn")
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=bias)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias) if cfg.ffn == "glu" else None
        self.act = {"relu": nn.ReLU(), "gelu": nn.GELU(approximate="tanh"), "gelu_exact": nn.GELU(),
                    "silu": nn.SiLU()}[cfg.activation]

    def forward(self, x: Tensor) -> Tensor:
        if self.w3 is None:
            return self.w2(self.act(self.w1(x)))
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """One decoder layer: attention and feed-forward, each with a residual connection, and a norm
    before (pre-norm) or after (post-norm) each residual branch. A parallel block adds both
    branches, read from the same residual, at once."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.pre_norm = cfg.norm_placement == "pre"
        self.parallel = cfg.block != "sequential"
        self.norm1, self.attn = _norm(cfg), Attention(cfg)
        # parallel_shared_norm: one norm feeds both branches, so there is no norm2.
        self.norm2 = None if cfg.block == "parallel_shared_norm" else _norm(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x: Tensor) -> Tensor:
        if self.parallel:
            h = self.norm1(x)
            return self.attn(h) + self.ffn(h if self.norm2 is None else self.norm2(x)) + x
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
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=cfg.lm_head_bias)
        # Tied output layer with a bias: the bias is the only output-layer parameter of its own.
        self.tied_bias = nn.Parameter(torch.zeros(cfg.vocab_size)) if cfg.tie_embeddings and cfg.lm_head_bias else None

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids (B, T) -> logits (B, T, V)."""
        t = token_ids.shape[1]
        x = self.tok_emb(token_ids)
        if self.cfg.embed_scale is not None:
            x = x * torch.tensor(self.cfg.embed_scale, dtype=x.dtype)
        if self.cfg.positional_encoding == "learned":
            x = x + self.pos_emb(torch.arange(t, device=token_ids.device))
        elif self.cfg.positional_encoding == "sinusoidal":
            x = x + self.pos_table[:t].to(x.dtype)
        # positional_encoding == "rope": rotary positions inside each Attention, nothing added here.
        # positional_encoding == "none": no position code; order comes from the causal mask only.
        for block in self.layers:
            x = block(x)
        x = self.norm_f(x)
        if self.lm_head is None:
            return F.linear(x, self.tok_emb.weight, self.tied_bias)   # tied: logits = x @ tok_emb^T (+ b)
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
            if block.norm2 is not None:
                named.update(_norm_params(f"{p}.norm2", block.norm2))
            for name in ("w1", "w2", "w3"):
                linear = getattr(block.ffn, name)
                if linear is None:                       # w3 only in a gated feed-forward
                    continue
                named[f"{p}.ffn.{name}"] = linear.weight
                if linear.bias is not None:
                    named[f"{p}.ffn.b{name[1]}"] = linear.bias
        named.update(_norm_params("norm_f", self.norm_f))
        if self.lm_head is not None:
            named["lm_head"] = self.lm_head.weight
            if self.lm_head.bias is not None:
                named["lm_head_bias"] = self.lm_head.bias
        if self.tied_bias is not None:
            named["lm_head_bias"] = self.tied_bias
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
