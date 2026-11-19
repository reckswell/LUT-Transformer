"""Inference functions, non-LUT ("base") versions, written from mathematical base operations.

Every function here is forward-only (no saved state for backward) and uses only basic tensor
arithmetic: matmul (@), elementwise ops, reductions, exp, sqrt, indexing and reshapes. No
torch.nn, torch.nn.functional or autograd. The LUT versions of the same functions live in
python/LUT/inference_lut.py with identical signatures, and the toggles in python/LUT/host.py
pick one per stage.

Shape names used in comments:
    B  batch size            T  sequence length (new tokens)     S  total length incl. cache
    D  model width (d_model) H  number of heads                  Dh head width = D / H
    F  feed-forward width    V  vocabulary size

Status: signatures and documentation only. The bodies are written in window 1,
each one tested against its torch.nn / torch.nn.functional equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def _todo(name: str) -> NotImplementedError:
    return NotImplementedError(f"inference.{name} is written in window 1")


# ----------------------------------------------------------------------------------------------
# Token and position encodings
# ----------------------------------------------------------------------------------------------

def embedding(token_ids: Tensor, table: Tensor) -> Tensor:
    """Look up each token's vector: x[b, t, :] = table[token_ids[b, t], :].

    token_ids: (B, T) integer ids.  table: (V, D).  Returns (B, T, D).
    """
    raise _todo("embedding")


def sinusoidal_positions(length: int, d_model: int, start: int = 0, *, device=None, dtype=None) -> Tensor:
    """Fixed sinusoidal position codes for positions start .. start+length-1.

    pe[p, 2i]   = sin(p / 10000^(2i / D))
    pe[p, 2i+1] = cos(p / 10000^(2i / D))
    Returns (length, D). `start` lets decode steps continue from the cached length.
    DECISION (window 1): sinusoidal vs learned positions, see configs/model/.
    """
    raise _todo("sinusoidal_positions")


def learned_positions(length: int, table: Tensor, start: int = 0) -> Tensor:
    """Learned position vectors: rows start .. start+length-1 of table (P_max, D). Returns (length, D)."""
    raise _todo("learned_positions")


# ----------------------------------------------------------------------------------------------
# Linear layers and activations
# ----------------------------------------------------------------------------------------------

def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Affine map y = x @ W^T + b.

    x: (..., D_in).  weight: (D_out, D_in), the same layout as torch.nn.Linear so weights can be
    copied from the reference model.  bias: (D_out,) or None.  Returns (..., D_out).
    This is the main target for LUT replacement (see LUT/inference_lut.linear).
    """
    raise _todo("linear")


def relu(x: Tensor) -> Tensor:
    """ReLU: y = max(x, 0), elementwise."""
    raise _todo("relu")


def gelu(x: Tensor) -> Tensor:
    """GELU, tanh approximation: y = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).

    DECISION (window 1): ReLU vs GELU. ReLU is far cheaper in hardware.
    """
    raise _todo("gelu")


# ----------------------------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------------------------

def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm over the last axis.

    mu = mean(x), var = mean((x - mu)^2), y = gamma * (x - mu) / sqrt(var + eps) + beta.
    x: (..., D).  gamma, beta: (D,).  Returns (..., D).
    """
    raise _todo("layernorm")


def rmsnorm(x: Tensor, gamma: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last axis: y = gamma * x / sqrt(mean(x^2) + eps). No mean subtraction.

    DECISION (window 1): LayerNorm vs RMSNorm. RMSNorm drops one reduction, cheaper in hardware.
    """
    raise _todo("rmsnorm")


# ----------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------

def split_heads(x: Tensor, n_heads: int) -> Tensor:
    """(B, T, D) -> (B, H, T, Dh): view the model width as H heads of width Dh = D / H."""
    raise _todo("split_heads")


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, D): inverse of split_heads."""
    raise _todo("merge_heads")


def causal_mask(t_query: int, t_key: int, offset: int = 0, *, device=None) -> Tensor:
    """Boolean (t_query, t_key) mask, True where query i may attend to key j.

    Query i sits at absolute position offset + i, so it may see keys j <= offset + i.
    offset = number of tokens already in the KV cache (0 during prefill).
    """
    raise _todo("causal_mask")


def attention_scores(q: Tensor, k: Tensor, mask: Tensor | None = None) -> Tensor:
    """Scaled dot-product scores: s = (q @ k^T) / sqrt(Dh), masked positions set to -inf.

    q: (B, H, T, Dh).  k: (B, H, S, Dh).  mask: (T, S) boolean.  Returns (B, H, T, S).
    Both operands come from the input at run time, so this cannot be folded into a fixed
    weight table; it is the core of the LUT-attention question.
    """
    raise _todo("attention_scores")


def softmax(scores: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax: p = exp(s - max(s)) / sum(exp(s - max(s))) along `dim`.

    Subtracting the row max keeps exp() in range and doesn't change the result. Rows that are
    fully masked (all -inf) are not expected; causal masks always leave the diagonal visible.
    """
    raise _todo("softmax")


def attention_values(probs: Tensor, v: Tensor) -> Tensor:
    """Weight the values by the attention probabilities: out = probs @ v.

    probs: (B, H, T, S).  v: (B, H, S, Dh).  Returns (B, H, T, Dh).
    """
    raise _todo("attention_values")


# ----------------------------------------------------------------------------------------------
# KV cache and sampling (decode)
# ----------------------------------------------------------------------------------------------

@dataclass
class KVCache:
    """Per-layer cache of keys and values for incremental decoding.

    keys[l], values[l]: (B, H, S_max, Dh) preallocated buffers; `length` = tokens filled so far.
    During decode every new token reads the whole cache, which is why attention cost grows with
    context and is memory-bandwidth bound.
    """

    keys: list[Tensor] = field(default_factory=list)
    values: list[Tensor] = field(default_factory=list)
    length: int = 0

    @classmethod
    def allocate(cls, n_layers: int, batch: int, n_heads: int, max_len: int, head_dim: int,
                 *, device=None, dtype=None) -> "KVCache":
        """Preallocate zeroed buffers for n_layers layers."""
        raise _todo("KVCache.allocate")

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Write k_new, v_new (B, H, T, Dh) at positions length .. length+T-1 of `layer` and return
        views of all filled keys and values, (B, H, length+T, Dh). Call advance(T) after the last
        layer of a step."""
        raise _todo("KVCache.append")

    def advance(self, n_tokens: int) -> None:
        """Mark n_tokens more positions as filled (once per forward step, after all layers)."""
        raise _todo("KVCache.advance")


def sample_next_token(logits: Tensor, temperature: float = 1.0, top_k: int | None = None,
                      generator: torch.Generator | None = None) -> Tensor:
    """Pick the next token id from the last position's logits (B, V). Returns (B,).

    temperature == 0 means greedy argmax. Otherwise divide logits by temperature, keep the top_k
    largest if given, softmax, and draw one sample per row.
    """
    raise _todo("sample_next_token")
