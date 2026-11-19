"""Inference functions, non-LUT ("base") versions, written from mathematical base operations.

Every function here is forward-only (no saved state for backward) and uses only basic tensor
arithmetic: matmul (@), elementwise ops (+, -, *, /, exp, sqrt, tanh, sin, cos, comparisons),
reductions (sum, max), indexing and reshapes. No torch.nn, torch.nn.functional or autograd. The
LUT versions of the same functions live in python/LUT/inference_lut.py with identical signatures,
and the toggles in python/LUT/host.py pick one per stage.

Shape names used in comments:
    B  batch size            T  sequence length (new tokens)     S  total length incl. cache
    D  model width (d_model) H  number of heads                  Dh head width = D / H
    F  feed-forward width    V  vocabulary size

Each function is tested against its torch.nn / torch.nn.functional equivalent in
python/tests/test_ops_against_torch.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


# ----------------------------------------------------------------------------------------------
# Token and position encodings
# ----------------------------------------------------------------------------------------------

def embedding(token_ids: Tensor, table: Tensor) -> Tensor:
    """Look up each token's vector: x[b, t, :] = table[token_ids[b, t], :].

    token_ids: (B, T) integer ids.  table: (V, D).  Returns (B, T, D).

    This is a plain row read, with no arithmetic. In hardware it is a memory read of one D-wide
    row per token, which is why it is already "a LUT".
    """
    # Indexing a (V, D) tensor with a (B, T) integer tensor picks one row per id, giving (B, T, D).
    return table[token_ids]


def sinusoidal_positions(length: int, d_model: int, start: int = 0, *, device=None, dtype=torch.float32) -> Tensor:
    """Fixed sinusoidal position codes for positions start .. start+length-1.

    pe[p, 2i]   = sin(p / 10000^(2i / D))
    pe[p, 2i+1] = cos(p / 10000^(2i / D))
    Returns (length, D). `start` lets decode steps continue from the cached length.

    Each pair of columns (2i, 2i+1) is a rotation at its own frequency w_i = 10000^(-2i/D), from
    one turn per 2*pi positions (i = 0) down to one turn per about 2*pi*10000 positions.
    The table is a constant, worked out in float64 and rounded once to `dtype` (the activations'
    dtype, float32 by default) at the end.
    """
    # p: the absolute positions, as a column (length, 1).
    p = torch.arange(start, start + length, device=device, dtype=torch.float64)[:, None]
    # 2i for i = 0, 1, ...: the even column indices, as a row (1, ceil(D/2)).
    two_i = torch.arange(0, d_model, 2, device=device, dtype=torch.float64)[None, :]
    # w_i = 10000^(-2i/D) = exp(-(2i/D) * ln 10000). Computed in float64 so large positions keep
    # their precision, then cast at the end.
    w = torch.exp(-(two_i / d_model) * math.log(10000.0))
    angle = p * w                                             # (length, ceil(D/2)), angle[p, i] = p * w_i
    pe = torch.zeros(length, d_model, device=device, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(angle)                            # even columns
    pe[:, 1::2] = torch.cos(angle[:, : d_model // 2])         # odd columns (one fewer if D is odd)
    return pe.to(dtype)


def learned_positions(length: int, table: Tensor, start: int = 0) -> Tensor:
    """Learned position vectors: rows start .. start+length-1 of table (P_max, D). Returns (length, D).

    A learned table only covers P_max positions (GPT-2: 1,024); asking past the end is an error
    rather than a silent wrap.
    """
    if start < 0 or start + length > table.shape[0]:
        raise ValueError(f"positions {start}..{start + length - 1} are outside the table's {table.shape[0]} rows")
    return table[start:start + length]


def rope_tables(length: int, head_dim: int, theta: float = 10000.0, start: int = 0, *, device=None,
                dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Rotary position tables (cos, sin) for positions start .. start+length-1, each (length, Dh/2).

    Rotary positions (RoPE, used by Llama) don't add anything to the embedding. Instead, inside
    every attention layer, each query and key is rotated by an angle that grows with its position
    (the `rope` stage below). Pair i of a head's Dh columns, i = 0 .. Dh/2 - 1, turns at frequency
        w_i = theta^(-2i / Dh)                    (theta = 10,000 for Llama)
    so at position p it is rotated by the angle p * w_i, and
        cos[p, i] = cos(p * w_i),   sin[p, i] = sin(p * w_i).
    Pair 0 turns once per 2*pi positions; the last pair about once per 2*pi*theta positions.

    The table is a constant (a ROM in hardware). It is built in float32 exactly the way Llama's own
    code builds it (llama2.c's precompute_freqs_cis, which trained stories15M, and Hugging Face's
    LlamaRotaryEmbedding, which serves it): w in float32 on the CPU, then angle = p * w and cos, sin
    in float32. Then it is cast once to `dtype`. So the constants are, bit for bit, the ones the
    checkpoint was trained and is served with. (A float64-built table would differ by up to about
    2e-5 at position 255, because p * w is rounded to float32 there; sinusoidal_positions is built
    in float64 instead because it has no reference implementation to match.)
    """
    if head_dim % 2:
        raise ValueError(f"rotary positions need an even head width, got {head_dim}")
    # w_i for i = 0 .. Dh/2 - 1: the even column indices 0, 2, 4, ... divided by Dh, as a power of theta.
    w = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))      # (Dh/2,)
    p = torch.arange(start, start + length, dtype=torch.float32, device=device)              # (length,)
    angle = p[:, None] * w.to(device)[None, :]                    # (length, Dh/2), angle[p, i] = p * w_i
    return torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotary position embedding: rotate each pair of a head's columns by its position's angle.

    x: (B, H, T, Dh) queries or keys.  cos, sin: (T, Dh/2) from rope_tables for these T positions.
    Returns (B, H, T, Dh).

    Pairs follow Hugging Face's "split halves" convention: column i pairs with column i + Dh/2.
    With x1 = x[..., :Dh/2] and x2 = x[..., Dh/2:], each pair (x1_i, x2_i) is a 2-D vector turned
    by the angle p * w_i:
        y1 = x1 cos - x2 sin
        y2 = x2 cos + x1 sin
    Why this encodes position: a rotation by a then the inverse of b is a rotation by a - b, so after
    rotating q at position p and k at position s, each pair's contribution to q . k depends only on
    p - s. Attention scores see relative positions, and nothing is added to the residual stream.

    Convention: llama2.c (and Meta's original Llama code) pairs adjacent columns (2i, 2i+1) instead.
    The two are the same model once the rows of Wq and Wk are reordered within each head, which is
    what Hugging Face's conversion script does (reference/hf_llama.permute_rotary_rows). We load the
    Hugging Face checkpoint, so this function uses split halves; tests/test_llama_profile.py checks
    that both conventions agree on correspondingly ordered weights and that the wrong pairing fails.

    Cost: 4 multiplies and 2 adds per pair. Per token and layer, q and k each have D/2 pairs, so
    4D multiplies and 2D adds. cos and sin come from a table indexed by (position, pair).
    """
    mid = x.shape[-1] // 2                                     # Dh/2, where the second half starts
    x1, x2 = x[..., :mid], x[..., mid:]
    # cos, sin (T, Dh/2) broadcast over the batch and head axes.
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


# ----------------------------------------------------------------------------------------------
# Linear layers and activations
# ----------------------------------------------------------------------------------------------

def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Affine map y = x @ W^T + b.

    y[..., o] = sum over i of x[..., i] * W[o, i], plus b[o].

    x: (..., D_in).  weight: (D_out, D_in), the same layout as torch.nn.Linear so weights can be
    copied from the reference model.  bias: (D_out,) or None.  Returns (..., D_out).
    Cost: N * D_in * D_out multiply-accumulates for N rows of x. This is the main target for LUT
    replacement (see LUT/inference_lut.linear), because W is fixed after training.
    """
    # x (..., D_in) @ W^T (D_in, D_out) -> (..., D_out). matmul sums over the shared D_in axis.
    y = x @ weight.transpose(0, 1)
    if bias is not None:
        y = y + bias                                          # (D_out,) broadcasts over the leading axes
    return y


def relu(x: Tensor) -> Tensor:
    """ReLU: y = max(x, 0), elementwise.

    In hardware this is the sign bit choosing between x and 0: no arithmetic at all.
    """
    return torch.where(x > 0, x, torch.zeros_like(x))


# sqrt(2 / pi), the constant inside GELU's tanh approximation.
_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x: Tensor) -> Tensor:
    """GELU, tanh approximation: y = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).

    The exact GELU is x * Phi(x), where Phi is the standard normal CDF. The tanh form is the
    approximation GPT-2 was trained with (torch's F.gelu(approximate="tanh")); it differs from
    the exact form by under about 1e-3. Either way, once the input is quantized to b bits the
    whole function becomes a 2^b-entry table, so the choice of formula costs nothing in hardware.
    """
    inner = _GELU_C * (x + 0.044715 * x * x * x)              # sqrt(2/pi) * (x + 0.044715 x^3)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def silu(x: Tensor) -> Tensor:
    """SiLU (also called swish), Llama's activation: y = x * sigmoid(x) = x / (1 + exp(-x)), elementwise.

    Like GELU it is a smooth ReLU: about x for large x, about 0 (slightly negative, minimum -0.28 at
    x = -1.28) for large negative x. For very negative x, exp(-x) overflows to inf and y becomes -0,
    which is the right limit. Once the input is quantized to b bits it is a 2^b-entry table.
    """
    return x / (1.0 + torch.exp(-x))


def glu_gate(a: Tensor, b: Tensor) -> Tensor:
    """The gate of a gated feed-forward (GLU): y = a * b, elementwise.

    a = act(h W1^T) and b = h W3^T, both (..., F). With act = SiLU this is Llama's SwiGLU:
        FFN(h) = W2 (silu(W1 h) * (W3 h)).
    One multiply per element (F per token per layer). Unlike the linear layers, both operands are
    produced at run time, as in QK^T and PV, so no weight can be folded into a table for it.
    """
    return a * b


# ----------------------------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------------------------

def layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm over the last axis.

    mu = mean(x), var = mean((x - mu)^2), y = gamma * (x - mu) / sqrt(var + eps) + beta.
    x: (..., D).  gamma, beta: (D,).  Returns (..., D).

    var is the biased (population) variance, dividing by D rather than D - 1, as torch does.
    Two reductions per row (the mean, then the variance) and one reciprocal square root.
    """
    d = x.shape[-1]
    mu = x.sum(dim=-1, keepdim=True) / d                      # (..., 1) mean of each row
    centered = x - mu                                         # (..., D)
    var = (centered * centered).sum(dim=-1, keepdim=True) / d # (..., 1) mean squared deviation
    x_hat = centered / torch.sqrt(var + eps)                  # (..., D) zero mean, unit variance
    return gamma * x_hat + beta                               # per-feature scale and shift


def rmsnorm(x: Tensor, gamma: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last axis: y = gamma * x / sqrt(mean(x^2) + eps). No mean subtraction.

    One reduction per row instead of LayerNorm's two, and no shift (beta).
    """
    d = x.shape[-1]
    mean_sq = (x * x).sum(dim=-1, keepdim=True) / d           # (..., 1) mean of the squares
    return gamma * (x / torch.sqrt(mean_sq + eps))


# ----------------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------------

def split_heads(x: Tensor, n_heads: int) -> Tensor:
    """(B, T, D) -> (B, H, T, Dh): view the model width as H heads of width Dh = D / H.

    Head h owns columns h*Dh .. (h+1)*Dh - 1 of D. No arithmetic, only a re-indexing.
    """
    b, t, d = x.shape
    return x.reshape(b, t, n_heads, d // n_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, D): inverse of split_heads (head h goes back to its Dh columns)."""
    b, h, t, dh = x.shape
    return x.transpose(1, 2).reshape(b, t, h * dh)


def causal_mask(t_query: int, t_key: int, offset: int = 0, *, device=None) -> Tensor:
    """Boolean (t_query, t_key) mask, True where query i may attend to key j.

    Query i sits at absolute position offset + i, so it may see keys j <= offset + i.
    offset = number of tokens already in the KV cache (0 during prefill).
    """
    i = torch.arange(t_query, device=device)[:, None]         # (t_query, 1) query rows
    j = torch.arange(t_key, device=device)[None, :]           # (1, t_key) key columns
    return j <= i + offset


def attention_scores(q: Tensor, k: Tensor, mask: Tensor | None = None) -> Tensor:
    """Scaled dot-product scores: s = (q @ k^T) / sqrt(Dh), masked positions set to -inf.

    s[b, h, i, j] = sum over d of q[b, h, i, d] * k[b, h, j, d], divided by sqrt(Dh).

    q: (B, H, T, Dh).  k: (B, H, S, Dh).  mask: (T, S) boolean.  Returns (B, H, T, S).
    Cost: B * H * T * S * Dh multiply-accumulates. Both operands come from the input at run time,
    so this cannot be folded into a fixed weight table; it is the core of the LUT-attention
    question. The 1/sqrt(Dh) scale is kept as its own step so it stays visible when quantized;
    with Dh = 64 it is 1/8, a 3-bit shift.
    """
    dh = q.shape[-1]
    s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)             # (B, H, T, Dh) @ (B, H, Dh, S)
    if mask is not None:
        # -inf becomes exactly 0 after softmax's exp, so masked keys get no weight.
        s = torch.where(mask, s, torch.full_like(s, float("-inf")))
    return s


def softmax(scores: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax: p = exp(s - max(s)) / sum(exp(s - max(s))) along `dim`.

    Subtracting the row max keeps exp() in range (every exponent is <= 0, so every term is in
    (0, 1]) and doesn't change the result, because the common factor exp(-max) cancels between
    the numerator and the sum. Per score: one subtract, one exp, one add into the row sum and one
    divide. Rows that are fully masked (all -inf) give NaN, as in torch; causal masks always
    leave the diagonal visible, so they don't occur.
    """
    m = scores.amax(dim=dim, keepdim=True)                    # row max
    e = torch.exp(scores - m)                                 # in (0, 1]; masked entries -> 0
    return e / e.sum(dim=dim, keepdim=True)


def attention_values(probs: Tensor, v: Tensor) -> Tensor:
    """Weight the values by the attention probabilities: out = probs @ v.

    out[b, h, i, d] = sum over j of probs[b, h, i, j] * v[b, h, j, d].

    probs: (B, H, T, S).  v: (B, H, S, Dh).  Returns (B, H, T, Dh).
    Cost: B * H * T * S * Dh multiply-accumulates, the same as attention_scores. Like QK^T, both
    operands are produced at run time.
    """
    return probs @ v


# ----------------------------------------------------------------------------------------------
# KV cache and sampling (decode)
# ----------------------------------------------------------------------------------------------

@dataclass
class KVCache:
    """Per-layer cache of keys and values for incremental decoding.

    keys[l], values[l]: (B, H, S_max, Dh) preallocated buffers; `length` = tokens filled so far.
    During decode every new token reads the whole cache, which is why attention cost grows with
    context and is memory-bandwidth bound: one decode step reads 2 * n_layers * B * H * length * Dh
    cached values.
    """

    keys: list[Tensor] = field(default_factory=list)
    values: list[Tensor] = field(default_factory=list)
    length: int = 0

    @classmethod
    def allocate(cls, n_layers: int, batch: int, n_heads: int, max_len: int, head_dim: int,
                 *, device=None, dtype=None) -> "KVCache":
        """Preallocate zeroed buffers for n_layers layers. Preallocating (rather than
        concatenating each step) mirrors the fixed on-chip or HBM buffer a hardware design has."""
        shape = (batch, n_heads, max_len, head_dim)
        return cls(
            keys=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)],
            values=[torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)],
            length=0,
        )

    @property
    def max_len(self) -> int:
        return self.keys[0].shape[2] if self.keys else 0

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Write k_new, v_new (B, H, T, Dh) at positions length .. length+T-1 of `layer` and return
        views of all filled keys and values, (B, H, length+T, Dh). Call advance(T) after the last
        layer of a step."""
        t = k_new.shape[2]
        end = self.length + t
        if end > self.max_len:
            raise ValueError(f"KV cache full: {self.length} + {t} tokens > max_len {self.max_len}")
        self.keys[layer][:, :, self.length:end] = k_new
        self.values[layer][:, :, self.length:end] = v_new
        return self.keys[layer][:, :, :end], self.values[layer][:, :, :end]

    def advance(self, n_tokens: int) -> None:
        """Mark n_tokens more positions as filled (once per forward step, after all layers)."""
        if self.length + n_tokens > self.max_len:
            raise ValueError(f"KV cache full: {self.length} + {n_tokens} tokens > max_len {self.max_len}")
        self.length += n_tokens


def sample_next_token(logits: Tensor, temperature: float = 1.0, top_k: int | None = None,
                      generator: torch.Generator | None = None) -> Tensor:
    """Pick the next token id from the last position's logits (B, V). Returns (B,).

    temperature == 0 means greedy argmax. Otherwise divide logits by temperature, keep the top_k
    largest if given, softmax, and draw one sample per row.

    The draw is inverse-CDF sampling: with u ~ Uniform(0, 1) and c = cumsum(p), the sample is the
    first index v with c[v] > u, which is the number of entries with c[v] <= u. Index v is chosen
    with probability c[v] - c[v-1] = p[v].
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    z = logits / temperature
    if top_k is not None and top_k < z.shape[-1]:
        kth_largest = torch.topk(z, top_k, dim=-1).values[..., -1:]       # (B, 1)
        z = torch.where(z >= kth_largest, z, torch.full_like(z, float("-inf")))
    p = softmax(z, dim=-1)                                                # (B, V)
    c = torch.cumsum(p, dim=-1)                                           # (B, V), c[..., -1] ~= 1
    u = torch.rand(p.shape[0], 1, generator=generator, device=p.device, dtype=p.dtype)
    idx = (c <= u).sum(dim=-1)
    # If rounding leaves c[-1] slightly below u, idx would be V; clamp to the last index.
    return idx.clamp(max=p.shape[-1] - 1)
