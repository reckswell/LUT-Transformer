"""Baseline transformer host: the model definition and the train / evaluate / generate entry points.

This is the file to read to see how the model is put together, the way a torch.nn.Module's
forward() reads. It owns the configuration, the parameters, the list of stages, and the order in
which the stage functions are called. The math of each stage lives in train.py and inference.py,
written from base operations. There are no LUTs here: the LUT model is python/LUT/host.py, which
has its own copy of this composition (so it can be restructured later) and calls the functions
its per-stage toggles pick.

Which device runs which task (train, prefill, decode) comes from a hardware config.

    cd python
    python -m baseline.host info     --model configs/model/core.yaml --hw configs/hw/gpu.yaml
    python -m baseline.host info     --model configs/model/gpt2.yaml
    python -m baseline.host train    ...same flags...
    python -m baseline.host evaluate ...
    python -m baseline.host generate ... --prompt "Once upon a time"

Status: inference (forward, prefill, decode_step, generate) and training (forward_train, backward,
train_step) work and are checked against reference/torch_model.py and autograd. The train / evaluate /
generate commands come next; they need a tokenizer and dataset (D2).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import yaml
from torch import Tensor

from baseline import inference, train
from common import precision

PYTHON_DIR = Path(__file__).resolve().parents[1]   # python/, where configs/ lives
Mode = Literal["inference", "train"]

# The float dtype of the parameters, and so of every activation, position code, KV cache entry and
# logit computed from them. float32 is the base metric (Alpha, 2026-11-19): quantization comes
# later (D4). Tests that check the math against references use float64 explicitly; that is a
# check, not a different baseline. GPU runs also turn TF32 off (common/precision.py).
DTYPE = torch.float32


def _todo(name: str) -> NotImplementedError:
    return NotImplementedError(f"baseline.host.{name} is written in window 1")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Model shape and architecture choices (configs/model/*.yaml).

    The defaults are the barebones core (Alpha, 2026-11-19): the fewest components a decoder-only
    transformer needs to train and to carry the attention claim. Every other component is an
    option a model profile switches on; configs/model/gpt2.yaml is the first profile.

      core default                    why it is the barebones choice
      norm = rmsnorm                  one reduction per row and no shift (LayerNorm: two and a shift)
      norm_placement = pre            trains stably at depth without warm-up tricks
      activation = relu               a select on the sign bit; outputs are never negative
      positional_encoding = sinusoidal  no learned table and no length limit ("none" is also allowed)
      tie_embeddings = True           one V x D table serves as input embedding and output layer
      bias = False                    no adds at accumulator width in the integer pipeline
    """

    name: str = "core"
    vocab_size: int | None = None      # set from the tokenizer once the dataset is chosen
    context_length: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    norm: str = "rmsnorm"              # rmsnorm | layernorm
    norm_placement: str = "pre"        # pre | post
    activation: str = "relu"           # relu | gelu
    positional_encoding: str = "sinusoidal"  # sinusoidal | learned | none
    tie_embeddings: bool = True
    bias: bool = False
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.norm not in ("layernorm", "rmsnorm"):
            raise ValueError(f"norm must be layernorm or rmsnorm, got {self.norm!r}")
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be relu or gelu, got {self.activation!r}")
        if self.norm_placement not in ("pre", "post"):
            raise ValueError(f"norm_placement must be pre or post, got {self.norm_placement!r}")
        if self.positional_encoding not in ("sinusoidal", "learned", "none"):
            raise ValueError(f"positional_encoding must be sinusoidal, learned or none, got {self.positional_encoding!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransformerConfig":
        data = load_yaml(path)
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        config = cls(**known)
        config.validate()
        return config


def load_configs(model_path: str | Path, hw_path: str | Path) -> tuple[TransformerConfig, dict]:
    """Read the model config and the hardware config (device per task)."""
    model = TransformerConfig.from_yaml(model_path)
    hw = load_yaml(hw_path)
    devices = hw.get("devices") or {}
    for task in ("train", "prefill", "decode"):
        if task not in devices:
            raise ValueError(f"{hw_path}: devices.{task} is missing")
    return model, hw


# ----------------------------------------------------------------------------------------------
# Stages: the named steps of the model, each one a function in inference.py / train.py
# ----------------------------------------------------------------------------------------------
#
# Stage name -> function name. "{norm}" and "{activation}" are filled from the model config
# (layernorm or rmsnorm, relu or gelu). In training mode the functions are <name>_forward and
# <name>_backward. The LUT host toggles these same stages, so a stage is the unit that can be
# swapped for a LUT version (and later, the unit that becomes an HLS kernel).

LAYER_STAGES: dict[str, str] = {        # inside every decoder layer, in the order they run
    "norm": "{norm}",                   # used twice per layer (before attention, before the FFN)
    "qkv_proj": "linear",               # Q, K, V projections
    "attn_scores": "attention_scores",  # QK^T / sqrt(Dh), causal mask
    "softmax": "softmax",
    "attn_values": "attention_values",  # PV
    "out_proj": "linear",
    "ffn_up": "linear",
    "ffn_act": "{activation}",
    "ffn_down": "linear",
}
MODEL_STAGES: dict[str, str] = {        # outside the layers
    "embedding": "embedding",
    "final_norm": "{norm}",
    "lm_head": "linear",
}
STAGES: dict[str, str] = {**LAYER_STAGES, **MODEL_STAGES}


def function_name(stage: str, cfg: TransformerConfig) -> str:
    """The function a stage calls, for example "norm" -> "rmsnorm" under the core config."""
    return STAGES[stage].format(norm=cfg.norm, activation=cfg.activation)


@dataclass
class ModelOps:
    """The function every stage calls, as the composition below uses them.

    layers[i][stage] is the function for `stage` inside decoder layer i; embedding, final_norm
    and lm_head sit outside the layers. In "inference" mode each entry is one function; in
    "train" mode each is a (forward, backward) pair. The baseline fills every entry from
    inference.py / train.py; the LUT host swaps individual entries for LUT functions.
    """

    embedding: Any
    layers: list[dict[str, Any]]
    final_norm: Any
    lm_head: Any


def base_function(stage: str, cfg: TransformerConfig, mode: Mode) -> Any:
    """The baseline function for one stage (or its (forward, backward) pair in train mode)."""
    name = function_name(stage, cfg)
    if mode == "inference":
        return _get(inference, name)
    return (_get(train, f"{name}_forward"), _get(train, f"{name}_backward"))


def base_ops(cfg: TransformerConfig, mode: Mode) -> ModelOps:
    """Every stage on its baseline function."""
    return ModelOps(
        embedding=base_function("embedding", cfg, mode),
        layers=[{stage: base_function(stage, cfg, mode) for stage in LAYER_STAGES} for _ in range(cfg.n_layers)],
        final_norm=base_function("final_norm", cfg, mode),
        lm_head=base_function("lm_head", cfg, mode),
    )


def _get(module: Any, name: str) -> Callable[..., Any]:
    fn = getattr(module, name, None)
    if fn is None:
        raise ValueError(f"{module.__name__} has no function {name!r}")
    return fn


# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------

def param_shapes(cfg: TransformerConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter's name and shape, in a fixed order, named like the reference model so
    weights can be copied in both directions (reference/torch_model.copy_params_*):

        tok_emb                      (V, D)
        pos_emb                      (context_length, D)   only if positional_encoding = learned
        layers.{i}.norm1.gamma/beta  (D,)                  beta only for layernorm
        layers.{i}.attn.wq/wk/wv/wo  (D, D)                plus .bq/.bk/.bv/.bo (D,) if bias
        layers.{i}.norm2.gamma/beta  (D,)
        layers.{i}.ffn.w1 (F, D), .w2 (D, F)               plus .b1 (F,), .b2 (D,) if bias
        norm_f.gamma/beta            (D,)
        lm_head                      (V, D)                only if not tie_embeddings

    Weight matrices are (D_out, D_in), the torch.nn.Linear layout that inference.linear expects.
    """
    if cfg.vocab_size is None:
        raise ValueError("cfg.vocab_size must be set (it comes from the tokenizer)")
    v, d, f = cfg.vocab_size, cfg.d_model, cfg.d_ff
    shapes: dict[str, tuple[int, ...]] = {"tok_emb": (v, d)}
    if cfg.positional_encoding == "learned":
        shapes["pos_emb"] = (cfg.context_length, d)

    def norm(prefix: str) -> None:
        shapes[f"{prefix}.gamma"] = (d,)
        if cfg.norm == "layernorm":
            shapes[f"{prefix}.beta"] = (d,)

    for i in range(cfg.n_layers):
        p = f"layers.{i}"
        norm(f"{p}.norm1")
        for name in ("q", "k", "v", "o"):
            shapes[f"{p}.attn.w{name}"] = (d, d)
            if cfg.bias:
                shapes[f"{p}.attn.b{name}"] = (d,)
        norm(f"{p}.norm2")
        shapes[f"{p}.ffn.w1"] = (f, d)
        if cfg.bias:
            shapes[f"{p}.ffn.b1"] = (f,)
        shapes[f"{p}.ffn.w2"] = (d, f)
        if cfg.bias:
            shapes[f"{p}.ffn.b2"] = (d,)
    norm("norm_f")
    if not cfg.tie_embeddings:
        shapes["lm_head"] = (v, d)
    return shapes


def _is_bias(name: str) -> bool:
    """attn.bq/bk/bv/bo and ffn.b1/b2."""
    last = name.rsplit(".", 1)[-1]
    return last in ("bq", "bk", "bv", "bo", "b1", "b2")


def init_params(cfg: TransformerConfig, *, seed: int, device: str = "cpu", dtype: torch.dtype = DTYPE) -> dict[str, Tensor]:
    """Create every parameter in param_shapes(cfg) as a plain tensor in one flat dict.

    Initialization, GPT-2's scheme (DECISION: a placeholder until Alpha picks one; it only
    matters for training, since pretrained profiles load their own weights):
      - every weight matrix and embedding ~ Normal(0, 0.02)
      - the two projections that write into the residual stream (attn.wo and ffn.w2) ~
        Normal(0, 0.02 / sqrt(2 * n_layers)). Each block adds two terms to the residual sum, so
        scaling them down keeps the sum's variance from growing with depth.
      - biases 0, norm gamma 1, norm beta 0 (so every norm starts as a plain standardization)
    Random numbers come from one seeded CPU generator, drawn in param_shapes order, and are moved
    to `device` afterwards, so the same seed gives the same weights on the CPU and the GPU.
    Every tensor is created in `dtype` (DTYPE, float32, unless a test asks for float64).
    """
    gen = torch.Generator().manual_seed(seed)
    std, residual_std = 0.02, 0.02 / math.sqrt(2 * cfg.n_layers)
    params: dict[str, Tensor] = {}
    for name, shape in param_shapes(cfg).items():
        if name.endswith(".gamma"):
            t = torch.ones(shape)
        elif name.endswith(".beta") or _is_bias(name):
            t = torch.zeros(shape)
        elif name.endswith(".attn.wo") or name.endswith(".ffn.w2"):
            t = torch.randn(shape, generator=gen) * residual_std
        else:
            t = torch.randn(shape, generator=gen) * std
        params[name] = t.to(device=device, dtype=dtype)
    return params


def count_params(cfg: TransformerConfig) -> dict[str, int]:
    """Parameter counts from the shapes alone (nothing is allocated): the total, and the split
    between embeddings (token, position and an untied output layer) and the decoder layers. A
    tied embedding counts once."""
    counts = {"embeddings": 0, "layers": 0, "final_norm": 0}
    for name, shape in param_shapes(cfg).items():
        group = "layers" if name.startswith("layers.") else "final_norm" if name.startswith("norm_f") else "embeddings"
        counts[group] += math.prod(shape)
    counts["total"] = sum(counts.values())
    return counts


# ----------------------------------------------------------------------------------------------
# Model composition (the "transformer definition")
# ----------------------------------------------------------------------------------------------
#
# One decoder block i, pre-norm placement (post-norm moves each norm after its residual add),
# with L = ops.layers[i]:
#
#   h  = L["norm"](x)                                  # norm1
#   q, k, v = L["qkv_proj"](h, Wq), (h, Wk), (h, Wv)   # three projections, then split_heads
#   s  = L["attn_scores"](q, k, causal_mask)           # QK^T / sqrt(Dh), masked
#   p  = L["softmax"](s)
#   a  = L["attn_values"](p, v)                        # PV, then merge_heads
#   x  = x + L["out_proj"](a, Wo)                      # residual
#   h  = L["norm"](x)                                  # norm2
#   x  = x + L["ffn_down"](L["ffn_act"](L["ffn_up"](h, W1)), W2)   # residual
#
# Whole model: tokens -> ops.embedding (+ positions) -> n_layers blocks -> ops.final_norm ->
# ops.lm_head -> logits. Training runs this with the *_forward functions, keeps every cache, and
# then calls the *_backward functions in exactly the reverse order. Inference splits it into
# prefill (the whole prompt at once, filling the KV cache) and decode (one token at a time,
# reading the cache). Every function below takes `ops`, the function each stage calls.
# LUT/host.py has its own copy of this composition; tests/test_lut_wiring.py checks that, with
# every stage on BASE, the copy gives bit-identical results.
#
# split_heads, merge_heads, causal_mask and the position codes are not stages: they re-index
# tensors or build constants and do no arithmetic on activations, so they never get a LUT and are
# called from baseline/inference.py directly.

def _norm(fn: Callable, x: Tensor, params: dict, prefix: str, cfg: TransformerConfig) -> Tensor:
    """Call a norm stage: LayerNorm takes (x, gamma, beta), RMSNorm takes (x, gamma)."""
    if cfg.norm == "layernorm":
        return fn(x, params[f"{prefix}.gamma"], params[f"{prefix}.beta"])
    return fn(x, params[f"{prefix}.gamma"])


def _embed(params: dict, token_ids: Tensor, start: int, ops: ModelOps, cfg: TransformerConfig) -> Tensor:
    """x = token embedding + position code for tokens at absolute positions start .. start+T-1.
    token_ids (B, T) -> (B, T, D). With positional_encoding = none, order comes only from the
    causal mask."""
    x = ops.embedding(token_ids, params["tok_emb"])
    t = token_ids.shape[1]
    if cfg.positional_encoding == "learned":
        x = x + inference.learned_positions(t, params["pos_emb"], start)
    elif cfg.positional_encoding == "sinusoidal":
        x = x + inference.sinusoidal_positions(t, cfg.d_model, start, device=x.device, dtype=x.dtype)
    return x


def _attention(layer: dict, params: dict, prefix: str, h: Tensor, cfg: TransformerConfig,
               cache: inference.KVCache | None, layer_index: int) -> Tensor:
    """Multi-head causal self-attention on the normed input h (B, T, D). Returns (B, T, D).

    With a cache, the new keys and values are appended first and the queries attend to every
    cached position plus the new ones (offset = tokens already cached)."""
    t = h.shape[1]
    q = layer["qkv_proj"](h, params[f"{prefix}.wq"], params.get(f"{prefix}.bq"))   # (B, T, D)
    k = layer["qkv_proj"](h, params[f"{prefix}.wk"], params.get(f"{prefix}.bk"))
    v = layer["qkv_proj"](h, params[f"{prefix}.wv"], params.get(f"{prefix}.bv"))
    q, k, v = (inference.split_heads(z, cfg.n_heads) for z in (q, k, v))           # (B, H, T, Dh)
    offset = 0
    if cache is not None:
        offset = cache.length
        k, v = cache.append(layer_index, k, v)                                      # (B, H, offset+T, Dh)
    mask = inference.causal_mask(t, k.shape[2], offset, device=h.device)            # (T, offset+T)
    s = layer["attn_scores"](q, k, mask)                                            # (B, H, T, offset+T)
    probs = layer["softmax"](s)
    a = layer["attn_values"](probs, v)                                              # (B, H, T, Dh)
    return layer["out_proj"](inference.merge_heads(a), params[f"{prefix}.wo"], params.get(f"{prefix}.bo"))


def _feed_forward(layer: dict, params: dict, prefix: str, h: Tensor) -> Tensor:
    """down(act(up(h))): (B, T, D) -> (B, T, F) -> (B, T, D)."""
    u = layer["ffn_up"](h, params[f"{prefix}.w1"], params.get(f"{prefix}.b1"))
    return layer["ffn_down"](layer["ffn_act"](u), params[f"{prefix}.w2"], params.get(f"{prefix}.b2"))


def _block(i: int, x: Tensor, params: dict, ops: ModelOps, cfg: TransformerConfig,
           cache: inference.KVCache | None) -> Tensor:
    """Decoder layer i: attention and feed-forward, each added to the residual stream x (B, T, D)."""
    layer, p = ops.layers[i], f"layers.{i}"

    def norm(z: Tensor, which: str) -> Tensor:
        return _norm(layer["norm"], z, params, f"{p}.{which}", cfg)

    if cfg.norm_placement == "pre":
        x = x + _attention(layer, params, f"{p}.attn", norm(x, "norm1"), cfg, cache, i)
        return x + _feed_forward(layer, params, f"{p}.ffn", norm(x, "norm2"))
    x = norm(x + _attention(layer, params, f"{p}.attn", x, cfg, cache, i), "norm1")
    return norm(x + _feed_forward(layer, params, f"{p}.ffn", x), "norm2")


def _residual_stream(params: dict, token_ids: Tensor, ops: ModelOps, cfg: TransformerConfig,
                     cache: inference.KVCache | None) -> Tensor:
    """Embedding, then every decoder layer. Returns x (B, T, D) before the final norm. With a
    cache, appends this step's keys and values and advances it by T."""
    start = cache.length if cache is not None else 0
    x = _embed(params, token_ids, start, ops, cfg)
    for i in range(cfg.n_layers):
        x = _block(i, x, params, ops, cfg, cache)
    if cache is not None:
        cache.advance(token_ids.shape[1])
    return x


def _head(params: dict, x: Tensor, ops: ModelOps, cfg: TransformerConfig) -> Tensor:
    """Final norm, then the output layer: (..., D) -> logits (..., V). With tied embeddings the
    output weight is the token table itself, logits = x @ tok_emb^T."""
    h = _norm(ops.final_norm, x, params, "norm_f", cfg)
    weight = params["tok_emb"] if cfg.tie_embeddings else params["lm_head"]
    return ops.lm_head(h, weight)


def forward(params: dict, token_ids: Tensor, ops: ModelOps, cfg: TransformerConfig) -> Tensor:
    """The whole sequence at once with no cache: token_ids (B, T) -> logits (B, T, V) for every
    position. Used for evaluation (perplexity) and to check against reference/torch_model.py.

    token_ids are integer indices (int64) into the token table, not values: they pick rows of
    tok_emb and are never converted to float or quantized. Everything computed from them has the
    parameters' dtype (DTYPE, float32)."""
    return _head(params, _residual_stream(params, token_ids, ops, cfg, None), ops, cfg)


def new_cache(params: dict, cfg: TransformerConfig, batch: int, max_len: int) -> inference.KVCache:
    """An empty KV cache for `batch` sequences of up to max_len tokens, on the parameters' device
    and dtype."""
    ref = params["tok_emb"]
    return inference.KVCache.allocate(cfg.n_layers, batch, cfg.n_heads, max_len, cfg.head_dim,
                                      device=ref.device, dtype=ref.dtype)


def prefill(params: dict, token_ids: Tensor, cache: inference.KVCache, ops: ModelOps, cfg: TransformerConfig) -> Tensor:
    """Run the prompt (B, T) through the model at once, filling the KV cache. Returns the last
    position's logits (B, V). Only the last position goes through the final norm and the output
    layer, because generation needs nothing else (the output layer is the largest matmul in a
    small model)."""
    x = _residual_stream(params, token_ids, ops, cfg, cache)
    return _head(params, x[:, -1], ops, cfg)


def decode_step(params: dict, token_ids: Tensor, cache: inference.KVCache, ops: ModelOps, cfg: TransformerConfig) -> Tensor:
    """Run one new token per sequence (B, 1) against the cache, appending its key and value.
    Returns logits (B, V). Each call reads the whole cache, so its attention cost grows with the
    number of tokens so far."""
    if token_ids.shape[1] != 1:
        raise ValueError(f"decode_step takes one token per sequence, got shape {tuple(token_ids.shape)}")
    x = _residual_stream(params, token_ids, ops, cfg, cache)
    return _head(params, x[:, -1], ops, cfg)


def generate(params: dict, prompt_ids: Tensor, max_new_tokens: int, ops: ModelOps, cfg: TransformerConfig, *,
             temperature: float = 1.0, top_k: int | None = None,
             generator: torch.Generator | None = None) -> Tensor:
    """prefill once, then sample_next_token + decode_step max_new_tokens times.
    prompt_ids (B, T0) -> (B, T0 + max_new_tokens), the prompt followed by the new tokens.
    temperature 0 is greedy. With learned positions, T0 + max_new_tokens may not exceed
    context_length.

    Everything runs on the parameters' device for now. Running prefill and decode on different
    devices (the hardware config) means moving the cache between them; that comes with the
    hardware split.
    """
    batch, t0 = prompt_ids.shape
    cache = new_cache(params, cfg, batch, t0 + max_new_tokens)
    tokens = [prompt_ids]
    logits = prefill(params, prompt_ids, cache, ops, cfg)
    for step in range(max_new_tokens):
        next_id = inference.sample_next_token(logits, temperature, top_k, generator)[:, None]   # (B, 1)
        tokens.append(next_id)
        if step < max_new_tokens - 1:          # the last token doesn't need a forward pass
            logits = decode_step(params, next_id, cache, ops, cfg)
    return torch.cat(tokens, dim=1)


# ----------------------------------------------------------------------------------------------
# Training: the same composition with *_forward functions, then *_backward in reverse order
# ----------------------------------------------------------------------------------------------
#
# In "train" mode every ops entry is a (forward, backward) pair. forward_train runs the model
# exactly as forward() does, keeping each stage's cache; backward() walks the same steps in reverse,
# applying the chain rule. Two rules from the composition carry over to the gradients:
#   - a residual add x_out = x_in + branch(x_in) sends d_x_out both straight to x_in and through
#     the branch, and the two contributions are summed;
#   - a parameter used twice (the tied embedding: input lookup and output layer) gets the sum of
#     both gradients.
# split_heads and merge_heads are re-indexings, so each one's backward is the other.

def _norm_forward(pair: tuple, x: Tensor, params: dict, prefix: str, cfg: TransformerConfig) -> tuple[Tensor, Any]:
    if cfg.norm == "layernorm":
        return pair[0](x, params[f"{prefix}.gamma"], params[f"{prefix}.beta"])
    return pair[0](x, params[f"{prefix}.gamma"])


def _norm_backward(pair: tuple, dy: Tensor, cache: Any, prefix: str, grads: dict, cfg: TransformerConfig) -> Tensor:
    """Run a norm's backward, store d_gamma (and d_beta) in grads, return dx."""
    out = pair[1](dy, cache)
    grads[f"{prefix}.gamma"] = out[1]
    if cfg.norm == "layernorm":
        grads[f"{prefix}.beta"] = out[2]
    return out[0]


def _linear_backward(pair: tuple, dy: Tensor, cache: Any, w_name: str, b_name: str, grads: dict) -> Tensor:
    """Run a linear stage's backward, store dW (and db) in grads, return dx."""
    dx, dw, db = pair[1](dy, cache)
    grads[w_name] = dw
    if db is not None:
        grads[b_name] = db
    return dx


def _attention_train(layer: dict, params: dict, p: str, h: Tensor, cfg: TransformerConfig) -> tuple[Tensor, dict]:
    """Attention forward with caches (no KV cache: training sees whole sequences)."""
    t = h.shape[1]
    c: dict[str, Any] = {}
    qkv = {}
    for name in ("q", "k", "v"):
        out, c[name] = layer["qkv_proj"][0](h, params[f"{p}.w{name}"], params.get(f"{p}.b{name}"))
        qkv[name] = inference.split_heads(out, cfg.n_heads)
    mask = inference.causal_mask(t, t, device=h.device)
    s, c["scores"] = layer["attn_scores"][0](qkv["q"], qkv["k"], mask)
    probs, c["softmax"] = layer["softmax"][0](s)
    a, c["values"] = layer["attn_values"][0](probs, qkv["v"])
    out, c["out"] = layer["out_proj"][0](inference.merge_heads(a), params[f"{p}.wo"], params.get(f"{p}.bo"))
    return out, c


def _attention_backward(layer: dict, d_out: Tensor, c: dict, p: str, grads: dict, cfg: TransformerConfig) -> Tensor:
    """Reverse of _attention_train. Returns d_h, the gradient for the attention input."""
    da = _linear_backward(layer["out_proj"], d_out, c["out"], f"{p}.wo", f"{p}.bo", grads)
    d_probs, dv = layer["attn_values"][1](inference.split_heads(da, cfg.n_heads), c["values"])
    ds = layer["softmax"][1](d_probs, c["softmax"])
    dq, dk = layer["attn_scores"][1](ds, c["scores"])
    d_h = 0
    for name, d_heads in (("q", dq), ("k", dk), ("v", dv)):
        d_h = d_h + _linear_backward(layer["qkv_proj"], inference.merge_heads(d_heads), c[name],
                                     f"{p}.w{name}", f"{p}.b{name}", grads)
    return d_h


def _ffn_train(layer: dict, params: dict, p: str, h: Tensor) -> tuple[Tensor, dict]:
    c: dict[str, Any] = {}
    u, c["up"] = layer["ffn_up"][0](h, params[f"{p}.w1"], params.get(f"{p}.b1"))
    z, c["act"] = layer["ffn_act"][0](u)
    out, c["down"] = layer["ffn_down"][0](z, params[f"{p}.w2"], params.get(f"{p}.b2"))
    return out, c


def _ffn_backward(layer: dict, d_out: Tensor, c: dict, p: str, grads: dict) -> Tensor:
    dz = _linear_backward(layer["ffn_down"], d_out, c["down"], f"{p}.w2", f"{p}.b2", grads)
    du = layer["ffn_act"][1](dz, c["act"])
    return _linear_backward(layer["ffn_up"], du, c["up"], f"{p}.w1", f"{p}.b1", grads)


def forward_train(params: dict, token_ids: Tensor, ops: ModelOps, cfg: TransformerConfig) -> tuple[Tensor, dict]:
    """Full forward pass with caches for backward. Returns (logits (B, T, V), caches)."""
    if cfg.dropout:
        raise NotImplementedError("dropout > 0 is not wired into forward_train yet (every config uses 0.0)")
    caches: dict[str, Any] = {"T": token_ids.shape[1], "layers": []}
    x, caches["embedding"] = ops.embedding[0](token_ids, params["tok_emb"])
    t = token_ids.shape[1]
    if cfg.positional_encoding == "learned":
        x = x + params["pos_emb"][:t]
    elif cfg.positional_encoding == "sinusoidal":
        x = x + inference.sinusoidal_positions(t, cfg.d_model, device=x.device, dtype=x.dtype)
    for i in range(cfg.n_layers):
        layer, p, c = ops.layers[i], f"layers.{i}", {}
        if cfg.norm_placement == "pre":
            h, c["norm1"] = _norm_forward(layer["norm"], x, params, f"{p}.norm1", cfg)
            attn, c["attn"] = _attention_train(layer, params, f"{p}.attn", h, cfg)
            x = x + attn
            h, c["norm2"] = _norm_forward(layer["norm"], x, params, f"{p}.norm2", cfg)
            ffn, c["ffn"] = _ffn_train(layer, params, f"{p}.ffn", h)
            x = x + ffn
        else:
            attn, c["attn"] = _attention_train(layer, params, f"{p}.attn", x, cfg)
            x, c["norm1"] = _norm_forward(layer["norm"], x + attn, params, f"{p}.norm1", cfg)
            ffn, c["ffn"] = _ffn_train(layer, params, f"{p}.ffn", x)
            x, c["norm2"] = _norm_forward(layer["norm"], x + ffn, params, f"{p}.norm2", cfg)
        caches["layers"].append(c)
    h, caches["final_norm"] = _norm_forward(ops.final_norm, x, params, "norm_f", cfg)
    weight = params["tok_emb"] if cfg.tie_embeddings else params["lm_head"]
    logits, caches["lm_head"] = ops.lm_head[0](h, weight)
    return logits, caches


def backward(d_logits: Tensor, caches: dict, ops: ModelOps, cfg: TransformerConfig) -> dict[str, Tensor]:
    """Backpropagate d_logits through the model in reverse order. Returns {param name: gradient},
    with residual-branch gradients summed and tied-embedding gradients added together."""
    grads: dict[str, Tensor] = {}
    dh, d_head, _ = ops.lm_head[1](d_logits, caches["lm_head"])
    dx = _norm_backward(ops.final_norm, dh, caches["final_norm"], "norm_f", grads, cfg)
    for i in reversed(range(cfg.n_layers)):
        layer, p, c = ops.layers[i], f"layers.{i}", caches["layers"][i]
        if cfg.norm_placement == "pre":
            # x_out = x_mid + ffn(norm2(x_mid)): dx reaches x_mid directly and through the branch.
            d_branch = _ffn_backward(layer, dx, c["ffn"], f"{p}.ffn", grads)
            dx = dx + _norm_backward(layer["norm"], d_branch, c["norm2"], f"{p}.norm2", grads, cfg)
            # x_mid = x_in + attn(norm1(x_in))
            d_branch = _attention_backward(layer, dx, c["attn"], f"{p}.attn", grads, cfg)
            dx = dx + _norm_backward(layer["norm"], d_branch, c["norm1"], f"{p}.norm1", grads, cfg)
        else:
            # x_out = norm2(x_mid + ffn(x_mid)), then x_mid = norm1(x_in + attn(x_in))
            d_sum = _norm_backward(layer["norm"], dx, c["norm2"], f"{p}.norm2", grads, cfg)
            dx = d_sum + _ffn_backward(layer, d_sum, c["ffn"], f"{p}.ffn", grads)
            d_sum = _norm_backward(layer["norm"], dx, c["norm1"], f"{p}.norm1", grads, cfg)
            dx = d_sum + _attention_backward(layer, d_sum, c["attn"], f"{p}.attn", grads, cfg)
    d_table = ops.embedding[1](dx, caches["embedding"])
    if cfg.tie_embeddings:
        grads["tok_emb"] = d_table + d_head          # used twice: input lookup and output layer
    else:
        grads["tok_emb"], grads["lm_head"] = d_table, d_head
    if cfg.positional_encoding == "learned":
        # x = tok + pos_emb[:T] broadcast over the batch, so each used row gets dx summed over B.
        t = caches["T"]
        d_pos = torch.zeros(cfg.context_length, cfg.d_model, device=dx.device, dtype=dx.dtype)
        d_pos[:t] = dx.sum(dim=0)
        grads["pos_emb"] = d_pos
    return grads


def train_step(params: dict, batch: Any, ops: ModelOps, opt_state: dict, cfg: TransformerConfig, *, lr: float,
               max_grad_norm: float = 1.0, weight_decay: float = 0.1) -> dict:
    """forward_train -> cross_entropy -> backward -> clip_grad_norm -> adamw_step.
    batch is (inputs, targets), both (B, T). Updates params in place.
    Returns {"loss", "grad_norm", "tokens", "lr"} for training_loop."""
    inputs, targets = batch
    logits, caches = forward_train(params, inputs, ops, cfg)
    loss, loss_cache = train.cross_entropy_forward(logits, targets)
    grads = backward(train.cross_entropy_backward(loss_cache), caches, ops, cfg)
    grad_norm = train.clip_grad_norm(grads, max_grad_norm)
    train.adamw_step(params, grads, opt_state, lr=lr, weight_decay=weight_decay)
    return {"loss": float(loss), "grad_norm": grad_norm, "tokens": int(targets.numel()), "lr": lr}


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the baseline and LUT hosts."""
    parser.add_argument("command", choices=["info", "train", "evaluate", "generate"])
    parser.add_argument("--model", default=str(PYTHON_DIR / "configs/model/core.yaml"))
    parser.add_argument("--hw", default=str(PYTHON_DIR / "configs/hw/gpu.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)


def _cmd_info(cfg: TransformerConfig, hw: dict, fp: dict) -> None:
    print(f"model   {cfg.name}: {asdict(cfg)}")
    print(f"hw      {hw.get('name', '?')}: {hw['devices']}")
    print(f"dtype   {DTYPE}; float32 matmul precision {fp['float32_matmul_precision']}, "
          f"TF32 {'on' if fp['cuda_matmul_allow_tf32'] or fp['cudnn_allow_tf32'] else 'off'}")
    print()
    print(f"{'stage':<13}function")
    for stage in STAGES:
        print(f"{stage:<13}{function_name(stage, cfg)}")
    if cfg.vocab_size is None:
        print("\nparameters: vocab_size not set yet (it comes from the tokenizer)")
    else:
        counts = count_params(cfg)
        print(f"\nparameters: {counts['total']:,} total = {counts['embeddings']:,} embeddings "
              f"+ {counts['layers']:,} in {cfg.n_layers} layers + {counts['final_norm']:,} final norm")
    base_ops(cfg, "inference")   # raises if a function is missing
    base_ops(cfg, "train")
    print("\nall stage functions exist for inference and training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baseline (non-LUT) transformer host")
    add_common_args(parser)
    args = parser.parse_args(argv)

    fp = precision.full_fp32()          # full float32 math, no TF32, before anything runs
    cfg, hw = load_configs(args.model, args.hw)
    if args.command == "info":
        _cmd_info(cfg, hw, fp)
        return
    raise SystemExit(f"'{args.command}' is written in window 1. Try 'info' for now.")


if __name__ == "__main__":
    main()
