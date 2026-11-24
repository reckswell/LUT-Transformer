"""Hugging Face model families mapped onto our stages: one checked profile per family.

A control, not part of our model (like hf_gpt2.py and hf_llama.py): the Hugging Face model is
imported and never modified; each profile only reads its config and state dict and says how the
same computation runs as our stages (baseline/host.py STAGES), so the from-scratch model, the LUT
model and every LUT/host.py slot run the family's real weights.

    model_type  checkpoints (examples)        what it adds to the core, as TransformerConfig options
    llama       Llama 1/2/3 (plain RoPE),     RMSNorm, SwiGLU, rotary; grouped-query attention
                TinyLlama, stories15M         (n_kv_heads), head_size, attention/MLP biases
    mistral     Mistral 7B v0.1-v0.3          llama + sliding-window attention (attention_window)
    qwen2       Qwen2, Qwen2.5                llama + biases on Q/K/V only (qkv_bias), sliding window
                                              when every layer uses it
    gemma       Gemma 1 (2B, 7B)              llama + RMSNorm weight stored as w with gamma = 1 + w,
                                              embedding scale sqrt(d_model), tanh GELU in the gated
                                              feed-forward, 256-wide heads, multi-query on 2B
    gpt_neox    Pythia, GPT-NeoX-20B          LayerNorm, exact GELU, parallel block with two norms,
                                              partial rotary (25 % of each head), fused QKV weight
    phi         Phi-1, Phi-1.5, Phi-2         LayerNorm, tanh GELU, parallel block with one shared
                                              norm, partial rotary, biases everywhere, output bias
    opt         OPT 125M-66B (not 350M)       LayerNorm, ReLU, learned positions (offset 2), biases

Each profile has three parts, as reference/hf_profiles.Profile wants:
  config_from_hf(hf_config, name)  our TransformerConfig. It checks every setting it reads and
                                   raises Unsupported listing all of the settings it can't model,
                                   rather than running them wrongly.
  params_from_hf(state_dict, cfg)  our parameter dict, in host.param_shapes(cfg) names and
                                   layouts. Every entry of the state dict must be used (except
                                   known constant buffers), so a checkpoint with a part the profile
                                   doesn't model (an extra norm, say) fails instead of dropping it.
  load(model_name, ...)            from the local checkpoint folder, never the network by default.

reference/inspect_model.py uses the same pieces to list what any model uses that we don't model
yet, and to check a profile against a small random copy of the model. tests/test_hf_families.py
checks every profile against Hugging Face on small random models (no download).

Stages that are not new functions: grouped-query attention repeats key/value heads (a re-indexing,
inference.repeat_kv), partial rotary calls the rope stage on the rotated columns only, the
sliding window changes the mask, the parallel block adds both branches to the residual, and the
embedding scale is a constant multiply (folded into the token table in hardware unless the table
is shared with the output layer). Gemma's 1 + w is added once here, offline, so the norm stage is
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch
from torch import Tensor

from baseline import host
from reference.local_cache import CHECKPOINT_DIR, use_local_hf_cache

# Hugging Face activation names (transformers' ACT2FN) -> ours.
ACTIVATIONS = {
    "silu": "silu", "swish": "silu",
    "relu": "relu",
    "gelu": "gelu_exact",                # GELUActivation: x Phi(x) with erf
    "gelu_new": "gelu",                  # NewGELUActivation: the tanh form, same formula as ours
    "gelu_pytorch_tanh": "gelu",         # F.gelu(approximate="tanh")
    "gelu_fast": "gelu",                 # the tanh form with sqrt(2/pi) rounded to 0.7978845608 (3e-12 apart)
}


class Unsupported(ValueError):
    """A config uses settings our stages don't model. problems lists every one, not just the first."""

    def __init__(self, family: str, problems: list[str]) -> None:
        self.family, self.problems = family, list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{family}: {len(self.problems)} setting(s) not modeled:\n{lines}")


class Findings:
    """Collects what a config mapper can't model (problems) and what it adapts (notes)."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.problems: list[str] = []
        self.notes: list[str] = []

    def require(self, ok: bool, problem: str) -> None:
        if not ok:
            self.problems.append(problem)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def done(self) -> None:
        if self.problems:
            raise Unsupported(self.family, self.problems)


# ----------------------------------------------------------------------------------------------
# Config pieces shared by the families
# ----------------------------------------------------------------------------------------------

def activation(found: Findings, hf_name: str | None) -> str:
    """Our activation for a Hugging Face activation name."""
    ours = ACTIVATIONS.get(hf_name)
    found.require(ours is not None, f"activation {hf_name!r} is not modeled (modeled: {sorted(ACTIVATIONS)})")
    return ours or "relu"


def rope_settings(hf_config: Any) -> dict[str, Any]:
    """rope_type, rope_theta and partial_rotary_factor. transformers 5 keeps them in
    config.rope_parameters; older configs have rope_theta, rope_scaling and partial_rotary_factor
    (or GPT-NeoX's rotary_pct / rotary_emb_base) at the top level."""
    parameters = dict(getattr(hf_config, "rope_parameters", None) or {})
    if not parameters:
        scaling = getattr(hf_config, "rope_scaling", None) or {}
        parameters = {"rope_type": scaling.get("rope_type", scaling.get("type", "default")),
                      "rope_theta": getattr(hf_config, "rope_theta", getattr(hf_config, "rotary_emb_base", 10000.0)),
                      "partial_rotary_factor": getattr(hf_config, "partial_rotary_factor",
                                                       getattr(hf_config, "rotary_pct", 1.0))}
    parameters.setdefault("rope_type", "default")
    parameters.setdefault("partial_rotary_factor", 1.0)
    return parameters


def rotary(found: Findings, hf_config: Any, head_dim: int) -> dict[str, Any]:
    """rope_theta and rotary_dim (None when the whole head rotates)."""
    rope = rope_settings(hf_config)
    found.require(rope["rope_type"] == "default",
                  f"rotary variant {rope['rope_type']!r} is not modeled (only plain RoPE; scaled variants change the angle table)")
    rotated = int(head_dim * float(rope["partial_rotary_factor"]))        # as Hugging Face computes it
    found.require(rotated % 2 == 0 and rotated > 0, f"rotary width {rotated} must be even and positive")
    return {"rope_theta": float(rope["rope_theta"]), "rotary_dim": None if rotated == head_dim else rotated}


def heads(hf_config: Any) -> dict[str, Any]:
    """n_heads, n_kv_heads (None without grouped-query attention) and head_size (None when it is
    d_model / n_heads)."""
    n_heads = hf_config.num_attention_heads
    n_kv = getattr(hf_config, "num_key_value_heads", None) or n_heads
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // n_heads
    return {"n_heads": n_heads, "n_kv_heads": None if n_kv == n_heads else n_kv,
            "head_size": None if head_dim * n_heads == hf_config.hidden_size else head_dim}


def head_dim_of(kw: dict[str, Any], d_model: int) -> int:
    return kw["head_size"] if kw["head_size"] is not None else d_model // kw["n_heads"]


def window(found: Findings, hf_config: Any) -> int | None:
    """attention_window for a sliding-window config, None for full attention. Mixed layers (some
    sliding, some full: Qwen2 with max_window_layers, Gemma 2/3) are not modeled."""
    size = getattr(hf_config, "sliding_window", None)
    types = getattr(hf_config, "layer_types", None)
    if types:
        kinds = set(types)
        found.require(kinds <= {"full_attention", "sliding_attention"}, f"layer types {sorted(kinds)} are not modeled")
        found.require(len(kinds) == 1, f"layers mix {sorted(kinds)}; one attention kind for every layer is modeled")
        return size if kinds == {"sliding_attention"} else None
    return size


def _no(found: Findings, hf_config: Any, *names: str) -> None:
    """Settings that, when present and truthy, change the computation in ways we don't model."""
    for name in names:
        value = getattr(hf_config, name, None)
        found.require(not value, f"{name} = {value!r} is not modeled")


# ----------------------------------------------------------------------------------------------
# State dicts
# ----------------------------------------------------------------------------------------------

class StateDict:
    """A Hugging Face state dict that remembers which entries were read, so params_from_hf can
    refuse a checkpoint with entries no stage uses. `ignore` lists name endings of constant
    buffers (rotary frequency tables, old causal-mask buffers)."""

    def __init__(self, state_dict: dict[str, Tensor], prefix: str, ignore: tuple[str, ...] = ()) -> None:
        self.sd = {name: t.detach() for name, t in state_dict.items()}
        self.prefix, self.ignore, self.used = prefix, ignore, set()

    def get(self, name: str, *, top: bool = False) -> Tensor | None:
        key = name if top else self.prefix + name
        if key in self.sd:
            self.used.add(key)
            return self.sd[key]
        return None

    def __call__(self, name: str, *, top: bool = False) -> Tensor:
        t = self.get(name, top=top)
        if t is None:
            raise KeyError(f"state dict has no {name if top else self.prefix + name}")
        return t

    def unused(self) -> list[str]:
        return sorted(k for k in self.sd if k not in self.used and not k.endswith(self.ignore))


def finish(params: dict[str, Tensor], cfg: host.TransformerConfig, sd: StateDict) -> dict[str, Tensor]:
    """Check the export against host.param_shapes(cfg) and that every state-dict entry was used.
    Returns contiguous detached copies."""
    left = sd.unused()
    if left:
        raise ValueError(f"state dict entries no stage uses (a part this profile doesn't model): {left}")
    expected = host.param_shapes(cfg)
    if params.keys() != expected.keys():
        raise KeyError(f"names differ from param_shapes: {sorted(params.keys() ^ expected.keys())}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(params[name].shape)} != expected {shape}")
    return {name: t.contiguous().clone() for name, t in params.items()}


def output_layer(params: dict[str, Tensor], cfg: host.TransformerConfig, sd: StateDict, name: str = "lm_head.weight") -> None:
    """lm_head when untied; when tied, check the stored copy (if any) equals the token table."""
    head = sd.get(name, top=True)
    if cfg.tie_embeddings:
        if head is not None and not torch.equal(head, params["tok_emb"]):
            raise ValueError(f"config says tied embeddings but {name} differs from the token table")
    elif head is None:
        raise KeyError(f"untied embeddings but the state dict has no {name}")
    else:
        params["lm_head"] = head


def _norm(params: dict[str, Tensor], ours: str, sd: StateDict, theirs: str, *, offset: float = 0.0,
          d: int | None = None) -> None:
    """A norm's weight (and bias for LayerNorm). offset: Gemma stores w with gamma = 1 + w.
    A LayerNorm without weights (OPT's elementwise_affine = False) gets gamma 1 and beta 0."""
    w = sd.get(f"{theirs}.weight")
    if w is None:
        params[f"{ours}.gamma"] = torch.ones(d)
        params[f"{ours}.beta"] = torch.zeros(d)
        return
    params[f"{ours}.gamma"] = w + offset if offset else w
    b = sd.get(f"{theirs}.bias")
    if b is not None:
        params[f"{ours}.beta"] = b


def _linear(params: dict[str, Tensor], ours_w: str, ours_b: str, sd: StateDict, theirs: str) -> None:
    params[ours_w] = sd(f"{theirs}.weight")
    b = sd.get(f"{theirs}.bias")
    if b is not None:
        params[ours_b] = b


# ----------------------------------------------------------------------------------------------
# Llama-style decoders: llama, mistral, qwen2, gemma
# ----------------------------------------------------------------------------------------------

def _llama_style(found: Findings, hf: Any, *, biases: tuple[bool, bool, bool], name: str) -> dict[str, Any]:
    kw = heads(hf)
    d = hf.hidden_size
    act_name = getattr(hf, "hidden_act", None) or getattr(hf, "hidden_activation", None)
    qkv_bias, out_bias, ffn_bias = biases
    kw.update(
        name=name,
        vocab_size=hf.vocab_size,
        context_length=hf.max_position_embeddings,       # the trained context; RoPE itself has no limit
        d_model=d,
        n_layers=hf.num_hidden_layers,
        d_ff=hf.intermediate_size,
        norm="rmsnorm",
        norm_eps=float(hf.rms_norm_eps),
        norm_placement="pre",
        activation=activation(found, act_name),
        ffn="glu",
        positional_encoding="rope",
        tie_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
        bias=False,
        qkv_bias=qkv_bias or None,
        out_bias=out_bias or None,
        ffn_bias=ffn_bias or None,
        attention_window=window(found, hf),
        **rotary(found, hf, head_dim_of(kw, d)),
    )
    _no(found, hf, "attn_logit_softcapping", "final_logit_softcapping", "qk_layernorm", "use_qk_norm",
        "num_local_experts", "num_experts")
    return kw


def _config(family: str, mapper: Callable[[Findings, Any, str], dict[str, Any]]) -> Callable[..., host.TransformerConfig]:
    def config_from_hf(hf_config: Any, name: str | None = None) -> host.TransformerConfig:
        found = Findings(family)
        kw = mapper(found, hf_config, name or family)
        found.done()
        cfg = host.TransformerConfig(**kw)
        cfg.validate()
        return cfg
    config_from_hf.__name__ = f"{family}_config_from_hf"
    return config_from_hf


def _findings(family: str, mapper: Callable, hf_config: Any) -> Findings:
    found = Findings(family)
    try:
        mapper(found, hf_config, family)
    except Exception as err:                  # a missing attribute is a finding too
        found.problems.append(f"config could not be read: {type(err).__name__}: {err}")
    return found


def _llama_map(found: Findings, hf: Any, name: str) -> dict[str, Any]:
    attention_bias = bool(getattr(hf, "attention_bias", False))
    return _llama_style(found, hf, biases=(attention_bias, attention_bias, bool(getattr(hf, "mlp_bias", False))), name=name)


def _mistral_map(found: Findings, hf: Any, name: str) -> dict[str, Any]:
    return _llama_style(found, hf, biases=(False, False, False), name=name)


def _qwen2_map(found: Findings, hf: Any, name: str) -> dict[str, Any]:
    return _llama_style(found, hf, biases=(True, False, False), name=name)    # Q/K/V biases only (modeling_qwen2.py)


def _gemma_map(found: Findings, hf: Any, name: str) -> dict[str, Any]:
    attention_bias = bool(getattr(hf, "attention_bias", False))
    kw = _llama_style(found, hf, biases=(attention_bias, attention_bias, False), name=name)
    # Gemma multiplies the embedding by sqrt(d_model) held as a float32 tensor (GemmaTextScaledWordEmbedding),
    # so the constant is sqrt(d) rounded to float32.
    kw["embed_scale"] = float(torch.tensor(hf.hidden_size ** 0.5, dtype=torch.float32))
    return kw


def _llama_style_params(state_dict: dict[str, Tensor], cfg: host.TransformerConfig, *, norm_offset: float = 0.0) -> dict[str, Tensor]:
    """Hugging Face Llama-style names (model.embed_tokens, model.layers.{i}.input_layernorm,
    self_attn.{q,k,v,o}_proj, post_attention_layernorm, mlp.{gate,up,down}_proj, model.norm,
    lm_head) onto ours. gate_proj -> w1 (through the activation), up_proj -> w3, down_proj -> w2.
    q_proj and k_proj are already in the rotate_half row order our rope uses."""
    sd = StateDict(state_dict, "model.", ignore=("rotary_emb.inv_freq",))
    params: dict[str, Tensor] = {"tok_emb": sd("embed_tokens.weight")}
    for i in range(cfg.n_layers):
        h, p = f"layers.{i}", f"layers.{i}"
        _norm(params, f"{p}.norm1", sd, f"{h}.input_layernorm", offset=norm_offset)
        for ours, theirs in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
            _linear(params, f"{p}.attn.w{ours}", f"{p}.attn.b{ours}", sd, f"{h}.self_attn.{theirs}")
        _norm(params, f"{p}.norm2", sd, f"{h}.post_attention_layernorm", offset=norm_offset)
        for ours, theirs in (("1", "gate_proj"), ("3", "up_proj"), ("2", "down_proj")):
            _linear(params, f"{p}.ffn.w{ours}", f"{p}.ffn.b{ours}", sd, f"{h}.mlp.{theirs}")
    _norm(params, "norm_f", sd, "norm", offset=norm_offset)
    output_layer(params, cfg, sd)
    return finish(params, cfg, sd)


def llama_params_from_hf(state_dict: dict[str, Tensor], cfg: host.TransformerConfig) -> dict[str, Tensor]:
    return _llama_style_params(state_dict, cfg)


def gemma_params_from_hf(state_dict: dict[str, Tensor], cfg: host.TransformerConfig) -> dict[str, Tensor]:
    """As llama, with every RMSNorm gamma = 1 + the stored weight (GemmaRMSNorm multiplies by 1 + w)."""
    return _llama_style_params(state_dict, cfg, norm_offset=1.0)


# ----------------------------------------------------------------------------------------------
# GPT-NeoX (Pythia)
# ----------------------------------------------------------------------------------------------

def _gpt_neox_map(found: Findings, hf: Any, name: str) -> dict[str, Any]:
    kw = heads(hf)
    d = hf.hidden_size
    bias = bool(getattr(hf, "attention_bias", True))
    kw.update(
        name=name, vocab_size=hf.vocab_size, context_length=hf.max_position_embeddings, d_model=d,
        n_layers=hf.num_hidden_layers, d_ff=hf.intermediate_size,
        norm="layernorm", norm_eps=float(hf.layer_norm_eps), norm_placement="pre",
        activation=activation(found, hf.hidden_act), ffn="mlp", positional_encoding="rope",
        block="parallel" if getattr(hf, "use_parallel_residual", True) else "sequential",
        tie_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
        bias=True, qkv_bias=bias, out_bias=bias, ffn_bias=True,             # dense_h_to_4h / 4h_to_h always biased
        **rotary(found, hf, head_dim_of(kw, d)),
    )
    found.require(kw["n_kv_heads"] is None, "GPT-NeoX with fewer key/value heads is not modeled")
    found.require(kw["head_size"] is None, "GPT-NeoX head width must be hidden_size / num_attention_heads")
    return kw


def gpt_neox_params_from_hf(state_dict: dict[str, Tensor], cfg: host.TransformerConfig) -> dict[str, Tensor]:
    """GPT-NeoX names onto ours. The fused query_key_value weight (3 D, D) holds, per head, that
    head's q rows, then its k rows, then its v rows (the output is viewed as (..., H, 3 Dh) and
    split in three, modeling_gpt_neox.py), so it is cut per head:
        rows viewed as (H, 3, Dh, D): wq = [:, 0], wk = [:, 1], wv = [:, 2], each back to (H Dh, D).
    attention.dense -> wo; mlp.dense_h_to_4h -> w1, dense_4h_to_h -> w2; input_layernorm -> norm1,
    post_attention_layernorm -> norm2; final_layer_norm -> norm_f; lm_head (embed_out before
    transformers 5) -> lm_head."""
    sd = StateDict(state_dict, "gpt_neox.", ignore=("rotary_emb.inv_freq", "attention.bias", "attention.masked_bias"))
    h_, dh = cfg.n_heads, cfg.head_dim
    params: dict[str, Tensor] = {"tok_emb": sd("embed_in.weight")}
    for i in range(cfg.n_layers):
        h, p = f"layers.{i}", f"layers.{i}"
        _norm(params, f"{p}.norm1", sd, f"{h}.input_layernorm")
        _norm(params, f"{p}.norm2", sd, f"{h}.post_attention_layernorm")
        w = sd(f"{h}.attention.query_key_value.weight").reshape(h_, 3, dh, -1)
        b = sd.get(f"{h}.attention.query_key_value.bias")
        b = None if b is None else b.reshape(h_, 3, dh)
        for j, name in enumerate("qkv"):
            params[f"{p}.attn.w{name}"] = w[:, j].reshape(h_ * dh, -1)
            if b is not None:
                params[f"{p}.attn.b{name}"] = b[:, j].reshape(h_ * dh)
        _linear(params, f"{p}.attn.wo", f"{p}.attn.bo", sd, f"{h}.attention.dense")
        _linear(params, f"{p}.ffn.w1", f"{p}.ffn.b1", sd, f"{h}.mlp.dense_h_to_4h")
        _linear(params, f"{p}.ffn.w2", f"{p}.ffn.b2", sd, f"{h}.mlp.dense_4h_to_h")
    _norm(params, "norm_f", sd, "final_layer_norm")
    # transformers 5 names the output layer lm_head; checkpoints saved by older versions, embed_out.
    output_layer(params, cfg, sd, "embed_out.weight" if "embed_out.weight" in sd.sd else "lm_head.weight")
    return finish(params, cfg, sd)


# ----------------------------------------------------------------------------------------------
# Phi (1, 1.5, 2)
# ----------------------------------------------------------------------------------------------

def _phi_map(found: Findings, hf: Any, name: str) -> dict[str, Any]:
    kw = heads(hf)
    d = hf.hidden_size
    kw.update(
        name=name, vocab_size=hf.vocab_size, context_length=hf.max_position_embeddings, d_model=d,
        n_layers=hf.num_hidden_layers, d_ff=hf.intermediate_size,
        norm="layernorm", norm_eps=float(hf.layer_norm_eps), norm_placement="pre",
        activation=activation(found, hf.hidden_act), ffn="mlp", positional_encoding="rope",
        block="parallel_shared_norm",                    # one input_layernorm feeds attention and MLP
        tie_embeddings=bool(getattr(hf, "tie_word_embeddings", False)),
        bias=True, lm_head_bias=True,                    # every Linear in modeling_phi.py has a bias
        **rotary(found, hf, head_dim_of(kw, d)),
    )
    _no(found, hf, "qk_layernorm")
    return kw


def phi_params_from_hf(state_dict: dict[str, Tensor], cfg: host.TransformerConfig) -> dict[str, Tensor]:
    """Phi names onto ours: input_layernorm -> norm1 (the only norm of the layer),
    self_attn.{q,k,v}_proj -> wq/wk/wv, self_attn.dense -> wo, mlp.fc1 -> w1, mlp.fc2 -> w2,
    final_layernorm -> norm_f, lm_head (weight and bias) -> lm_head, lm_head_bias."""
    sd = StateDict(state_dict, "model.", ignore=("rotary_emb.inv_freq",))
    params: dict[str, Tensor] = {"tok_emb": sd("embed_tokens.weight")}
    for i in range(cfg.n_layers):
        h, p = f"layers.{i}", f"layers.{i}"
        _norm(params, f"{p}.norm1", sd, f"{h}.input_layernorm")
        for ours, theirs in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "dense")):
            _linear(params, f"{p}.attn.w{ours}", f"{p}.attn.b{ours}", sd, f"{h}.self_attn.{theirs}")
        _linear(params, f"{p}.ffn.w1", f"{p}.ffn.b1", sd, f"{h}.mlp.fc1")
        _linear(params, f"{p}.ffn.w2", f"{p}.ffn.b2", sd, f"{h}.mlp.fc2")
    _norm(params, "norm_f", sd, "final_layernorm")
    output_layer(params, cfg, sd)
    params["lm_head_bias"] = sd("lm_head.bias", top=True)
    return finish(params, cfg, sd)


# ----------------------------------------------------------------------------------------------
# OPT
# ----------------------------------------------------------------------------------------------

OPT_POSITION_OFFSET = 2      # OPTLearnedPositionalEmbedding looks up position + 2


def _opt_map(found: Findings, hf: Any, name: str) -> dict[str, Any]:
    found.require(getattr(hf, "word_embed_proj_dim", hf.hidden_size) == hf.hidden_size,
                  f"word_embed_proj_dim {getattr(hf, 'word_embed_proj_dim', None)} != hidden_size {hf.hidden_size}: "
                  "project_in / project_out layers are not modeled (OPT-350M)")
    found.require(getattr(hf, "do_layer_norm_before", True), "post-norm OPT (do_layer_norm_before = False, OPT-350M) is not modeled")
    _no(found, hf, "_remove_final_layer_norm")
    bias = bool(getattr(hf, "enable_bias", True))
    return dict(
        name=name, vocab_size=hf.vocab_size, context_length=hf.max_position_embeddings, d_model=hf.hidden_size,
        n_heads=hf.num_attention_heads, n_layers=hf.num_hidden_layers, d_ff=hf.ffn_dim,
        norm="layernorm", norm_eps=1e-5, norm_placement="pre",               # nn.LayerNorm's default eps
        activation=activation(found, hf.activation_function), ffn="mlp", positional_encoding="learned",
        tie_embeddings=bool(getattr(hf, "tie_word_embeddings", True)), bias=bias,
    )


def opt_params_from_hf(state_dict: dict[str, Tensor], cfg: host.TransformerConfig) -> dict[str, Tensor]:
    """OPT names onto ours: decoder.embed_positions rows 2.. -> pos_emb (position p is row p + 2),
    self_attn_layer_norm -> norm1, self_attn.{q,k,v,out}_proj -> wq/wk/wv/wo, final_layer_norm
    (the layer's second norm) -> norm2, fc1 -> w1, fc2 -> w2, decoder.final_layer_norm -> norm_f.
    OPT scales q by 1/sqrt(Dh) right after q_proj; our attention_scores divides QK^T by sqrt(Dh)
    instead, the same product."""
    sd = StateDict(state_dict, "model.decoder.")
    d = cfg.d_model
    params: dict[str, Tensor] = {"tok_emb": sd("embed_tokens.weight"),
                                 "pos_emb": sd("embed_positions.weight")[OPT_POSITION_OFFSET:]}
    for i in range(cfg.n_layers):
        h, p = f"layers.{i}", f"layers.{i}"
        _norm(params, f"{p}.norm1", sd, f"{h}.self_attn_layer_norm", d=d)
        for ours, theirs in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")):
            _linear(params, f"{p}.attn.w{ours}", f"{p}.attn.b{ours}", sd, f"{h}.self_attn.{theirs}")
        _norm(params, f"{p}.norm2", sd, f"{h}.final_layer_norm", d=d)
        _linear(params, f"{p}.ffn.w1", f"{p}.ffn.b1", sd, f"{h}.fc1")
        _linear(params, f"{p}.ffn.w2", f"{p}.ffn.b2", sd, f"{h}.fc2")
    _norm(params, "norm_f", sd, "final_layer_norm", d=d)
    output_layer(params, cfg, sd)
    return finish(params, cfg, sd)


# ----------------------------------------------------------------------------------------------
# Loading and the family table
# ----------------------------------------------------------------------------------------------

def loader(model_type: str, config_from_hf: Callable, params_from_hf: Callable) -> Callable:
    """load(model_name, ...) for a family: the checkpoint from the local folder (never the network
    unless local_files_only=False), our config and params, and the Hugging Face model (eval mode,
    on the CPU) as the ground truth, all in `dtype`."""
    def load(model_name: str, *, device: str = "cpu", dtype: torch.dtype = host.DTYPE,
             cache_dir: str | Path = CHECKPOINT_DIR, local_files_only: bool = True,
             revision: str | None = None) -> tuple[host.TransformerConfig, dict[str, Tensor], Any]:
        use_local_hf_cache()
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir), dtype=dtype,
                                                        local_files_only=local_files_only, revision=revision).eval()
        if hf_model.config.model_type != model_type:
            raise ValueError(f"{model_name} is a {hf_model.config.model_type!r} model, not {model_type!r}")
        if next(hf_model.parameters()).dtype != dtype:
            hf_model = hf_model.to(dtype)
        cfg = config_from_hf(hf_model.config, name=Path(str(model_name)).name or model_type)
        params = params_from_hf(hf_model.state_dict(), cfg)
        return cfg, {k: t.to(device=device, dtype=dtype) for k, t in params.items()}, hf_model
    load.__name__ = f"load_{model_type}"
    return load


class Family(NamedTuple):
    config_from_hf: Callable[..., host.TransformerConfig]
    params_from_hf: Callable[[dict[str, Tensor], host.TransformerConfig], dict[str, Tensor]]
    load: Callable[..., tuple[host.TransformerConfig, dict[str, Tensor], Any]]
    mapper: Callable[[Findings, Any, str], dict[str, Any]]


def _family(model_type: str, mapper: Callable, params_from_hf: Callable) -> Family:
    config_from_hf = _config(model_type, mapper)
    return Family(config_from_hf, params_from_hf, loader(model_type, config_from_hf, params_from_hf), mapper)


llama_config_from_hf = _config("llama", _llama_map)

# model_type -> Family. llama is listed for inspect_model; hf_profiles keeps hf_llama.load_llama
# for it (pinned stories15M revision).
FAMILIES: dict[str, Family] = {
    "llama": Family(llama_config_from_hf, llama_params_from_hf,
                    loader("llama", llama_config_from_hf, llama_params_from_hf), _llama_map),
    "mistral": _family("mistral", _mistral_map, llama_params_from_hf),
    "qwen2": _family("qwen2", _qwen2_map, llama_params_from_hf),
    "gemma": _family("gemma", _gemma_map, gemma_params_from_hf),
    "gpt_neox": _family("gpt_neox", _gpt_neox_map, gpt_neox_params_from_hf),
    "phi": _family("phi", _phi_map, phi_params_from_hf),
    "opt": _family("opt", _opt_map, opt_params_from_hf),
}


def findings(hf_config: Any) -> Findings | None:
    """What a profiled family's mapper can't model in this config (None if the family has no profile here)."""
    family = FAMILIES.get(getattr(hf_config, "model_type", None))
    return None if family is None else _findings(hf_config.model_type, family.mapper, hf_config)
