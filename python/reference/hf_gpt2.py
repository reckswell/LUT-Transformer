"""Hugging Face GPT-2, imported and never modified, plus the export of its weights into our layout.

A control, not part of our model: it lives in reference/ with the other library-built models, so
baseline/ and LUT/ hold only from-scratch code. (Formerly common/checkpoints.py.)

A model profile is a config (python/configs/model/<name>.yaml) plus a loader that maps the
published checkpoint's names and layouts onto ours (host.init_params), so the from-scratch code
(and the LUT host's toggles) run the real weights. The published model itself, imported from
Hugging Face `transformers`, is the profile's ground truth; we don't reimplement it. Nothing here
patches or subclasses a Hugging Face class: the "hf" arm in eval/ runs the model exactly as
`transformers` ships it.

Profiles so far: GPT-2 (openai-community/gpt2 and its larger sizes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from baseline import host

from reference.local_cache import CHECKPOINT_DIR, use_local_hf_cache

# Downloaded weights, configs and tokenizer files go to CHECKPOINT_DIR, in
# ../LUT-Transformer-local/checkpoints by default ($LUT_CHECKPOINT_DIR overrides it); Hugging
# Face's own cache goes to ../LUT-Transformer-local/hf-home. Never the repository or ~/.cache
# (Alpha, 2026-11-19). Every from_pretrained call below passes cache_dir=CHECKPOINT_DIR.


# ----------------------------------------------------------------------------------------------
# GPT-2
# ----------------------------------------------------------------------------------------------

def gpt2_config_from_hf(hf_config: Any) -> host.TransformerConfig:
    """Our TransformerConfig for a Hugging Face GPT2Config.

    GPT-2 is the core plus LayerNorm (eps 1e-5), GELU (the tanh form, which Hugging Face calls
    "gelu_new"), learned positions, biases and tied embeddings, with pre-norm placement. Raises if
    the checkpoint uses a setting our functions don't model, rather than running it wrongly.
    """
    if hf_config.activation_function not in ("gelu_new", "gelu_pytorch_tanh"):
        raise ValueError(f"GPT-2 activation {hf_config.activation_function!r} is not the tanh GELU")
    if abs(hf_config.layer_norm_epsilon - 1e-5) > 1e-12:
        raise ValueError(f"layer_norm_epsilon {hf_config.layer_norm_epsilon} != 1e-5, which inference.layernorm assumes")
    if getattr(hf_config, "scale_attn_by_inverse_layer_idx", False) or getattr(hf_config, "reorder_and_upcast_attn", False):
        raise ValueError("attention variants (scale_attn_by_inverse_layer_idx, reorder_and_upcast_attn) are not modeled")
    cfg = host.TransformerConfig(
        name="gpt2",
        vocab_size=hf_config.vocab_size,
        context_length=hf_config.n_positions,
        d_model=hf_config.n_embd,
        n_heads=hf_config.n_head,
        n_layers=hf_config.n_layer,
        d_ff=hf_config.n_inner or 4 * hf_config.n_embd,
        norm="layernorm",
        norm_placement="pre",
        activation="gelu",
        positional_encoding="learned",
        tie_embeddings=True,
        bias=True,
    )
    cfg.validate()
    return cfg


def gpt2_params_from_hf(state_dict: dict[str, Tensor], cfg: host.TransformerConfig) -> dict[str, Tensor]:
    """Map a Hugging Face GPT-2 state dict onto host.param_shapes(cfg) names and layouts.

        Hugging Face name (prefix "transformer." optional)   ours
        wte.weight            (V, D)                          tok_emb
        wpe.weight            (P, D)                          pos_emb
        h.{i}.ln_1.weight / .bias                             layers.{i}.norm1.gamma / .beta
        h.{i}.attn.c_attn.weight (D, 3D) / .bias (3D,)        layers.{i}.attn.wq, wk, wv / bq, bk, bv
        h.{i}.attn.c_proj.weight (D, D) / .bias               layers.{i}.attn.wo / bo
        h.{i}.ln_2.weight / .bias                             layers.{i}.norm2.gamma / .beta
        h.{i}.mlp.c_fc.weight (D, F) / .bias (F,)             layers.{i}.ffn.w1 (F, D) / b1
        h.{i}.mlp.c_proj.weight (F, D) / .bias (D,)           layers.{i}.ffn.w2 (D, F) / b2
        ln_f.weight / .bias                                   norm_f.gamma / .beta

    Two layout changes:
      - GPT-2's linear layers are "Conv1D" modules computing y = x @ W + b with W stored as
        (D_in, D_out). inference.linear computes y = x @ W^T + b with W stored as (D_out, D_in),
        so every matrix is transposed.
      - c_attn fuses the three projections: its output columns 0..D-1 are Q, D..2D-1 are K and
        2D..3D-1 are V. They are split into separate matrices so each is its own qkv_proj call.
    lm_head.weight is the same tensor as wte (tied) and is skipped, as are the causal-mask buffers
    (attn.bias, attn.masked_bias) some checkpoints carry. Returns detached copies, and raises if
    any of our parameters is missing or has the wrong shape.
    """
    sd = {name.removeprefix("transformer."): t.detach() for name, t in state_dict.items()}
    d = cfg.d_model
    params: dict[str, Tensor] = {"tok_emb": sd["wte.weight"], "pos_emb": sd["wpe.weight"]}
    for i in range(cfg.n_layers):
        h, p = f"h.{i}", f"layers.{i}"
        params[f"{p}.norm1.gamma"], params[f"{p}.norm1.beta"] = sd[f"{h}.ln_1.weight"], sd[f"{h}.ln_1.bias"]
        w_qkv, b_qkv = sd[f"{h}.attn.c_attn.weight"], sd[f"{h}.attn.c_attn.bias"]       # (D, 3D), (3D,)
        for j, name in enumerate(("q", "k", "v")):
            params[f"{p}.attn.w{name}"] = w_qkv[:, j * d:(j + 1) * d].t()                 # (D, D) as (out, in)
            params[f"{p}.attn.b{name}"] = b_qkv[j * d:(j + 1) * d]
        params[f"{p}.attn.wo"], params[f"{p}.attn.bo"] = sd[f"{h}.attn.c_proj.weight"].t(), sd[f"{h}.attn.c_proj.bias"]
        params[f"{p}.norm2.gamma"], params[f"{p}.norm2.beta"] = sd[f"{h}.ln_2.weight"], sd[f"{h}.ln_2.bias"]
        params[f"{p}.ffn.w1"], params[f"{p}.ffn.b1"] = sd[f"{h}.mlp.c_fc.weight"].t(), sd[f"{h}.mlp.c_fc.bias"]
        params[f"{p}.ffn.w2"], params[f"{p}.ffn.b2"] = sd[f"{h}.mlp.c_proj.weight"].t(), sd[f"{h}.mlp.c_proj.bias"]
    params["norm_f.gamma"], params["norm_f.beta"] = sd["ln_f.weight"], sd["ln_f.bias"]

    expected = host.param_shapes(cfg)
    if params.keys() != expected.keys():
        raise KeyError(f"names differ from param_shapes: {sorted(params.keys() ^ expected.keys())}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(params[name].shape)} != expected {shape}")
    return {name: t.contiguous().clone() for name, t in params.items()}


def load_gpt2(model_name: str = "openai-community/gpt2", *, device: str = "cpu",
              dtype: torch.dtype = host.DTYPE,
              cache_dir: str | Path = CHECKPOINT_DIR,
              local_files_only: bool = False) -> tuple[host.TransformerConfig, dict[str, Tensor], Any]:
    """Load a pretrained GPT-2 from Hugging Face (downloaded once into CHECKPOINT_DIR).

    model_name is a Hugging Face repo id, or a folder that holds a saved model (config.json and the
    weights). local_files_only=True never touches the network: a model that isn't already in
    cache_dir raises instead of downloading (the benchmark sweep uses this unless --download).

    Returns (cfg, params, hf_model): our config, our parameter dict on `device`, and the
    Hugging Face GPT2LMHeadModel itself (in eval mode, on the CPU) as the ground truth. Both are
    in `dtype` (float32 by default, the base metric), so the two models compare like for like.
    transformers 5 loads a checkpoint in its own stored dtype unless told otherwise (dtype="auto"),
    so the dtype is passed explicitly and then checked. Needs the `transformers` package.
    """
    use_local_hf_cache()                 # raises if huggingface_hub's cache points anywhere else
    from transformers import GPT2LMHeadModel

    hf_model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=str(cache_dir), dtype=dtype,
                                               local_files_only=local_files_only).eval()
    if next(hf_model.parameters()).dtype != dtype:
        hf_model = hf_model.to(dtype)
    cfg = gpt2_config_from_hf(hf_model.config)
    params = gpt2_params_from_hf(hf_model.state_dict(), cfg)
    params = {name: t.to(device=device, dtype=dtype) for name, t in params.items()}
    return cfg, params, hf_model


def load_gpt2_tokenizer(model_name: str = "openai-community/gpt2", *, cache_dir: str | Path = CHECKPOINT_DIR) -> Any:
    """GPT-2's byte-pair tokenizer from Hugging Face, downloaded once into CHECKPOINT_DIR. Its ids
    are int64 indices into the 50,257-row token table. Needs the `transformers` package."""
    use_local_hf_cache()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
