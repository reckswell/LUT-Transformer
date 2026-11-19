"""Hugging Face Llama (LlamaForCausalLM), imported and never modified, plus the export of its weights
into our layout.

A control, not part of our model: it lives in reference/ with the other library-built models, so
baseline/ and LUT/ hold only from-scratch code. Like reference/hf_gpt2.py, a profile is a config
(python/configs/model/stories15m.yaml) plus a loader that maps the published checkpoint's names
and layouts onto ours (host.init_params), so the from-scratch code (and the LUT host's toggles)
run the real weights. The published model, imported from `transformers`, is the ground truth.

Profiles so far: stories15M, Karpathy's 15.2M-parameter Llama trained on TinyStories, in the
Hugging Face format published as Xenova/llama2.c-stories15M (pinned revision below; installed in
../LUT-Transformer-local/checkpoints, verified bit for bit against Karpathy's stories15M.pt).
Other Llama-architecture checkpoints load the same way if they use only what we model (see
llama_config_from_hf).

Rotary convention, which the weights depend on:
  - Meta's original code and llama2.c rotate adjacent column pairs (2i, 2i+1) of each head.
  - Hugging Face's Llama rotates column i with column i + Dh/2 ("rotate_half", split halves).
  - Hugging Face's conversion script (convert_llama_weights_to_hf.py) makes the two the same model
    by reordering the rows of Wq and Wk within each head: Meta row 2i becomes row i and row 2i+1
    becomes row i + Dh/2 (permute_rotary_rows below). Wv, Wo and everything else are unchanged.
  The checkpoint we load is in the Hugging Face layout, so baseline/inference.rope uses split
  halves and the weights load with no reordering. tests/test_llama_profile.py checks this three
  ways: our model matches LlamaForCausalLM; the interleaved rotation on llama2.c-ordered weights
  equals split halves on Hugging Face-ordered weights; the wrong pairing changes the output.

Downloads: nothing is downloaded here by default (local_files_only=True). The weights, config and
tokenizer are read from CHECKPOINT_DIR (../LUT-Transformer-local/checkpoints), as every
from_pretrained call passes cache_dir=CHECKPOINT_DIR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from baseline import host

from reference.local_cache import CHECKPOINT_DIR, use_local_hf_cache

STORIES15M = "Xenova/llama2.c-stories15M"
STORIES15M_REVISION = "17c2f1eabe1e163acc15ad35e225794e7b907682"   # installed 2026-11-19


# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------

def _rope_settings(hf_config: Any) -> tuple[str, float]:
    """(rope_type, theta) from a LlamaConfig: transformers 5 keeps them in config.rope_parameters;
    older configs have rope_theta and rope_scaling (None for plain RoPE)."""
    parameters = getattr(hf_config, "rope_parameters", None)
    if parameters:
        return parameters.get("rope_type", "default"), float(parameters["rope_theta"])
    scaling = getattr(hf_config, "rope_scaling", None)
    kind = "default" if not scaling else scaling.get("rope_type", scaling.get("type", "?"))
    return kind, float(getattr(hf_config, "rope_theta", 10000.0))


def llama_config_from_hf(hf_config: Any, name: str = "llama") -> host.TransformerConfig:
    """Our TransformerConfig for a Hugging Face LlamaConfig.

    Llama is the core plus RMSNorm with the checkpoint's eps (1e-5 for stories15M), SwiGLU (SiLU in
    a gated feed-forward), rotary positions (theta from the config), no biases, and tied or untied
    embeddings as the checkpoint says (stories15M ties them), with pre-norm placement.
    Raises if the checkpoint uses something our functions don't model, rather than running it
    wrongly: grouped-query attention (fewer key/value heads than query heads), scaled rotary
    variants (rope_type other than default), attention or MLP biases, a head width that isn't
    d_model / n_heads, or an activation other than SiLU.
    """
    if hf_config.hidden_act != "silu":
        raise ValueError(f"Llama activation {hf_config.hidden_act!r} is not silu")
    n_heads = hf_config.num_attention_heads
    n_kv = getattr(hf_config, "num_key_value_heads", None) or n_heads
    if n_kv != n_heads:
        raise ValueError(f"grouped-query attention ({n_kv} key/value heads for {n_heads} query heads) is not modeled")
    rope_type, theta = _rope_settings(hf_config)
    if rope_type != "default":
        raise ValueError(f"rotary variant {rope_type!r} is not modeled (only plain RoPE)")
    if getattr(hf_config, "attention_bias", False) or getattr(hf_config, "mlp_bias", False):
        raise ValueError("attention or MLP biases are not modeled for Llama")
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is not None and head_dim * n_heads != hf_config.hidden_size:
        raise ValueError(f"head_dim {head_dim} x {n_heads} heads != hidden_size {hf_config.hidden_size}")
    cfg = host.TransformerConfig(
        name=name,
        vocab_size=hf_config.vocab_size,
        context_length=hf_config.max_position_embeddings,   # the trained context; RoPE itself has no limit
        d_model=hf_config.hidden_size,
        n_heads=n_heads,
        n_layers=hf_config.num_hidden_layers,
        d_ff=hf_config.intermediate_size,
        norm="rmsnorm",
        norm_eps=float(hf_config.rms_norm_eps),
        norm_placement="pre",
        activation="silu",
        ffn="glu",
        positional_encoding="rope",
        rope_theta=theta,
        tie_embeddings=bool(hf_config.tie_word_embeddings),
        bias=False,
    )
    cfg.validate()
    return cfg


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------

def llama_params_from_hf(state_dict: dict[str, Tensor], cfg: host.TransformerConfig) -> dict[str, Tensor]:
    """Map a Hugging Face Llama state dict onto host.param_shapes(cfg) names and layouts.

        Hugging Face name (prefix "model." optional)       ours
        embed_tokens.weight              (V, D)            tok_emb
        layers.{i}.input_layernorm.weight (D,)             layers.{i}.norm1.gamma
        layers.{i}.self_attn.q_proj.weight (D, D)          layers.{i}.attn.wq   (rows in rotate_half order)
        layers.{i}.self_attn.k_proj.weight                 layers.{i}.attn.wk   (rows in rotate_half order)
        layers.{i}.self_attn.v_proj.weight                 layers.{i}.attn.wv
        layers.{i}.self_attn.o_proj.weight                 layers.{i}.attn.wo
        layers.{i}.post_attention_layernorm.weight         layers.{i}.norm2.gamma
        layers.{i}.mlp.gate_proj.weight  (F, D)            layers.{i}.ffn.w1    (goes through SiLU)
        layers.{i}.mlp.up_proj.weight    (F, D)            layers.{i}.ffn.w3    (the gate's other input)
        layers.{i}.mlp.down_proj.weight  (D, F)            layers.{i}.ffn.w2
        norm.weight                                        norm_f.gamma
        lm_head.weight                   (V, D)            lm_head, only if untied

    No layout changes: Hugging Face's nn.Linear weights are already (D_out, D_in), as
    inference.linear expects, and q_proj / k_proj are already in the rotate_half row order our rope
    uses (see the module docstring). With tied embeddings lm_head.weight is the embedding itself
    and is skipped (checked to be equal). Returns detached copies, and raises if any of our
    parameters is missing or has the wrong shape.
    """
    sd = {name.removeprefix("model."): t.detach() for name, t in state_dict.items()}
    params: dict[str, Tensor] = {"tok_emb": sd["embed_tokens.weight"]}
    for i in range(cfg.n_layers):
        h, p = f"layers.{i}", f"layers.{i}"
        params[f"{p}.norm1.gamma"] = sd[f"{h}.input_layernorm.weight"]
        for ours, theirs in (("wq", "q_proj"), ("wk", "k_proj"), ("wv", "v_proj"), ("wo", "o_proj")):
            params[f"{p}.attn.{ours}"] = sd[f"{h}.self_attn.{theirs}.weight"]
        params[f"{p}.norm2.gamma"] = sd[f"{h}.post_attention_layernorm.weight"]
        params[f"{p}.ffn.w1"] = sd[f"{h}.mlp.gate_proj.weight"]
        params[f"{p}.ffn.w3"] = sd[f"{h}.mlp.up_proj.weight"]
        params[f"{p}.ffn.w2"] = sd[f"{h}.mlp.down_proj.weight"]
    params["norm_f.gamma"] = sd["norm.weight"]
    head = sd.get("lm_head.weight")
    if cfg.tie_embeddings:
        if head is not None and not torch.equal(head, params["tok_emb"]):
            raise ValueError("config says tied embeddings but lm_head.weight differs from embed_tokens.weight")
    else:
        if head is None:
            raise KeyError("untied embeddings but the state dict has no lm_head.weight")
        params["lm_head"] = head

    expected = host.param_shapes(cfg)
    if params.keys() != expected.keys():
        raise KeyError(f"names differ from param_shapes: {sorted(params.keys() ^ expected.keys())}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(params[name].shape)} != expected {shape}")
    return {name: t.contiguous().clone() for name, t in params.items()}


def permute_rotary_rows(w: Tensor, n_heads: int) -> Tensor:
    """Reorder the rows of a Meta / llama2.c Wq or Wk into Hugging Face's rotate_half order, as
    convert_llama_weights_to_hf.py does. w: (n_heads * Dh, D_in) -> same shape.

    Within each head, Meta row 2i (first of rotation pair i) moves to row i and row 2i+1 moves to
    row i + Dh/2, so the adjacent pairs become split-half pairs with the same frequency w_i.
    Viewing the rows as (head, i, j) with Meta row index 2i + j, the new index is j * Dh/2 + i:
    a transpose of the (i, j) axes.
    """
    rows, d_in = w.shape
    dh = rows // n_heads
    return w.reshape(n_heads, dh // 2, 2, d_in).transpose(1, 2).reshape(rows, d_in)


def unpermute_rotary_rows(w: Tensor, n_heads: int) -> Tensor:
    """The inverse of permute_rotary_rows: Hugging Face row order back to Meta / llama2.c order."""
    rows, d_in = w.shape
    dh = rows // n_heads
    return w.reshape(n_heads, 2, dh // 2, d_in).transpose(1, 2).reshape(rows, d_in)


def params_from_llama2c(state_dict: dict[str, Tensor], cfg: host.TransformerConfig, *,
                        hf_row_order: bool = True) -> dict[str, Tensor]:
    """Map a llama2.c checkpoint's model state dict (Karpathy's stories15M.pt, key "model") onto our
    names. hf_row_order=True reorders Wq and Wk rows with permute_rotary_rows, so the result equals
    the Hugging Face export and runs with our split-halves rope; False keeps llama2.c's order (only
    right with an interleaved rotation). Used to check provenance and the rotary convention.

        llama2.c name                         ours
        tok_embeddings.weight                 tok_emb (output.weight is the same tensor: tied)
        layers.{i}.attention_norm.weight      layers.{i}.norm1.gamma
        layers.{i}.attention.wq/wk/wv/wo      layers.{i}.attn.wq/wk/wv/wo
        layers.{i}.ffn_norm.weight            layers.{i}.norm2.gamma
        layers.{i}.feed_forward.w1/w2/w3      layers.{i}.ffn.w1/w2/w3
        norm.weight                           norm_f.gamma
    """
    sd = {name.removeprefix("_orig_mod."): t.detach() for name, t in state_dict.items()}
    params: dict[str, Tensor] = {"tok_emb": sd["tok_embeddings.weight"]}
    for i in range(cfg.n_layers):
        p = f"layers.{i}"
        params[f"{p}.norm1.gamma"] = sd[f"{p}.attention_norm.weight"]
        for name in ("q", "k", "v", "o"):
            w = sd[f"{p}.attention.w{name}.weight"]
            if hf_row_order and name in ("q", "k"):
                w = permute_rotary_rows(w, cfg.n_heads)
            params[f"{p}.attn.w{name}"] = w
        params[f"{p}.norm2.gamma"] = sd[f"{p}.ffn_norm.weight"]
        for name in ("w1", "w2", "w3"):
            params[f"{p}.ffn.{name}"] = sd[f"{p}.feed_forward.{name}.weight"]
    params["norm_f.gamma"] = sd["norm.weight"]
    if not cfg.tie_embeddings:
        params["lm_head"] = sd["output.weight"]
    if params.keys() != host.param_shapes(cfg).keys():
        raise KeyError(f"names differ from param_shapes: {sorted(params.keys() ^ host.param_shapes(cfg).keys())}")
    return {name: t.contiguous().clone() for name, t in params.items()}


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

def load_llama(model_name: str = STORIES15M, *, device: str = "cpu", dtype: torch.dtype = host.DTYPE,
               cache_dir: str | Path = CHECKPOINT_DIR, local_files_only: bool = True,
               revision: str | None = None) -> tuple[host.TransformerConfig, dict[str, Tensor], Any]:
    """Load a pretrained Llama from the local checkpoint folder (stories15M by default).

    model_name is a Hugging Face repo id or a folder holding a saved model. local_files_only=True
    (the default) never touches the network: a model that isn't already in cache_dir raises. For
    stories15M the pinned revision is used unless another is given.

    Returns (cfg, params, hf_model): our config, our parameter dict on `device`, and the Hugging
    Face LlamaForCausalLM itself (eval mode, on the CPU) as the ground truth, both in `dtype`
    (float32 by default, the base metric). transformers 5 would otherwise load the checkpoint's
    stored dtype, so dtype is passed and then checked. Needs the `transformers` package.
    """
    use_local_hf_cache()                 # raises if huggingface_hub's cache points anywhere else
    from transformers import LlamaForCausalLM

    if revision is None and model_name == STORIES15M:
        revision = STORIES15M_REVISION
    hf_model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=str(cache_dir), dtype=dtype,
                                                local_files_only=local_files_only, revision=revision).eval()
    if next(hf_model.parameters()).dtype != dtype:
        hf_model = hf_model.to(dtype)
    name = "stories15m" if model_name in (STORIES15M,) or "stories15M" in str(model_name) else "llama"
    cfg = llama_config_from_hf(hf_model.config, name=name)
    params = llama_params_from_hf(hf_model.state_dict(), cfg)
    params = {key: t.to(device=device, dtype=dtype) for key, t in params.items()}
    return cfg, params, hf_model


def load_llama_tokenizer(model_name: str = STORIES15M, *, cache_dir: str | Path = CHECKPOINT_DIR,
                         local_files_only: bool = True, revision: str | None = None) -> Any:
    """The checkpoint's SentencePiece tokenizer (32,000 tokens for stories15M; ids are int64 indices
    into the token table), read from cache_dir. It prepends the BOS token (id 1), as llama2.c's
    run.c does. Needs the `transformers` package."""
    use_local_hf_cache()
    from transformers import AutoTokenizer

    if revision is None and model_name == STORIES15M:
        revision = STORIES15M_REVISION
    return AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=local_files_only,
                                         revision=revision)
