"""LUT transformer host: the LUT model's own wiring, with per-stage toggles for what each stage does.

Read the PARAMETERS block first, right below the imports. Like a SystemVerilog module's parameter
list, it holds every setting as an uppercase global that can be changed at the top of the file:
first the shared numbers (bit widths, LUT shape), then one line per stage of the transformer
saying whether it runs the baseline function (baseline/inference.py, baseline/train.py) or a LUT
architecture from LUT/inference_lut.py, with which table shape, in which decoder layers, and with
what input quantization. Each stage has its own line, so a design can mix LUT, quantized and
baseline stages freely; there is no master switch (Alpha, 2026-11-19). The globals are read each
time lut_ops() builds the model, so a script or test can override one the way SystemVerilog
overrides a parameter with #(.SOFTMAX(...)):

    from LUT import host as lut_host
    lut_host.SOFTMAX = Stage(lut=LUTArch(...))
    ops = lut_host.lut_ops(cfg, "inference")

This file has its own copy of the model's wiring (the order of operations: forward, prefill,
decode_step, generate, forward_train, backward, train_step), separate from baseline/host.py
(Alpha's decision D9, 2026-11-19). The LUT model may later restructure the design, for example to
pipeline or unroll stages, and that work happens here without touching the baseline. For now the
wiring is a line-for-line copy of baseline/host.py's, so with every stage on BASE the two models
give bit-identical results; tests/test_lut_wiring.py checks that (control arm 2). When the wiring
here is changed on purpose, that test switches to a float tolerance with a stated reason.

Every stage line is a set of slots, one per technique (LUT/stages.py Stage): a LUT architecture
(lut), quantization of each run-time input (quant, and quant_b for the second operand of QK^T, PV
and the gate), offline preparation of the weights (weight), smoothing (smooth) and replacement
nonlinear functions (alt: the softmax exp, the norms' rsqrt, the activation). KV_CACHE sets how the
KV cache stores keys and values, and CALIBRATION names the calibrated ranges that static scales
and smoothing use. Each slot defaults to the baseline, and a stage with every slot at its default
runs the baseline function itself, so the defaults are bit-identical to baseline/host.py and run
exactly as fast (tests/test_lut_slots.py, bench/slot_parity.py). The techniques themselves are
functions in LUT/techniques.py (quantization, D4, Alpha's) and LUT/nonlinear.py.

What this file still takes from baseline/ is listed at the imports, each with the reason it is
not wiring. The baseline building blocks themselves (baseline/inference.py and train.py) are what
a BASE stage means, so every stage starts from the same functions in both models.

    cd python
    python -m LUT.host info                       # print the toggles and check every function exists
    python -m LUT.host info --preset all_base     # a named set of toggles from PRESETS
    python -m LUT.host info --preset int8_w8a8    # a precision preset (LUT/presets.py)
    python -m LUT.host train | evaluate | generate  ...   (window 1 and 2)

In hardware each stage setting is a different circuit, so an HLS port turns these globals into
compile-time constants (template or #define parameters), not run-time switches.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import torch
from torch import Tensor

# Shared with the baseline, and why none of it is wiring (tests/test_lut_wiring.py checks that
# this file uses nothing else from baseline/):
#   base.TransformerConfig, load_configs, add_common_args
#       the model's shape and architecture options, read from the same YAML files, and the shared
#       command-line flags. Both models describe the same transformer.
#   base.param_shapes, init_params (init_params wrapped below with this file's DTYPE)
#       the parameter names and layouts. The LUT model reads the same weights as the baseline, so
#       the same trained or pretrained weights can run in every control arm; LUT tables are built
#       from these weights.
#   base.Mode, ModelOps, STAGES (also used by LUT/ops_registry.py, with LAYER_STAGES and
#   function_name)
#       the stage names the toggles refer to and the container that holds each stage's function.
#       If the LUT wiring is restructured into different stages, give it its own list here.
#   inference.KVCache, split_heads, merge_heads, causal_mask, sinusoidal_positions,
#   learned_positions, rope_tables, sample_next_token
#       helpers that re-index tensors, build constants (rope_tables: the rotary cos/sin table),
#       store the cache or pick the next token.
#       They do no arithmetic on activations, so they are not stages and never get a LUT.
#   train.cross_entropy_forward / _backward, clip_grad_norm, adamw_step
#       the loss and the optimizer: what training does with the model's output, not part of the
#       model's wiring.
from baseline import host as base
from baseline import inference, train
from common import precision
from common.quant import QuantSpec  # noqa: F401  (for QuantSpec(...) in the stage lines below)
from LUT import observer as calibration
from LUT import ops_registry, presets
from LUT.stages import BASE, KV_BASE, KVStore, LUTArch, Stage  # noqa: F401  (LUTArch, KVStore: for the lines below)

# ==============================================================================================
# PARAMETERS: edit these (like a SystemVerilog module's parameter list)
# ==============================================================================================
#
# Current setting (Alpha, 2026-11-19): every stage is BASE, so the model runs on plain float32
# (DTYPE) everywhere and nothing is quantized. That is the base metric; quantization comes later.
#
# DTYPE is the float dtype of the parameters, activations, KV cache and logits, and so of every
# stage input that isn't quantized. The other numbers only take effect in a stage line that uses
# them; with every stage on BASE they change nothing. Their names map onto existing fields of
# QuantSpec (common/quant.py) and LUTArch (LUT/stages.py).

DTYPE          = torch.float32      # float dtype everywhere a stage isn't quantized (TF32 off: common/precision.py)
DATA_BITS      = 8                  # QuantSpec.bits: integer width of a quantized stage input
DATA_SIGNED    = True               # QuantSpec.signed: two's complement
ACC_BITS       = 32                 # QuantSpec.acc_bits: accumulator width of a quantized multiply-accumulate
                                    # (scales, granularity, static or dynamic, percentile: the other
                                    #  QuantSpec fields, common/quant.py; the technique is decision D4)
LUT_ARCH       = "<architecture>"   # LUTArch.name: a key of inference_lut.ARCHITECTURES (none until D5)
LUT_INPUTS     = 2                  # LUTArch.inputs_per_table, k
LUT_INPUT_BITS = 4                  # LUTArch.bits_per_input, b: address bits = k * b = 8, so 256 entries
LUT_OUT_BITS   = 16                 # LUTArch.out_bits: width of one table entry
LUT_ACC_BITS   = 24                 # LUTArch.acc_bits: integer accumulator for summed table outputs

# One line per stage, in the order the model runs them. A stage the model doesn't have (ROPE in a
# model without rotary positions, FFN_GATE without a gated feed-forward) is listed as "not in this
# model" by `info` and does nothing. Each is BASE or Stage(...) with any of these slots (each
# optional, each defaulting to the baseline; LUT/stages.py has the details):
#   lut      a LUT architecture instead of the baseline function (decision D5)
#   quant    quantize the stage's first (or only) run-time input, with or without a LUT (D4)
#   quant_b  quantize the second run-time input: k in ATTN_SCORES, v in ATTN_VALUES, W3 h in FFN_GATE
#   weight   prepare the stage's weights once, offline (a QuantSpec: quantized weights)
#   smooth   SmoothQuant-style smoothing of a linear stage's input, alpha in [0, 1]
#   alt      replacement nonlinear functions: {"exp": ...} in SOFTMAX, {"rsqrt": ...} in NORM and
#            FINAL_NORM, {"act": ...} in FFN_ACT (names in LUT/nonlinear.py)
#   layers   which decoder layers (counted from 0) get the setting; the others run BASE
# A line can also be a list of Stages for different settings in different layers; a later entry
# wins for the layers it names. A mixed design, for example:
#
#   ATTN_VALUES = Stage(lut=LUTArch(LUT_ARCH, LUT_INPUTS, LUT_INPUT_BITS, LUT_OUT_BITS, LUT_ACC_BITS),
#                       layers=(1, 2, 3))                         # a LUT in layers 1 to 3, BASE in layer 0
#   QKV_PROJ    = Stage(quant=QuantSpec(bits=DATA_BITS, percentile=99.99, acc_bits=ACC_BITS),
#                       weight=QuantSpec(bits=DATA_BITS, granularity="per_channel", axis=0))   # W8A8, no LUT
#   ATTN_SCORES = Stage(quant=QuantSpec(bits=8), quant_b=QuantSpec(bits=4))    # Q at 8 bits, K at 4
#   FFN_DOWN    = [Stage(quant=QuantSpec(bits=8)), Stage(quant=QuantSpec(bits=4), layers=(2,))]
#   SOFTMAX     = BASE                                            # baseline function, float inputs
#
# A stage that names something not written yet (an unknown architecture, a technique in
# LUT/techniques.py still pending D4) or a static scale without CALIBRATION fails when lut_ops()
# builds the model, before a run starts.

EMBEDDING   = BASE   # embedding     token lookup             BASE | Stage(lut, quant)  (layers ignored)
NORM        = BASE   # norm          LayerNorm or RMSNorm     BASE | Stage(lut, quant, layers)  (both norms in a layer)
QKV_PROJ    = BASE   # qkv_proj      Q, K, V projections      BASE | Stage(lut, quant, layers)
ROPE        = BASE   # rope          rotary rotation of Q, K  BASE | Stage(lut, quant, layers)  (rotary-position models only, e.g. Llama; 4 multiplies per pair)
ATTN_SCORES = BASE   # attn_scores   QK^T / sqrt(Dh)          BASE | Stage(lut, quant, layers)  (both operands made at run time)
SOFTMAX     = BASE   # softmax       exp, sum, divide         BASE | Stage(lut, quant, layers)
ATTN_VALUES = BASE   # attn_values   PV                       BASE | Stage(lut, quant, layers)  (both operands made at run time)
OUT_PROJ    = BASE   # out_proj      attention output         BASE | Stage(lut, quant, layers)
FFN_UP      = BASE   # ffn_up        D -> F linear (W1, W3)   BASE | Stage(lut, quant, layers)
FFN_ACT     = BASE   # ffn_act       ReLU, GELU or SiLU       BASE | Stage(lut, quant, layers)
FFN_GATE    = BASE   # ffn_gate      act(W1 h) * (W3 h)       BASE | Stage(lut, quant, layers)  (gated feed-forward only, e.g. Llama's SwiGLU; both operands made at run time)
FFN_DOWN    = BASE   # ffn_down      F -> D linear            BASE | Stage(lut, quant, layers)
FINAL_NORM  = BASE   # final_norm    norm before the output   BASE | Stage(lut, quant)  (layers ignored)
LM_HEAD     = BASE   # lm_head       output layer             BASE | Stage(lut, quant)  (layers ignored; most of a small model's per-token compute)

# How the KV cache stores keys and values: KV_BASE (as they are, like the baseline) or
# KVStore(k=QuantSpec(...), v=QuantSpec(...), k_before_rope=False, layers=None). k and v are set
# separately; k_before_rope stores unrotated keys and rotates them on every read.
KV_CACHE    = KV_BASE

# Calibrated ranges for static scales and smoothing: None, or a file written by
# calibrate(...).save(path) on calibration text (never the evaluation text). Only read when a
# stage line asks for a static scale or smoothing.
CALIBRATION = None

# Training: False trains the float baseline and applies the LUT stages afterwards, for inference
# only (post-training, the plan's default). True runs the LUT stages from LUT/train_lut.py in the
# forward pass, with surrogate gradients (LUT-aware training). This is the one global setting here,
# by Alpha's choice (2026-11-19): it stays global until the LUT design (D5) exists, and becomes a
# per-stage choice once a LUT stage can be trained through with surrogate gradients. The plan cuts
# LUT-aware training first if time runs short.
TRAIN_WITH_LUTS = False

# ==============================================================================================
# End of PARAMETERS
# ==============================================================================================


def init_params(cfg: base.TransformerConfig, *, seed: int, device: str = "cpu") -> dict[str, Tensor]:
    """The LUT model's parameters: the baseline's names, layouts and initialization, created in
    this file's DTYPE. The same seed gives the same weights as base.init_params at that dtype."""
    return base.init_params(cfg, seed=seed, device=device, dtype=DTYPE)


def parameter_toggles() -> dict[str, Any]:
    """The stage lines above as {stage name: Stage (or a list of Stages)}, plus "kv_cache": KV_CACHE,
    read at the moment of the call, so an override like lut_host.SOFTMAX = Stage(...) takes effect
    the next time lut_ops() runs."""
    return {
        "norm": NORM,
        "qkv_proj": QKV_PROJ,
        "rope": ROPE,
        "attn_scores": ATTN_SCORES,
        "softmax": SOFTMAX,
        "attn_values": ATTN_VALUES,
        "out_proj": OUT_PROJ,
        "ffn_up": FFN_UP,
        "ffn_act": FFN_ACT,
        "ffn_gate": FFN_GATE,
        "ffn_down": FFN_DOWN,
        "embedding": EMBEDDING,
        "final_norm": FINAL_NORM,
        "lm_head": LM_HEAD,
        "kv_cache": KV_CACHE,
    }


# Named sets of toggles that don't come from the PARAMETERS block. "all_base" is every stage on
# BASE: the LUT wiring with the baseline building blocks, control arm 2 in tests/test_lut_wiring.py
# and eval/. The others come from LUT/presets.py: the four low-precision configurations (FP16
# inputs, FP8, INT8 W8A8, INT4 W4A8) and the advisor's one-point-at-a-time sweep (names starting
# "sweep_"). They never override the stage lines above; they are only used when asked for by name.
PRESETS: dict[str, dict[str, Any]] = {
    "all_base": {stage: BASE for stage in base.STAGES},
    **presets.precision_presets(),
    **presets.sweep_presets(),
}


def preset_toggles(name: str = "parameters") -> dict[str, Any]:
    """"parameters" (the default) reads the PARAMETERS block now; any other name is a PRESETS entry."""
    return parameter_toggles() if name == "parameters" else PRESETS[name]


def lut_ops(cfg: base.TransformerConfig, mode: base.Mode, toggles: dict[str, Any] | None = None,
            *, train_with_luts: bool | None = None, observer: calibration.Observer | None = None,
            ranges: dict[str, calibration.Ranges] | None = None) -> base.ModelOps:
    """The function each stage calls in the wiring below, with the toggled stages swapped in.

    toggles None reads the PARAMETERS block at this moment. In "train" mode with train_with_luts
    False (the TRAIN_WITH_LUTS default), every stage trains on its baseline function and the LUTs
    are applied afterwards, for inference only.
    observer: record every stage input for calibration (see calibrate()); outputs are unchanged.
    ranges: calibrated ranges for static scales and smoothing; None reads the CALIBRATION file,
    if one is set.
    """
    chosen = parameter_toggles() if toggles is None else toggles
    train_with_luts = TRAIN_WITH_LUTS if train_with_luts is None else train_with_luts
    if mode == "train" and not train_with_luts:
        chosen = PRESETS["all_base"]
    if ranges is None and CALIBRATION is not None:
        ranges = calibration.load(CALIBRATION)
    return ops_registry.resolve_ops(chosen, cfg, mode, observer=observer, ranges=ranges)


def calibrate(params: dict, cfg: base.TransformerConfig, batches: Any, *, kv: KVStore = KV_BASE,
              observer: calibration.Observer | None = None) -> calibration.Observer:
    """Run the model with every stage on BASE over `batches` (token ids, each (B, T)) and record the
    range of every stage input in every layer, every linear stage's weights and what each layer
    writes into the KV cache. kv: the KV layout to record (KVStore(k_before_rope=True) records
    unrotated keys). Returns the Observer; observer.save(path) writes the file CALIBRATION names."""
    obs = observer if observer is not None else calibration.Observer()
    toggles = {**PRESETS["all_base"], "kv_cache": KVStore(k_before_rope=kv.k_before_rope, layers=kv.layers)}
    ops = lut_ops(cfg, "inference", toggles, observer=obs, ranges={})
    with torch.no_grad():
        for ids in batches:
            forward(params, ids, ops, cfg)
    return obs


# ----------------------------------------------------------------------------------------------
# Wiring: the LUT model's order of operations
# ----------------------------------------------------------------------------------------------
#
# Copied from baseline/host.py on 2026-11-19 and kept line for line until a deliberate change.
# The functions here call each stage through `ops` (built by lut_ops from the PARAMETERS block).
# One decoder block i, pre-norm placement (post-norm moves each norm after its residual add),
# with L = ops.layers[i]:
#
#   h  = L["norm"](x)                                  # norm1
#   q, k, v = L["qkv_proj"](h, Wq), (h, Wk), (h, Wv)   # three projections, then split_heads
#   q, k = L["rope"](q, cos, sin), L["rope"](k, cos, sin)   # rotary positions only (Llama)
#   k, v = L["kv_store"](k, v)                         # only if KV_CACHE stores them differently
#   s  = L["attn_scores"](q, k, causal_mask)           # QK^T / sqrt(Dh), masked
#   p  = L["softmax"](s)
#   a  = L["attn_values"](p, v)                        # PV, then merge_heads
#   x  = x + L["out_proj"](a, Wo)                      # residual
#   h  = L["norm"](x)                                  # norm2
#   z  = L["ffn_act"](L["ffn_up"](h, W1))
#   z  = L["ffn_gate"](z, L["ffn_up"](h, W3))          # gated feed-forward only (Llama's SwiGLU)
#   x  = x + L["ffn_down"](z, W2)                      # residual
#
# With rotary positions, k is rotated before it goes into the KV cache, so the cache holds rotated
# keys and a decode step only rotates its own new q and k. KVStore(k_before_rope=True) stores the
# unrotated keys instead and rotates every cached key by its own position on each read. The
# "kv_store" entry exists only in layers whose KV_CACHE setting isn't KV_BASE (or while
# calibrating), so with the defaults this is the baseline's order of operations exactly.
#
# Whole model: tokens -> ops.embedding (+ positions) -> n_layers blocks -> ops.final_norm ->
# ops.lm_head -> logits. Every stage is called through `ops`, never directly, so the toggles
# reach every stage. A restructured design (fused or pipelined stages) changes these functions
# and, if it needs new stages, the stage list and LUT/ops_registry.py.

def _norm(fn: Callable, x: Tensor, params: dict, prefix: str, cfg: base.TransformerConfig) -> Tensor:
    """Call a norm stage: LayerNorm takes (x, gamma, beta, eps), RMSNorm takes (x, gamma, eps)."""
    if cfg.norm == "layernorm":
        return fn(x, params[f"{prefix}.gamma"], params[f"{prefix}.beta"], cfg.norm_epsilon)
    return fn(x, params[f"{prefix}.gamma"], cfg.norm_epsilon)


def _rope_tables(length: int, start: int, cfg: base.TransformerConfig, like: Tensor) -> tuple[Tensor, Tensor] | None:
    """The rotary (cos, sin) tables for positions start .. start+length-1, each (length, Dh/2), on
    like's device and dtype; None unless positional_encoding = rope. Built once per forward step
    and shared by every layer, like a position ROM."""
    if cfg.positional_encoding != "rope":
        return None
    return inference.rope_tables(length, cfg.rotary_columns, cfg.rope_theta, start, device=like.device, dtype=like.dtype)


def _rotate(fn: Callable, x: Tensor, rope: tuple[Tensor, Tensor], cfg: base.TransformerConfig) -> Tensor:
    """Call the rope stage on the first cfg.rotary_columns columns of each head and pass the
    others through unchanged (partial rotary: GPT-NeoX, Phi). With every column rotated (the
    default) this is the rope stage itself. The split and the join are re-indexings."""
    rd = cfg.rotary_columns
    if rd == x.shape[-1]:
        return fn(x, *rope)
    return torch.cat((fn(x[..., :rd], *rope), x[..., rd:]), dim=-1)


def _shared_kv(x: Tensor, cfg: base.TransformerConfig) -> Tensor:
    """Grouped-query attention: each key/value head serves n_heads / n_kv_heads query heads, so
    repeat it that many times, (B, Hkv, S, Dh) -> (B, H, S, Dh), as Hugging Face's repeat_kv does.
    A re-indexing. Without grouped-query attention, x itself."""
    if cfg.kv_heads == cfg.n_heads:
        return x
    return inference.repeat_kv(x, cfg.n_heads // cfg.kv_heads)


def _attention_mask(t_query: int, t_key: int, offset: int, cfg: base.TransformerConfig, device: Any) -> Tensor:
    """The causal mask, (t_query, t_key) boolean. With a sliding window, query i (at position
    offset + i) also loses every key at least attention_window positions back, which is the causal
    mask shifted by the window, inverted. A comparison of positions: no arithmetic on activations."""
    mask = inference.causal_mask(t_query, t_key, offset, device=device)
    if cfg.attention_window is not None:
        mask = mask & ~inference.causal_mask(t_query, t_key, offset - cfg.attention_window, device=device)
    return mask


def _embed(params: dict, token_ids: Tensor, start: int, ops: base.ModelOps, cfg: base.TransformerConfig) -> Tensor:
    """x = token embedding + position code for tokens at absolute positions start .. start+T-1.
    token_ids (B, T) -> (B, T, D). Rotary positions add nothing here (they rotate q and k inside
    attention)."""
    x = ops.embedding(token_ids, params["tok_emb"])
    if cfg.embed_scale is not None:
        x = x * cfg.embed_scale                                  # Gemma: sqrt(d_model)
    t = token_ids.shape[1]
    if cfg.positional_encoding == "learned":
        x = x + inference.learned_positions(t, params["pos_emb"], start)
    elif cfg.positional_encoding == "sinusoidal":
        x = x + inference.sinusoidal_positions(t, cfg.d_model, start, device=x.device, dtype=x.dtype)
    return x


def _attention(layer: dict, params: dict, prefix: str, h: Tensor, cfg: base.TransformerConfig,
               cache: inference.KVCache | None, layer_index: int, rope: tuple[Tensor, Tensor] | None) -> Tensor:
    """Multi-head causal self-attention on the normed input h (B, T, D). Returns (B, T, D).
    rope is the (cos, sin) tables for these T positions, or None without rotary positions. With a
    cache, the new keys and values are appended first and the queries attend to every cached
    position plus the new ones (offset = tokens already cached)."""
    t = h.shape[1]
    q = layer["qkv_proj"](h, params[f"{prefix}.wq"], params.get(f"{prefix}.bq"))   # (B, T, D)
    k = layer["qkv_proj"](h, params[f"{prefix}.wk"], params.get(f"{prefix}.bk"))
    v = layer["qkv_proj"](h, params[f"{prefix}.wv"], params.get(f"{prefix}.bv"))
    q = inference.split_heads(q, cfg.n_heads)                                       # (B, H, T, Dh)
    k, v = inference.split_heads(k, cfg.kv_heads), inference.split_heads(v, cfg.kv_heads)   # (B, Hkv, T, Dh)
    store = layer.get("kv_store")      # how keys and values are stored (KV_CACHE); None: as they are
    k_before_rope = store is not None and store.k_before_rope
    if rope is not None:                                                            # rotate q and k, not v
        q = _rotate(layer["rope"], q, rope, cfg)
        if not k_before_rope:
            k = _rotate(layer["rope"], k, rope, cfg)
    if store is not None:
        k, v = store(k, v)                                                          # what the cache gives back
    offset = 0
    if cache is not None:
        offset = cache.length
        k, v = cache.append(layer_index, k, v)                                      # (B, Hkv, offset+T, Dh)
    if k_before_rope and rope is not None:                                          # rotate every stored key
        k = _rotate(layer["rope"], k, _rope_tables(k.shape[2], 0, cfg, k), cfg)     # by its own position
    k, v = _shared_kv(k, cfg), _shared_kv(v, cfg)                                   # (B, H, offset+T, Dh)
    mask = _attention_mask(t, k.shape[2], offset, cfg, h.device)                    # (T, offset+T)
    s = layer["attn_scores"](q, k, mask)                                            # (B, H, T, offset+T)
    probs = layer["softmax"](s)
    a = layer["attn_values"](probs, v)                                              # (B, H, T, Dh)
    return layer["out_proj"](inference.merge_heads(a), params[f"{prefix}.wo"], params.get(f"{prefix}.bo"))


def _feed_forward(layer: dict, params: dict, prefix: str, h: Tensor, cfg: base.TransformerConfig) -> Tensor:
    """mlp: down(act(up(h, W1))). glu: down(gate(act(up(h, W1)), up(h, W3))), which with SiLU is
    Llama's SwiGLU. (B, T, D) -> (B, T, F) -> (B, T, D)."""
    z = layer["ffn_act"](layer["ffn_up"](h, params[f"{prefix}.w1"], params.get(f"{prefix}.b1")))
    if cfg.ffn == "glu":
        z = layer["ffn_gate"](z, layer["ffn_up"](h, params[f"{prefix}.w3"], params.get(f"{prefix}.b3")))
    return layer["ffn_down"](z, params[f"{prefix}.w2"], params.get(f"{prefix}.b2"))


def _block(i: int, x: Tensor, params: dict, ops: base.ModelOps, cfg: base.TransformerConfig,
           cache: inference.KVCache | None, rope: tuple[Tensor, Tensor] | None) -> Tensor:
    """Decoder layer i: attention and feed-forward, each added to the residual stream x (B, T, D)."""
    layer, p = ops.layers[i], f"layers.{i}"

    def norm(z: Tensor, which: str) -> Tensor:
        return _norm(layer["norm"], z, params, f"{p}.{which}", cfg)

    if cfg.parallel_block:                               # both branches read x (GPT-NeoX, Phi)
        h = norm(x, "norm1")
        attn = _attention(layer, params, f"{p}.attn", h, cfg, cache, i, rope)
        h = h if cfg.block == "parallel_shared_norm" else norm(x, "norm2")
        return x + (attn + _feed_forward(layer, params, f"{p}.ffn", h, cfg))
    if cfg.norm_placement == "pre":
        x = x + _attention(layer, params, f"{p}.attn", norm(x, "norm1"), cfg, cache, i, rope)
        return x + _feed_forward(layer, params, f"{p}.ffn", norm(x, "norm2"), cfg)
    x = norm(x + _attention(layer, params, f"{p}.attn", x, cfg, cache, i, rope), "norm1")
    return norm(x + _feed_forward(layer, params, f"{p}.ffn", x, cfg), "norm2")


def _residual_stream(params: dict, token_ids: Tensor, ops: base.ModelOps, cfg: base.TransformerConfig,
                     cache: inference.KVCache | None) -> Tensor:
    """Embedding, then every decoder layer. Returns x (B, T, D) before the final norm. With a
    cache, appends this step's keys and values and advances it by T."""
    start = cache.length if cache is not None else 0
    x = _embed(params, token_ids, start, ops, cfg)
    rope = _rope_tables(token_ids.shape[1], start, cfg, x)       # None unless rotary positions
    for i in range(cfg.n_layers):
        x = _block(i, x, params, ops, cfg, cache, rope)
    if cache is not None:
        cache.advance(token_ids.shape[1])
    return x


def _head(params: dict, x: Tensor, ops: base.ModelOps, cfg: base.TransformerConfig) -> Tensor:
    """Final norm, then the output layer: (..., D) -> logits (..., V). With tied embeddings the
    output weight is the token table itself, logits = x @ tok_emb^T."""
    h = _norm(ops.final_norm, x, params, "norm_f", cfg)
    weight = params["tok_emb"] if cfg.tie_embeddings else params["lm_head"]
    return ops.lm_head(h, weight, params.get("lm_head_bias"))


def forward(params: dict, token_ids: Tensor, ops: base.ModelOps, cfg: base.TransformerConfig) -> Tensor:
    """The whole sequence at once with no cache: token_ids (B, T) -> logits (B, T, V).
    token_ids are integer indices (int64) into the token table, not values; they are never
    converted to float or quantized. Everything computed from them has the parameters' dtype."""
    return _head(params, _residual_stream(params, token_ids, ops, cfg, None), ops, cfg)


def new_cache(params: dict, cfg: base.TransformerConfig, batch: int, max_len: int) -> inference.KVCache:
    """An empty KV cache for `batch` sequences of up to max_len tokens, on the parameters' device
    and dtype."""
    ref = params["tok_emb"]
    return inference.KVCache.allocate(cfg.n_layers, batch, cfg.kv_heads, max_len, cfg.head_dim,
                                      device=ref.device, dtype=ref.dtype)


def prefill(params: dict, token_ids: Tensor, cache: inference.KVCache, ops: base.ModelOps,
            cfg: base.TransformerConfig) -> Tensor:
    """Run the prompt (B, T) at once, filling the KV cache. Returns the last position's logits
    (B, V); only that position goes through the final norm and the output layer."""
    x = _residual_stream(params, token_ids, ops, cfg, cache)
    return _head(params, x[:, -1], ops, cfg)


def decode_step(params: dict, token_ids: Tensor, cache: inference.KVCache, ops: base.ModelOps,
                cfg: base.TransformerConfig) -> Tensor:
    """Run one new token per sequence (B, 1) against the cache, appending its key and value.
    Returns logits (B, V). Each call reads the whole cache."""
    if token_ids.shape[1] != 1:
        raise ValueError(f"decode_step takes one token per sequence, got shape {tuple(token_ids.shape)}")
    x = _residual_stream(params, token_ids, ops, cfg, cache)
    return _head(params, x[:, -1], ops, cfg)


def generate(params: dict, prompt_ids: Tensor, max_new_tokens: int, ops: base.ModelOps,
             cfg: base.TransformerConfig, *, temperature: float = 1.0, top_k: int | None = None,
             generator: torch.Generator | None = None) -> Tensor:
    """prefill once, then sample_next_token + decode_step max_new_tokens times.
    prompt_ids (B, T0) -> (B, T0 + max_new_tokens). temperature 0 is greedy."""
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
# Training wiring: *_forward functions, then *_backward in reverse order (copied from
# baseline/host.py on 2026-11-19)
# ----------------------------------------------------------------------------------------------
#
# In "train" mode every ops entry is a (forward, backward) pair. A residual add sends its
# gradient both straight through and through the branch (the two are summed); the tied
# embedding gets the sum of its input-lookup and output-layer gradients, and h gets the sum of
# its W1 and W3 paths in a gated feed-forward. split_heads and
# merge_heads are re-indexings, so each one's backward is the other.

def _norm_forward(pair: tuple, x: Tensor, params: dict, prefix: str, cfg: base.TransformerConfig) -> tuple[Tensor, Any]:
    if cfg.norm == "layernorm":
        return pair[0](x, params[f"{prefix}.gamma"], params[f"{prefix}.beta"], cfg.norm_epsilon)
    return pair[0](x, params[f"{prefix}.gamma"], cfg.norm_epsilon)


def _norm_backward(pair: tuple, dy: Tensor, cache: Any, prefix: str, grads: dict, cfg: base.TransformerConfig) -> Tensor:
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


def _rotate_forward(forward: Callable, x: Tensor, rope: tuple[Tensor, Tensor], cfg: base.TransformerConfig) -> tuple[Tensor, Any]:
    """_rotate in training: rope_forward on the rotated columns, the rest passed through."""
    rd = cfg.rotary_columns
    if rd == x.shape[-1]:
        return forward(x, *rope)
    y, cache = forward(x[..., :rd], *rope)
    return torch.cat((y, x[..., rd:]), dim=-1), cache


def _rotate_backward(backward: Callable, dy: Tensor, cache: Any, cfg: base.TransformerConfig) -> Tensor:
    """Reverse of _rotate_forward: rope_backward on the rotated columns, the rest passed through."""
    rd = cfg.rotary_columns
    if rd == dy.shape[-1]:
        return backward(dy, cache)
    return torch.cat((backward(dy[..., :rd], cache), dy[..., rd:]), dim=-1)


def _shared_kv_backward(d: Tensor, cfg: base.TransformerConfig) -> Tensor:
    """Reverse of _shared_kv: each key/value head gets the sum of the gradients of the query heads
    that shared it, (B, H, S, Dh) -> (B, Hkv, S, Dh)."""
    if cfg.kv_heads == cfg.n_heads:
        return d
    return train.repeat_kv_backward(d, cfg.n_heads // cfg.kv_heads)


def _attention_train(layer: dict, params: dict, p: str, h: Tensor, cfg: base.TransformerConfig,
                     rope: tuple[Tensor, Tensor] | None) -> tuple[Tensor, dict]:
    """Attention forward with caches (no KV cache: training sees whole sequences)."""
    t = h.shape[1]
    c: dict[str, Any] = {}
    qkv = {}
    for name in ("q", "k", "v"):
        out, c[name] = layer["qkv_proj"][0](h, params[f"{p}.w{name}"], params.get(f"{p}.b{name}"))
        qkv[name] = inference.split_heads(out, cfg.n_heads if name == "q" else cfg.kv_heads)
    if rope is not None:
        qkv["q"], c["rope_q"] = _rotate_forward(layer["rope"][0], qkv["q"], rope, cfg)
        qkv["k"], c["rope_k"] = _rotate_forward(layer["rope"][0], qkv["k"], rope, cfg)
    mask = _attention_mask(t, t, 0, cfg, h.device)
    s, c["scores"] = layer["attn_scores"][0](qkv["q"], _shared_kv(qkv["k"], cfg), mask)
    probs, c["softmax"] = layer["softmax"][0](s)
    a, c["values"] = layer["attn_values"][0](probs, _shared_kv(qkv["v"], cfg))
    out, c["out"] = layer["out_proj"][0](inference.merge_heads(a), params[f"{p}.wo"], params.get(f"{p}.bo"))
    return out, c


def _attention_backward(layer: dict, d_out: Tensor, c: dict, p: str, grads: dict, cfg: base.TransformerConfig) -> Tensor:
    """Reverse of _attention_train. Returns d_h, the gradient for the attention input."""
    da = _linear_backward(layer["out_proj"], d_out, c["out"], f"{p}.wo", f"{p}.bo", grads)
    d_probs, dv = layer["attn_values"][1](inference.split_heads(da, cfg.n_heads), c["values"])
    ds = layer["softmax"][1](d_probs, c["softmax"])
    dq, dk = layer["attn_scores"][1](ds, c["scores"])
    dk, dv = _shared_kv_backward(dk, cfg), _shared_kv_backward(dv, cfg)   # sum over the heads sharing each
    if "rope_q" in c:                                    # undo the rotation (rope_backward)
        dq = _rotate_backward(layer["rope"][1], dq, c["rope_q"], cfg)
        dk = _rotate_backward(layer["rope"][1], dk, c["rope_k"], cfg)
    d_h = 0
    for name, d_heads in (("q", dq), ("k", dk), ("v", dv)):
        d_h = d_h + _linear_backward(layer["qkv_proj"], inference.merge_heads(d_heads), c[name],
                                     f"{p}.w{name}", f"{p}.b{name}", grads)
    return d_h


def _ffn_train(layer: dict, params: dict, p: str, h: Tensor, cfg: base.TransformerConfig) -> tuple[Tensor, dict]:
    c: dict[str, Any] = {}
    u, c["up"] = layer["ffn_up"][0](h, params[f"{p}.w1"], params.get(f"{p}.b1"))
    z, c["act"] = layer["ffn_act"][0](u)
    if cfg.ffn == "glu":
        u3, c["up3"] = layer["ffn_up"][0](h, params[f"{p}.w3"], params.get(f"{p}.b3"))
        z, c["gate"] = layer["ffn_gate"][0](z, u3)
    out, c["down"] = layer["ffn_down"][0](z, params[f"{p}.w2"], params.get(f"{p}.b2"))
    return out, c


def _ffn_backward(layer: dict, d_out: Tensor, c: dict, p: str, grads: dict, cfg: base.TransformerConfig) -> Tensor:
    """Reverse of _ffn_train. Returns d_h; with a gate, h fed both W1 and W3, so both paths add."""
    dz = _linear_backward(layer["ffn_down"], d_out, c["down"], f"{p}.w2", f"{p}.b2", grads)
    if cfg.ffn == "glu":
        dz, du3 = layer["ffn_gate"][1](dz, c["gate"])    # dz now is d(act output), du3 is d(W3 h)
        d_h3 = _linear_backward(layer["ffn_up"], du3, c["up3"], f"{p}.w3", f"{p}.b3", grads)
    du = layer["ffn_act"][1](dz, c["act"])
    d_h = _linear_backward(layer["ffn_up"], du, c["up"], f"{p}.w1", f"{p}.b1", grads)
    if cfg.ffn == "glu":
        d_h = d_h + d_h3
    return d_h


def forward_train(params: dict, token_ids: Tensor, ops: base.ModelOps, cfg: base.TransformerConfig) -> tuple[Tensor, dict]:
    """Full forward pass with caches for backward. Returns (logits (B, T, V), caches)."""
    if cfg.dropout:
        raise NotImplementedError("dropout > 0 is not wired into forward_train yet (every config uses 0.0)")
    caches: dict[str, Any] = {"T": token_ids.shape[1], "layers": []}
    x, caches["embedding"] = ops.embedding[0](token_ids, params["tok_emb"])
    if cfg.embed_scale is not None:
        x = x * cfg.embed_scale
    t = token_ids.shape[1]
    if cfg.positional_encoding == "learned":
        x = x + params["pos_emb"][:t]
    elif cfg.positional_encoding == "sinusoidal":
        x = x + inference.sinusoidal_positions(t, cfg.d_model, device=x.device, dtype=x.dtype)
    rope = _rope_tables(t, 0, cfg, x)                    # None unless rotary positions
    for i in range(cfg.n_layers):
        layer, p, c = ops.layers[i], f"layers.{i}", {}
        if cfg.parallel_block:
            h, c["norm1"] = _norm_forward(layer["norm"], x, params, f"{p}.norm1", cfg)
            attn, c["attn"] = _attention_train(layer, params, f"{p}.attn", h, cfg, rope)
            if cfg.block == "parallel":
                h, c["norm2"] = _norm_forward(layer["norm"], x, params, f"{p}.norm2", cfg)
            ffn, c["ffn"] = _ffn_train(layer, params, f"{p}.ffn", h, cfg)
            x = x + (attn + ffn)
        elif cfg.norm_placement == "pre":
            h, c["norm1"] = _norm_forward(layer["norm"], x, params, f"{p}.norm1", cfg)
            attn, c["attn"] = _attention_train(layer, params, f"{p}.attn", h, cfg, rope)
            x = x + attn
            h, c["norm2"] = _norm_forward(layer["norm"], x, params, f"{p}.norm2", cfg)
            ffn, c["ffn"] = _ffn_train(layer, params, f"{p}.ffn", h, cfg)
            x = x + ffn
        else:
            attn, c["attn"] = _attention_train(layer, params, f"{p}.attn", x, cfg, rope)
            x, c["norm1"] = _norm_forward(layer["norm"], x + attn, params, f"{p}.norm1", cfg)
            ffn, c["ffn"] = _ffn_train(layer, params, f"{p}.ffn", x, cfg)
            x, c["norm2"] = _norm_forward(layer["norm"], x + ffn, params, f"{p}.norm2", cfg)
        caches["layers"].append(c)
    h, caches["final_norm"] = _norm_forward(ops.final_norm, x, params, "norm_f", cfg)
    weight = params["tok_emb"] if cfg.tie_embeddings else params["lm_head"]
    logits, caches["lm_head"] = ops.lm_head[0](h, weight, params.get("lm_head_bias"))
    return logits, caches


def backward(d_logits: Tensor, caches: dict, ops: base.ModelOps, cfg: base.TransformerConfig) -> dict[str, Tensor]:
    """Backpropagate d_logits through the model in reverse order. Returns {param name: gradient},
    with residual-branch gradients summed and tied-embedding gradients added together."""
    grads: dict[str, Tensor] = {}
    dh, d_head, d_head_bias = ops.lm_head[1](d_logits, caches["lm_head"])
    if d_head_bias is not None:
        grads["lm_head_bias"] = d_head_bias
    dx = _norm_backward(ops.final_norm, dh, caches["final_norm"], "norm_f", grads, cfg)
    for i in reversed(range(cfg.n_layers)):
        layer, p, c = ops.layers[i], f"layers.{i}", caches["layers"][i]
        if cfg.block == "parallel":
            # x_out = x_in + (attn(norm1 x_in) + ffn(norm2 x_in)): dx reaches x_in directly and through both.
            d_branch = _ffn_backward(layer, dx, c["ffn"], f"{p}.ffn", grads, cfg)
            d_ffn = _norm_backward(layer["norm"], d_branch, c["norm2"], f"{p}.norm2", grads, cfg)
            d_branch = _attention_backward(layer, dx, c["attn"], f"{p}.attn", grads, cfg)
            dx = dx + (_norm_backward(layer["norm"], d_branch, c["norm1"], f"{p}.norm1", grads, cfg) + d_ffn)
        elif cfg.block == "parallel_shared_norm":
            # x_out = x_in + (attn(h) + ffn(h)), h = norm1 x_in: h's gradient is the sum of both branches'.
            d_h = _attention_backward(layer, dx, c["attn"], f"{p}.attn", grads, cfg)
            d_h = d_h + _ffn_backward(layer, dx, c["ffn"], f"{p}.ffn", grads, cfg)
            dx = dx + _norm_backward(layer["norm"], d_h, c["norm1"], f"{p}.norm1", grads, cfg)
        elif cfg.norm_placement == "pre":
            # x_out = x_mid + ffn(norm2(x_mid)): dx reaches x_mid directly and through the branch.
            d_branch = _ffn_backward(layer, dx, c["ffn"], f"{p}.ffn", grads, cfg)
            dx = dx + _norm_backward(layer["norm"], d_branch, c["norm2"], f"{p}.norm2", grads, cfg)
            # x_mid = x_in + attn(norm1(x_in))
            d_branch = _attention_backward(layer, dx, c["attn"], f"{p}.attn", grads, cfg)
            dx = dx + _norm_backward(layer["norm"], d_branch, c["norm1"], f"{p}.norm1", grads, cfg)
        else:
            # x_out = norm2(x_mid + ffn(x_mid)), then x_mid = norm1(x_in + attn(x_in))
            d_sum = _norm_backward(layer["norm"], dx, c["norm2"], f"{p}.norm2", grads, cfg)
            dx = d_sum + _ffn_backward(layer, d_sum, c["ffn"], f"{p}.ffn", grads, cfg)
            d_sum = _norm_backward(layer["norm"], dx, c["norm1"], f"{p}.norm1", grads, cfg)
            dx = d_sum + _attention_backward(layer, d_sum, c["attn"], f"{p}.attn", grads, cfg)
    d_table = ops.embedding[1](dx if cfg.embed_scale is None else dx * cfg.embed_scale, caches["embedding"])
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


def train_step(params: dict, batch: Any, ops: base.ModelOps, opt_state: dict, cfg: base.TransformerConfig, *,
               lr: float, max_grad_norm: float = 1.0, weight_decay: float = 0.1) -> dict:
    """forward_train -> cross_entropy -> backward -> clip_grad_norm -> adamw_step.
    batch is (inputs, targets), both (B, T). Updates params in place.
    Returns {"loss", "grad_norm", "tokens", "lr"}."""
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

def _cmd_info(cfg: base.TransformerConfig, hw: dict, preset: str, fp: dict) -> None:
    chosen = preset_toggles(preset)
    print(f"model   {cfg.name}: n_layers {cfg.n_layers}, d_model {cfg.d_model}, norm {cfg.norm}, "
          f"activation {cfg.activation}, ffn {cfg.ffn}, positions {cfg.positional_encoding}")
    print(f"hw      {hw.get('name', '?')}: {hw['devices']}")
    print(f"dtype   {DTYPE}; float32 matmul precision {fp['float32_matmul_precision']}, "
          f"TF32 {'on' if fp['cuda_matmul_allow_tf32'] or fp['cudnn_allow_tf32'] else 'off'}")
    print(f"preset  {preset}    train_with_luts: {TRAIN_WITH_LUTS}    calibration: {CALIBRATION or 'none'}")
    print()
    print(f"{'stage':<13}{'runs':<26}{'k x b -> out':<14}{'table bits':>11}  {'layers':<8}quant")
    for row in ops_registry.describe(chosen, cfg):
        print(f"{row['stage']:<13}{row['runs']:<26}{row['shape']:<14}{row['table_bits']:>11}  {row['layers']:<8}{row['quant']}")
    for mode in ("inference", "train"):
        try:
            lut_ops(cfg, mode, chosen)
            print(f"{mode}: every selected function exists")
        except (ValueError, NotImplementedError) as error:
            print(f"{mode}: not available with these toggles: {error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LUT transformer host")
    base.add_common_args(parser)
    parser.add_argument("--preset", default="parameters", choices=["parameters", *sorted(PRESETS)],
                        help="which toggles to use (default: the PARAMETERS block)")
    args = parser.parse_args(argv)

    fp = precision.full_fp32()          # full float32 math, no TF32, before anything runs
    cfg, hw = base.load_configs(args.model, args.hw)
    if args.command == "info":
        _cmd_info(cfg, hw, args.preset, fp)
        return
    raise SystemExit(f"'{args.command}' is written in windows 1 and 2. Try 'info' for now.")


if __name__ == "__main__":
    main()
