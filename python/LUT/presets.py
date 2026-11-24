"""Named slot settings for the LUT model: the low-precision configurations the benchmarking thread
built as separate copies (2026-11-24), written as LUT/host.py stage lines, and a generator for the
advisor's sensitivity sweep (2026-11-24) that turns its table into one preset per point. LUT/host.py
adds all of them to PRESETS, so `python -m LUT.host info --preset int8_w8a8`, eval/harness.py and
bench/sweep.py can name them.

They are settings, not algorithms: they say which QuantSpec goes in which slot. Running one needs
the techniques in LUT/techniques.py (pending decision D4), and the static activation scales need
calibrated ranges (CALIBRATION in LUT/host.py). Until then lut_ops() refuses them at start-up with
the reason.

    preset        what it is (the benchmarking thread's copy of the same name)
    fp16_inputs   every stage input, weight and KV cache entry rounded to FP16 (no scale). The
                  computation between the roundings stays in DTYPE. The benchmark's "fp16" copy,
                  which also computes in FP16, is DTYPE = torch.float16 in LUT/host.py instead.
    fp8_e4m3      FP8 E4M3 per-tensor weights and inputs into every linear layer; the token table
                  in E4M3; everything else float. Activation range: max |x| from calibration.
    int8_w8a8     decision D4: INT8 per-output-channel weights, INT8 per-tensor static inputs into
                  every linear layer (99.99th-percentile range), 32-bit accumulators; the token
                  table in INT8 per row.
    int4_w4a8     INT4 weights in groups of 32 with a 4-bit multiplier per group on a per-row
                  scale, INT8 inputs as int8_w8a8; the output layer and the token table in INT8.

The sweep (sweep_presets): one point at a time at reduced precision, every other stage BASE.
    sweep_w{8,6,4}_{stage}    the weights of qkv_proj, out_proj, ffn_up (W1 and W3), ffn_down, lm_head
    sweep_a{8,6,4}_{stage}    the run-time input of the same five linear stages
    sweep_{q,k,p,v}{8,6,4}    one attention operand: Q or K into QK^T, P or V into PV
    sweep_kv_{k,v}{8,4,3,2}   how the KV cache stores keys or values
The norms, softmax internals, SiLU and the residual stream stay as they are (float here; the
advisor keeps them at 16-bit fixed point for now). The QuantSpec templates below fix everything
but the width; they follow D4 where it decides, and the comments mark what it doesn't.
per_layer() turns a point into one preset per layer, and combine() merges points (pairs, greedy
steps).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from common.quant import QuantSpec
from LUT.stages import KVStore, Stage

LINEAR = ("qkv_proj", "out_proj", "ffn_up", "ffn_down", "lm_head")
BLOCK_LINEAR = ("qkv_proj", "out_proj", "ffn_up", "ffn_down")

# Formats (common/quant.py QuantSpec).
W8 = QuantSpec(bits=8, granularity="per_channel", axis=0)                     # D4: per output channel
A8 = QuantSpec(bits=8, percentile=99.99, acc_bits=32)                         # D4: per tensor, static; 99.99th percentile
W4_G32 = QuantSpec(bits=4, granularity="per_group", axis=1, group_size=32, scale_bits=4)
E4M3 = QuantSpec(bits=8, kind="float", exp_bits=4)                            # per tensor, static (max |x| for inputs)
FP16 = QuantSpec(bits=16, kind="float", exp_bits=5, scale_mode="none")

# Sweep templates: only bits changes from point to point.
SWEEP_WEIGHT = W8
SWEEP_INPUT = QuantSpec(bits=8, acc_bits=32)                                  # D4; max |x| (the percentile is not decided)
SWEEP_PROBS = replace(SWEEP_INPUT, signed=False)                              # P is in [0, 1]: unsigned (not a D4 decision)
SWEEP_KV = QuantSpec(bits=8)                                                  # per tensor, static: D4 says nothing about the KV cache
WEIGHT_BITS = INPUT_BITS = (8, 6, 4)
KV_BITS = (8, 4, 3, 2)
ATTENTION_OPERANDS = {"q": ("attn_scores", "quant"), "k": ("attn_scores", "quant_b"),
                      "p": ("attn_values", "quant"), "v": ("attn_values", "quant_b")}


def precision_presets() -> dict[str, dict[str, Any]]:
    """The four low-precision configurations as toggles (module docstring)."""
    two_inputs = ("attn_scores", "attn_values", "ffn_gate")
    with_weight = ("embedding", "norm", "final_norm", *LINEAR)
    all_inputs = ("norm", "qkv_proj", "rope", "attn_scores", "softmax", "attn_values", "out_proj", "ffn_up",
                  "ffn_act", "ffn_gate", "ffn_down", "final_norm", "lm_head")
    fp16 = {stage: Stage(quant=FP16 if stage in all_inputs else None,
                         quant_b=FP16 if stage in two_inputs else None,
                         weight=FP16 if stage in with_weight else None)
            for stage in ("embedding", *all_inputs)}
    fp16["kv_cache"] = KVStore(k=FP16, v=FP16)
    return {
        "fp16_inputs": fp16,
        "fp8_e4m3": {**{s: Stage(quant=E4M3, weight=E4M3) for s in LINEAR}, "embedding": Stage(weight=E4M3)},
        "int8_w8a8": {**{s: Stage(quant=A8, weight=W8) for s in LINEAR}, "embedding": Stage(weight=W8)},
        "int4_w4a8": {**{s: Stage(quant=A8, weight=W4_G32) for s in BLOCK_LINEAR},
                      "lm_head": Stage(quant=A8, weight=W8), "embedding": Stage(weight=W8)},
    }


def sweep_presets(weight: QuantSpec = SWEEP_WEIGHT, inputs: QuantSpec = SWEEP_INPUT, probs: QuantSpec = SWEEP_PROBS,
                  kv: QuantSpec = SWEEP_KV) -> dict[str, dict[str, Any]]:
    """The advisor's table as presets, one point each (module docstring). The templates set every
    QuantSpec field but bits."""
    out: dict[str, dict[str, Any]] = {}
    for stage in LINEAR:
        for bits in WEIGHT_BITS:
            out[f"sweep_w{bits}_{stage}"] = {stage: Stage(weight=replace(weight, bits=bits))}
    for stage in LINEAR:
        for bits in INPUT_BITS:
            out[f"sweep_a{bits}_{stage}"] = {stage: Stage(quant=replace(inputs, bits=bits))}
    for operand, (stage, slot) in ATTENTION_OPERANDS.items():
        template = probs if operand == "p" else inputs
        if slot == "quant_b":
            template = replace(template, acc_bits=None)        # the accumulator width goes on the first operand
        for bits in INPUT_BITS:
            out[f"sweep_{operand}{bits}"] = {stage: Stage(**{slot: replace(template, bits=bits)})}
    for which in ("k", "v"):
        for bits in KV_BITS:
            out[f"sweep_kv_{which}{bits}"] = {"kv_cache": KVStore(**{which: replace(kv, bits=bits)})}
    return out


def per_layer(toggles: dict[str, Any], n_layers: int) -> list[dict[str, Any]]:
    """One copy of `toggles` per decoder layer i, with every setting restricted to layer i
    (layers=(i,)). Stages outside the layers (embedding, final_norm, lm_head) ignore layers, so a
    point on them stays the same in every copy."""
    copies = []
    for i in range(n_layers):
        one: dict[str, Any] = {}
        for key, line in toggles.items():
            items = list(line) if isinstance(line, (list, tuple)) else [line]
            restricted = [replace(item, layers=(i,)) for item in items]
            one[key] = restricted if isinstance(line, (list, tuple)) else restricted[0]
        copies.append(one)
    return copies


def merge(a: Stage, b: Stage) -> Stage:
    """Two Stages for the same layers as one: each slot from b where b sets it, else from a."""
    fields = {f: getattr(b, f) if getattr(b, f) not in (None, {}) else getattr(a, f)
              for f in ("lut", "quant", "quant_b", "weight", "smooth", "alt")}
    return Stage(layers=b.layers, **fields)


def combine(*points: dict[str, Any]) -> dict[str, Any]:
    """Several points at once (a pair, or the next greedy step). Two points on the same stage and
    the same layers merge slot by slot (sweep_q8 + sweep_k4: Q at 8 bits and K at 4 in one Stage);
    on different layers they become a list, so each keeps its own layers (a later entry wins on a
    layer both name, stages.py). KV cache settings merge field by field."""
    out: dict[str, Any] = {}
    for point in points:
        for key, line in point.items():
            if key not in out:
                out[key] = line
            elif key == "kv_cache":                         # merge field by field; the later point wins
                a, b = out[key], line
                out[key] = KVStore(k=b.k if b.k is not None else a.k, v=b.v if b.v is not None else a.v,
                                   k_before_rope=a.k_before_rope or b.k_before_rope,
                                   layers=b.layers if b.layers is not None else a.layers)
            else:
                merged = list(out[key]) if isinstance(out[key], (list, tuple)) else [out[key]]
                for entry in (list(line) if isinstance(line, (list, tuple)) else [line]):
                    same = [j for j, m in enumerate(merged) if m.layers == entry.layers]
                    if same:
                        merged[same[-1]] = merge(merged[same[-1]], entry)
                    else:
                        merged.append(entry)
                out[key] = merged[0] if len(merged) == 1 else merged
    return out
