"""The precision techniques (LUT/techniques.py, decision D4) and the integer path of the LUT model.

1. Rounding and formats: half-to-even, saturation and wrap-around, unsigned, E4M3 saturating at
   448 instead of turning into NaN.
2. Weights: INT8 per output channel, INT4 in groups of 32 with a 4-bit multiplier per group,
   each against the formula written out independently here.
3. The integer kernels are exact: a linear layer, QK^T, PV and the GLU gate on integer codes
   equal an int64 reference (torch.equal), and the accumulator width wraps or saturates the
   integer sums exactly as int64 arithmetic does. A float stage with an accumulator width is
   refused.
4. The presets run end to end with calibrated ranges, take the integer path where they should,
   and stay close to the float model.
5. Against the benchmarking copies (python/precision, kept outside this repository; skipped where
   that folder isn't present): given the same activation ranges, int8_w8a8 and int4_w4a8 give the
   copies' logits bit for bit.
"""

import json
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from baseline import host as base  # noqa: E402
from common.quant import QuantSpec  # noqa: E402
from LUT import host as lut_host  # noqa: E402
from LUT import observer as calibration  # noqa: E402
from LUT import ops_registry, presets, techniques  # noqa: E402
from LUT.stages import KVStore, Stage  # noqa: E402
from LUT.techniques import Quantized, Site  # noqa: E402
from test_model_against_reference import _setup as setup_models  # noqa: E402
from test_model_against_reference import _tokens as tokens  # noqa: E402

SITE = Site("qkv_proj", 0, "x")


def gen(seed):
    return torch.Generator().manual_seed(seed)


# ----------------------------------------------------------------------------------------------
# 1. Rounding and formats
# ----------------------------------------------------------------------------------------------

def test_rounding_is_half_to_even_and_saturates_or_wraps():
    v = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, 126.6, 127.5, 300.0, -129.0])
    assert round_list(v, QuantSpec(bits=8)) == [0, 2, 2, 0, -2, 127, 127, 127, -128]
    assert round_list(v, QuantSpec(bits=8, saturate=False)) == [0, 2, 2, 0, -2, 127, -128, 44, 127]
    assert round_list(v, QuantSpec(bits=8, rounding="nearest_away"))[:5] == [1, 2, 3, -1, -2]
    assert round_list(torch.tensor([-3.0, 0.4, 255.7, 300.0]), QuantSpec(bits=8, signed=False)) == [0, 0, 255, 255]


def round_list(v, spec):
    return [int(x) for x in techniques.round_to_int(v, spec)]


def test_e4m3_saturates_at_448_and_rounds_to_its_grid():
    spec = QuantSpec(bits=8, kind="float", exp_bits=4)
    out = techniques.round_to_float(torch.tensor([500.0, -1e6, 1.0, 0.3, 17.0]), spec)
    assert out[:3].tolist() == [448.0, -448.0, 1.0]
    assert torch.equal(out[3:], torch.tensor([0.3, 17.0]).to(torch.float8_e4m3fn).float())
    assert torch.isfinite(out).all()


def test_static_dynamic_and_fixed_point_scales():
    x = torch.tensor([[1.0, -4.0], [0.5, 2.0]])
    ranges = calibration.Ranges(absmax=8.0, percentiles={99.99: 6.35})
    static = techniques.input_scale(x, QuantSpec(bits=8), SITE, ranges)
    assert static.dtype == torch.float32 and float(static) == pytest.approx(8.0 / 127)
    assert float(static) == float(torch.tensor(8.0 / 127, dtype=torch.float32))      # divided in double, as the copies do
    assert float(techniques.input_scale(x, QuantSpec(bits=8, percentile=99.99), SITE, ranges)) == pytest.approx(0.05)
    per_token = techniques.input_scale(x, QuantSpec(bits=8, granularity="per_token", scale_mode="dynamic"), SITE, None)
    assert per_token.flatten().tolist() == pytest.approx([4 / 127, 2 / 127])
    fixed = techniques.input_scale(x, QuantSpec(bits=16, scale_mode="none", frac_bits=8), SITE, None)
    assert float(fixed) == 2 ** -8
    q = techniques.quantize_input(torch.tensor([0.123456]), QuantSpec(bits=16, scale_mode="none", frac_bits=8), fixed, SITE)
    assert q.codes.tolist() == [32.0] and q.dequantize().tolist() == [0.125]


# ----------------------------------------------------------------------------------------------
# 2. Weights
# ----------------------------------------------------------------------------------------------

def test_int8_weights_per_output_channel():
    w = torch.randn(6, 64, generator=gen(0))
    q = techniques.prepare_weight(w, presets.W8, Site("qkv_proj", 0, "w"))
    s = w.abs().amax(dim=1) / 127.0
    assert torch.equal(q.scale.flatten(), s)
    assert torch.equal(q.codes, torch.round(w / s[:, None]).clamp(-128, 127))
    assert (q.codes.abs().amax(dim=1) == 127).all()                     # every row uses its full range
    assert ((q.dequantize() - w).abs() <= s[:, None] / 2 + 1e-7).all()


def test_int4_groups_with_a_four_bit_multiplier_per_group():
    w = torch.randn(5, 96, generator=gen(1))
    w[0, :32] *= 0.01                                                   # one quiet group: a small multiplier
    q = techniques.prepare_weight(w, presets.W4_G32, Site("ffn_up", 0, "w"))
    wg = w.reshape(5, 3, 32)
    s_g = wg.abs().amax(dim=-1) / 7.0
    s_row = s_g.amax(dim=-1)
    r = torch.clamp(torch.ceil(16 * s_g / s_row[:, None]), 1, 16)
    q4 = torch.round(wg / (r * (s_row / 16)[:, None])[..., None]).clamp(-8, 7)
    assert torch.equal(q.codes, (q4 * r[..., None]).reshape(5, 96))
    assert torch.equal(q.scale.flatten(), s_row / 16)
    assert int(r[0, 0]) == 1 and q.codes.abs().max() <= 128
    # r >= 16 s_g / s_row, so the group step r s_row / 16 is never finer than needed: nothing clips.
    assert (q4.abs() <= 7).all() and ((q.dequantize() - w).abs() <= (r * s_row[:, None] / 16).repeat_interleave(32, 1) / 2 + 1e-6).all()


def test_fp16_and_float_group_scales():
    w = torch.randn(4, 64, generator=gen(2))
    fp16 = techniques.prepare_weight(w, presets.FP16, Site("lm_head", None, "w"))
    assert torch.equal(fp16.dequantize(), w.half().float())
    g = techniques.prepare_weight(w, QuantSpec(bits=4, granularity="per_group", axis=1, group_size=32), Site("ffn_up", 0, "w"))
    assert g.scale.shape == w.shape and g.codes.abs().max() <= 8


def test_smoothing_factors_follow_smoothquant():
    act, wmax = torch.tensor([4.0, 1.0, 0.0]), torch.tensor([1.0, 0.25, 2.0])
    s = techniques.smoothing_factors(act, wmax, 0.5, SITE)
    assert s.tolist() == pytest.approx([2.0, 2.0, math.sqrt(1e-5) / math.sqrt(2.0)])


# ----------------------------------------------------------------------------------------------
# 3. Integer kernels and the accumulator
# ----------------------------------------------------------------------------------------------

def random_codes(shape, bits, seed):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return torch.randint(lo, hi + 1, shape, generator=gen(seed)).float()


def test_integer_linear_is_exact():
    x = Quantized(random_codes((3, 5, 64), 8, 3), torch.tensor(0.031, dtype=torch.float32), presets.A8)
    w = Quantized(random_codes((7, 64), 8, 4), torch.rand(7, 1, generator=gen(5)) * 0.01, presets.W8)
    bias = torch.randn(7, generator=gen(6))
    y = techniques.integer_stage("qkv_proj", [x, w, bias], {}, lambda acc: acc)
    acc = x.codes.long() @ w.codes.long().T                                     # int64 reference
    assert torch.equal(y, acc.float() * (x.scale * w.scale.flatten()) + bias)


def test_integer_attention_and_gate_are_exact():
    q = Quantized(random_codes((2, 3, 4, 8), 8, 7), torch.rand(2, 3, 4, 1, generator=gen(8)), QuantSpec(granularity="per_token"))
    k = Quantized(random_codes((2, 3, 6, 8), 8, 9), torch.tensor(0.5), QuantSpec())
    mask = torch.ones(4, 6, dtype=torch.bool).tril(2)
    s = techniques.integer_stage("attn_scores", [q, k, mask], {}, lambda a: a)
    ref = (q.codes.long() @ k.codes.long().transpose(-2, -1)).float() * (q.scale * k.scale) / math.sqrt(8)
    assert torch.equal(s, torch.where(mask, ref, torch.full_like(ref, float("-inf"))))
    p = Quantized(random_codes((2, 3, 4, 6), 8, 10).abs(), torch.tensor(1 / 255), QuantSpec(signed=False))
    v = Quantized(random_codes((2, 3, 6, 8), 8, 11), torch.rand(3, 1, 8, generator=gen(12)), QuantSpec(granularity="per_channel"))
    out = techniques.integer_stage("attn_values", [p, v], {}, lambda a: a)
    assert torch.equal(out, (p.codes.long() @ v.codes.long()).float() * (p.scale * v.scale))
    a, b = (Quantized(random_codes((2, 5), 8, s), torch.tensor(0.1), QuantSpec()) for s in (13, 14))
    assert torch.equal(techniques.integer_stage("ffn_gate", [a, b], {}, lambda t: t), (a.codes * b.codes) * (a.scale * b.scale))


@pytest.mark.parametrize("saturate", [True, False])
def test_accumulator_width_wraps_or_saturates_the_integer_sums(saturate):
    spec = QuantSpec(bits=8, acc_bits=12, saturate=saturate)
    x = Quantized(random_codes((4, 64), 8, 15), torch.tensor(1.0), spec)
    w = Quantized(random_codes((5, 64), 8, 16), torch.ones(5, 1), presets.W8)
    acc = x.codes.long() @ w.codes.long().T
    assert acc.abs().max() > 2047                                           # 12 bits overflow here
    hook = lambda a: techniques.accumulator(a, spec, SITE, (x, w))          # noqa: E731
    y = techniques.integer_stage("qkv_proj", [x, w], {}, hook)
    want = acc.clamp(-2048, 2047) if saturate else (acc + 2048) % 4096 - 2048
    assert torch.equal(y, want.float())


def test_an_accumulator_width_on_a_float_stage_is_refused():
    cfg, params, _ = setup_models("llama_like", dtype=torch.float32)
    toggles = {"qkv_proj": Stage(quant=QuantSpec(bits=8, scale_mode="dynamic", acc_bits=32))}   # weight stays float
    ops = lut_host.lut_ops(cfg, "inference", toggles)
    assert not ops.layers[0]["qkv_proj"].integer
    with pytest.raises(ValueError, match="acc_bits needs every input"):
        lut_host.forward(params, tokens(cfg, 1, 6), ops, cfg)


def test_kv_cache_per_token_codes_stay_within_half_a_step():
    x = torch.randn(2, 3, 5, 8, generator=gen(17))
    spec = QuantSpec(bits=4, granularity="per_token", scale_mode="dynamic")
    site = Site("kv_cache", 0, "k")
    stored = techniques.kv_encode(x, spec, site, None)
    assert stored.scale.shape == (2, 3, 5, 1) and stored.codes.abs().max() <= 8
    assert ((techniques.kv_decode(stored, spec, site) - x).abs() <= stored.scale / 2 + 1e-7).all()


# ----------------------------------------------------------------------------------------------
# 4. The presets end to end
# ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llama32():
    cfg, params, _ = setup_models("llama_like", dtype=torch.float32)
    ranges = lut_host.calibrate(params, cfg, [tokens(cfg, 2, 16, seed=30), tokens(cfg, 2, 16, seed=31)],
                                kv=KVStore(k=presets.SWEEP_KV, v=presets.SWEEP_KV)).ranges()
    return cfg, params, ranges


@pytest.mark.parametrize("preset, integer, close", [
    ("int8_w8a8", True, 0.05), ("int4_w4a8", True, 0.5), ("fp8_e4m3", False, 0.2), ("fp16_inputs", False, 2e-3),
    ("sweep_kv_k4", False, 0.5), ("sweep_q8", False, 0.05), ("sweep_w4_ffn_up", False, 0.5),
])
def test_presets_run_take_the_integer_path_and_stay_near_float(llama32, preset, integer, close):
    cfg, params, ranges = llama32
    ids = tokens(cfg, 2, 20, seed=32)
    want = base.forward(params, ids, base.base_ops(cfg, "inference"), cfg)
    ops = lut_host.lut_ops(cfg, "inference", lut_host.preset_toggles(preset), ranges=ranges)
    stage = ops.layers[0]["qkv_proj"]
    assert getattr(stage, "integer", False) is integer
    got = lut_host.forward(params, ids, ops, cfg)
    assert not torch.equal(got, want)
    assert (got - want).abs().max() / want.abs().max() < close
    cache = lut_host.new_cache(params, cfg, 2, 20)                          # decode runs on the same path
    lut_host.prefill(params, ids[:, :12], cache, ops, cfg)
    assert torch.isfinite(lut_host.decode_step(params, ids[:, 12:13], cache, ops, cfg)).all()


def test_the_benchmarking_calibration_file_reads_as_ranges(tmp_path):
    data = {"act_amax_max": {"layers.0.attn.wq": 15.0, "layers.0.attn.wk": 15.0, "layers.0.attn.wv": 15.0,
                             "layers.0.attn.wo": 1.2, "layers.0.ffn.w1": 7.0, "layers.0.ffn.w3": 7.0,
                             "layers.0.ffn.w2": 7.5, "tok_emb": 9.0},
            "act_amax_p99_99": {"layers.0.attn.wq": 7.3, "layers.0.attn.wk": 7.3, "layers.0.attn.wv": 7.3,
                                "layers.0.attn.wo": 0.9, "layers.0.ffn.w1": 4.0, "layers.0.ffn.w3": 4.0,
                                "layers.0.ffn.w2": 1.25, "tok_emb": 5.0}}
    path = tmp_path / "act_amax_test.json"
    path.write_text(json.dumps(data))
    ranges = calibration.load(path)
    assert sorted(ranges) == ["ffn_down[0].x", "ffn_up[0].x", "lm_head.x", "out_proj[0].x", "qkv_proj[0].x"]
    assert ranges["qkv_proj[0].x"].absmax == 15.0 and ranges["qkv_proj[0].x"].percentile(99.99) == 7.3
    assert ranges["lm_head.x"].percentile(99.99) == 5.0
    data["act_amax_max"]["layers.0.attn.wk"] = 14.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sharing that input"):
        calibration.load(path)


# ----------------------------------------------------------------------------------------------
# 5. Against the benchmarking copies (outside this repository)
# ----------------------------------------------------------------------------------------------

def weight_amax(ranges, cfg, which):
    """The copies' {weight name: range} from our ranges (which: absmax or the 99.99th percentile)."""
    site = {"wq": "qkv_proj", "wk": "qkv_proj", "wv": "qkv_proj", "wo": "out_proj", "w1": "ffn_up", "w3": "ffn_up",
            "w2": "ffn_down"}
    out = {}
    for i in range(cfg.n_layers):
        for w, stage in site.items():
            kind = "attn" if w in ("wq", "wk", "wv", "wo") else "ffn"
            out[f"layers.{i}.{kind}.{w}"] = which(ranges[f"{stage}[{i}].x"])
    out["tok_emb" if cfg.tie_embeddings else "lm_head"] = which(ranges["lm_head.x"])
    return out


@pytest.mark.parametrize("copy, preset", [("int8", "int8_w8a8"), ("int4", "int4_w4a8")])
def test_presets_give_the_benchmarking_copies_logits_bit_for_bit(llama32, copy, preset):
    copies = pytest.importorskip("precision.copies", reason="the benchmarking copies (python/precision, outside this repository)")
    cfg, params, observed = llama32
    data = {"act_amax_max": weight_amax(observed, cfg, lambda r: r.absmax),
            "act_amax_p99_99": weight_amax(observed, cfg, lambda r: r.percentile(99.99))}
    ranges = calibration.from_weight_amax(data)
    theirs = copies.build(copy, params, cfg, act_amax=data["act_amax_p99_99"])
    ids = tokens(cfg, 2, 20, seed=33)
    want = base.forward(theirs.params, ids, theirs.ops, cfg)
    ops = lut_host.lut_ops(cfg, "inference", lut_host.preset_toggles(preset), ranges=ranges)
    assert torch.equal(lut_host.forward(params, ids, ops, cfg), want)
    cache_ours, cache_theirs = lut_host.new_cache(params, cfg, 2, 20), base.new_cache(params, cfg, 2, 20)
    assert torch.equal(lut_host.prefill(params, ids[:, :12], cache_ours, ops, cfg),
                       base.prefill(theirs.params, ids[:, :12], cache_theirs, theirs.ops, cfg))
    for t in range(12, 20):
        assert torch.equal(lut_host.decode_step(params, ids[:, t:t + 1], cache_ours, ops, cfg),
                           base.decode_step(theirs.params, ids[:, t:t + 1], cache_theirs, theirs.ops, cfg))
