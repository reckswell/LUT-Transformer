"""The LUT model's technique slots (LUT/stages.py Stage and KVStore, LUT/ops_registry.py).

    1. Every slot at its default: the baseline functions themselves, so bit-identical logits, KV
       cache and gradients and exactly the baseline's speed (no wrapper runs at all).
    2. Every slot switched on, with each technique (LUT/techniques.py) replaced by a stand-in that
       passes its tensor through unchanged: still bit-identical, for every architecture variant.
       So the plumbing itself never changes a number; only a technique does.
    3. Each slot reaches the tensor it names (stage, layer, operand, shape), with stand-ins that
       record or change values in a known way: separate Q/K and P/V specs, per-layer lists, a tied
       output layer prepared separately from the token lookup, smoothing's algebra, KV cache
       encode/decode and keys stored before the rotation.
    4. Calibration records every site without changing outputs, and saves and loads.
    5. Bad settings fail when the model is built: pending techniques, static scales without
       calibration, slots a stage doesn't have.
    6. The presets (FP16 / FP8 / INT8 / INT4 and the advisor's sweep) are valid settings.

The stand-ins are test scaffolding, not a quantization design (that is decision D4, Alpha's).
"""

import collections
import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from baseline import host as base  # noqa: E402
from baseline import inference  # noqa: E402
from common.quant import QuantSpec  # noqa: E402
from LUT import host as lut_host  # noqa: E402
from LUT import nonlinear, ops_registry, presets, techniques  # noqa: E402
from LUT import observer as calibration  # noqa: E402
from LUT.stages import BASE, KV_BASE, KVStore, Stage  # noqa: E402
from test_model_against_reference import VARIANTS  # noqa: E402
from test_model_against_reference import _setup as setup_models  # noqa: E402
from test_model_against_reference import _tokens as tokens  # noqa: E402

TECHNIQUES = ("input_scale", "quantize_input", "accumulator", "prepare_weight", "smoothing_factors",
              "kv_encode", "kv_decode")
STATIC8 = QuantSpec(bits=8, acc_bits=32)
DYN8 = QuantSpec(bits=8, scale_mode="dynamic")
W8 = QuantSpec(bits=8, granularity="per_channel", axis=0)


@pytest.fixture
def passthrough(monkeypatch):
    """Every technique replaced by a stand-in that returns its tensor unchanged; counts the calls."""
    calls = collections.Counter()

    def counted(name, fn):
        def run(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return run

    stand_ins = {
        "input_scale": lambda x, spec, site, ranges: torch.ones((), dtype=x.dtype, device=x.device),
        "quantize_input": lambda x, spec, scale, site: x,
        "accumulator": lambda y, spec, site, inputs: y,
        "prepare_weight": lambda w, spec, site: w,
        "smoothing_factors": lambda act, w, alpha, site: torch.ones_like(act),
        "kv_encode": lambda x, spec, site, ranges: {"x": x},
        "kv_decode": lambda stored, spec, site: stored["x"],
    }
    for name, fn in stand_ins.items():
        monkeypatch.setattr(techniques, name, counted(name, fn))
    return calls


def smoothable(stage, cfg):
    if stage in ("qkv_proj", "ffn_up"):
        return cfg.norm_placement == "pre"
    if stage == "ffn_down":
        return cfg.ffn == "glu"
    return stage in ("out_proj", "lm_head")


def every_slot_on(cfg):
    """Toggles with every slot of every stage set (alt only where the named function is the
    baseline math exactly: exp and the activation), and a coded KV cache."""
    toggles = {}
    for stage, operands in ops_registry.OPERANDS.items():
        kw = {}
        if operands:
            kw["quant"] = STATIC8
        if len(operands) == 2:
            kw["quant_b"] = DYN8
        if stage in ops_registry.WEIGHT_ARG:
            kw["weight"] = W8
        if stage in ops_registry.LINEAR_STAGES and smoothable(stage, cfg):
            kw["smooth"] = 0.5
        if stage == "softmax":
            kw["alt"] = {"exp": "exp"}
        if stage == "ffn_act":
            kw["alt"] = {"act": cfg.activation}
        toggles[stage] = Stage(**kw)
    toggles["kv_cache"] = KVStore(k=STATIC8, v=DYN8, k_before_rope=cfg.positional_encoding == "rope")
    return toggles


def calibrated(params, cfg, kv=KV_BASE):
    obs = lut_host.calibrate(params, cfg, [tokens(cfg, 2, 12, seed=40), tokens(cfg, 1, 16, seed=41)], kv=kv)
    return obs.ranges()


def run_all(params, cfg, ops, new_cache):
    """forward, prefill + decode (logits and the cache), greedy generation."""
    ids = tokens(cfg, batch=2, length=14)
    out = {"forward": base_or_lut(ops).forward(params, ids, ops, cfg)}
    cache = new_cache(params, cfg, 2, 14)
    out["prefill"] = base_or_lut(ops).prefill(params, ids[:, :9], cache, ops, cfg)
    for t in range(9, 14):
        out[f"decode{t}"] = base_or_lut(ops).decode_step(params, ids[:, t:t + 1], cache, ops, cfg)
    out["generate"] = base_or_lut(ops).generate(params, ids[:, :4], 5, ops, cfg, temperature=0)
    return out, cache


def base_or_lut(ops):
    return base if getattr(ops, "_baseline", False) else lut_host


def baseline_results(params, cfg):
    ops = base.base_ops(cfg, "inference")
    ops._baseline = True
    return run_all(params, cfg, ops, base.new_cache)


def assert_identical(got, want):
    for key in want:
        assert torch.equal(got[key], want[key]), key


# ----------------------------------------------------------------------------------------------
# 1. Defaults: the baseline functions themselves
# ----------------------------------------------------------------------------------------------

def test_default_slots_are_the_baseline_functions_with_no_wrapper():
    cfg = replace(base.TransformerConfig(vocab_size=37), n_layers=3, ffn="glu", activation="silu",
                  positional_encoding="rope")
    ops = lut_host.lut_ops(cfg, "inference")                          # the PARAMETERS block
    assert ops == base.base_ops(cfg, "inference")
    assert all("kv_store" not in layer for layer in ops.layers)
    # Explicit default Stage objects, per-layer lists of them and an explicit KV_BASE change nothing.
    toggles = {stage: [Stage(), Stage(layers=(1,))] for stage in base.STAGES}
    toggles["kv_cache"] = KVStore(layers=(0, 2))
    assert lut_host.lut_ops(cfg, "inference", toggles) == base.base_ops(cfg, "inference")
    for mode in ("inference", "train"):
        assert lut_host.lut_ops(cfg, mode, toggles, train_with_luts=True) == base.base_ops(cfg, mode)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_default_slots_give_bit_identical_logits_cache_and_gradients(variant):
    cfg, params, _ = setup_models(variant)
    want, b_cache = baseline_results(params, cfg)
    ops = lut_host.lut_ops(cfg, "inference", lut_host.parameter_toggles())
    got, l_cache = run_all(params, cfg, ops, lut_host.new_cache)
    assert_identical(got, want)
    for i in range(cfg.n_layers):
        assert torch.equal(l_cache.keys[i], b_cache.keys[i]) and torch.equal(l_cache.values[i], b_cache.values[i])
    ids, targets = tokens(cfg, 2, 12, seed=5), tokens(cfg, 2, 12, seed=6)
    for train_with_luts in (False, True):
        l_ops = lut_host.lut_ops(cfg, "train", lut_host.parameter_toggles(), train_with_luts=train_with_luts)
        b_ops = base.base_ops(cfg, "train")
        l_logits, l_caches = lut_host.forward_train(params, ids, l_ops, cfg)
        b_logits, b_caches = base.forward_train(params, ids, b_ops, cfg)
        assert torch.equal(l_logits, b_logits)
        _, loss_cache = base.train.cross_entropy_forward(b_logits, targets)
        d_logits = base.train.cross_entropy_backward(loss_cache)
        l_grads, b_grads = lut_host.backward(d_logits, l_caches, l_ops, cfg), base.backward(d_logits, b_caches, b_ops, cfg)
        assert all(torch.equal(l_grads[n], b_grads[n]) for n in b_grads)


# ----------------------------------------------------------------------------------------------
# 2. Every slot on, every technique a pass-through: still bit-identical
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_slot_on_with_pass_through_techniques_is_bit_identical(variant, passthrough):
    cfg, params, _ = setup_models(variant)
    toggles = every_slot_on(cfg)
    ranges = calibrated(params, cfg, toggles["kv_cache"])
    ops = lut_host.lut_ops(cfg, "inference", toggles, ranges=ranges)
    assert isinstance(ops.layers[0]["qkv_proj"], ops_registry.PrecisionStage) and "kv_store" in ops.layers[0]
    want, b_cache = baseline_results(params, cfg)
    got, l_cache = run_all(params, cfg, ops, lut_host.new_cache)
    assert_identical(got, want)
    if not toggles["kv_cache"].k_before_rope:
        assert all(torch.equal(l_cache.keys[i], b_cache.keys[i]) for i in range(cfg.n_layers))
    for name in TECHNIQUES:
        if name == "smoothing_factors" and not any(smoothable(s, cfg) for s in ops_registry.LINEAR_STAGES):
            continue
        assert passthrough[name] > 0, f"{name} was never called"
    assert passthrough["smoothing_factors"] == sum(smoothable(s, cfg) for s in ("qkv_proj", "out_proj", "ffn_up", "ffn_down")) \
        * cfg.n_layers + 1                                                    # once per site, then kept


def test_float32_every_slot_on_is_bit_identical(passthrough):
    cfg, params, _ = setup_models("llama_like", dtype=torch.float32)
    toggles = every_slot_on(cfg)
    ops = lut_host.lut_ops(cfg, "inference", toggles, ranges=calibrated(params, cfg, toggles["kv_cache"]))
    ids = tokens(cfg, 2, 16)
    assert torch.equal(lut_host.forward(params, ids, ops, cfg),
                       base.forward(params, ids, base.base_ops(cfg, "inference"), cfg))


# ----------------------------------------------------------------------------------------------
# 3. Each slot reaches the tensor it names
# ----------------------------------------------------------------------------------------------

@pytest.fixture
def recorder(monkeypatch, passthrough):
    """Pass-through techniques that also record (site key, spec, shape) for every quantized input."""
    seen = []

    def quantize_input(x, spec, scale, site):
        seen.append((site.key(), spec.bits, tuple(x.shape)))
        return x

    monkeypatch.setattr(techniques, "quantize_input", quantize_input)
    return seen


def test_attention_operands_get_separate_specs(recorder):
    cfg, params, _ = setup_models("llama_like")
    toggles = {"attn_scores": Stage(quant=replace(DYN8, bits=8), quant_b=replace(DYN8, bits=4)),
               "attn_values": Stage(quant=replace(DYN8, bits=6), quant_b=replace(DYN8, bits=3), layers=(1,))}
    ops = lut_host.lut_ops(cfg, "inference", toggles)
    cache = lut_host.new_cache(params, cfg, 1, 12)
    lut_host.prefill(params, tokens(cfg, 1, 7), cache, ops, cfg)
    lut_host.decode_step(params, tokens(cfg, 1, 1), cache, ops, cfg)
    h, dh = cfg.n_heads, cfg.head_dim
    assert ("attn_scores[0].q", 8, (1, h, 7, dh)) in recorder and ("attn_scores[0].k", 4, (1, h, 7, dh)) in recorder
    assert ("attn_scores[1].k", 4, (1, h, 8, dh)) in recorder                  # decode: k is the whole cache
    assert ("attn_values[1].p", 6, (1, h, 1, 8)) in recorder and ("attn_values[1].v", 3, (1, h, 8, dh)) in recorder
    assert not any(key.startswith("attn_values[0]") for key, _, _ in recorder)   # layer 0 stays BASE


def test_a_list_of_stages_sets_layers_differently(recorder):
    cfg, params, _ = setup_models("core_like")
    toggles = {"ffn_down": [Stage(quant=DYN8), Stage(quant=replace(DYN8, bits=4), layers=(1,))]}
    lut_host.forward(params, tokens(cfg, 1, 5), lut_host.lut_ops(cfg, "inference", toggles), cfg)
    assert {(key, bits) for key, bits, _ in recorder} == {("ffn_down[0].x", 8), ("ffn_down[1].x", 4)}
    rows = [r for r in ops_registry.describe(toggles, cfg) if r["stage"] == "ffn_down"]
    assert [(r["layers"], r["quant"]) for r in rows] == [("all", "x:int8/tensor/dynamic"), ("1", "x:int4/tensor/dynamic")]


def test_tied_output_layer_is_prepared_separately_from_the_token_lookup(monkeypatch, passthrough):
    cfg, params, _ = setup_models("core_like")                     # tied embeddings
    assert cfg.tie_embeddings
    monkeypatch.setattr(techniques, "prepare_weight",
                        lambda w, spec, site: w * 2 if site.stage == "lm_head" else w)
    original = params["tok_emb"].clone()
    toggles = {"lm_head": Stage(weight=W8), "embedding": Stage(weight=W8)}
    ids = tokens(cfg, 2, 9)
    logits = lut_host.forward(params, ids, lut_host.lut_ops(cfg, "inference", toggles), cfg)
    assert torch.equal(logits, 2 * base.forward(params, ids, base.base_ops(cfg, "inference"), cfg))
    assert torch.equal(params["tok_emb"], original)                  # the parameter itself is untouched


def test_prepared_weights_are_made_once_per_weight(monkeypatch, passthrough):
    cfg, params, _ = setup_models("llama_like")
    made = collections.Counter()

    def prepare_weight(w, spec, site):
        made[site.key()] += 1
        return w

    monkeypatch.setattr(techniques, "prepare_weight", prepare_weight)
    ops = lut_host.lut_ops(cfg, "inference", {"qkv_proj": Stage(weight=W8), "ffn_up": Stage(weight=W8),
                                              "norm": Stage(weight=W8)})
    for _ in range(3):
        lut_host.forward(params, tokens(cfg, 1, 6), ops, cfg)
    # wq, wk, wv; w1, w3; norm1 and norm2 gamma: one preparation each per layer, however many calls.
    assert made == {f"qkv_proj[{i}].w": 3 for i in range(cfg.n_layers)} | {f"ffn_up[{i}].w": 2 for i in range(cfg.n_layers)} \
        | {f"norm[{i}].w": 2 for i in range(cfg.n_layers)}


@pytest.mark.parametrize("exact", [True, False])
def test_smoothing_leaves_the_product_unchanged(monkeypatch, passthrough, exact):
    """x / s and W * s: with s a power of two the logits are bit-identical; with any positive s
    they agree to float64 rounding. This checks the slot's algebra, not a choice of s."""
    cfg, params, _ = setup_models("llama_like")
    gen = torch.Generator().manual_seed(7)

    def factors(act, w, alpha, site):
        if exact:
            return 2.0 ** torch.randint(-3, 4, act.shape, generator=gen).to(act.dtype)
        return 0.25 + 4 * torch.rand(act.shape, generator=gen, dtype=act.dtype)

    monkeypatch.setattr(techniques, "smoothing_factors", factors)
    toggles = {s: Stage(smooth=0.5) for s in ops_registry.LINEAR_STAGES}
    ops = lut_host.lut_ops(cfg, "inference", toggles, ranges=calibrated(params, cfg))
    ids = tokens(cfg, 2, 10)
    got, want = lut_host.forward(params, ids, ops, cfg), base.forward(params, ids, base.base_ops(cfg, "inference"), cfg)
    if exact:
        assert torch.equal(got, want)
    else:
        assert not torch.equal(got, want)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_kv_cache_stores_what_encode_and_decode_give_back(monkeypatch, passthrough):
    cfg, params, _ = setup_models("llama_like")
    monkeypatch.setattr(techniques, "kv_encode", lambda x, spec, site, ranges: torch.round(x * 16))
    monkeypatch.setattr(techniques, "kv_decode", lambda q, spec, site: q / 16)
    ops = lut_host.lut_ops(cfg, "inference", {"kv_cache": KVStore(v=DYN8, k=DYN8, layers=(1,))})
    ids = tokens(cfg, 2, 12)
    cache = lut_host.new_cache(params, cfg, 2, 12)
    logits = [lut_host.prefill(params, ids[:, :8], cache, ops, cfg)]
    logits += [lut_host.decode_step(params, ids[:, t:t + 1], cache, ops, cfg) for t in range(8, 12)]
    stored = cache.keys[1][:, :, :12] * 16
    assert torch.equal(stored, torch.round(stored)) and torch.equal(cache.values[1] * 16, torch.round(cache.values[1] * 16))
    assert not torch.equal(cache.keys[0] * 16, torch.round(cache.keys[0] * 16))        # layer 0 stores as is
    # The cached path and the whole-sequence forward store the same values, so they agree.
    full = lut_host.forward(params, ids, ops, cfg)
    torch.testing.assert_close(torch.stack(logits, 1), full[:, 7:], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(full, base.forward(params, ids, base.base_ops(cfg, "inference"), cfg))


def test_keys_stored_before_the_rotation_give_the_same_logits():
    cfg, params, _ = setup_models("llama_like")
    ops = lut_host.lut_ops(cfg, "inference", {"kv_cache": KVStore(k_before_rope=True)})
    assert all(layer["kv_store"].k_before_rope for layer in ops.layers)
    want, b_cache = baseline_results(params, cfg)
    got, l_cache = run_all(params, cfg, ops, lut_host.new_cache)
    assert_identical(got, want)
    n = l_cache.length
    cos, sin = inference.rope_tables(n, cfg.head_dim, cfg.rope_theta, device="cpu", dtype=params["tok_emb"].dtype)
    for i in range(cfg.n_layers):
        assert not torch.equal(l_cache.keys[i][:, :, :n], b_cache.keys[i][:, :, :n])    # unrotated in the cache
        assert torch.equal(inference.rope(l_cache.keys[i][:, :, :n], cos, sin), b_cache.keys[i][:, :, :n])
        assert torch.equal(l_cache.values[i], b_cache.values[i])


def test_nonlinear_slots():
    cfg, params, _ = setup_models("llama_like")
    ids = tokens(cfg, 2, 10)
    want = base.forward(params, ids, base.base_ops(cfg, "inference"), cfg)
    same = {"softmax": Stage(alt={"exp": "exp"}), "ffn_act": Stage(alt={"act": "silu"})}
    assert torch.equal(lut_host.forward(params, ids, lut_host.lut_ops(cfg, "inference", same), cfg), want)
    # The hardware form of the norm multiplies by an rsqrt where the baseline divides by a sqrt.
    rsqrt = {"norm": Stage(alt={"rsqrt": "rsqrt"}), "final_norm": Stage(alt={"rsqrt": "rsqrt"})}
    torch.testing.assert_close(lut_host.forward(params, ids, lut_host.lut_ops(cfg, "inference", rsqrt), cfg), want,
                               rtol=1e-13, atol=1e-13)
    # A replacement function (here a callable, as a quick experiment) really runs.
    coarse = {"softmax": Stage(alt={"exp": lambda x: torch.exp(torch.round(x * 4) / 4)})}
    assert not torch.allclose(lut_host.forward(params, ids, lut_host.lut_ops(cfg, "inference", coarse), cfg), want)
    rows = {r["stage"]: r for r in ops_registry.describe({**same, **rsqrt}, cfg)}
    assert rows["softmax"]["runs"] == "alt:exp=exp" and rows["norm"]["runs"] == "alt:rsqrt=rsqrt"


# ----------------------------------------------------------------------------------------------
# 4. Calibration
# ----------------------------------------------------------------------------------------------

def test_calibration_records_every_site_without_changing_outputs(tmp_path):
    cfg, params, _ = setup_models("llama_like")
    obs = calibration.Observer(max_samples=500)
    toggles = {**lut_host.PRESETS["all_base"]}
    ops = lut_host.lut_ops(cfg, "inference", toggles, observer=obs, ranges={})
    ids = tokens(cfg, 2, 12)
    assert torch.equal(lut_host.forward(params, ids, ops, cfg),
                       base.forward(params, ids, base.base_ops(cfg, "inference"), cfg))
    r = obs.ranges()
    d, h, dh, f = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.d_ff
    for i in range(cfg.n_layers):
        assert r[f"qkv_proj[{i}].x"].channel_absmax.shape == (d,)
        assert r[f"qkv_proj[{i}].w"].channel_absmax.shape == (d,) and r[f"qkv_proj[{i}].w"].calls == 3
        assert r[f"attn_scores[{i}].q"].channel_absmax.shape == (h, dh)
        assert r[f"softmax[{i}].s"].channel_absmax is None and math.isfinite(r[f"softmax[{i}].s"].absmax)
        assert r[f"ffn_down[{i}].x"].channel_absmax.shape == (f,)
        assert r[f"kv_cache[{i}].k"].channel_absmax.shape == (h, dh)
    x = r["lm_head.x"]
    assert x.percentile(50) <= x.percentile(99) <= x.percentile(100) == x.absmax and x.samples.numel() <= 500
    obs.save(tmp_path / "cal.json", text="test")
    loaded = calibration.load(tmp_path / "cal.json")
    assert loaded.keys() == r.keys()
    assert loaded["lm_head.x"].absmax == x.absmax and loaded["lm_head.x"].percentile(99.99) == x.percentile(99.99)
    assert torch.equal(loaded["qkv_proj[0].x"].channel_absmax, r["qkv_proj[0].x"].channel_absmax.float())


def test_calibration_file_feeds_static_scales(monkeypatch, passthrough, tmp_path):
    cfg, params, _ = setup_models("core_like")
    lut_host.calibrate(params, cfg, [tokens(cfg, 2, 10)]).save(tmp_path / "cal.json")
    got = {}
    monkeypatch.setattr(techniques, "input_scale",
                        lambda x, spec, site, ranges: got.setdefault(site.key(), ranges) and torch.ones(()))
    monkeypatch.setattr(lut_host, "CALIBRATION", str(tmp_path / "cal.json"))
    monkeypatch.setattr(lut_host, "QKV_PROJ", Stage(quant=STATIC8))
    lut_host.forward(params, tokens(cfg, 1, 4), lut_host.lut_ops(cfg, "inference"), cfg)
    assert set(got) == {f"qkv_proj[{i}].x" for i in range(cfg.n_layers)}
    assert all(isinstance(r, calibration.Ranges) and r.absmax > 0 for r in got.values())


# ----------------------------------------------------------------------------------------------
# 5. Bad settings fail when the model is built
# ----------------------------------------------------------------------------------------------

def test_pending_techniques_fail_at_startup(monkeypatch):
    # The techniques are written (D4); one put back to @pending must stop the model being built.
    cfg = base.TransformerConfig(vocab_size=37, n_layers=2)
    for name, toggles in (("quantize_input", {"qkv_proj": Stage(quant=DYN8)}),
                          ("prepare_weight", {"lm_head": Stage(weight=W8)}),
                          ("kv_encode", {"kv_cache": KVStore(k=DYN8)})):
        monkeypatch.setattr(techniques, name, techniques.pending("D4")(lambda *a: None))
        with pytest.raises(NotImplementedError, match="pending Alpha \\(decision D4\\)"):
            lut_host.lut_ops(cfg, "inference", toggles)
        monkeypatch.undo()


@pytest.mark.parametrize("toggles, message", [
    ({"embedding": Stage(quant=DYN8)}, "no run-time input"),
    ({"softmax": Stage(quant_b=DYN8)}, "one run-time input"),
    ({"attn_scores": Stage(quant=DYN8, quant_b=replace(DYN8, acc_bits=16))}, "acc_bits on quant"),
    ({"softmax": Stage(weight=W8)}, "has no weight"),
    ({"softmax": Stage(smooth=0.5)}, "linear stages"),
    ({"qkv_proj": Stage(smooth=1.5)}, "must be in"),
    ({"ffn_down": Stage(smooth=0.5)}, "gated feed-forward"),
    ({"qkv_proj": Stage(alt={"exp": "exp"})}, "no replaceable function"),
    ({"softmax": Stage(alt={"rsqrt": "rsqrt"})}, "alt takes"),
    ({"softmax": Stage(alt={"exp": "typo"})}, "no 'exp' function named 'typo'"),
    ({"kv_cache": KVStore(k_before_rope=True)}, "needs rotary positions"),
    ({"kv_cache": KVStore(layers=(9,))}, "out of range"),
    ({"qkv_proj": Stage(quant=QuantSpec(bits=8, kind="float"))}, "exp_bits"),
    ({"qkv_proj": "int8"}, "expected Stage"),
])
def test_bad_settings_fail_at_startup(toggles, message):
    cfg = base.TransformerConfig(vocab_size=37, n_layers=2)          # core: mlp, sinusoidal positions
    with pytest.raises(ValueError, match=message):
        lut_host.lut_ops(cfg, "inference", toggles)


def test_static_scale_and_smoothing_need_calibration(passthrough):
    cfg = base.TransformerConfig(vocab_size=37, n_layers=2)
    for toggles in ({"qkv_proj": Stage(quant=STATIC8)}, {"qkv_proj": Stage(smooth=0.5)},
                    {"kv_cache": KVStore(v=STATIC8)}):
        with pytest.raises(ValueError, match="needs calibrated ranges"):
            lut_host.lut_ops(cfg, "inference", toggles, ranges={})
    lut_host.lut_ops(cfg, "inference", {"qkv_proj": Stage(quant=DYN8)})        # dynamic needs none


def test_training_refuses_inference_only_slots():
    cfg = base.TransformerConfig(vocab_size=37, n_layers=2)
    toggles = {"qkv_proj": Stage(quant=DYN8)}
    assert lut_host.lut_ops(cfg, "train", toggles) == base.base_ops(cfg, "train")   # TRAIN_WITH_LUTS False
    with pytest.raises(NotImplementedError, match="quantization-aware"):
        lut_host.lut_ops(cfg, "train", toggles, train_with_luts=True)
    with pytest.raises(ValueError, match="inference mode"):
        lut_host.lut_ops(cfg, "train", {}, train_with_luts=True, observer=calibration.Observer())


# ----------------------------------------------------------------------------------------------
# 6. Presets
# ----------------------------------------------------------------------------------------------

def test_precision_and_sweep_presets_are_valid_settings(passthrough):
    cfg, params, _ = setup_models("llama_like")
    sweep = presets.sweep_presets()
    assert len(sweep) == 5 * 3 + 5 * 3 + 4 * 3 + 2 * 4
    assert {"fp16_inputs", "fp8_e4m3", "int8_w8a8", "int4_w4a8"} | set(sweep) <= set(lut_host.PRESETS)
    ranges = calibrated(params, cfg)
    ids = tokens(cfg, 1, 8)
    want = base.forward(params, ids, base.base_ops(cfg, "inference"), cfg)
    for name in ("fp16_inputs", "fp8_e4m3", "int8_w8a8", "int4_w4a8", *sweep):
        toggles = lut_host.PRESETS[name]
        ops = lut_host.lut_ops(cfg, "inference", toggles, ranges=ranges)
        assert torch.equal(lut_host.forward(params, ids, ops, cfg), want), name      # pass-through techniques
    rows = {r["stage"]: r["quant"] for r in ops_registry.describe(lut_host.PRESETS["int4_w4a8"], cfg)}
    assert rows["qkv_proj"] == "x:int8/tensor/p99.99/acc32,w:int4/g32/s4"
    assert rows["lm_head"] == "x:int8/tensor/p99.99/acc32,w:int8/channel"
    assert rows["embedding"] == "table:int8/channel" and rows["softmax"] == "float"
    kv_row = [r for r in ops_registry.describe(lut_host.PRESETS["fp16_inputs"], cfg) if r["stage"] == "kv_cache"]
    assert kv_row[0]["quant"] == "k:fp16,v:fp16"


def test_per_layer_and_combined_sweep_points():
    sweep = presets.sweep_presets()
    layers = presets.per_layer(sweep["sweep_k4"], 3)
    assert [t["attn_scores"].layers for t in layers] == [(0,), (1,), (2,)]
    assert layers[0]["attn_scores"].quant_b.bits == 4 and layers[0]["attn_scores"].quant is None
    both = presets.combine(sweep["sweep_q8"], sweep["sweep_k4"], sweep["sweep_kv_k4"], sweep["sweep_kv_v2"])
    assert both["attn_scores"] == Stage(quant=sweep["sweep_q8"]["attn_scores"].quant,
                                        quant_b=sweep["sweep_k4"]["attn_scores"].quant_b)   # same layers: one Stage
    split = presets.combine(sweep["sweep_q8"], presets.per_layer(sweep["sweep_k4"], 3)[1])
    assert [s.layers for s in split["attn_scores"]] == [None, (1,)]                      # different layers: a list
    assert both["kv_cache"].k.bits == 4 and both["kv_cache"].v.bits == 2
    cfg = replace(base.TransformerConfig(vocab_size=37), n_layers=3)
    ops_registry.check_toggles(both, cfg)


def test_quantspec_keeps_its_logged_fields_and_labels():
    spec = QuantSpec()
    d = spec.to_dict()
    assert list(d)[:8] == ["bits", "signed", "symmetric", "granularity", "axis", "group_size", "rounding", "saturate"]
    assert (d["bits"], d["granularity"], d["rounding"], d["saturate"]) == (8, "per_tensor", "nearest_even", True)
    assert QuantSpec(bits=8, kind="float", exp_bits=4).label() == "e4m3/tensor"
    assert QuantSpec(bits=16, kind="float", exp_bits=8, scale_mode="none").label() == "bf16"
    assert QuantSpec(bits=16, scale_mode="none", frac_bits=8).label() == "int16/q8"
    assert QuantSpec(bits=4, granularity="per_token", scale_mode="dynamic", saturate=False).label() == "int4/token/dynamic/wrap"
    with pytest.raises(ValueError):
        QuantSpec(bits=8, frac_bits=4).validate()
    with pytest.raises(ValueError):
        QuantSpec(granularity="per_group").validate()
