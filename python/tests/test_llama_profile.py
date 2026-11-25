"""The Llama profile (stories15M): our from-scratch model, configured as Llama and given Llama's
weights, against Hugging Face's LlamaForCausalLM (the imported ground truth), plus the rotary
convention the weights depend on.

These tests need `transformers` and skip without it. The real-weights tests skip unless stories15M
is installed in ../LUT-Transformer-local/checkpoints (nothing is downloaded here).

Tolerances, and why the Hugging Face comparison in float64 is done two ways:
  - Hugging Face's LlamaRMSNorm computes in float32 whatever the model's dtype: its forward casts
    the input to float32, normalizes, and casts back (transformers 5.23, modeling_llama.py,
    LlamaRMSNorm.forward). Its rotary table is also built in float32. So Hugging Face's "float64"
    Llama is float32 at every norm, and no float64 model can match it to 1e-9 (measured: 3.5e-7 on
    the small model below). Two checks instead, on a small random Llama in float64:
      1. Our model with a test-only norm that rounds the way Hugging Face's does (float32 inside),
         swapped into the norm stages through ModelOps (no Hugging Face code is changed), must match
         to rtol/atol 1e-9 (measured: 0.0, bit-identical). Everything else, the weight mapping, the
         rotary convention and table, SwiGLU, eps and the tied output layer, is therefore exact.
      2. Our own float64 RMSNorm must match to atol 1e-6: the size of float32 rounding at the norms.
    The float64 1e-9 check of the whole Llama math (forward, KV-cache decode, gradients against
    autograd, train steps, LUT bit-identity) is against reference/torch_model.py, which has no
    float32 islands: the llama_like and glu_rope_post_norm variants in test_model_against_reference.py,
    test_training_against_reference.py and test_lut_wiring.py.
  - Real stories15M weights in float32 (the base metric): logits within atol 2e-4 (measured 2.2e-5
    over 205 positions with logits up to 26), top-1 agreement 1.0, and identical greedy tokens.
  - The LUT host with every stage on BASE against the baseline: torch.equal, no tolerance.

Float64 here checks the math against a reference; it is not the base metric. Default runs are
float32 with TF32 off (DTYPE, common/precision.py, tests/test_fp32_defaults.py).
"""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from baseline import host  # noqa: E402
from baseline import inference  # noqa: E402
from eval.harness import Weights, compare, run_arm  # noqa: E402
from LUT import host as lut_host  # noqa: E402
from reference import hf_llama, hf_profiles  # noqa: E402
from reference.local_cache import CHECKPOINT_DIR  # noqa: E402
from test_model_against_reference import _setup as setup_models  # noqa: E402
from test_model_against_reference import _tokens as tokens  # noqa: E402

TIGHT = dict(rtol=1e-9, atol=1e-9)
HF_NORM_ROUNDING = dict(rtol=0, atol=1e-6)      # float32 inside Hugging Face's RMSNorm (see above)
REAL_FP32 = dict(rtol=0, atol=2e-4)


def hf_style_rmsnorm(x, gamma, eps):
    """Test-only stand-in for the norm stages: Hugging Face LlamaRMSNorm's rounding, which computes
    in float32 whatever x's dtype. Our model's own rmsnorm (baseline/inference.py) is not changed."""
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return gamma * h.to(x.dtype)


def with_hf_style_norms(ops):
    for layer in ops.layers:
        layer["norm"] = hf_style_rmsnorm
    ops.final_norm = hf_style_rmsnorm
    return ops


def _small_hf_llama(seed: int = 0, **overrides):
    """A randomly initialized two-layer Llama (no download), every weight perturbed away from
    Hugging Face's initial values (norm weights 1) so the mapping of each one is tested."""
    torch.manual_seed(seed)
    settings = dict(vocab_size=37, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=4, max_position_embeddings=24, rms_norm_eps=1e-5, tie_word_embeddings=True)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**{**settings, **overrides})).double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model


def _export(model):
    cfg = hf_llama.llama_config_from_hf(model.config)
    return cfg, hf_llama.llama_params_from_hf(model.state_dict(), cfg)


# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------

STORIES15M_CONFIG = dict(vocab_size=32000, hidden_size=288, intermediate_size=768, num_hidden_layers=6,
                         num_attention_heads=6, num_key_value_heads=6, max_position_embeddings=256,
                         rms_norm_eps=1e-5, tie_word_embeddings=True, hidden_act="silu")


def test_stories15m_yaml_profile_matches_the_checkpoint_config():
    from_yaml = host.TransformerConfig.from_yaml(host.PYTHON_DIR / "configs/model/stories15m.yaml")
    mapped = hf_llama.llama_config_from_hf(transformers.LlamaConfig(**STORIES15M_CONFIG), name="stories15m")
    assert asdict(from_yaml) == asdict(mapped)
    assert host.count_params(from_yaml)["total"] == 15_191_712           # tied embedding counted once
    assert from_yaml.head_dim == 48 and from_yaml.norm_epsilon == 1e-5
    installed = sorted(CHECKPOINT_DIR.glob("models--Xenova--llama2.c-stories15M/snapshots/*/config.json"))
    if installed:                                                         # the real file, when it's there
        real = transformers.AutoConfig.from_pretrained(installed[0].parent, cache_dir=str(CHECKPOINT_DIR),
                                                       local_files_only=True)
        assert asdict(hf_llama.llama_config_from_hf(real, name="stories15m")) == asdict(from_yaml)


def test_both_hosts_print_info_for_the_profile(capsys):
    model = str(host.PYTHON_DIR / "configs/model/stories15m.yaml")
    hw = str(host.PYTHON_DIR / "configs/hw/cpu.yaml")
    host.main(["info", "--model", model, "--hw", hw])
    lut_host.main(["info", "--model", model, "--hw", hw])
    out = capsys.readouterr().out
    assert "rope         rope" in out and "ffn_gate     glu_gate" in out and "ffn_act      silu" in out
    assert "15,191,712 total" in out and "not in this model" not in out
    host.main(["info", "--model", str(host.PYTHON_DIR / "configs/model/gpt2.yaml"), "--hw", hw])
    assert "(not used by this model)" in capsys.readouterr().out          # GPT-2 has no rope or gate


def test_loader_maps_the_newer_llama_options_and_rejects_settings_we_do_not_model():
    def config(**changes):
        base = dict(hidden_act="silu", num_attention_heads=4, num_key_value_heads=4, hidden_size=32,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, attention_bias=False,
                    mlp_bias=False, head_dim=8, vocab_size=37, max_position_embeddings=24, num_hidden_layers=2,
                    intermediate_size=48, rms_norm_eps=1e-5, tie_word_embeddings=True)
        return SimpleNamespace(**{**base, **changes})

    plain = hf_llama.llama_config_from_hf(config())
    assert plain.positional_encoding == "rope" and plain.n_kv_heads is None and plain.head_size is None
    # Grouped-query attention, wider heads and biases map onto their TransformerConfig options
    # (reference/hf_families.py; checked against Hugging Face in tests/test_hf_families.py).
    assert hf_llama.llama_config_from_hf(config(num_key_value_heads=2)).kv_heads == 2
    assert hf_llama.llama_config_from_hf(config(head_dim=16)).head_dim == 16
    biased = hf_llama.llama_config_from_hf(config(attention_bias=True, mlp_bias=True))
    assert biased.has_bias("qkv") and biased.has_bias("out") and biased.has_bias("ffn")
    for changes, message in ((dict(hidden_act="quick_gelu"), "activation 'quick_gelu'"),
                             (dict(rope_parameters={"rope_type": "linear", "rope_theta": 1e4, "factor": 2.0}), "rotary variant")):
        with pytest.raises(ValueError, match=message):
            hf_llama.llama_config_from_hf(config(**changes))
    # Every problem is listed, not only the first.
    with pytest.raises(ValueError, match="2 setting") as err:
        hf_llama.llama_config_from_hf(config(hidden_act="quick_gelu", rope_parameters={"rope_type": "llama3", "rope_theta": 5e5}))
    assert len(err.value.problems) == 2
    # Older configs keep rope_theta and rope_scaling at the top level.
    older = config(rope_parameters=None, rope_theta=500.0, rope_scaling=None)
    assert hf_llama.llama_config_from_hf(older).rope_theta == 500.0


# ----------------------------------------------------------------------------------------------
# Small random Llama in float64 against Hugging Face
# ----------------------------------------------------------------------------------------------

def test_llama_weights_run_in_our_code_and_match_hugging_face():
    model = _small_hf_llama()
    cfg, params = _export(model)
    ids = torch.randint(0, 37, (3, 24), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        theirs = model(ids).logits
    torch.testing.assert_close(host.forward(params, ids, with_hf_style_norms(host.base_ops(cfg, "inference")), cfg),
                               theirs, **TIGHT)
    torch.testing.assert_close(host.forward(params, ids, host.base_ops(cfg, "inference"), cfg), theirs,
                               **HF_NORM_ROUNDING)


def test_untied_llama_maps_its_output_layer():
    model = _small_hf_llama(seed=4, tie_word_embeddings=False)
    cfg, params = _export(model)
    assert not cfg.tie_embeddings and not torch.equal(params["lm_head"], params["tok_emb"])
    ids = torch.randint(0, 37, (2, 16), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        theirs = model(ids).logits
    torch.testing.assert_close(host.forward(params, ids, with_hf_style_norms(host.base_ops(cfg, "inference")), cfg),
                               theirs, **TIGHT)


def test_llama_through_the_lut_host_and_kv_cache():
    model = _small_hf_llama(seed=3)
    cfg, params = _export(model)
    ids = torch.randint(0, 37, (2, 16), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        theirs = model(ids).logits
    ops = with_hf_style_norms(lut_host.lut_ops(cfg, "inference", lut_host.PRESETS["all_base"]))
    cache = lut_host.new_cache(params, cfg, batch=2, max_len=16)
    # Prefill 12 tokens, then decode 4: the rotary angles continue from the cached length.
    torch.testing.assert_close(lut_host.prefill(params, ids[:, :12], cache, ops, cfg), theirs[:, 11], **TIGHT)
    for t in range(12, 16):
        torch.testing.assert_close(lut_host.decode_step(params, ids[:, t:t + 1], cache, ops, cfg), theirs[:, t], **TIGHT)


def test_eval_harness_runs_a_llama_in_every_arm():
    model = _small_hf_llama(seed=5)
    weights = Weights.from_hf(model, dtype=torch.float64)
    assert weights.cfg.positional_encoding == "rope" and weights.dtype == torch.float64
    ids = torch.randint(0, 37, (2, 20), generator=torch.Generator().manual_seed(6))
    fast = dict(repeats=0)
    hf, ours, lut = (run_arm("hf", weights, ids, **fast), run_arm("baseline", weights, ids, **fast),
                     run_arm("lut", weights, ids, preset="all_base", **fast))
    assert compare(ours, hf)["max_abs_logit_diff"] < 1e-6 and compare(ours, hf)["top1_agreement"] == 1.0
    assert torch.equal(lut.logits, ours.logits)
    assert {row["stage"]: row["runs"] for row in lut.modeled_cost}["rope"] == "base:rope"
    assert hf_profiles.export(model)[0] == weights.cfg


# ----------------------------------------------------------------------------------------------
# The rotary convention: split halves (Hugging Face) vs adjacent pairs (llama2.c / Meta)
# ----------------------------------------------------------------------------------------------

def interleaved_rope(x, cos, sin):
    """llama2.c's rotation (apply_rotary_emb in its model.py), written here for the test: column
    pairs (2i, 2i+1), pair i at frequency w_i. Swapped into the rope stage through ModelOps."""
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out


def _to_llama2c_rows(params, cfg):
    """The same weights with Wq and Wk rows in llama2.c's order (what Hugging Face's conversion undid)."""
    out = dict(params)
    for i in range(cfg.n_layers):
        for name in ("wq", "wk"):
            out[f"layers.{i}.attn.{name}"] = hf_llama.unpermute_rotary_rows(params[f"layers.{i}.attn.{name}"], cfg.n_heads)
    return out


def test_rotary_row_permutation_round_trips():
    w = torch.arange(6 * 8 * 5, dtype=torch.float64).reshape(48, 5)        # 6 heads of width 8
    assert torch.equal(hf_llama.unpermute_rotary_rows(hf_llama.permute_rotary_rows(w, 6), 6), w)
    moved = hf_llama.permute_rotary_rows(w, 6)
    # In head 1 (rows 8..15): Meta rows 8, 10, 12, 14 (first of each pair) become rows 8..11.
    assert torch.equal(moved[8:12], w[[8, 10, 12, 14]]) and torch.equal(moved[12:16], w[[9, 11, 13, 15]])


def test_both_conventions_are_the_same_model_and_the_wrong_pairing_is_not():
    cfg, params, _ = setup_models("llama_like")                            # float64, rows in our (HF) order
    meta_rows = _to_llama2c_rows(params, cfg)
    ids = tokens(cfg, batch=2, length=16)
    ours = host.base_ops(cfg, "inference")
    interleaved = host.base_ops(cfg, "inference")
    for layer in interleaved.layers:
        layer["rope"] = interleaved_rope
    reference = host.forward(params, ids, ours, cfg)
    # llama2.c's pairing on llama2.c's row order is the same model as split halves on HF's order.
    torch.testing.assert_close(host.forward(meta_rows, ids, interleaved, cfg), reference, **TIGHT)

    # The wrong pairing is caught at the attention scores, where rotary positions act.
    h = torch.randn(1, 10, cfg.d_model, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    cos, sin = inference.rope_tables(10, cfg.head_dim, cfg.rope_theta, dtype=torch.float64)

    def scores(p, rope_fn):
        q = rope_fn(inference.split_heads(inference.linear(h, p["layers.0.attn.wq"]), cfg.n_heads), cos, sin)
        k = rope_fn(inference.split_heads(inference.linear(h, p["layers.0.attn.wk"]), cfg.n_heads), cos, sin)
        return q @ k.transpose(-2, -1)

    right = scores(params, inference.rope)
    torch.testing.assert_close(scores(meta_rows, interleaved_rope), right, **TIGHT)
    assert (scores(meta_rows, inference.rope) - right).abs().max() > 1e-2 * right.abs().max()


# ----------------------------------------------------------------------------------------------
# Real stories15M weights (installed locally; nothing is downloaded)
# ----------------------------------------------------------------------------------------------

def _stories15m_installed() -> bool:
    return any(CHECKPOINT_DIR.glob("models--Xenova--llama2.c-stories15M/snapshots/*/model.safetensors"))


needs_stories15m = pytest.mark.skipif(not _stories15m_installed(),
                                      reason="stories15M is not installed in ../LUT-Transformer-local/checkpoints")


@pytest.fixture(scope="module")
def stories15m():
    cfg, params, hf_model = hf_llama.load_llama()
    tokenizer = hf_llama.load_llama_tokenizer()
    prompt = tokenizer("Once upon a time", return_tensors="pt").input_ids
    return cfg, params, hf_model, prompt


@needs_stories15m
def test_stories15m_matches_hugging_face_in_float32(stories15m):
    cfg, params, hf_model, prompt = stories15m
    assert prompt.tolist() == [[1, 9038, 2501, 263, 931]]                  # BOS, then "Once upon a time"
    assert cfg.name == "stories15m" and next(iter(params.values())).dtype == torch.float32
    ops = host.base_ops(cfg, "inference")
    with torch.no_grad():
        expected = hf_model.generate(prompt, max_new_tokens=64, do_sample=False)
    out = host.generate(params, prompt, 64, ops, cfg, temperature=0)
    n = expected.shape[1]                                                   # shorter only if it emitted EOS
    assert torch.equal(out[:, :n], expected)
    with torch.no_grad():
        theirs = hf_model(expected).logits
    ours = host.forward(params, expected, ops, cfg)
    torch.testing.assert_close(ours, theirs, **REAL_FP32)
    assert bool((ours.argmax(-1) == theirs.argmax(-1)).all())


@needs_stories15m
def test_stories15m_lut_host_all_base_is_bit_identical_to_the_baseline(stories15m):
    cfg, params, _, prompt = stories15m
    b_ops, l_ops = host.base_ops(cfg, "inference"), lut_host.lut_ops(cfg, "inference", lut_host.PRESETS["all_base"])
    ids = host.generate(params, prompt, 27, b_ops, cfg, temperature=0)      # 32 tokens
    assert torch.equal(lut_host.forward(params, ids, l_ops, cfg), host.forward(params, ids, b_ops, cfg))
    b_cache, l_cache = host.new_cache(params, cfg, 1, 32), lut_host.new_cache(params, cfg, 1, 32)
    assert torch.equal(lut_host.prefill(params, ids[:, :20], l_cache, l_ops, cfg),
                       host.prefill(params, ids[:, :20], b_cache, b_ops, cfg))
    for t in range(20, 32):
        assert torch.equal(lut_host.decode_step(params, ids[:, t:t + 1], l_cache, l_ops, cfg),
                           host.decode_step(params, ids[:, t:t + 1], b_cache, b_ops, cfg))
    assert torch.equal(lut_host.generate(params, prompt, 16, l_ops, cfg, temperature=0),
                       host.generate(params, prompt, 16, b_ops, cfg, temperature=0))


@needs_stories15m
def test_stories15m_wrong_rotary_pairing_changes_the_output(stories15m):
    cfg, params, hf_model, prompt = stories15m
    meta_rows = _to_llama2c_rows(params, cfg)
    interleaved = host.base_ops(cfg, "inference")
    for layer in interleaved.layers:
        layer["rope"] = interleaved_rope
    right = host.generate(params, prompt, 24, host.base_ops(cfg, "inference"), cfg, temperature=0)
    assert torch.equal(host.generate(meta_rows, prompt, 24, interleaved, cfg, temperature=0), right)
    wrong = host.generate(meta_rows, prompt, 24, host.base_ops(cfg, "inference"), cfg, temperature=0)
    assert not torch.equal(wrong, right)


@pytest.mark.skipif(not any(CHECKPOINT_DIR.glob("models--karpathy--tinyllamas/snapshots/*/stories15M.pt")),
                    reason="Karpathy's original stories15M.pt is not installed")
@needs_stories15m
def test_stories15m_equals_karpathys_original_after_the_rotary_permutation(stories15m):
    cfg, params, _, _ = stories15m
    (path,) = CHECKPOINT_DIR.glob("models--karpathy--tinyllamas/snapshots/*/stories15M.pt")
    original = torch.load(path, map_location="cpu", weights_only=True)["model"]
    converted = hf_llama.params_from_llama2c(original, cfg)
    assert converted.keys() == params.keys()
    for name in params:
        assert torch.equal(converted[name], params[name]), name
    unconverted = hf_llama.params_from_llama2c(original, cfg, hf_row_order=False)
    assert not torch.equal(unconverted["layers.0.attn.wq"], params["layers.0.attn.wq"])
