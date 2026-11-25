"""The family profiles (reference/hf_families.py) against Hugging Face, on small random models.

For each family a two-layer model is built from its transformers config class with random weights
(nothing is downloaded), every weight is moved away from its initial value (norm weights 1, or 0
for Gemma's 1 + w), and the export must run in our code with the same logits as the Hugging Face
model: the whole sequence at once, and prefill plus KV-cache decode against Hugging Face's full
forward. The LUT model with every stage on BASE must give the baseline's logits bit for bit. Each
family also runs in the variants its checkpoints use (grouped-query, multi-query, sliding window,
sequential GPT-NeoX, tied Phi, OPT without norm weights).

Tolerances. The LayerNorm families (gpt_neox, phi, opt) match Hugging Face's float64 model to
1e-9 (measured: about 1.5e-15). The RMSNorm families (llama, mistral, qwen2, gemma) can't: Hugging
Face normalizes in float32 whatever the model's dtype (see tests/test_llama_profile.py), so they
are compared at atol 2e-6 (measured: up to 4.4e-7 on logits up to about 4). A wrong mapping (a
swapped weight, the wrong rotary columns, a missed norm offset or embedding scale) moves logits
by 1e-2 or more, which the mutation test below checks.
The float64 1e-9 checks of the options themselves (forward, decode, gradients against autograd,
LUT bit-identity) are against reference/torch_model.py, in test_model_against_reference.py.
"""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from baseline import host  # noqa: E402
from LUT import host as lut_host  # noqa: E402
from reference import hf_families, hf_profiles  # noqa: E402

EXACT = dict(rtol=1e-9, atol=1e-9)
HF_FLOAT32_NORMS = dict(rtol=0, atol=2e-6)

SMALL = dict(vocab_size=61, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32)
ROPE_HALF = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}

# (model_type, settings) -> what the case covers. Every family's default shape, then its variants.
CASES = {
    "llama_gqa_bias": ("llama", dict(intermediate_size=48, num_key_value_heads=2, attention_bias=True,
                                     mlp_bias=True, tie_word_embeddings=False)),
    "llama_wide_heads": ("llama", dict(intermediate_size=48, head_dim=16, tie_word_embeddings=True)),
    "mistral_sliding": ("mistral", dict(intermediate_size=48, num_key_value_heads=2, sliding_window=6)),
    "mistral_full": ("mistral", dict(intermediate_size=48, num_key_value_heads=1, sliding_window=None)),
    "qwen2": ("qwen2", dict(intermediate_size=48, num_key_value_heads=2, tie_word_embeddings=True)),
    "qwen2_all_sliding": ("qwen2", dict(intermediate_size=48, num_key_value_heads=2, use_sliding_window=True,
                                        sliding_window=6, max_window_layers=0)),
    "gemma_mqa": ("gemma", dict(intermediate_size=48, num_key_value_heads=1, head_dim=16)),
    "gemma_mha": ("gemma", dict(intermediate_size=48, num_key_value_heads=4, head_dim=8)),
    "gpt_neox_parallel": ("gpt_neox", dict(intermediate_size=64, rope_parameters=dict(ROPE_HALF))),
    "gpt_neox_sequential": ("gpt_neox", dict(intermediate_size=64, use_parallel_residual=False,
                                             rope_parameters={**ROPE_HALF, "partial_rotary_factor": 0.25})),
    "phi": ("phi", dict(intermediate_size=64, rope_parameters={**ROPE_HALF, "partial_rotary_factor": 0.25})),
    "phi_gqa_tied": ("phi", dict(intermediate_size=64, num_key_value_heads=2, tie_word_embeddings=True,
                                 rope_parameters=dict(ROPE_HALF))),
    "opt": ("opt", dict(ffn_dim=64, word_embed_proj_dim=32)),
    "opt_no_norm_weights": ("opt", dict(ffn_dim=64, word_embed_proj_dim=32, layer_norm_elementwise_affine=False,
                                        enable_bias=False)),
}


def small_hf_model(model_type: str, settings: dict, seed: int = 0):
    """A random two-layer Hugging Face model of the family, in float64, every weight perturbed."""
    config = transformers.AutoConfig.for_model(model_type, **{**SMALL, **settings})
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(config).double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model


def tolerance(model_type: str) -> dict:
    return EXACT if model_type in ("gpt_neox", "phi", "opt") else HF_FLOAT32_NORMS


def tokens(vocab: int, batch: int, length: int, seed: int = 3):
    return torch.randint(0, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    model_type, settings = CASES[request.param]
    model = small_hf_model(model_type, settings)
    cfg, params = hf_profiles.export(model)
    return request.param, model_type, model, cfg, params


def test_every_family_has_a_profile():
    assert set(hf_families.FAMILIES) <= set(hf_profiles.PROFILES)
    assert {model_type for model_type, _ in CASES.values()} == set(hf_families.FAMILIES)


def test_logits_match_hugging_face(case):
    name, model_type, model, cfg, params = case
    ids = tokens(cfg.vocab_size, 2, 20)
    with torch.no_grad():
        theirs = model(ids).logits
    ours = host.forward(params, ids, host.base_ops(cfg, "inference"), cfg)
    torch.testing.assert_close(ours, theirs, **tolerance(model_type))


def test_prefill_and_decode_match_hugging_face(case):
    name, model_type, model, cfg, params = case
    ids = tokens(cfg.vocab_size, 2, 20, seed=4)
    with torch.no_grad():
        theirs = model(ids).logits
    ops = host.base_ops(cfg, "inference")
    cache = host.new_cache(params, cfg, 2, 20)
    torch.testing.assert_close(host.prefill(params, ids[:, :9], cache, ops, cfg), theirs[:, 8], **tolerance(model_type))
    for t in range(9, 20):
        torch.testing.assert_close(host.decode_step(params, ids[:, t:t + 1], cache, ops, cfg), theirs[:, t],
                                   **tolerance(model_type))
    assert cache.keys[0].shape[1] == cfg.kv_heads                  # the cache holds the key/value heads only


def test_lut_model_on_base_is_bit_identical_to_the_baseline(case):
    name, model_type, model, cfg, params = case
    ids = tokens(cfg.vocab_size, 2, 20, seed=5)
    base = host.forward(params, ids, host.base_ops(cfg, "inference"), cfg)
    assert torch.equal(lut_host.forward(params, ids, lut_host.lut_ops(cfg, "inference"), cfg), base)


def test_the_options_each_case_needs_are_set():
    def cfg_of(name):
        model_type, settings = CASES[name]
        return hf_profiles.profile_for(model_type).config_from_hf(
            transformers.AutoConfig.for_model(model_type, **{**SMALL, **settings}))

    assert cfg_of("llama_gqa_bias").kv_heads == 2 and cfg_of("llama_gqa_bias").has_bias("out")
    assert cfg_of("mistral_sliding").attention_window == 6 and cfg_of("mistral_full").attention_window is None
    qwen = cfg_of("qwen2")
    assert qwen.has_bias("qkv") and not qwen.has_bias("out") and not qwen.has_bias("ffn")
    assert cfg_of("qwen2_all_sliding").attention_window == 6
    gemma = cfg_of("gemma_mqa")
    assert gemma.kv_heads == 1 and gemma.head_dim == 16 and gemma.activation == "gelu"
    assert gemma.embed_scale == pytest.approx(32 ** 0.5, rel=1e-7)
    neox = cfg_of("gpt_neox_parallel")
    assert neox.block == "parallel" and neox.rotary_dim == 4 and neox.activation == "gelu_exact"
    assert cfg_of("gpt_neox_sequential").block == "sequential" and cfg_of("gpt_neox_sequential").rotary_dim == 2
    phi = cfg_of("phi")
    assert phi.block == "parallel_shared_norm" and phi.rotary_dim == 2 and phi.lm_head_bias
    assert cfg_of("opt").positional_encoding == "learned" and cfg_of("opt").activation == "relu"


def test_a_wrong_mapping_is_caught_at_this_tolerance():
    # Mutations a profile could plausibly get wrong; each must miss by far more than the tolerances.
    model = small_hf_model("gpt_neox", CASES["gpt_neox_parallel"][1])
    cfg, params = hf_profiles.export(model)
    ids = tokens(cfg.vocab_size, 2, 20)
    with torch.no_grad():
        theirs = model(ids).logits
    ops = host.base_ops(cfg, "inference")

    def off(p, c=cfg):
        return (host.forward(p, ids, ops, c) - theirs).abs().max().item()

    assert off(params) < 1e-9
    swapped = dict(params, **{"layers.0.attn.wq": params["layers.0.attn.wk"], "layers.0.attn.wk": params["layers.0.attn.wq"]})
    assert off(swapped) > 1e-2
    assert off(params, replace(cfg, rotary_dim=None)) > 1e-2                 # rotating every column
    assert off(params, replace(cfg, block="sequential")) > 1e-2
    gemma = small_hf_model("gemma", CASES["gemma_mqa"][1])
    gcfg, gparams = hf_profiles.export(gemma)
    with torch.no_grad():
        gtheirs = gemma(ids).logits
    no_offset = {k: (v - 1 if k.endswith(".gamma") else v) for k, v in gparams.items()}   # gamma = w, not 1 + w
    gops = host.base_ops(gcfg, "inference")
    assert (host.forward(gparams, ids, gops, gcfg) - gtheirs).abs().max().item() < 2e-6
    assert (host.forward(no_offset, ids, gops, gcfg) - gtheirs).abs().max().item() > 1e-2
    assert (host.forward(gparams, ids, gops, replace(gcfg, embed_scale=None)) - gtheirs).abs().max().item() > 1e-2


@pytest.mark.parametrize("model_type, settings, problem", [
    ("llama", dict(rope_parameters={"rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0, "low_freq_factor": 1.0,
                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 16}), "rotary variant 'llama3'"),
    ("qwen2", dict(use_sliding_window=True, sliding_window=6, max_window_layers=1), "layers mix"),
    ("phi", dict(qk_layernorm=True), "qk_layernorm"),
    ("opt", dict(word_embed_proj_dim=16), "project_in"),
    ("opt", dict(do_layer_norm_before=False), "post-norm OPT"),
    ("gpt_neox", dict(hidden_act="quick_gelu"), "activation 'quick_gelu'"),
])
def test_settings_we_do_not_model_are_refused_with_the_reason(model_type, settings, problem):
    config = transformers.AutoConfig.for_model(model_type, **{**SMALL, **settings})
    with pytest.raises(hf_families.Unsupported, match=problem):
        hf_profiles.profile_for(model_type).config_from_hf(config)


def test_a_state_dict_entry_no_stage_uses_is_refused():
    model = small_hf_model("qwen2", CASES["qwen2"][1])
    cfg, _ = hf_profiles.export(model)
    state = dict(model.state_dict())
    state["model.layers.0.self_attn.q_norm.weight"] = torch.ones(8, dtype=torch.float64)   # Qwen3's per-head norm
    with pytest.raises(ValueError, match="q_norm"):
        hf_profiles.profile_for("qwen2").params_from_hf(state, cfg)
