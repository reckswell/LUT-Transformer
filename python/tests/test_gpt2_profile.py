"""The GPT-2 profile: our core, configured as GPT-2 and given GPT-2's weights, must match
Hugging Face's GPT2LMHeadModel (the imported ground truth).

The tests here need the `transformers` package and skip without it. Only the last test needs the
real pretrained weights; it skips unless they are already in checkpoints/ (or LUT_GPT2_DOWNLOAD=1
allows downloading them, about 550 MB).

Tolerances: float64 rtol 1e-9 / atol 1e-9, as in test_model_against_reference.py (the two
implementations add the same terms in a different order, about 1e-15 per operation). The real
weights are compared in float32 at atol 1e-3 on logits of size about 100: float32 rounding over
12 layers of width 768 accumulates to about 1e-4 there.
"""

import os
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from baseline import host  # noqa: E402
from reference import hf_gpt2  # noqa: E402
from LUT import host as lut_host  # noqa: E402

TIGHT = dict(rtol=1e-9, atol=1e-9)


def _small_hf_gpt2(seed: int = 0):
    """A randomly initialized two-layer GPT-2 (no download), with every weight perturbed away
    from Hugging Face's initial values (LayerNorm 1 and 0, biases 0) so the mapping of each one
    is tested."""
    torch.manual_seed(seed)
    config = transformers.GPT2Config(vocab_size=37, n_positions=24, n_embd=32, n_layer=2, n_head=4,
                                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = transformers.GPT2LMHeadModel(config).double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model


def test_gpt2_yaml_profile_matches_hugging_face_gpt2_small():
    # GPT2Config() defaults are GPT-2 small, so this checks configs/model/gpt2.yaml without a download.
    from_yaml = host.TransformerConfig.from_yaml(host.PYTHON_DIR / "configs/model/gpt2.yaml")
    assert asdict(from_yaml) == asdict(hf_gpt2.gpt2_config_from_hf(transformers.GPT2Config()))
    assert host.count_params(from_yaml)["total"] == 124_439_808     # GPT-2 small, tied embedding counted once


def test_gpt2_weights_run_in_our_code_and_match_hugging_face():
    model = _small_hf_gpt2()
    cfg = hf_gpt2.gpt2_config_from_hf(model.config)
    params = hf_gpt2.gpt2_params_from_hf(model.state_dict(), cfg)
    ids = torch.randint(0, 37, (3, 24), generator=torch.Generator().manual_seed(1))
    ours = host.forward(params, ids, host.base_ops(cfg, "inference"), cfg)
    with torch.no_grad():
        theirs = model(ids).logits
    torch.testing.assert_close(ours, theirs, **TIGHT)


def test_gpt2_through_the_lut_host_and_kv_cache():
    model = _small_hf_gpt2(seed=3)
    cfg = hf_gpt2.gpt2_config_from_hf(model.config)
    params = hf_gpt2.gpt2_params_from_hf(model.state_dict(), cfg)
    ids = torch.randint(0, 37, (2, 16), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        theirs = model(ids).logits
    ops = lut_host.lut_ops(cfg, "inference", lut_host.PRESETS["all_base"])
    cache = lut_host.new_cache(params, cfg, batch=2, max_len=16)
    torch.testing.assert_close(lut_host.prefill(params, ids[:, :12], cache, ops, cfg), theirs[:, 11], **TIGHT)
    for t in range(12, 16):
        torch.testing.assert_close(lut_host.decode_step(params, ids[:, t:t + 1], cache, ops, cfg), theirs[:, t], **TIGHT)


def test_loader_rejects_settings_we_do_not_model():
    with pytest.raises(ValueError, match="tanh GELU"):
        hf_gpt2.gpt2_config_from_hf(transformers.GPT2Config(activation_function="relu"))
    with pytest.raises(ValueError, match="layer_norm_epsilon"):
        hf_gpt2.gpt2_config_from_hf(transformers.GPT2Config(layer_norm_epsilon=1e-6))


def _pretrained_available() -> bool:
    return os.environ.get("LUT_GPT2_DOWNLOAD") == "1" or any(hf_gpt2.CHECKPOINT_DIR.glob("models--openai-community--gpt2*"))


@pytest.mark.skipif(not _pretrained_available(), reason="GPT-2 weights not in checkpoints/ (set LUT_GPT2_DOWNLOAD=1 to fetch, about 550 MB)")
def test_pretrained_gpt2_small_matches_hugging_face():
    cfg, params, hf_model = hf_gpt2.load_gpt2()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=str(hf_gpt2.CHECKPOINT_DIR))
    ids = tokenizer("The quick brown fox jumps over the lazy dog because", return_tensors="pt").input_ids
    ours = host.forward(params, ids, host.base_ops(cfg, "inference"), cfg)
    with torch.no_grad():
        theirs = hf_model(ids).logits
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-3)
    out = host.generate(params, ids, 5, host.base_ops(cfg, "inference"), cfg, temperature=0)
    expected = hf_model.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    assert torch.equal(out, expected)
