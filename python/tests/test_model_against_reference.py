"""The from-scratch model (baseline/host.py) against the torch.nn reference (reference/torch_model.py).

Given the same weights and tokens, both must produce the same logits, for every architecture
option the config allows. Prefill plus decode with the KV cache must give the same logits as
running the whole sequence at once, and greedy generation must pick the same tokens as the
reference recomputing everything at each step.

Tolerances: float64 comparisons use rtol 1e-9 / atol 1e-9. The two models add the same terms in
a different order, which in float64 moves results by about 1e-15 per operation, so anything
above 1e-9 is a real difference. The float32 comparison uses rtol 1e-4 / atol 1e-4: a few layers
of float32 rounding (about 1e-7 relative each) on logits of size about 1 stays well inside it.
"""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from baseline import host  # noqa: E402
from reference import torch_model as ref  # noqa: E402
from LUT import host as lut_host  # noqa: E402

SMALL = host.TransformerConfig(name="test", vocab_size=37, context_length=24, d_model=32, n_heads=4,
                               n_layers=2, d_ff=64)

# Every architecture switch in TransformerConfig appears in at least one of these.
VARIANTS = {
    "gpt2_like": dict(norm="layernorm", norm_placement="pre", activation="gelu",
                      positional_encoding="learned", tie_embeddings=True, bias=True),
    "core_like": dict(norm="rmsnorm", norm_placement="pre", activation="relu",
                      positional_encoding="sinusoidal", tie_embeddings=True, bias=False),
    "no_positions": dict(norm="rmsnorm", norm_placement="pre", activation="relu",
                         positional_encoding="none", tie_embeddings=False, bias=False),
    "post_norm": dict(norm="layernorm", norm_placement="post", activation="relu",
                      positional_encoding="sinusoidal", tie_embeddings=False, bias=True),
}
TIGHT = dict(rtol=1e-9, atol=1e-9)


def _setup(variant: str, dtype=torch.float64, seed: int = 0):
    """A config, from-scratch parameters and a reference model holding the same weights.

    Norm gammas and betas and all biases are moved away from their 1 / 0 initial values, so a
    mistake in how they are applied cannot hide behind an identity."""
    cfg = replace(SMALL, **VARIANTS[variant])
    cfg.validate()
    params = host.init_params(cfg, seed=seed, dtype=dtype)
    gen = torch.Generator().manual_seed(seed + 1)
    for name, tensor in params.items():
        if name.endswith(".gamma"):
            tensor.copy_(1 + 0.2 * torch.randn(tensor.shape, generator=gen, dtype=dtype))
        elif name.endswith(".beta") or ".b" in name:
            tensor.copy_(0.1 * torch.randn(tensor.shape, generator=gen, dtype=dtype))
    model = ref.ReferenceTransformer(cfg).to(dtype)
    ref.copy_params_from_scratch(model, params)
    return cfg, params, model.eval()


def _tokens(cfg, batch: int, length: int, seed: int = 2):
    return torch.randint(0, cfg.vocab_size, (batch, length), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_forward_matches_reference(variant):
    cfg, params, model = _setup(variant)
    ids = _tokens(cfg, batch=3, length=cfg.context_length)
    ops = host.base_ops(cfg, "inference")
    torch.testing.assert_close(host.forward(params, ids, ops, cfg), ref.reference_logits(model, ids), **TIGHT)


def test_forward_matches_reference_in_float32():
    cfg, params, model = _setup("gpt2_like", dtype=torch.float32)
    ids = _tokens(cfg, batch=2, length=16)
    ops = host.base_ops(cfg, "inference")
    torch.testing.assert_close(host.forward(params, ids, ops, cfg), ref.reference_logits(model, ids), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_prefill_then_decode_matches_full_forward(variant):
    # Prefill 10 tokens, then decode the remaining 6 one at a time. The logits after each step
    # must equal the full forward pass's logits at that position.
    cfg, params, _ = _setup(variant)
    ids = _tokens(cfg, batch=2, length=16)
    ops = host.base_ops(cfg, "inference")
    full = host.forward(params, ids, ops, cfg)
    cache = host.new_cache(params, cfg, batch=2, max_len=16)
    torch.testing.assert_close(host.prefill(params, ids[:, :10], cache, ops, cfg), full[:, 9], **TIGHT)
    for t in range(10, 16):
        logits = host.decode_step(params, ids[:, t:t + 1], cache, ops, cfg)
        torch.testing.assert_close(logits, full[:, t], **TIGHT)
    assert cache.length == 16


@pytest.mark.parametrize("variant", ["gpt2_like", "core_like"])
def test_greedy_generate_matches_reference_rollout(variant):
    cfg, params, model = _setup(variant)
    prompt = _tokens(cfg, batch=2, length=5)
    ops = host.base_ops(cfg, "inference")
    out = host.generate(params, prompt, 8, ops, cfg, temperature=0)
    expected = prompt
    for _ in range(8):   # the reference recomputes the whole sequence for every new token
        next_id = ref.reference_logits(model, expected)[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat([expected, next_id], dim=1)
    assert torch.equal(out, expected)


def test_lut_host_with_every_stage_on_base_gives_identical_logits():
    cfg, params, _ = _setup("gpt2_like")
    ids = _tokens(cfg, batch=2, length=12)
    base_logits = host.forward(params, ids, host.base_ops(cfg, "inference"), cfg)
    lut_ops = lut_host.lut_ops(cfg, "inference", lut_host.PRESETS["all_base"])
    assert torch.equal(lut_host.forward(params, ids, lut_ops, cfg), base_logits)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_parameter_names_round_trip(variant):
    cfg, params, model = _setup(variant)
    exported = ref.copy_params_to_scratch(model)
    assert exported.keys() == params.keys()
    for name in params:
        assert torch.equal(exported[name], params[name]), name


def test_init_params_is_reproducible_and_scaled():
    cfg = replace(SMALL, **VARIANTS["gpt2_like"], n_layers=8, d_model=256, d_ff=1024, vocab_size=1000)
    a, b = host.init_params(cfg, seed=3), host.init_params(cfg, seed=3)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["tok_emb"], host.init_params(cfg, seed=4)["tok_emb"])
    # Standard deviations: 0.02 for ordinary weights, 0.02 / sqrt(2 * 8) = 0.005 for residual outputs.
    assert abs(a["layers.0.attn.wq"].std().item() - 0.02) < 0.001
    assert abs(a["layers.0.attn.wo"].std().item() - 0.005) < 0.0003
    assert abs(a["layers.0.ffn.w2"].std().item() - 0.005) < 0.0003
    assert torch.equal(a["norm_f.gamma"], torch.ones(256)) and torch.equal(a["layers.0.attn.bq"], torch.zeros(256))


def test_learned_positions_cap_generation_at_context_length():
    cfg, params, _ = _setup("gpt2_like")
    ops = host.base_ops(cfg, "inference")
    prompt = _tokens(cfg, batch=1, length=cfg.context_length - 2)
    with pytest.raises(ValueError, match="outside"):
        host.generate(params, prompt, 4, ops, cfg, temperature=0)


def test_sinusoidal_and_no_positions_generate_past_context_length():
    for variant in ("core_like", "no_positions"):
        cfg, params, _ = _setup(variant)
        ops = host.base_ops(cfg, "inference")
        out = host.generate(params, _tokens(cfg, 1, cfg.context_length), 4, ops, cfg, temperature=0)
        assert out.shape == (1, cfg.context_length + 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("variant", ["gpt2_like", "core_like"])
def test_forward_on_cuda_matches_reference_on_cuda(variant):
    # float32 on the GPU (the RTX 5080 on alpha-pc). Same 1e-4 tolerance as the CPU float32 test:
    # the GPU kernels sum in a different order again, but by the same few float32 steps.
    cfg, params, model = _setup(variant, dtype=torch.float32)
    params = {name: t.cuda() for name, t in params.items()}
    model = model.cuda()
    ids = _tokens(cfg, batch=2, length=16).cuda()
    ops = host.base_ops(cfg, "inference")
    torch.testing.assert_close(host.forward(params, ids, ops, cfg), ref.reference_logits(model, ids), rtol=1e-4, atol=1e-4)
    out = host.generate(params, ids[:, :5], 6, ops, cfg, temperature=0)
    assert out.device.type == "cuda" and out.shape == (2, 11)
