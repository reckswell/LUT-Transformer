"""The ops the Llama profile adds (baseline/inference.py and baseline/train.py): rope_tables, rope,
silu and glu_gate.

Forward against torch.nn.functional and, for the rotary pieces, against Hugging Face's own
LlamaRotaryEmbedding and apply_rotary_pos_emb (skipped without transformers). Backward against
autograd on an independently written formula. Everything is float64 at rtol 1e-9 / atol 1e-9:
both sides agree to about 1e-15 there, so only a correct formula passes.

The rotary table is built in float32 on purpose (the constants Llama was trained with, see
inference.rope_tables), so the tests that check the table's formula allow float32 rounding of the
angle, and the tests of the rotation itself use exact float64 tables built here.

Float64 here checks the math against a reference; it is not the base metric. Default runs are
float32 with TF32 off (DTYPE, common/precision.py, tests/test_fp32_defaults.py).
"""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from baseline import inference, train  # noqa: E402

D64 = torch.float64
TIGHT = dict(rtol=1e-9, atol=1e-9)


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


def _leaf(*shape, gen, scale=1.0):
    return (torch.randn(*shape, generator=gen, dtype=D64) * scale).requires_grad_(True)


def _exact_tables(length: int, head_dim: int, theta: float = 10000.0, start: int = 0):
    """cos(p w_i), sin(p w_i) computed in float64 from the formula (not inference.rope_tables)."""
    p = torch.arange(start, start + length, dtype=D64)[:, None]
    w = theta ** (-torch.arange(0, head_dim, 2, dtype=D64) / head_dim)
    return torch.cos(p * w), torch.sin(p * w)


def _rotate_half_reference(x, cos, sin):
    """Hugging Face's formula, written here independently: x * [cos, cos] + rotate_half(x) * [sin, sin]."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * torch.cat((cos, cos), -1) + torch.cat((-x2, x1), -1) * torch.cat((sin, sin), -1)


# ----------------------------------------------------------------------------------------------
# Rotary tables and rotation
# ----------------------------------------------------------------------------------------------

def test_rope_tables_follow_the_formula_and_the_start_offset():
    cos, sin = inference.rope_tables(10, 8, 10000.0, start=3, dtype=D64)
    exact_cos, exact_sin = _exact_tables(10, 8, start=3)
    assert cos.shape == sin.shape == (10, 4) and cos.dtype == D64
    # Built in float32: the angle p * w_i is rounded to float32 (about 6e-8 relative), so at
    # positions up to 12 the table is within about 1e-6 of the float64 formula.
    torch.testing.assert_close(cos, exact_cos, rtol=0, atol=2e-6)
    torch.testing.assert_close(sin, exact_sin, rtol=0, atol=2e-6)
    full_cos, full_sin = inference.rope_tables(13, 8, 10000.0, dtype=D64)
    assert torch.equal(cos, full_cos[3:]) and torch.equal(sin, full_sin[3:])   # decode steps continue the table
    with pytest.raises(ValueError, match="even head width"):
        inference.rope_tables(4, 7)


def test_rope_tables_equal_hugging_face_llama_rotary_embedding_bit_for_bit():
    transformers = pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = transformers.LlamaConfig(hidden_size=288, num_attention_heads=6, max_position_embeddings=256)
    rotary = LlamaRotaryEmbedding(config)                     # stories15M's head width (48) and theta
    positions = torch.arange(300)[None]                       # past the trained context too
    for dtype in (torch.float32, D64):
        hf_cos, hf_sin = rotary(torch.zeros(1, dtype=dtype), positions)        # (1, 300, 48)
        cos, sin = inference.rope_tables(300, 48, config.rope_parameters["rope_theta"], dtype=dtype)
        assert torch.equal(hf_cos[0], torch.cat((cos, cos), -1)), dtype
        assert torch.equal(hf_sin[0], torch.cat((sin, sin), -1)), dtype


def test_rope_matches_hugging_face_apply_rotary_pos_emb(gen):
    pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    q, k = torch.randn(2, 3, 7, 12, generator=gen, dtype=D64), torch.randn(2, 3, 7, 12, generator=gen, dtype=D64)
    cos, sin = inference.rope_tables(7, 12, start=5, dtype=D64)
    hf_q, hf_k = apply_rotary_pos_emb(q, k, torch.cat((cos, cos), -1)[None], torch.cat((sin, sin), -1)[None])
    # The same products and sums in the same order (a + (-b) is exactly a - b), so bit for bit.
    assert torch.equal(inference.rope(q, cos, sin), hf_q)
    assert torch.equal(inference.rope(k, cos, sin), hf_k)


def test_rope_is_a_rotation_that_encodes_relative_position(gen):
    q, k = torch.randn(1, 1, 1, 16, generator=gen, dtype=D64), torch.randn(1, 1, 1, 16, generator=gen, dtype=D64)
    cos, sin = _exact_tables(40, 16)

    def score(p, s):
        return float((inference.rope(q, cos[p:p + 1], sin[p:p + 1]) * inference.rope(k, cos[s:s + 1], sin[s:s + 1])).sum())

    # q at p and k at s: the score depends only on p - s.
    assert score(5, 2) == pytest.approx(score(31, 28), abs=1e-12)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)
    # Each pair keeps its length, so the whole vector does too.
    y = inference.rope(q, cos[9:10], sin[9:10])
    torch.testing.assert_close(y.norm(), q.norm(), **TIGHT)
    pair_len = lambda t: t[..., :8] ** 2 + t[..., 8:] ** 2   # noqa: E731
    torch.testing.assert_close(pair_len(y), pair_len(q), **TIGHT)


def test_rope_backward_matches_autograd(gen):
    x = _leaf(2, 3, 7, 8, gen=gen)
    cos, sin = inference.rope_tables(7, 8, start=4, dtype=D64)
    dy = torch.randn(2, 3, 7, 8, generator=gen, dtype=D64)
    y, cache = train.rope_forward(x.detach(), cos, sin)
    ref = _rotate_half_reference(x, cos, sin)
    ref.backward(dy)
    torch.testing.assert_close(y, ref.detach(), **TIGHT)
    torch.testing.assert_close(train.rope_forward(x.detach(), cos, sin)[0], inference.rope(x.detach(), cos, sin), **TIGHT)
    torch.testing.assert_close(train.rope_backward(dy, cache), x.grad, **TIGHT)


# ----------------------------------------------------------------------------------------------
# SiLU and the SwiGLU gate
# ----------------------------------------------------------------------------------------------

def test_silu_matches_torch_including_extremes(gen):
    x = torch.cat([torch.randn(200, generator=gen, dtype=D64) * 6,
                   torch.tensor([-1000.0, -60.0, -1e-300, 0.0, 1e-300, 60.0, 1000.0], dtype=D64)])
    y = inference.silu(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, F.silu(x), **TIGHT)
    torch.testing.assert_close(train.silu_forward(x)[0], F.silu(x), **TIGHT)
    assert y.min() > -0.2785 and y.min() < -0.27   # the minimum, about -0.2785 at x = -1.2785


def test_silu_backward_matches_autograd(gen):
    x = torch.cat([torch.randn(300, generator=gen, dtype=D64) * 5, torch.tensor([-800.0, 0.0, 800.0], dtype=D64)])
    x.requires_grad_(True)
    dy = torch.randn(x.shape, generator=gen, dtype=D64)
    _, cache = train.silu_forward(x.detach())
    F.silu(x).backward(dy)
    torch.testing.assert_close(train.silu_backward(dy, cache), x.grad, **TIGHT)


def test_glu_gate_forward_and_backward(gen):
    a, b = _leaf(4, 5, 6, gen=gen), _leaf(4, 5, 6, gen=gen)
    dy = torch.randn(4, 5, 6, generator=gen, dtype=D64)
    y, cache = train.glu_gate_forward(a.detach(), b.detach())
    torch.mul(a, b).backward(dy)
    assert torch.equal(y, inference.glu_gate(a.detach(), b.detach()))
    da, db = train.glu_gate_backward(dy, cache)
    torch.testing.assert_close(da, a.grad, **TIGHT)
    torch.testing.assert_close(db, b.grad, **TIGHT)


def test_swiglu_feed_forward_chain_matches_autograd(gen):
    # The composition baseline/host.py runs: w2(silu(w1 h) * (w3 h)), through the *_forward and
    # *_backward functions, against F.linear / F.silu with autograd. Checks dh (summed over the W1
    # and W3 paths) and dW1, dW2, dW3.
    h, w1, w3, w2 = _leaf(2, 5, 8, gen=gen), _leaf(12, 8, gen=gen), _leaf(12, 8, gen=gen), _leaf(8, 12, gen=gen)
    dy = torch.randn(2, 5, 8, generator=gen, dtype=D64)
    u, c_up = train.linear_forward(h.detach(), w1.detach())
    z, c_act = train.silu_forward(u)
    u3, c_up3 = train.linear_forward(h.detach(), w3.detach())
    g, c_gate = train.glu_gate_forward(z, u3)
    y, c_down = train.linear_forward(g, w2.detach())
    ref = F.linear(F.silu(F.linear(h, w1)) * F.linear(h, w3), w2)
    ref.backward(dy)
    torch.testing.assert_close(y, ref.detach(), **TIGHT)
    dg, dw2, _ = train.linear_backward(dy, c_down)
    dz, du3 = train.glu_gate_backward(dg, c_gate)
    dh3, dw3, _ = train.linear_backward(du3, c_up3)
    dh1, dw1, _ = train.linear_backward(train.silu_backward(dz, c_act), c_up)
    for mine, theirs in ((dh1 + dh3, h.grad), (dw1, w1.grad), (dw2, w2.grad), (dw3, w3.grad)):
        torch.testing.assert_close(mine, theirs, **TIGHT)


def _rope_tables_per_call(length, head_dim, theta=10000.0, start=0, *, device=None, dtype=torch.float32):
    """rope_tables as it was before 2026-11-24 (rebuilt on every call, w copied from the CPU each
    time): the reference the cached table must equal bit for bit."""
    w = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    p = torch.arange(start, start + length, dtype=torch.float32, device=device)
    angle = p[:, None] * w.to(device)[None, :]
    return torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device"))])
@pytest.mark.parametrize("dtype", [torch.float32, D64])
def test_cached_rope_tables_equal_the_per_call_tables_bit_for_bit(device, dtype):
    inference.clear_rope_tables()
    for length in (1, 2, 7, 64, 256):
        for start in range(0, 1024 - length + 1, 1 if length == 1 else 37):
            cos, sin = inference.rope_tables(length, 48, 10000.0, start, device=device, dtype=dtype)
            ref_cos, ref_sin = _rope_tables_per_call(length, 48, 10000.0, start, device=device, dtype=dtype)
            assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin), (length, start)
            assert cos.device.type == device and cos.dtype == dtype


def test_the_rope_table_is_built_once_and_grows_when_needed():
    inference.clear_rope_tables()
    a, _ = inference.rope_tables(1, 8, start=5)
    b, _ = inference.rope_tables(1, 8, start=200)
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()        # one 256-row table, sliced
    assert len(inference._ROPE_TABLES) == 1 and next(iter(inference._ROPE_TABLES.values()))[0].shape == (256, 4)
    c, _ = inference.rope_tables(4, 8, start=300)                                     # past row 255: rebuilt larger
    assert next(iter(inference._ROPE_TABLES.values()))[0].shape == (512, 4)
    assert torch.equal(c, _rope_tables_per_call(4, 8, start=300)[0])
    inference.rope_tables(1, 8, dtype=D64)                                            # another dtype: its own table
    assert len(inference._ROPE_TABLES) == 2
    inference.clear_rope_tables()
    assert not inference._ROPE_TABLES
