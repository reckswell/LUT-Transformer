"""From-scratch ops vs PyTorch. Forward is compared to torch.nn.functional in float32; backward
is compared to autograd in float64. A test whose function is not written yet skips (via _call)
until it is, then becomes the check that the function is right.

Float64 here checks the math against a reference; it is not the base metric. Default runs are
float32 with TF32 off (DTYPE, common/precision.py, tests/test_fp32_defaults.py).
"""

import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from baseline import inference, train  # noqa: E402


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:
        pytest.skip(str(error))


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


def test_linear_forward(gen):
    x, w, b = torch.randn(2, 5, 8, generator=gen), torch.randn(6, 8, generator=gen), torch.randn(6, generator=gen)
    torch.testing.assert_close(_call(inference.linear, x, w, b), F.linear(x, w, b), rtol=1e-5, atol=1e-6)


def test_softmax_forward(gen):
    s = torch.randn(2, 3, 4, 7, generator=gen) * 10
    torch.testing.assert_close(_call(inference.softmax, s), torch.softmax(s, dim=-1), rtol=1e-5, atol=1e-7)


def test_layernorm_forward(gen):
    x, g, b = torch.randn(2, 5, 8, generator=gen), torch.randn(8, generator=gen), torch.randn(8, generator=gen)
    torch.testing.assert_close(_call(inference.layernorm, x, g, b), F.layer_norm(x, (8,), g, b, 1e-5), rtol=1e-5, atol=1e-5)


def test_relu_forward(gen):
    x = torch.randn(4, 9, generator=gen)
    torch.testing.assert_close(_call(inference.relu, x), F.relu(x))


# Tolerances for the forward tests below: float32 with rtol 1e-5 and atol 1e-6 (atol 1e-5 where a
# result sums many terms). Our ops and torch's kernels add the same terms in a different order,
# so they differ by a few float32 ulps (about 1e-7 relative each); 1e-5 leaves room for that and
# still catches any real formula error, which shows up at 1e-3 or more.

def test_linear_forward_without_bias(gen):
    x, w = torch.randn(3, 8, generator=gen), torch.randn(5, 8, generator=gen)
    torch.testing.assert_close(inference.linear(x, w), F.linear(x, w), rtol=1e-5, atol=1e-6)


def test_softmax_is_stable_for_large_scores(gen):
    s = torch.randn(4, 16, generator=gen) * 1e4         # exp(1e4) overflows without the max shift
    p = inference.softmax(s)
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p, torch.softmax(s, dim=-1), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(p.sum(-1), torch.ones(4))


def test_gelu_forward_matches_torch_tanh_approximation(gen):
    x = torch.randn(4, 33, generator=gen) * 4
    torch.testing.assert_close(inference.gelu(x), F.gelu(x, approximate="tanh"), rtol=1e-5, atol=1e-6)


def test_rmsnorm_forward(gen):
    x, g = torch.randn(2, 5, 8, generator=gen), torch.randn(8, generator=gen)
    torch.testing.assert_close(inference.rmsnorm(x, g), F.rms_norm(x, (8,), g, 1e-6), rtol=1e-5, atol=1e-5)


def test_layernorm_forward_with_large_offset(gen):
    # Rows far from zero: the mean subtraction must happen before squaring, or float32 loses it.
    # Tolerance 1e-4: near 1000 one float32 step is 6.1e-5, so the two means can differ by about
    # that much (measured max difference 6.1e-5).
    x = torch.randn(3, 64, generator=gen) + 1000.0
    g, b = torch.ones(64), torch.zeros(64)
    torch.testing.assert_close(inference.layernorm(x, g, b), F.layer_norm(x, (64,), g, b, 1e-5), rtol=1e-4, atol=1e-4)


def test_embedding_forward(gen):
    table = torch.randn(11, 6, generator=gen)
    ids = torch.randint(0, 11, (2, 7), generator=gen)
    torch.testing.assert_close(inference.embedding(ids, table), F.embedding(ids, table))


@pytest.mark.parametrize("d_model", [8, 7])
def test_sinusoidal_positions_match_the_formula(d_model):
    # The obviously-correct slow version: one Python math call per entry.
    start, length = 5, 6
    ref = torch.zeros(length, d_model)
    for row, p in enumerate(range(start, start + length)):
        for col in range(d_model):
            two_i = col - (col % 2)
            angle = p / 10000 ** (two_i / d_model)
            ref[row, col] = math.sin(angle) if col % 2 == 0 else math.cos(angle)
    torch.testing.assert_close(inference.sinusoidal_positions(length, d_model, start), ref, rtol=1e-5, atol=1e-6)


def test_learned_positions_slice_and_bounds(gen):
    table = torch.randn(10, 4, generator=gen)
    torch.testing.assert_close(inference.learned_positions(3, table, start=6), table[6:9])
    with pytest.raises(ValueError, match="outside"):
        inference.learned_positions(3, table, start=8)


def test_split_and_merge_heads(gen):
    x = torch.randn(2, 5, 12, generator=gen)
    heads = inference.split_heads(x, n_heads=3)
    assert heads.shape == (2, 3, 5, 4)
    torch.testing.assert_close(heads[1, 2, 4], x[1, 4, 8:12])       # head 2 owns columns 8..11
    torch.testing.assert_close(inference.merge_heads(heads), x)


@pytest.mark.parametrize("t_query,t_key,offset", [(5, 5, 0), (3, 7, 4), (1, 9, 8)])
def test_causal_mask(t_query, t_key, offset):
    ref = torch.ones(t_query, t_key, dtype=torch.bool).tril(diagonal=offset)
    assert torch.equal(inference.causal_mask(t_query, t_key, offset), ref)


def _attention(q, k, v, mask):
    """The from-scratch attention path, in the order the model runs it."""
    s = inference.attention_scores(q, k, mask)
    return inference.attention_values(inference.softmax(s), v)


def test_attention_prefill_matches_sdpa(gen):
    q, k, v = (torch.randn(2, 3, 6, 8, generator=gen) for _ in range(3))
    mask = inference.causal_mask(6, 6)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(_attention(q, k, v, mask), ref, rtol=1e-5, atol=1e-5)


def test_attention_with_cached_keys_matches_sdpa(gen):
    # 3 new queries at positions 4..6 against 7 keys (4 cached + 3 new).
    q = torch.randn(2, 3, 3, 8, generator=gen)
    k, v = (torch.randn(2, 3, 7, 8, generator=gen) for _ in range(2))
    mask = inference.causal_mask(3, 7, offset=4)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(_attention(q, k, v, mask), ref, rtol=1e-5, atol=1e-5)


def test_attention_scores_are_minus_inf_where_masked(gen):
    q, k = torch.randn(1, 1, 4, 8, generator=gen), torch.randn(1, 1, 4, 8, generator=gen)
    s = inference.attention_scores(q, k, inference.causal_mask(4, 4))
    assert torch.isneginf(s[0, 0, 0, 1:]).all() and torch.isfinite(s[0, 0, 3]).all()
    torch.testing.assert_close(s[0, 0, 3, 2], (q[0, 0, 3] @ k[0, 0, 2]) / 8 ** 0.5)


def test_kv_cache_matches_concatenation(gen):
    cache = inference.KVCache.allocate(n_layers=2, batch=2, n_heads=3, max_len=8, head_dim=4)
    steps = [torch.randn(2, 3, t, 4, generator=gen) for t in (5, 1, 1)]   # prefill 5, then 2 decode steps
    for x in steps:
        for layer in range(2):
            keys, values = cache.append(layer, x, -x)
        cache.advance(x.shape[2])
    full = torch.cat(steps, dim=2)
    assert cache.length == 7
    torch.testing.assert_close(keys, full)
    torch.testing.assert_close(values, -full)
    with pytest.raises(ValueError, match="full"):
        cache.append(0, torch.zeros(2, 3, 2, 4), torch.zeros(2, 3, 2, 4))


def test_sample_next_token(gen):
    logits = torch.tensor([[0.0, 1.0, 2.0, 0.5], [3.0, -1.0, 0.0, 2.9]])
    assert inference.sample_next_token(logits, temperature=0).tolist() == [2, 0]
    assert inference.sample_next_token(logits, top_k=1, generator=gen).tolist() == [2, 0]
    # Sampled frequencies match softmax(logits) within about 4 standard errors (n = 20,000).
    n = 20_000
    draws = torch.stack([inference.sample_next_token(logits, generator=gen) for _ in range(n)])
    for row in range(2):
        freq = torch.bincount(draws[:, row], minlength=4).double() / n
        torch.testing.assert_close(freq, torch.softmax(logits[row].double(), -1), rtol=0, atol=0.015)
    # top_k = 2 never picks outside the two largest logits.
    draws = torch.stack([inference.sample_next_token(logits, top_k=2, generator=gen) for _ in range(500)])
    assert set(draws[:, 0].tolist()) <= {1, 2} and set(draws[:, 1].tolist()) <= {0, 3}


def test_linear_backward_matches_autograd(gen):
    x = torch.randn(2, 5, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    w = torch.randn(6, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6, generator=gen, dtype=torch.float64, requires_grad=True)
    dy = torch.randn(2, 5, 6, generator=gen, dtype=torch.float64)
    y, cache = _call(train.linear_forward, x.detach(), w.detach(), b.detach())
    dx, dw, db = train.linear_backward(dy, cache)
    F.linear(x, w, b).backward(dy)
    torch.testing.assert_close(y, F.linear(x, w, b).detach())
    for mine, ref in ((dx, x.grad), (dw, w.grad), (db, b.grad)):
        torch.testing.assert_close(mine, ref)


def test_softmax_backward_matches_autograd(gen):
    s = torch.randn(2, 3, 4, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    dp = torch.randn(2, 3, 4, 7, generator=gen, dtype=torch.float64)
    p, cache = _call(train.softmax_forward, s.detach())
    ds = train.softmax_backward(dp, cache)
    torch.softmax(s, dim=-1).backward(dp)
    torch.testing.assert_close(ds, s.grad)


def test_cross_entropy_matches_torch(gen):
    logits = torch.randn(2, 5, 11, generator=gen, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 11, (2, 5), generator=gen)
    loss, cache = _call(train.cross_entropy_forward, logits.detach(), targets)
    d_logits = train.cross_entropy_backward(cache)
    ref = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
    ref.backward()
    torch.testing.assert_close(loss, ref.detach())
    torch.testing.assert_close(d_logits, logits.grad)
