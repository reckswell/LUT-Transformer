"""Hand-written backward functions (baseline/train.py) against autograd.

Each test runs the same math through torch.nn.functional (or plain tensor ops) with autograd,
backpropagates the same random upstream gradient, and compares. Everything is float64, where both
sides agree to about 1e-15, so the default float64 tolerances of torch.testing.assert_close
(rtol 1e-7, atol 1e-7) would only pass a correct formula. The forward outputs are compared too,
since train.py repeats the forward math on purpose (separate from inference.py).
"""

import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from baseline import train  # noqa: E402

D64 = torch.float64


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


def _leaf(*shape, gen, scale=1.0, offset=0.0):
    return (torch.randn(*shape, generator=gen, dtype=D64) * scale + offset).requires_grad_(True)


def _check(ours, refs):
    for mine, ref in zip(ours, refs):
        torch.testing.assert_close(mine, ref)


def test_embedding(gen):
    table = _leaf(11, 6, gen=gen)
    ids = torch.tensor([[1, 4, 4, 0], [9, 1, 1, 1]])          # repeats must sum
    dx = torch.randn(2, 4, 6, generator=gen, dtype=D64)
    x, cache = train.embedding_forward(ids, table.detach())
    F.embedding(ids, table).backward(dx)
    torch.testing.assert_close(x, table.detach()[ids])
    torch.testing.assert_close(train.embedding_backward(dx, cache), table.grad)


@pytest.mark.parametrize("bias", [True, False])
def test_linear(gen, bias):
    x, w = _leaf(2, 5, 8, gen=gen), _leaf(6, 8, gen=gen)
    b = _leaf(6, gen=gen) if bias else None
    dy = torch.randn(2, 5, 6, generator=gen, dtype=D64)
    y, cache = train.linear_forward(x.detach(), w.detach(), b.detach() if bias else None)
    ref = F.linear(x, w, b)
    ref.backward(dy)
    torch.testing.assert_close(y, ref.detach())
    dx, dw, db = train.linear_backward(dy, cache)
    _check((dx, dw), (x.grad, w.grad))
    if bias:
        torch.testing.assert_close(db, b.grad)
    else:
        assert db is None


@pytest.mark.parametrize("name,ref_fn", [("relu", F.relu), ("gelu", lambda x: F.gelu(x, approximate="tanh"))])
def test_activations(gen, name, ref_fn):
    x = _leaf(4, 33, gen=gen, scale=3.0)
    dy = torch.randn(4, 33, generator=gen, dtype=D64)
    y, cache = getattr(train, f"{name}_forward")(x.detach())
    ref = ref_fn(x)
    ref.backward(dy)
    torch.testing.assert_close(y, ref.detach())
    torch.testing.assert_close(getattr(train, f"{name}_backward")(dy, cache), x.grad)


def test_layernorm(gen):
    x, g, b = _leaf(2, 5, 8, gen=gen, offset=3.0), _leaf(8, gen=gen, offset=1.0), _leaf(8, gen=gen)
    dy = torch.randn(2, 5, 8, generator=gen, dtype=D64)
    y, cache = train.layernorm_forward(x.detach(), g.detach(), b.detach())
    ref = F.layer_norm(x, (8,), g, b, 1e-5)
    ref.backward(dy)
    torch.testing.assert_close(y, ref.detach())
    _check(train.layernorm_backward(dy, cache), (x.grad, g.grad, b.grad))


def test_rmsnorm(gen):
    x, g = _leaf(2, 5, 8, gen=gen, offset=1.0), _leaf(8, gen=gen, offset=1.0)
    dy = torch.randn(2, 5, 8, generator=gen, dtype=D64)
    y, cache = train.rmsnorm_forward(x.detach(), g.detach())
    ref = F.rms_norm(x, (8,), g, 1e-6)
    ref.backward(dy)
    torch.testing.assert_close(y, ref.detach())
    _check(train.rmsnorm_backward(dy, cache), (x.grad, g.grad))


def test_softmax(gen):
    s = _leaf(2, 3, 4, 7, gen=gen, scale=5.0)
    dp = torch.randn(2, 3, 4, 7, generator=gen, dtype=D64)
    p, cache = train.softmax_forward(s.detach())
    torch.softmax(s, dim=-1).backward(dp)
    torch.testing.assert_close(p, torch.softmax(s.detach(), -1))
    torch.testing.assert_close(train.softmax_backward(dp, cache), s.grad)


def test_attention_chain_matches_autograd(gen):
    # scores -> softmax -> values, with a causal mask and cached keys (offset 2), as the model runs it.
    q, k, v = _leaf(2, 3, 4, 8, gen=gen), _leaf(2, 3, 6, 8, gen=gen), _leaf(2, 3, 6, 8, gen=gen)
    mask = torch.arange(6)[None, :] <= torch.arange(4)[:, None] + 2
    d_out = torch.randn(2, 3, 4, 8, generator=gen, dtype=D64)

    s, c_s = train.attention_scores_forward(q.detach(), k.detach(), mask)
    p, c_p = train.softmax_forward(s)
    out, c_v = train.attention_values_forward(p, v.detach())
    d_probs, dv = train.attention_values_backward(d_out, c_v)
    dq, dk = train.attention_scores_backward(train.softmax_backward(d_probs, c_p), c_s)

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    ref.backward(d_out)
    torch.testing.assert_close(out, ref.detach())
    _check((dq, dk, dv), (q.grad, k.grad, v.grad))


@pytest.mark.parametrize("p", [0.0, 0.3])
def test_dropout(gen, p):
    x = torch.randn(1000, generator=gen, dtype=D64)
    y, cache = train.dropout_forward(x, p, generator=torch.Generator().manual_seed(1))
    dy = torch.randn(1000, generator=gen, dtype=D64)
    dx = train.dropout_backward(dy, cache)
    if p == 0.0:
        assert y is x and dx is dy
        return
    keep = cache["keep"]
    torch.testing.assert_close(y, torch.where(keep, x / (1 - p), torch.zeros_like(x)))
    torch.testing.assert_close(dx, torch.where(keep, dy / (1 - p), torch.zeros_like(dy)))
    assert abs(keep.double().mean().item() - (1 - p)) < 0.05


def test_cross_entropy_with_ignored_tokens(gen):
    logits = _leaf(2, 5, 11, gen=gen, scale=4.0)
    targets = torch.randint(0, 11, (2, 5), generator=gen)
    targets[0, 1] = targets[1, 4] = -100                      # not counted
    loss, cache = train.cross_entropy_forward(logits.detach(), targets)
    ref = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1), ignore_index=-100)
    ref.backward()
    torch.testing.assert_close(loss, ref.detach())
    torch.testing.assert_close(train.cross_entropy_backward(cache), logits.grad)


def test_cross_entropy_is_stable_for_large_logits():
    logits = torch.tensor([[[1000.0, 0.0, -1000.0]]], dtype=D64)
    loss, _ = train.cross_entropy_forward(logits, torch.tensor([[1]]))
    torch.testing.assert_close(loss, torch.tensor(1000.0, dtype=D64))


def test_clip_grad_norm_matches_torch(gen):
    grads = {"a": torch.randn(3, 4, generator=gen, dtype=D64), "b": torch.randn(5, generator=gen, dtype=D64)}
    tensors = [g.clone().requires_grad_(False) for g in grads.values()]
    leaves = [torch.zeros_like(t, requires_grad=True) for t in tensors]
    for leaf, t in zip(leaves, tensors):
        leaf.grad = t.clone()
    ref_norm = torch.nn.utils.clip_grad_norm_(leaves, max_norm=1.0)
    norm = train.clip_grad_norm(grads, max_norm=1.0)
    assert math.isclose(norm, ref_norm.item(), rel_tol=1e-12)
    for g, leaf in zip(grads.values(), leaves):
        torch.testing.assert_close(g, leaf.grad)


def test_adamw_matches_torch_optim_for_five_steps(gen):
    params = {"w": torch.randn(4, 3, generator=gen, dtype=D64), "gamma": torch.randn(3, generator=gen, dtype=D64)}
    ref_w = params["w"].clone().requires_grad_(True)
    ref_g = params["gamma"].clone().requires_grad_(True)
    # torch's AdamW with the same split: weight decay on matrices only.
    opt = torch.optim.AdamW([{"params": [ref_w], "weight_decay": 0.1}, {"params": [ref_g], "weight_decay": 0.0}],
                            lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    state: dict = {}
    for _ in range(5):
        grads = {"w": torch.randn(4, 3, generator=gen, dtype=D64), "gamma": torch.randn(3, generator=gen, dtype=D64)}
        ref_w.grad, ref_g.grad = grads["w"].clone(), grads["gamma"].clone()
        opt.step()
        train.adamw_step(params, grads, state, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    torch.testing.assert_close(params["w"], ref_w.detach())
    torch.testing.assert_close(params["gamma"], ref_g.detach())
    assert state["t"] == 5


def test_lr_schedule():
    kw = dict(max_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.1)
    assert train.lr_at(0, **kw) == pytest.approx(0.1)          # first warm-up step
    assert train.lr_at(9, **kw) == pytest.approx(1.0)          # end of warm-up
    assert train.lr_at(60, **kw) == pytest.approx(0.55)        # halfway through the cosine
    assert train.lr_at(110, **kw) == pytest.approx(0.1)        # end
    assert train.lr_at(500, **kw) == pytest.approx(0.1)        # stays at min_lr afterwards
