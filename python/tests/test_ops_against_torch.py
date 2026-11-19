"""From-scratch ops vs PyTorch. Each test skips until its function is written (window 1), then
becomes the check that the function is right. Forward is compared to torch.nn.functional in
float32; backward is compared to autograd in float64."""

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
