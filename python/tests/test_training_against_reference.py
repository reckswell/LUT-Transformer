"""Model-level training (host.forward_train, host.backward, host.train_step) against the torch.nn
reference trained with autograd and torch.optim.

Float64 throughout, with rtol 1e-9 / atol 1e-9 as in test_model_against_reference.py: the two
sides add the same terms in a different order (about 1e-15 per operation), so anything larger
is a wrong gradient.
"""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from baseline import host  # noqa: E402
from reference import torch_model as ref  # noqa: E402
from test_model_against_reference import VARIANTS  # noqa: E402
from test_model_against_reference import _setup as setup_models  # noqa: E402
from test_model_against_reference import _tokens as tokens  # noqa: E402

TIGHT = dict(rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_gradients_match_autograd_on_the_reference(variant):
    cfg, params, model = setup_models(variant)
    ids, targets = tokens(cfg, 3, 20, seed=5), tokens(cfg, 3, 20, seed=6)
    targets[0, :4] = -100                                      # some ignored tokens
    ops = host.base_ops(cfg, "train")

    logits, caches = host.forward_train(params, ids, ops, cfg)
    loss, loss_cache = host.train.cross_entropy_forward(logits, targets)
    grads = host.backward(host.train.cross_entropy_backward(loss_cache), caches, ops, cfg)

    ref_logits = model(ids)
    ref_loss = F.cross_entropy(ref_logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
    ref_loss.backward()
    torch.testing.assert_close(logits, ref_logits.detach(), **TIGHT)
    torch.testing.assert_close(loss, ref_loss.detach(), **TIGHT)
    named = model.scratch_params()
    assert grads.keys() == named.keys()
    for name, param in named.items():
        torch.testing.assert_close(grads[name], param.grad, **TIGHT, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize("variant", ["gpt2_like", "core_like"])
def test_three_train_steps_match_torch_adamw(variant):
    cfg, params, model = setup_models(variant)
    ops = host.base_ops(cfg, "train")
    named = model.scratch_params()
    # Same weight-decay split as train.default_decay: matrices and embeddings only.
    decay = [p for p in named.values() if p.dim() >= 2]
    no_decay = [p for p in named.values() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
    state: dict = {}
    for step in range(3):
        batch = (tokens(cfg, 2, 16, seed=10 + step), tokens(cfg, 2, 16, seed=20 + step))
        metrics = host.train_step(params, batch, ops, state, cfg, lr=3e-3, max_grad_norm=0.5)
        opt.zero_grad()
        loss = F.cross_entropy(model(batch[0]).reshape(-1, cfg.vocab_size), batch[1].reshape(-1))
        loss.backward()
        ref_norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 0.5)
        opt.step()
        assert metrics["loss"] == pytest.approx(loss.item(), rel=1e-9)
        assert metrics["grad_norm"] == pytest.approx(ref_norm.item(), rel=1e-9)
    for name, param in named.items():
        torch.testing.assert_close(params[name], param.detach(), **TIGHT, msg=lambda m, n=name: f"{n}: {m}")


def test_training_overfits_one_batch():
    # A sanity check that the pieces learn: 60 steps on one batch should cut the loss a lot.
    cfg, params, _ = setup_models("core_like", dtype=torch.float32)
    ops = host.base_ops(cfg, "train")
    batch = (tokens(cfg, 4, 16, seed=1), tokens(cfg, 4, 16, seed=2))
    state: dict = {}
    first = host.train_step(params, batch, ops, state, cfg, lr=1e-2)["loss"]
    for _ in range(59):
        last = host.train_step(params, batch, ops, state, cfg, lr=1e-2)["loss"]
    assert last < 0.3 * first
