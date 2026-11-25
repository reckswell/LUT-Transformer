"""The eval harness runs every arm on the same tokens and weights, and the controls hold:
  1. hf vs baseline on a random-init Hugging Face GPT-2 (skips without transformers): float64
     rtol/atol 1e-9, as in test_gpt2_profile.py (the same terms summed in a different order).
  2. lut "all_base" vs baseline: bit-identical (torch.equal), as in test_lut_wiring.py.
  3. one stage set through a PARAMETERS global: wired through, but no LUT function exists until
     D5, so for now it must fail loudly rather than run the baseline silently.

Float64 here checks the math against a reference; it is not the base metric. Default runs are
float32 with TF32 off (DTYPE, common/precision.py, tests/test_fp32_defaults.py).
"""

import math

import pytest

torch = pytest.importorskip("torch")

from baseline import host as base  # noqa: E402
from eval.harness import Weights, compare, next_token_loss, run_arm  # noqa: E402
from LUT import host as lut_host  # noqa: E402
from LUT import inference_lut  # noqa: E402
from LUT.stages import LUTArch, Stage  # noqa: E402
from test_model_against_reference import _setup as setup_models  # noqa: E402
from test_model_against_reference import _tokens as tokens  # noqa: E402

FAST = dict(repeats=2, warmup=1)


def test_core_arms_baseline_and_lut():
    cfg, params, _ = setup_models("core_like")
    weights, ids = Weights(cfg, params), tokens(cfg, batch=2, length=16)

    baseline = run_arm("baseline", weights, ids, **FAST)
    arm2 = run_arm("lut", weights, ids, preset="all_base", **FAST)
    defaults = run_arm("lut", weights, ids, **FAST)        # the PARAMETERS block, all BASE today

    assert baseline.logits.shape == (2, 16, cfg.vocab_size)
    assert torch.equal(arm2.logits, baseline.logits) and torch.equal(defaults.logits, baseline.logits)
    assert compare(arm2, baseline) == {"max_abs_logit_diff": 0.0, "top1_agreement": 1.0, "perplexity_ratio": 1.0}
    assert baseline.timing["repeats"] == 2 and baseline.timing["median_ms"] > 0
    assert baseline.modeled_cost is None and baseline.preset is None
    assert [row["stage"] for row in arm2.modeled_cost] == list(base.STAGES)
    runs = {row["stage"]: row["runs"] for row in arm2.modeled_cost}
    assert {stage for stage, what in runs.items() if what == "not in this model"} == {"rope", "ffn_gate"}
    assert all(what.startswith("base:") for stage, what in runs.items() if stage not in ("rope", "ffn_gate"))
    with pytest.raises(ValueError, match="hf arm needs"):
        run_arm("hf", weights, ids, repeats=0)


def test_next_token_loss_by_hand():
    # Uniform logits over V tokens: every prediction has probability 1/V, so loss = ln V.
    logits, ids = torch.zeros(2, 5, 7), torch.randint(0, 7, (2, 5))
    assert next_token_loss(logits, ids) == pytest.approx(math.log(7), rel=1e-12)
    # A confident correct prediction at every position: loss near 0.
    sure = torch.full((1, 4, 3), -50.0)
    ids = torch.tensor([[0, 1, 2, 1]])
    for t in range(3):
        sure[0, t, ids[0, t + 1]] = 50.0
    assert next_token_loss(sure, ids) < 1e-30


def test_gpt2_arms_hf_baseline_and_lut():
    pytest.importorskip("transformers")
    from test_gpt2_profile import _small_hf_gpt2

    weights = Weights.from_hf(_small_hf_gpt2(seed=4), dtype=torch.float64)   # float64: checks the math
    ids = tokens(weights.cfg, batch=2, length=20, seed=9)

    hf = run_arm("hf", weights, ids, **FAST)
    baseline = run_arm("baseline", weights, ids, **FAST)
    arm2 = run_arm("lut", weights, ids, preset="all_base", **FAST)

    torch.testing.assert_close(baseline.logits, hf.logits, rtol=1e-9, atol=1e-9)        # control 1
    assert torch.equal(arm2.logits, baseline.logits)                                     # control 2
    assert compare(baseline, hf)["top1_agreement"] == 1.0
    assert baseline.perplexity == pytest.approx(hf.perplexity, rel=1e-9)


def test_one_stage_through_a_parameter_global_waits_for_d5(monkeypatch):
    # Control 3's mechanism: set one stage like an SV parameter override and run the lut arm.
    # No LUT function exists until D5, so it must raise, not quietly run the baseline.
    cfg, params, _ = setup_models("core_like")
    monkeypatch.setitem(inference_lut.ARCHITECTURES, "dummy", "test stand-in")
    monkeypatch.setattr(lut_host, "SOFTMAX", Stage(lut=LUTArch("dummy", 2, 4, 16, 24)))
    with pytest.raises(NotImplementedError, match="D5"):
        run_arm("lut", Weights(cfg, params), tokens(cfg, 1, 8), repeats=0)
