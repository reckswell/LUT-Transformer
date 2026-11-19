"""Control arm 2: the LUT model's own wiring (LUT/host.py) with every stage on BASE, against the
baseline model (baseline/host.py).

LUT/host.py has its own copy of the order of operations (decision D9), so it can be restructured
later. While the copy is unchanged, the same baseline functions run in the same order on the same
weights, so every result must be bit-identical (torch.equal, no tolerance): logits, KV-cache
decode, greedy generation, gradients and parameters after training steps, for every architecture
variant. This is the guard on the copy: any drift between the two wirings fails here.

When the LUT wiring is restructured on purpose (for example a reordered sum after unrolling), the
float rounding changes while the math does not. Then set WIRING_MATCH to a tolerance dict and say
in WIRING_MATCH_REASON what changed and why that tolerance is right.

The other tests here check the PARAMETERS block (its globals are read when lut_ops runs, one
stage changes without touching the others) and that LUT/host.py takes nothing from baseline/
that is wiring.

Float64 here checks the math against a reference; it is not the base metric. Default runs are
float32 with TF32 off (DTYPE, common/precision.py, tests/test_fp32_defaults.py).
"""

import ast
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from baseline import host as base  # noqa: E402
from baseline import inference  # noqa: E402
from LUT import host as lut_host  # noqa: E402
from LUT import inference_lut  # noqa: E402
from LUT.stages import BASE, LUTArch, Stage  # noqa: E402
from test_model_against_reference import VARIANTS  # noqa: E402
from test_model_against_reference import _setup as setup_models  # noqa: E402
from test_model_against_reference import _tokens as tokens  # noqa: E402

# How LUT wiring with every stage on BASE must compare with the baseline wiring:
#   "exact"            torch.equal (bit-identical)
#   dict(rtol=, atol=) torch.testing.assert_close with those tolerances
WIRING_MATCH: str | dict = "exact"
WIRING_MATCH_REASON = "LUT/host.py's wiring is a line-for-line copy of baseline/host.py's (2026-11-19)"

LUT_HOST_FILE = Path(lut_host.__file__)


def assert_same(lut_value, base_value, what: str) -> None:
    """Compare a LUT-wiring result with the baseline's under WIRING_MATCH."""
    if WIRING_MATCH == "exact":
        assert torch.equal(lut_value, base_value), f"{what}: LUT wiring differs from baseline ({WIRING_MATCH_REASON})"
    else:
        torch.testing.assert_close(lut_value, base_value, **WIRING_MATCH, msg=lambda m: f"{what}: {m} ({WIRING_MATCH_REASON})")


def _arm2_ops(cfg, mode):
    return lut_host.lut_ops(cfg, mode, lut_host.PRESETS["all_base"])


# ----------------------------------------------------------------------------------------------
# Bit-identical results, every variant
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_forward_prefill_decode_and_generate_match_baseline(variant):
    cfg, params, _ = setup_models(variant)
    ids = tokens(cfg, batch=2, length=16)
    b_ops, l_ops = base.base_ops(cfg, "inference"), _arm2_ops(cfg, "inference")

    assert_same(lut_host.forward(params, ids, l_ops, cfg), base.forward(params, ids, b_ops, cfg), "forward logits")

    b_cache, l_cache = base.new_cache(params, cfg, 2, 16), lut_host.new_cache(params, cfg, 2, 16)
    assert_same(lut_host.prefill(params, ids[:, :10], l_cache, l_ops, cfg),
                base.prefill(params, ids[:, :10], b_cache, b_ops, cfg), "prefill logits")
    for t in range(10, 16):
        assert_same(lut_host.decode_step(params, ids[:, t:t + 1], l_cache, l_ops, cfg),
                    base.decode_step(params, ids[:, t:t + 1], b_cache, b_ops, cfg), f"decode logits at {t}")
    assert l_cache.length == b_cache.length == 16
    for i in range(cfg.n_layers):
        assert_same(l_cache.keys[i], b_cache.keys[i], f"cached keys, layer {i}")
        assert_same(l_cache.values[i], b_cache.values[i], f"cached values, layer {i}")

    prompt = ids[:, :5]
    assert torch.equal(lut_host.generate(params, prompt, 6, l_ops, cfg, temperature=0),
                       base.generate(params, prompt, 6, b_ops, cfg, temperature=0))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_gradients_and_train_steps_match_baseline(variant):
    cfg, params, _ = setup_models(variant)
    ids, targets = tokens(cfg, 3, 20, seed=5), tokens(cfg, 3, 20, seed=6)
    b_ops, l_ops = base.base_ops(cfg, "train"), _arm2_ops(cfg, "train")

    b_logits, b_caches = base.forward_train(params, ids, b_ops, cfg)
    l_logits, l_caches = lut_host.forward_train(params, ids, l_ops, cfg)
    assert_same(l_logits, b_logits, "training logits")
    _, loss_cache = base.train.cross_entropy_forward(b_logits, targets)
    d_logits = base.train.cross_entropy_backward(loss_cache)
    b_grads = base.backward(d_logits, b_caches, b_ops, cfg)
    l_grads = lut_host.backward(d_logits, l_caches, l_ops, cfg)
    assert l_grads.keys() == b_grads.keys()
    for name in b_grads:
        assert_same(l_grads[name], b_grads[name], f"gradient {name}")

    # Two optimizer steps from the same start, each model updating its own copy of the weights.
    b_params = {k: v.clone() for k, v in params.items()}
    l_params = {k: v.clone() for k, v in params.items()}
    b_state, l_state = {}, {}
    for step in range(2):
        batch = (tokens(cfg, 2, 16, seed=10 + step), tokens(cfg, 2, 16, seed=20 + step))
        b_metrics = base.train_step(b_params, batch, b_ops, b_state, cfg, lr=3e-3, max_grad_norm=0.5)
        l_metrics = lut_host.train_step(l_params, batch, l_ops, l_state, cfg, lr=3e-3, max_grad_norm=0.5)
        assert l_metrics == b_metrics
    for name in b_params:
        assert_same(l_params[name], b_params[name], f"parameter {name} after 2 steps")


def test_float32_forward_matches_baseline():
    cfg, params, _ = setup_models("core_like", dtype=torch.float32)
    ids = tokens(cfg, batch=2, length=16)
    assert_same(lut_host.forward(params, ids, _arm2_ops(cfg, "inference"), cfg),
                base.forward(params, ids, base.base_ops(cfg, "inference"), cfg), "float32 forward logits")


# ----------------------------------------------------------------------------------------------
# The PARAMETERS block
# ----------------------------------------------------------------------------------------------

DUMMY = LUTArch("dummy", inputs_per_table=2, bits_per_input=4, out_bits=16, acc_bits=24)


@pytest.fixture
def dummy_arch(monkeypatch):
    monkeypatch.setitem(inference_lut.ARCHITECTURES, "dummy", "test stand-in")


def test_default_parameters_are_the_baseline_building_blocks():
    # The committed PARAMETERS block (every stage BASE) gives exactly the baseline functions and
    # bit-identical logits: the defaults are control arm 2.
    cfg, params, _ = setup_models("gpt2_like")
    assert all(stage == BASE for stage in lut_host.parameter_toggles().values())
    ops = lut_host.lut_ops(cfg, "inference")
    assert ops == base.base_ops(cfg, "inference")
    assert ops.layers[0]["softmax"] is inference.softmax and ops.lm_head is inference.linear
    ids = tokens(cfg, batch=2, length=12)
    assert_same(lut_host.forward(params, ids, ops, cfg),
                base.forward(params, ids, base.base_ops(cfg, "inference"), cfg), "default-parameter logits")


def test_changing_one_global_changes_only_that_stage(monkeypatch, dummy_arch):
    cfg = replace(base.TransformerConfig(vocab_size=37), n_layers=3)
    monkeypatch.setattr(lut_host, "SOFTMAX", Stage(lut=DUMMY, layers=(0, 2)))   # like #(.SOFTMAX(...)) in SV
    ops = lut_host.lut_ops(cfg, "inference")                                      # globals read now
    baseline = base.base_ops(cfg, "inference")
    for i, layer in enumerate(ops.layers):
        for stage, fn in layer.items():
            if stage == "softmax" and i in (0, 2):
                assert fn.func is inference_lut.softmax and fn.keywords["arch"] is DUMMY
            else:
                assert fn is baseline.layers[i][stage], (i, stage)
    assert (ops.embedding, ops.final_norm, ops.lm_head) == (baseline.embedding, baseline.final_norm, baseline.lm_head)
    assert lut_host.PRESETS["all_base"]["softmax"] == BASE                        # presets are untouched


def test_parameters_block_comes_right_after_the_imports():
    # Like a SystemVerilog parameter list: only the docstring and imports sit above it.
    tree = ast.parse(LUT_HOST_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DTYPE" for t in node.targets):
            break
        is_docstring = isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
        assert is_docstring or isinstance(node, (ast.Import, ast.ImportFrom)), f"line {node.lineno} sits above PARAMETERS"
    else:
        pytest.fail("no DTYPE assignment: the PARAMETERS block is missing")
    stage_globals = {stage.upper() for stage in base.STAGES}
    assert stage_globals <= set(vars(lut_host)), stage_globals - set(vars(lut_host))


# ----------------------------------------------------------------------------------------------
# The LUT wiring is its own: nothing from baseline/ that is wiring
# ----------------------------------------------------------------------------------------------

# Names LUT/host.py may use from baseline/, matching the reasons written at its imports.
SHARED_FROM_BASELINE = {
    "base": {"TransformerConfig", "Mode", "ModelOps", "STAGES", "load_configs", "add_common_args", "init_params"},
    "inference": {"KVCache", "split_heads", "merge_heads", "causal_mask", "sinusoidal_positions",
                  "learned_positions", "rope_tables", "sample_next_token"},
    "train": {"cross_entropy_forward", "cross_entropy_backward", "clip_grad_norm", "adamw_step"},
}
WIRING = ("forward", "new_cache", "prefill", "decode_step", "generate", "forward_train", "backward", "train_step")


def test_lut_host_uses_only_shared_non_wiring_pieces_from_baseline():
    tree = ast.parse(LUT_HOST_FILE.read_text())
    used: dict[str, set[str]] = {name: set() for name in SHARED_FROM_BASELINE}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in used:
            used[node.value.id].add(node.attr)
    for module, names in used.items():
        extra = names - SHARED_FROM_BASELINE[module]
        assert not extra, f"LUT/host.py uses {module}.{sorted(extra)} from baseline/; wiring belongs in LUT/host.py"


def test_lut_wiring_functions_are_defined_in_lut_host():
    for name in WIRING:
        fn = getattr(lut_host, name)
        assert fn.__module__ == "LUT.host" and fn is not getattr(base, name), name
