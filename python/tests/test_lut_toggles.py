"""The LUT host's toggles pick the right function for every stage and layer, and the table-size
and integer-range arithmetic is right. A dummy architecture stands in for the real LUT design
(decision D5) so the switching can be tested before any LUT function exists."""

import pytest

torch = pytest.importorskip("torch")

from baseline import host as base  # noqa: E402
from baseline import inference, train  # noqa: E402
from common.quant import QuantSpec, accumulator_bits, int_range  # noqa: E402
from LUT import host as lut_host  # noqa: E402
from LUT import inference_lut, ops_registry, train_lut  # noqa: E402
from LUT.host import BASE, LUTArch, Stage  # noqa: E402
from LUT.inference_lut import LUTCost, lut_entries, lut_size_bits  # noqa: E402

CFG = base.TransformerConfig(n_layers=4, norm="layernorm", activation="relu")
DUMMY = LUTArch("dummy", inputs_per_table=2, bits_per_input=4, out_bits=16, acc_bits=24)


@pytest.fixture
def dummy_arch(monkeypatch):
    monkeypatch.setitem(inference_lut.ARCHITECTURES, "dummy", "test stand-in")


def _is_lut(fn, lut_fn):
    """True if fn is lut_fn with the dummy architecture bound by the registry."""
    return getattr(fn, "func", None) is lut_fn and fn.keywords.get("arch") is DUMMY


def test_all_base_matches_the_baseline_model():
    for preset in ("all_base", "parameters"):
        for mode in ("inference", "train"):
            assert lut_host.lut_ops(CFG, mode, lut_host.preset_toggles(preset)) == base.base_ops(CFG, mode)
    assert lut_host.lut_ops(CFG, "inference") == base.base_ops(CFG, "inference")   # the PARAMETERS defaults
    ops = base.base_ops(CFG, "inference")
    assert ops.layers[0]["qkv_proj"] is inference.linear and ops.layers[3]["ffn_act"] is inference.relu
    assert ops.final_norm is inference.layernorm and ops.lm_head is inference.linear
    assert base.base_ops(CFG, "train").layers[1]["softmax"] == (train.softmax_forward, train.softmax_backward)


def test_lut_toggle_switches_only_that_stage_and_those_layers(dummy_arch):
    ops = lut_host.lut_ops(CFG, "inference", {"attn_values": Stage(lut=DUMMY, layers=(1, 3))})
    assert [_is_lut(layer["attn_values"], inference_lut.attention_values) for layer in ops.layers] == [False, True, False, True]
    assert ops.layers[0]["attn_values"] is inference.attention_values
    assert all(layer["attn_scores"] is inference.attention_scores for layer in ops.layers)


def test_lut_toggle_on_a_stage_outside_the_layers(dummy_arch):
    ops = lut_host.lut_ops(CFG, "inference", {"lm_head": Stage(lut=DUMMY)})
    assert _is_lut(ops.lm_head, inference_lut.linear)
    assert all(layer["qkv_proj"] is inference.linear for layer in ops.layers)


def test_training_ignores_luts_unless_train_with_luts(dummy_arch):
    toggles = {"softmax": Stage(lut=DUMMY)}
    assert lut_host.lut_ops(CFG, "train", toggles, train_with_luts=False) == base.base_ops(CFG, "train")
    forward, backward = lut_host.lut_ops(CFG, "train", toggles, train_with_luts=True).layers[0]["softmax"]
    assert _is_lut(forward, train_lut.softmax_forward) and backward is train_lut.softmax_backward
    with pytest.raises(ValueError, match="has no function 'embedding_forward'"):
        lut_host.lut_ops(CFG, "train", {"embedding": Stage(lut=DUMMY)}, train_with_luts=True)


def test_model_config_picks_norm_and_activation():
    cfg = base.TransformerConfig(norm="rmsnorm", activation="gelu")
    ops = lut_host.lut_ops(cfg, "inference", {})
    assert ops.layers[0]["norm"] is inference.rmsnorm and ops.layers[0]["ffn_act"] is inference.gelu
    assert ops.final_norm is inference.rmsnorm


def test_bad_toggles_fail_at_startup(dummy_arch):
    with pytest.raises(ValueError, match="unknown stage"):
        ops_registry.resolve_ops({"atention": BASE}, CFG, "inference")
    with pytest.raises(ValueError, match="unknown LUT architecture"):
        ops_registry.resolve_ops({"softmax": Stage(lut=LUTArch("typo", 1, 8, 16))}, CFG, "inference")
    with pytest.raises(ValueError, match="out of range"):
        ops_registry.resolve_ops({"softmax": Stage(lut=DUMMY, layers=(4,))}, CFG, "inference")
    with pytest.raises(ValueError, match="needs calibrated ranges"):    # a static scale with no CALIBRATION
        ops_registry.resolve_ops({"qkv_proj": Stage(quant=QuantSpec(bits=8))}, CFG, "inference")


def test_toggles_list_every_stage_and_describe_reports_table_size(dummy_arch):
    assert set(lut_host.parameter_toggles()) == set(base.STAGES) | {"kv_cache"}
    rows = {row["stage"]: row for row in ops_registry.describe({"ffn_up": Stage(lut=DUMMY, layers=(0,))}, CFG)}
    assert rows["ffn_up"]["runs"] == "lut:dummy" and rows["ffn_up"]["table_bits"] == "4,096"
    assert rows["ffn_up"]["layers"] == "0" and rows["ffn_down"]["runs"] == "base:linear"
    assert rows["embedding"]["layers"] == "-"


def test_repository_configs_load():
    cfg, hw = base.load_configs(base.PYTHON_DIR / "configs/model/core.yaml", base.PYTHON_DIR / "configs/hw/gpu.yaml")
    assert cfg.head_dim * cfg.n_heads == cfg.d_model and hw["devices"]["decode"] == "cuda"


def test_lut_size_grows_exponentially():
    assert lut_entries(1, 4) == 16
    assert lut_entries(2, 4) == 256
    assert lut_size_bits(2, 4, 16) == 4096 == DUMMY.table_bits()
    cost = LUTCost()
    cost.add_table(2, 4, 16, count=3)
    assert cost.tables == 3 and cost.table_bits == 3 * 4096


def test_integer_ranges_and_accumulators():
    assert int_range(8, signed=True) == (-128, 127)
    assert int_range(4, signed=False) == (0, 15)
    assert accumulator_bits(8, 8, 256) == 24
    assert accumulator_bits(8, 8, 1) == 16
