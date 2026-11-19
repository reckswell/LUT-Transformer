"""Float32 is the base metric (Alpha, 2026-11-19): every default run path computes in plain
float32, with TF32 off, and nothing is quantized until a PARAMETERS line asks for it.

Checked here for each arm's default path: the parameters, the KV cache, the logits (and, for
training, the gradients and updated parameters) are float32; token ids are int64 indices; the
eval harness records the dtype and the TF32 flags; the model code never uses half precision or
autocast. The other test files compare against references in float64 on purpose: that checks the
math (rounding stays near 1e-15, so a 1e-9 tolerance catches real errors) and is not the base
metric.
"""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from baseline import host as base  # noqa: E402
from bench import runlog  # noqa: E402
from common import precision  # noqa: E402
from eval.harness import Weights, run_arm  # noqa: E402
from LUT import host as lut_host  # noqa: E402

PYTHON_DIR = Path(__file__).resolve().parents[1]
F32 = torch.float32
CFG = base.TransformerConfig(vocab_size=37, context_length=24, d_model=32, n_heads=4, n_layers=2, d_ff=64)


def _ids(batch=2, length=12, seed=0):
    return torch.randint(0, CFG.vocab_size, (batch, length), generator=torch.Generator().manual_seed(seed))


def _all_f32(tensors) -> bool:
    return all(t.dtype == F32 for t in tensors)


def test_dtype_settings_are_float32():
    assert base.DTYPE == F32 and lut_host.DTYPE == F32
    assert all(stage.quant is None and stage.lut is None for stage in lut_host.parameter_toggles().values())


def test_baseline_default_run_is_float32():
    params = base.init_params(CFG, seed=0)
    assert _all_f32(params.values())
    ops, ids = base.base_ops(CFG, "inference"), _ids()
    assert ids.dtype == torch.int64
    assert base.forward(params, ids, ops, CFG).dtype == F32
    cache = base.new_cache(params, CFG, batch=2, max_len=13)
    assert base.prefill(params, ids, cache, ops, CFG).dtype == F32
    assert base.decode_step(params, ids[:, :1], cache, ops, CFG).dtype == F32
    assert _all_f32(cache.keys + cache.values)
    # Training keeps everything float32: gradients and the updated parameters.
    train_ops = base.base_ops(CFG, "train")
    logits, caches = base.forward_train(params, ids, train_ops, CFG)
    _, loss_cache = base.train.cross_entropy_forward(logits, _ids(seed=1))
    grads = base.backward(base.train.cross_entropy_backward(loss_cache), caches, train_ops, CFG)
    assert logits.dtype == F32 and _all_f32(grads.values())
    base.train_step(params, (ids, _ids(seed=1)), train_ops, {}, CFG, lr=1e-3)
    assert _all_f32(params.values())


def test_lut_default_run_is_float32():
    params = lut_host.init_params(CFG, seed=0)
    assert _all_f32(params.values())
    assert all(torch.equal(params[k], v) for k, v in base.init_params(CFG, seed=0).items())
    ops, ids = lut_host.lut_ops(CFG, "inference"), _ids()
    assert lut_host.forward(params, ids, ops, CFG).dtype == F32
    cache = lut_host.new_cache(params, CFG, batch=2, max_len=12)
    assert lut_host.prefill(params, ids, cache, ops, CFG).dtype == F32
    assert _all_f32(cache.keys + cache.values)


def test_eval_arms_run_float32_with_tf32_off():
    weights = Weights(CFG, base.init_params(CFG, seed=0))
    for arm, preset in (("baseline", "parameters"), ("lut", "parameters"), ("lut", "all_base")):
        result = run_arm(arm, weights, _ids(), preset=preset, repeats=0)
        assert result.dtype == "torch.float32" and result.logits.dtype == F32
        assert result.precision["float32_matmul_precision"] == "highest"
        assert result.precision["cuda_matmul_allow_tf32"] is False and result.precision["cudnn_allow_tf32"] is False
    with pytest.raises(TypeError, match="int64"):
        run_arm("baseline", weights, _ids().float(), repeats=0)


def test_full_fp32_turns_tf32_off_and_metadata_records_it():
    saved = precision.precision_info()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        info = precision.full_fp32()
        assert info == {"default_dtype": "torch.float32", "float32_matmul_precision": "highest",
                        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}
        recorded = runlog.system_info()["torch"]
        assert {k: recorded[k] for k in info} == info
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
        torch.set_float32_matmul_precision(saved["float32_matmul_precision"])


def test_model_code_never_uses_half_precision_or_autocast():
    forbidden = {"float16", "half", "bfloat16", "autocast", "amp"}
    problems = []
    for folder in ("baseline", "LUT", "reference", "eval", "common"):
        for path in sorted((PYTHON_DIR / folder).glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
                if name in forbidden:
                    problems.append(f"{path.relative_to(PYTHON_DIR)}:{node.lineno}: {name}")
    assert not problems, problems


def test_hf_weights_default_to_float32(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from reference import hf_gpt2
    from test_gpt2_profile import _small_hf_gpt2

    weights = Weights.from_hf(_small_hf_gpt2(seed=1))          # built in float64, converted by default
    assert weights.dtype == F32
    result = run_arm("hf", weights, torch.randint(0, 37, (2, 10)), repeats=0)
    assert result.dtype == "torch.float32" and result.logits.dtype == F32

    # load_gpt2 asks transformers for float32 explicitly (transformers 5 defaults to the
    # checkpoint's own dtype), and converts if it gets something else back.
    seen = {}

    def fake_from_pretrained(name, **kwargs):
        seen.update(kwargs)
        return _small_hf_gpt2(seed=2)                           # float64, as if "auto" had been used

    monkeypatch.setattr(transformers.GPT2LMHeadModel, "from_pretrained", staticmethod(fake_from_pretrained))
    _, params, hf_model = hf_gpt2.load_gpt2()
    assert seen["dtype"] == F32
    assert _all_f32(params.values()) and _all_f32(hf_model.parameters())
