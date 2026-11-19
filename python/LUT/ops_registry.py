"""Turn the LUT host's toggles (its PARAMETERS block, or a named preset) into the functions its
wiring calls.

Start from every stage on its baseline function (baseline/host.py:base_ops, the same building
blocks the baseline model uses), then, for each stage
the toggles give a LUT architecture, swap in the same-named function from inference_lut.py (or the
<name>_forward / <name>_backward pair from train_lut.py in training), with the architecture bound
to it, in the layers the toggle names. Everything is checked here, at start-up, so a typo in the
toggles fails before a run starts rather than halfway through it.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from baseline import host as base
from LUT import inference_lut, train_lut

if TYPE_CHECKING:
    from LUT.stages import Stage


def check_toggles(toggles: dict[str, "Stage"], cfg: base.TransformerConfig) -> None:
    """Raise ValueError for an unknown stage, an unknown architecture or a layer out of range."""
    unknown = set(toggles) - set(base.STAGES)
    if unknown:
        raise ValueError(f"unknown stage(s) in toggles: {sorted(unknown)}; known: {list(base.STAGES)}")
    for stage, choice in toggles.items():
        if choice.lut is not None and choice.lut.name not in inference_lut.ARCHITECTURES:
            known = sorted(inference_lut.ARCHITECTURES) or "none yet (decision D5)"
            raise ValueError(f"stage {stage!r}: unknown LUT architecture {choice.lut.name!r}; known: {known}")
        if choice.layers is not None:
            bad = [i for i in choice.layers if not 0 <= i < cfg.n_layers]
            if bad:
                raise ValueError(f"stage {stage!r}: layers {bad} out of range for {cfg.n_layers} layers")


def lut_function(stage: str, choice: "Stage", cfg: base.TransformerConfig, mode: base.Mode) -> Any:
    """The LUT function for one stage, with its architecture bound as arch=...
    (or the bound (forward, backward) pair in train mode)."""
    name = base.function_name(stage, cfg)
    if mode == "inference":
        return functools.partial(_get(inference_lut, name, stage), arch=choice.lut)
    forward = _get(train_lut, f"{name}_forward", stage)
    backward = _get(train_lut, f"{name}_backward", stage)
    return (functools.partial(forward, arch=choice.lut), backward)


def resolve_ops(toggles: dict[str, "Stage"], cfg: base.TransformerConfig, mode: base.Mode) -> base.ModelOps:
    """ModelOps for LUT/host.py's wiring: every stage on its baseline function, then each toggled
    stage swapped in. Stages are independent; nothing here overrides a toggle."""
    check_toggles(toggles, cfg)
    ops = base.base_ops(cfg, mode)
    for stage, choice in toggles.items():
        if choice.quant is not None:
            raise NotImplementedError(f"stage {stage!r}: input quantization is written in window 2 (decision D4)")
        if choice.lut is None:
            continue
        fn = lut_function(stage, choice, cfg, mode)
        if stage in base.LAYER_STAGES:
            for i in (range(cfg.n_layers) if choice.layers is None else choice.layers):
                ops.layers[i][stage] = fn
        else:
            setattr(ops, stage, fn)
    return ops


def describe(toggles: dict[str, "Stage"], cfg: base.TransformerConfig) -> list[dict[str, str]]:
    """One row per stage (stages the toggles leave out run the baseline), for printing and for
    run metadata: what runs, the table shape and size, the layers, and the input quantization.
    A stage the model doesn't have (rope without rotary positions, ffn_gate without a gated
    feed-forward) says "not in this model"."""
    rows = []
    for stage in base.STAGES:
        choice = toggles.get(stage)
        lut = choice.lut if choice else None
        quant = choice.quant if choice else None
        layers = choice.layers if choice and choice.layers is not None else None
        if not base.stage_used(stage, cfg):
            runs = "not in this model"       # e.g. rope without rotary positions; the toggle does nothing
        else:
            runs = f"lut:{lut.name}" if lut else f"base:{base.function_name(stage, cfg)}"
        rows.append({
            "stage": stage,
            "runs": runs,
            "shape": f"{lut.inputs_per_table} x {lut.bits_per_input} -> {lut.out_bits}" if lut else "-",
            "table_bits": f"{lut.table_bits():,}" if lut else "-",
            "layers": ("-" if stage not in base.LAYER_STAGES
                       else "all" if layers is None else ",".join(map(str, layers))),
            "quant": f"int{quant.bits}" if quant else "float",
        })
    return rows


def _get(module: Any, name: str, stage: str) -> Any:
    fn = getattr(module, name, None)
    if fn is None:
        raise ValueError(f"stage {stage!r}: {module.__name__} has no function {name!r}")
    return fn
