"""LUT transformer host: the baseline transformer, with toggles for what each stage does with LUTs.

Read the TOGGLES block first. It lists every stage of the transformer and says, for each one,
whether it runs the baseline function (baseline/inference.py, baseline/train.py) or a LUT
architecture from LUT/inference_lut.py, with which table shape, in which decoder layers, and with
what input quantization. To change what the LUT model does, edit a line there (or add a named set
of toggles to PRESETS); nothing else in the code needs to change.

The order of operations is not copied here. The LUT model runs baseline/host.py's composition
(forward_train, prefill, decode_step, generate) with the functions these toggles choose. That keeps
the LUT and baseline models identical except for the toggled stages, so a difference in accuracy,
speed or modeled cost comes from the LUTs and not from a drifted copy of the model. With every
stage on BASE, the LUT model must match baseline/host.py exactly (a test checks this).

    cd python
    python -m LUT.host info                       # print the toggles and check every function exists
    python -m LUT.host info --preset all_base     # a named set of toggles from PRESETS
    python -m LUT.host train | evaluate | generate  ...   (window 1 and 2)

In hardware each toggle is a different circuit, so an HLS port turns these into compile-time
constants rather than run-time switches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from baseline import host as base
from common.quant import QuantSpec
from LUT import inference_lut, ops_registry


# ----------------------------------------------------------------------------------------------
# What a toggle can say
# ----------------------------------------------------------------------------------------------

@dataclass
class LUTArch:
    """One LUT architecture and the shape of its tables.

    name              which architecture: a key of LUT/inference_lut.ARCHITECTURES
    inputs_per_table  k, how many values together index one table
    bits_per_input    b, the bit width of each of those values
    out_bits          the width of each table entry
    acc_bits          the width of the integer accumulator that sums table outputs (None: no sum)
    options           anything specific to this architecture, passed to it unchanged

    One table has 2^(k*b) entries of out_bits bits, so 2^(k*b) * out_bits bits of storage.
    For example k = 2, b = 4, out_bits = 16 gives 256 entries and 4,096 bits.
    """

    name: str
    inputs_per_table: int
    bits_per_input: int
    out_bits: int
    acc_bits: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def entries(self) -> int:
        return inference_lut.lut_entries(self.inputs_per_table, self.bits_per_input)

    def table_bits(self) -> int:
        return inference_lut.lut_size_bits(self.inputs_per_table, self.bits_per_input, self.out_bits)


@dataclass
class Stage:
    """What one stage does.

    lut     None runs the baseline function. A LUTArch runs that LUT architecture instead.
    quant   None keeps float inputs. A QuantSpec quantizes this stage's inputs first, with or
            without a LUT, so quantization and LUTs can sit at different stages (decision D4).
    layers  the decoder layers the choice applies to, counted from 0 (None: all layers). Other
            layers run the baseline. Ignored for embedding, final_norm and lm_head.
    """

    lut: LUTArch | None = None
    quant: QuantSpec | None = None
    layers: tuple[int, ...] | None = None


BASE = Stage()   # the baseline function, float inputs, every layer


# ----------------------------------------------------------------------------------------------
# TOGGLES: edit these
# ----------------------------------------------------------------------------------------------
#
# Every stage is listed. LUT/inference_lut.ARCHITECTURES is empty until the LUT design (decision
# D5) is implemented, so every stage is on BASE for now. To switch one stage, replace BASE, for
# example (the architecture name is a placeholder until D5):
#
#   "attn_values": Stage(lut=LUTArch("<architecture>", inputs_per_table=2, bits_per_input=4,
#                                    out_bits=16, acc_bits=24), layers=(1, 2, 3)),

TOGGLES: dict[str, Stage] = {
    # Inside every decoder layer (all layers unless `layers` says otherwise)
    "norm":        BASE,   # LayerNorm or RMSNorm, both places in the block
    "qkv_proj":    BASE,   # Q, K, V projections: fixed weights after training
    "attn_scores": BASE,   # QK^T / sqrt(Dh): both operands are made at run time
    "softmax":     BASE,   # exp, sum and divide per score
    "attn_values": BASE,   # PV: both operands are made at run time
    "out_proj":    BASE,
    "ffn_up":      BASE,
    "ffn_act":     BASE,   # ReLU or GELU
    "ffn_down":    BASE,
    # Outside the layers
    "embedding":   BASE,   # already a lookup into the token table
    "final_norm":  BASE,
    "lm_head":     BASE,   # the output layer: most of a small model's per-token compute
}

# False: train the float baseline, then build the tables from the trained weights for inference
# (post-training, the plan's default). True: LUT-aware training, where the forward pass runs the
# LUT stages from LUT/train_lut.py with surrogate gradients. An open question for Alpha; the plan
# cuts it first if time runs short.
TRAIN_WITH_LUTS = False

# Named sets of toggles, for sweeps over LUT architectures. Pick one with --preset.
PRESETS: dict[str, dict[str, Stage]] = {
    "toggles": TOGGLES,
    "all_base": {stage: BASE for stage in base.STAGES},
}


# ----------------------------------------------------------------------------------------------
# Building the model's functions from the toggles
# ----------------------------------------------------------------------------------------------

def lut_ops(cfg: base.TransformerConfig, mode: base.Mode, toggles: dict[str, Stage] | None = None,
            *, train_with_luts: bool | None = None) -> base.ModelOps:
    """The functions baseline/host.py's composition calls, with the toggled stages swapped in.

    In "train" mode with train_with_luts False, every stage trains on its baseline function and
    the LUTs are applied afterwards, for inference only.
    """
    toggles = TOGGLES if toggles is None else toggles
    train_with_luts = TRAIN_WITH_LUTS if train_with_luts is None else train_with_luts
    if mode == "train" and not train_with_luts:
        toggles = PRESETS["all_base"]
    return ops_registry.resolve_ops(toggles, cfg, mode)


# The composition itself is baseline/host.py's, run with lut_ops(...) in place of base_ops(...).
forward_train = base.forward_train
backward = base.backward
train_step = base.train_step
prefill = base.prefill
decode_step = base.decode_step
generate = base.generate


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

def _cmd_info(cfg: base.TransformerConfig, hw: dict, preset: str) -> None:
    toggles = PRESETS[preset]
    print(f"model   {cfg.name}: n_layers {cfg.n_layers}, d_model {cfg.d_model}, norm {cfg.norm}, activation {cfg.activation}")
    print(f"hw      {hw.get('name', '?')}: {hw['devices']}")
    print(f"preset  {preset}    train_with_luts: {TRAIN_WITH_LUTS}")
    print()
    print(f"{'stage':<13}{'runs':<26}{'k x b -> out':<14}{'table bits':>11}  {'layers':<8}quant")
    for row in ops_registry.describe(toggles, cfg):
        print(f"{row['stage']:<13}{row['runs']:<26}{row['shape']:<14}{row['table_bits']:>11}  {row['layers']:<8}{row['quant']}")
    for mode in ("inference", "train"):
        try:
            lut_ops(cfg, mode, toggles)
            print(f"{mode}: every selected function exists")
        except (ValueError, NotImplementedError) as error:
            print(f"{mode}: not available with these toggles: {error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LUT transformer host")
    base.add_common_args(parser)
    parser.add_argument("--preset", default="toggles", choices=sorted(PRESETS),
                        help="which named set of toggles to use (default: the TOGGLES block)")
    args = parser.parse_args(argv)

    cfg, hw = base.load_configs(args.model, args.hw)
    if args.command == "info":
        _cmd_info(cfg, hw, args.preset)
        return
    raise SystemExit(f"'{args.command}' is written in windows 1 and 2. Try 'info' for now.")


if __name__ == "__main__":
    main()
