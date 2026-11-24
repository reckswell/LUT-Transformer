"""Speed parity of the LUT model's slots (2026-11-24, Alpha: every technique isolated as a slot that
runs the base model's algorithm "with the same speeds").

Times the same work on three arms, interleaved round by round so drift (clock boost, other load)
hits every arm alike, and checks the logits:

    baseline       baseline/host.py with base_ops
    lut_default    LUT/host.py with every slot at its default (the PARAMETERS block as committed)
    lut_all_slots  LUT/host.py with every slot switched on and every technique replaced by a
                   pass-through (--all-slots; the cost of the plumbing alone, not of a technique)

lut_default runs the baseline's own functions, so it should match baseline to within timing noise
and give bit-identical logits; lut_all_slots must also give bit-identical logits and shows what
the wrappers cost when every slot is in use. Weights are random (speed does not depend on their
values), so nothing needs to be installed.

    cd python
    python -m bench.slot_parity                                    # stories15m shape, CPU
    python -m bench.slot_parity --device cuda --all-slots          # GPU, with the third arm
    python -m bench.slot_parity --model configs/model/gpt2.yaml --vocab 50257

Each run writes results/runs/<run_id>/ (bench/runlog.py) and prints the run id. Timing is PyTorch
emulation on this machine; it says nothing about FPGA speed (agent.md 2.6).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import replace
from typing import Any, Callable

import torch

from baseline import host as base
from bench import runlog, timing
from common import precision
from common.quant import QuantSpec
from LUT import host as lut_host
from LUT import ops_registry, techniques
from LUT.stages import KVStore, Stage

ARMS = ("baseline", "lut_default", "lut_all_slots")


def pass_through_techniques() -> dict[str, Callable]:
    """Every LUT/techniques.py function as a stand-in that returns its tensor unchanged."""
    return {
        "input_scale": lambda x, spec, site, ranges: torch.ones((), dtype=x.dtype, device=x.device),
        "quantize_input": lambda x, spec, scale, site: x,
        "accumulator": lambda y, spec, site, inputs: y,
        "prepare_weight": lambda w, spec, site: w,
        "smoothing_factors": lambda act, w, alpha, site: torch.ones_like(act),
        "kv_encode": lambda x, spec, site, ranges: x,
        "kv_decode": lambda stored, spec, site: stored,
    }


def all_slots_on(cfg: base.TransformerConfig) -> dict[str, Any]:
    """Every slot of every stage set (dynamic scales, so no calibration is needed)."""
    spec = QuantSpec(bits=8, scale_mode="dynamic", acc_bits=32)
    toggles: dict[str, Any] = {}
    for stage, operands in ops_registry.OPERANDS.items():
        kw: dict[str, Any] = {}
        if operands:
            kw["quant"] = spec
        if len(operands) == 2:
            kw["quant_b"] = replace(spec, acc_bits=None)
        if stage in ops_registry.WEIGHT_ARG:
            kw["weight"] = QuantSpec(bits=8, granularity="per_channel", axis=0)
        if stage == "softmax":
            kw["alt"] = {"exp": "exp"}
        if stage == "ffn_act":
            kw["alt"] = {"act": cfg.activation}
        toggles[stage] = Stage(**kw)
    toggles["kv_cache"] = KVStore(k=spec, v=spec)
    return toggles


def build_arms(cfg: base.TransformerConfig, all_slots: bool) -> dict[str, tuple[Any, Any]]:
    """{arm: (host module, ops)}."""
    arms = {"baseline": (base, base.base_ops(cfg, "inference")),
            "lut_default": (lut_host, lut_host.lut_ops(cfg, "inference"))}
    if all_slots:
        saved = {name: getattr(techniques, name) for name in pass_through_techniques()}
        try:
            for name, fn in pass_through_techniques().items():
                setattr(techniques, name, fn)
            arms["lut_all_slots"] = (lut_host, lut_host.lut_ops(cfg, "inference", all_slots_on(cfg)))
        finally:
            for name, fn in saved.items():
                setattr(techniques, name, fn)
    return arms


def workloads(host: Any, ops: Any, params: dict, cfg: base.TransformerConfig, ids: torch.Tensor,
              decode_tokens: int) -> dict[str, Callable[[], Any]]:
    """forward on the whole batch; prefill of the prompt then decode_tokens greedy decode steps."""
    prompt = ids[:, : ids.shape[1] - decode_tokens]

    def forward() -> Any:
        return host.forward(params, ids, ops, cfg)

    def prefill_decode() -> Any:
        cache = host.new_cache(params, cfg, ids.shape[0], ids.shape[1])
        logits = host.prefill(params, prompt, cache, ops, cfg)
        for t in range(prompt.shape[1], ids.shape[1]):
            logits = host.decode_step(params, ids[:, t:t + 1], cache, ops, cfg)
        return logits

    return {"forward": forward, "prefill_decode": prefill_decode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=str(base.PYTHON_DIR / "configs/model/stories15m.yaml"))
    parser.add_argument("--vocab", type=int, default=None, help="vocab_size if the config leaves it unset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=128, help="tokens per sequence (forward); prompt + decode")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=8, help="interleaved rounds; each times every arm")
    parser.add_argument("--repeats", type=int, default=5, help="timed calls per arm per round")
    parser.add_argument("--all-slots", action="store_true", help="add the lut_all_slots arm")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args(argv)

    fp = precision.full_fp32()
    cfg = base.TransformerConfig.from_yaml(args.model)
    if cfg.vocab_size is None:
        cfg = replace(cfg, vocab_size=args.vocab or 32000)
    params = lut_host.init_params(cfg, seed=0, device=args.device)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.length), generator=torch.Generator().manual_seed(1))
    ids = ids.to(args.device)
    arms = build_arms(cfg, args.all_slots)

    run = runlog.Run("slot_parity", configs={"model": args.model, "cfg": cfg.__dict__, "args": vars(args)},
                     devices={"inference": args.device}, results_dir=args.results_dir)
    # Same logits on every arm (pass-through techniques for lut_all_slots).
    reference = {name: fn() for name, fn in workloads(base, arms["baseline"][1], params, cfg, ids, args.decode_tokens).items()}
    identical = {}
    for arm, (host, ops) in arms.items():
        outs = {name: fn() for name, fn in workloads(host, ops, params, cfg, ids, args.decode_tokens).items()}
        identical[arm] = all(torch.equal(outs[k], reference[k]) for k in reference)

    samples: dict[tuple[str, str], list[float]] = {}
    for _ in range(args.rounds):
        for arm, (host, ops) in arms.items():
            for name, fn in workloads(host, ops, params, cfg, ids, args.decode_tokens).items():
                stats = timing.time_fn(fn, device=args.device, warmup=1, repeats=args.repeats)
                samples.setdefault((arm, name), []).extend(stats.samples_ms)

    summary: dict[str, Any] = {"identical_logits": identical, "fp32": fp, "workloads": {}}
    print(f"run {run.run_id}: {cfg.name}, B={args.batch} T={args.length} (decode {args.decode_tokens}), "
          f"{args.device}, {args.rounds} rounds x {args.repeats}")
    print(f"{'workload':<16}{'arm':<15}{'median ms':>10}{'vs baseline':>13}  identical logits")
    for name in ("forward", "prefill_decode"):
        base_median = statistics.median(samples[("baseline", name)])
        rows = {}
        for arm in arms:
            med = statistics.median(samples[(arm, name)])
            rows[arm] = {"median_ms": med, "ratio_to_baseline": med / base_median, "n": len(samples[(arm, name)]),
                         "min_ms": min(samples[(arm, name)])}
            print(f"{name:<16}{arm:<15}{med:>10.3f}{med / base_median:>12.3f}x  {identical[arm]}")
        summary["workloads"][name] = rows
    run.finish(summary)
    print(f"saved {run.dir}")
    return 0 if all(identical.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
