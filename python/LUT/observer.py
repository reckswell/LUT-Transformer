"""Calibration: record the range of every stage input in every layer while the model runs, without
changing anything it computes. Static scales (QuantSpec scale_mode "static") and smoothing
(Stage.smooth) are computed from these ranges.

    obs = lut_host.calibrate(params, cfg, batches)        # the FP32 model on calibration text
    obs.save("calibration.json")                          # then CALIBRATION = "calibration.json" in LUT/host.py
    obs.ranges()["qkv_proj[0].x"].percentile(99.99)

Calibration text must not overlap the evaluation text (use the training split).

What is recorded, per site (LUT/techniques.py's Site: stage, layer, operand; key "qkv_proj[0].x"):
    min, max, absmax     over every finite value seen (masked scores of -inf are left out)
    channel_absmax       the largest |x| per channel: per last-axis channel for (..., D) inputs,
                         per (head, channel) for q, k, v and the rope input (B, H, T, Dh); none for
                         the softmax scores and probabilities, whose last axis is key position
    percentiles          from a uniform subsample of |x| of at most max_samples values per site
                         (exact up to that sampling); saved files keep PERCENTILES
    weights              operand "w" of a linear stage: the largest |W[:, j]| per input channel j over
                         every weight seen at that site (wq, wk and wv share qkv_proj's site), which
                         smoothing needs
    kv_cache[i].k / .v   what layer i writes into the KV cache (rotated keys, or unrotated with
                         KVStore(k_before_rope=True))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from LUT.techniques import Site

PERCENTILES = (50.0, 90.0, 99.0, 99.9, 99.99, 99.999, 100.0)
FORMAT = "lut-calibration-1"
NO_CHANNELS = ("s", "p")                       # softmax scores and probabilities
HEAD_CHANNELS = ("q", "k", "v")                # (B, H, T, Dh): channels are (H, Dh)


@dataclass
class Ranges:
    """The recorded range of one site. samples is None for ranges read from a file, which carry
    their percentiles precomputed instead."""

    calls: int = 0
    count: int = 0
    min: float = float("inf")
    max: float = float("-inf")
    absmax: float = 0.0
    channel_absmax: Tensor | None = None
    samples: Tensor | None = None
    percentiles: dict[float, float] = field(default_factory=dict)

    def percentile(self, p: float) -> float:
        """The p-th percentile of |x| (p = 100 is absmax)."""
        if p >= 100.0:
            return self.absmax
        if self.samples is not None and self.samples.numel():
            k = min(self.samples.numel(), max(1, round(p / 100.0 * self.samples.numel())))
            return float(self.samples.kthvalue(k).values)
        if p in self.percentiles:
            return self.percentiles[p]
        raise KeyError(f"percentile {p} was not saved; saved: {sorted(self.percentiles)}")

    def to_dict(self) -> dict[str, Any]:
        pct = {f"{p:g}": self.percentile(p) for p in PERCENTILES} if self.samples is not None else \
            {f"{p:g}": v for p, v in self.percentiles.items()}
        return {"calls": self.calls, "count": self.count, "min": self.min, "max": self.max, "absmax": self.absmax,
                "channel_absmax": None if self.channel_absmax is None else self.channel_absmax.tolist(),
                "percentiles": pct}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ranges":
        ch = data.get("channel_absmax")
        return cls(calls=data["calls"], count=data["count"], min=data["min"], max=data["max"], absmax=data["absmax"],
                   channel_absmax=None if ch is None else torch.tensor(ch, dtype=torch.float32),
                   percentiles={float(p): float(v) for p, v in data.get("percentiles", {}).items()})


class Observer:
    """Collects Ranges per site. LUT/ops_registry.py wraps every stage with record() calls when
    lut_ops(..., observer=obs) builds the model; the stage's own function and inputs are unchanged."""

    def __init__(self, max_samples: int = 1 << 20) -> None:
        self.max_samples = max_samples
        self._ranges: dict[str, Ranges] = {}
        self._parts: dict[str, list[Tensor]] = {}       # |x| subsamples waiting to be joined
        self._stride: dict[str, int] = {}               # keep every stride-th value of each call

    @torch.no_grad()
    def record(self, site: Site, x: Tensor) -> None:
        """Add one stage input (any shape, float) to its site's ranges."""
        key = site.key()
        r = self._ranges.setdefault(key, Ranges())
        values = x.detach().to(torch.float32)
        flat = values.flatten()
        finite = flat[torch.isfinite(flat)]
        r.calls += 1
        r.count += finite.numel()
        if finite.numel():
            r.min = min(r.min, float(finite.min()))
            r.max = max(r.max, float(finite.max()))
            r.absmax = max(r.absmax, float(finite.abs().max()))
        ch = self._channels(site.operand, values)
        if ch is not None:
            ch = ch.cpu()
            r.channel_absmax = ch if r.channel_absmax is None or r.channel_absmax.shape != ch.shape \
                else torch.maximum(r.channel_absmax, ch)
        self._sample(key, finite.abs())

    @torch.no_grad()
    def record_weight(self, site: Site, w: Tensor) -> None:
        """A linear stage's weight (D_out, D_in): keep the largest |W[:, j]| per input channel."""
        key = site.key()
        r = self._ranges.setdefault(key, Ranges())
        col = w.detach().to(torch.float32).abs().amax(dim=0).cpu()
        r.calls += 1
        r.count += w.numel()
        r.absmax = max(r.absmax, float(col.max()))
        r.channel_absmax = col if r.channel_absmax is None else torch.maximum(r.channel_absmax, col)

    def ranges(self) -> dict[str, Ranges]:
        """Every site's Ranges, keyed by Site.key()."""
        for key, parts in self._parts.items():
            if parts:
                r = self._ranges[key]
                joined = torch.cat(([r.samples] if r.samples is not None else []) + parts)
                r.samples = joined
                parts.clear()
        return self._ranges

    def save(self, path: str | Path, **meta: Any) -> None:
        """Write the ranges (with PERCENTILES precomputed, without the samples) as JSON. meta (the
        calibration text, its token count, the checkpoint) is stored with them."""
        data = {"format": FORMAT, "meta": meta, "sites": {k: r.to_dict() for k, r in sorted(self.ranges().items())}}
        Path(path).write_text(json.dumps(data, indent=1))

    # -- helpers --------------------------------------------------------------------------------

    @staticmethod
    def _channels(operand: str, x: Tensor) -> Tensor | None:
        if operand in NO_CHANNELS or x.dim() == 0:
            return None
        if x.dim() == 4 and operand in HEAD_CHANNELS + ("x",):        # q, k, v, rope input: (B, H, T, Dh)
            return x.abs().amax(dim=(0, 2))
        return x.abs().reshape(-1, x.shape[-1]).amax(dim=0)

    def _sample(self, key: str, values: Tensor) -> None:
        stride = self._stride.get(key, 1)
        parts = self._parts.setdefault(key, [])
        parts.append(values[::stride].cpu())
        held = sum(p.numel() for p in parts)
        r = self._ranges[key]
        held += 0 if r.samples is None else r.samples.numel()
        while held > self.max_samples:                    # halve what is kept, and keep half as much from now on
            joined = torch.cat(([r.samples] if r.samples is not None else []) + parts)[::2]
            r.samples, parts[:] = joined, []
            stride *= 2
            held = joined.numel()
        self._stride[key] = stride


def load(path: str | Path) -> dict[str, Ranges]:
    """Ranges saved by Observer.save, keyed by site."""
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    return {k: Ranges.from_dict(v) for k, v in data["sites"].items()}
