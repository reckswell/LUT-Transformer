"""Turn the LUT host's toggles (its PARAMETERS block, or a named preset) into the functions its
wiring calls.

Start from every stage on its baseline function (baseline/host.py:base_ops, the same building
blocks the baseline model uses). Then, stage by stage and layer by layer, look at the slots the
toggles set (LUT/stages.py Stage):
  - none set: keep the baseline function itself. Nothing is wrapped and nothing extra runs, so the
    stage is bit-identical to the baseline and exactly as fast.
  - lut: the same-named function from inference_lut.py (or the <name>_forward / <name>_backward
    pair from train_lut.py in training), with the architecture bound to it.
  - alt: the stage function from nonlinear.py with the named functions plugged in.
  - quant, quant_b, weight, smooth: the function above, wrapped so that its inputs are quantized
    and its weight prepared first (LUT/techniques.py does each step).
The KV cache setting (toggles["kv_cache"], a KVStore) adds a "kv_store" entry to the layers it
covers, which LUT/host.py's attention wiring calls where keys and values are written; layers
without one store them as they are. An Observer (calibration) wraps every stage with a recorder
that reads the inputs and changes nothing.

Everything is checked here, at start-up, so a typo in the toggles, a slot a stage doesn't have, a
static scale without calibration or a technique still pending fails before a run starts rather
than halfway through it.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Iterable

from baseline import host as base
from common.quant import QuantSpec
from LUT import inference_lut, nonlinear, techniques, train_lut
from LUT.stages import BASE, KV_BASE, KVStore, Stage
from LUT.techniques import Site

if TYPE_CHECKING:
    from LUT.observer import Observer, Ranges

# The run-time inputs of each stage, in argument order; Stage.quant is the first, Stage.quant_b the
# second. The embedding's input is token ids (integers), so it has none.
OPERANDS: dict[str, tuple[str, ...]] = {
    "embedding": (), "norm": ("x",), "qkv_proj": ("x",), "rope": ("x",), "attn_scores": ("q", "k"),
    "softmax": ("s",), "attn_values": ("p", "v"), "out_proj": ("x",), "ffn_up": ("x",), "ffn_act": ("x",),
    "ffn_gate": ("a", "b"), "ffn_down": ("x",), "final_norm": ("x",), "lm_head": ("x",),
}
# Stages with a weight argument (its position, its operand name), for Stage.weight.
WEIGHT_ARG: dict[str, tuple[int, str]] = {
    "embedding": (1, "table"), "norm": (1, "w"), "final_norm": (1, "w"), "qkv_proj": (1, "w"),
    "out_proj": (1, "w"), "ffn_up": (1, "w"), "ffn_down": (1, "w"), "lm_head": (1, "w"),
}
LINEAR_STAGES = ("qkv_proj", "out_proj", "ffn_up", "ffn_down", "lm_head")
# Where a smoothing factor folds in hardware (Stage.smooth), and what the fold needs.
SMOOTH_FOLD: dict[str, str] = {
    "qkv_proj": "norm1 gamma (pre-norm)", "ffn_up": "norm2 gamma (pre-norm)", "lm_head": "final norm gamma",
    "out_proj": "wv rows", "ffn_down": "w3 rows (gated feed-forward)",
}
TOGGLE_KEYS = (*base.STAGES, "kv_cache")


# ----------------------------------------------------------------------------------------------
# Reading the toggles
# ----------------------------------------------------------------------------------------------

def entries(line: Any) -> list[Stage]:
    """A stage line as a list of Stages (None: no entries, so BASE everywhere)."""
    if line is None:
        return []
    return list(line) if isinstance(line, (list, tuple)) else [line]


def layer_stage(line: Any, layer: int | None) -> Stage:
    """The Stage that applies in decoder `layer` (None: a stage outside the layers, where layers=
    is ignored): the last entry whose layers include it, else BASE."""
    chosen = BASE
    for entry in entries(line):
        if layer is None or entry.layers is None or layer in entry.layers:
            chosen = entry
    return chosen


def check_toggles(toggles: dict[str, Any], cfg: base.TransformerConfig) -> None:
    """Raise ValueError for an unknown stage, an unknown architecture or function, a layer out of
    range, a slot the stage doesn't have, or a setting the model can't carry out."""
    unknown = set(toggles) - set(TOGGLE_KEYS)
    if unknown:
        raise ValueError(f"unknown stage(s) in toggles: {sorted(unknown)}; known: {list(TOGGLE_KEYS)}")
    for stage in base.STAGES:
        for choice in entries(toggles.get(stage)):
            if not isinstance(choice, Stage):
                raise ValueError(f"stage {stage!r}: expected Stage or a list of Stages, got {type(choice).__name__}")
            _check_stage(stage, choice, cfg)
    kv = toggles.get("kv_cache", KV_BASE)
    if not isinstance(kv, KVStore):
        raise ValueError(f"kv_cache: expected KVStore, got {type(kv).__name__}")
    _check_layers("kv_cache", kv.layers, cfg)
    for spec in (kv.k, kv.v):
        if spec is not None:
            spec.validate()
    if kv.k_before_rope and cfg.positional_encoding != "rope":
        raise ValueError("kv_cache: k_before_rope needs rotary positions (positional_encoding = rope)")


def _check_layers(stage: str, layers: Iterable[int] | None, cfg: base.TransformerConfig) -> None:
    if layers is not None:
        bad = [i for i in layers if not 0 <= i < cfg.n_layers]
        if bad:
            raise ValueError(f"stage {stage!r}: layers {bad} out of range for {cfg.n_layers} layers")


def _check_stage(stage: str, choice: Stage, cfg: base.TransformerConfig) -> None:
    if choice.lut is not None and choice.lut.name not in inference_lut.ARCHITECTURES:
        known = sorted(inference_lut.ARCHITECTURES) or "none yet (decision D5)"
        raise ValueError(f"stage {stage!r}: unknown LUT architecture {choice.lut.name!r}; known: {known}")
    _check_layers(stage, choice.layers, cfg)
    n_inputs = len(OPERANDS[stage])
    if choice.quant is not None:
        if n_inputs < 1:
            raise ValueError(f"stage {stage!r} has no run-time input to quantize (its input is token ids)")
        choice.quant.validate()
    if choice.quant_b is not None:
        if n_inputs < 2:
            raise ValueError(f"stage {stage!r} has one run-time input; quant_b is for attn_scores, "
                             f"attn_values and ffn_gate")
        choice.quant_b.validate()
        if choice.quant_b.acc_bits is not None:
            raise ValueError(f"stage {stage!r}: set acc_bits on quant (the first operand), not quant_b")
    if choice.weight is not None:
        if stage not in WEIGHT_ARG:
            raise ValueError(f"stage {stage!r} has no weight; weight= is for {sorted(WEIGHT_ARG)}")
        if isinstance(choice.weight, QuantSpec):
            choice.weight.validate()
    if choice.smooth is not None:
        if stage not in LINEAR_STAGES:
            raise ValueError(f"stage {stage!r}: smooth= is for the linear stages {LINEAR_STAGES}")
        if not 0.0 <= choice.smooth <= 1.0:
            raise ValueError(f"stage {stage!r}: smooth (alpha) must be in [0, 1], got {choice.smooth}")
        if stage in ("qkv_proj", "ffn_up") and cfg.norm_placement != "pre":
            raise ValueError(f"stage {stage!r}: smoothing folds into the preceding norm, which needs pre-norm")
        if stage == "ffn_down" and cfg.ffn != "glu":
            raise ValueError("stage 'ffn_down': smoothing folds into w3, which only a gated feed-forward has")
    if choice.alt:
        if choice.lut is not None:
            raise ValueError(f"stage {stage!r}: set lut or alt, not both")
        slot = nonlinear.STAGE_SLOTS.get(stage)
        if slot is None:
            raise ValueError(f"stage {stage!r} has no replaceable function; alt= is for {sorted(nonlinear.STAGE_SLOTS)}")
        if set(choice.alt) != {slot}:
            raise ValueError(f"stage {stage!r}: alt takes {{{slot!r}: name}}, got keys {sorted(choice.alt)}")
        nonlinear.lookup(slot, choice.alt[slot])


# ----------------------------------------------------------------------------------------------
# Building the ops
# ----------------------------------------------------------------------------------------------

def lut_function(stage: str, choice: Stage, cfg: base.TransformerConfig, mode: base.Mode) -> Any:
    """The LUT function for one stage, with its architecture bound as arch=...
    (or the bound (forward, backward) pair in train mode)."""
    name = base.function_name(stage, cfg)
    if mode == "inference":
        return functools.partial(_get(inference_lut, name, stage), arch=choice.lut)
    forward = _get(train_lut, f"{name}_forward", stage)
    backward = _get(train_lut, f"{name}_backward", stage)
    return (functools.partial(forward, arch=choice.lut), backward)


def resolve_ops(toggles: dict[str, Any], cfg: base.TransformerConfig, mode: base.Mode, *,
                observer: "Observer | None" = None, ranges: "dict[str, Ranges] | None" = None) -> base.ModelOps:
    """ModelOps for LUT/host.py's wiring: every stage on its baseline function, then each stage
    whose slots the toggles set rebuilt as the module docstring says. Stages are independent;
    nothing here overrides a toggle. ranges: calibrated ranges by site (LUT/observer.py), needed by
    static scales and smoothing."""
    check_toggles(toggles, cfg)
    kv = toggles.get("kv_cache", KV_BASE)
    if mode == "train":
        _check_trainable(toggles, kv, observer)
    ops = base.base_ops(cfg, mode)
    for stage in base.STAGES:
        line = toggles.get(stage)
        if stage in base.LAYER_STAGES:
            for i in range(cfg.n_layers):
                ops.layers[i][stage] = _build(stage, i, layer_stage(line, i), ops.layers[i][stage], cfg, mode,
                                              observer, ranges)
        else:
            setattr(ops, stage, _build(stage, None, layer_stage(line, None), getattr(ops, stage), cfg, mode,
                                       observer, ranges))
    if mode == "inference":
        for i in range(cfg.n_layers):
            covered = kv.layers is None or i in kv.layers
            if (covered and not kv.is_base()) or observer is not None:
                ops.layers[i]["kv_store"] = KVStoreStage(i, kv if covered else KV_BASE, observer, ranges)
    return ops


def _check_trainable(toggles: dict[str, Any], kv: KVStore, observer: Any) -> None:
    """Training runs the baseline or a LUT (train_lut.py); the precision and alt slots and the KV
    store are inference-only until Alpha decides how training sees them (TRAIN_WITH_LUTS)."""
    if observer is not None:
        raise ValueError("calibration (observer) runs in inference mode")
    for stage in base.STAGES:
        for choice in entries(toggles.get(stage)):
            if not choice.is_base() and (choice.lut is None or choice.quant or choice.quant_b or choice.weight
                                         or choice.smooth is not None or choice.alt):
                raise NotImplementedError(
                    f"stage {stage!r}: quant / quant_b / weight / smooth / alt in training (quantization-aware "
                    f"training) is pending Alpha's decision; train with TRAIN_WITH_LUTS = False")
    if not kv.is_base():
        raise NotImplementedError("kv_cache settings apply to inference only")


def _build(stage: str, layer: int | None, choice: Stage, base_fn: Any, cfg: base.TransformerConfig,
           mode: base.Mode, observer: Any, ranges: Any) -> Any:
    """The function for one stage in one layer."""
    if (choice.is_base() and observer is None) or not base.stage_used(stage, cfg):
        return base_fn                                   # the baseline function itself: no wrapper
    fn = base_fn
    if choice.lut is not None:
        fn = lut_function(stage, choice, cfg, mode)
    elif choice.alt:
        fn = nonlinear.stage_function(stage, cfg.norm, choice.alt)
    if mode == "train":
        return fn
    if choice.quant or choice.quant_b or choice.weight is not None or choice.smooth is not None:
        fn = PrecisionStage(fn, stage, layer, choice, ranges)
    if observer is not None:
        fn = ObservedStage(fn, stage, layer, observer)
    return fn


class PrecisionStage:
    """A stage function with its precision slots applied around it, in the order hardware has
    them: smoothing (x / s, W * s), the weight's offline preparation, quantization of each run-time
    input, the stage itself, then the accumulator width on its output. The techniques are bound
    and checked when the model is built; static scales, smoothing factors and prepared weights
    are computed on first use and kept (weights keyed by the tensor, so each tied copy and each
    norm's gamma gets its own)."""

    def __init__(self, fn: Callable, stage: str, layer: int | None, choice: Stage, ranges: Any) -> None:
        self.fn, self.stage, self.layer, self.choice = fn, stage, layer, choice
        where = stage if layer is None else f"{stage} layer {layer}"
        self.specs = [(i, spec, Site(stage, layer, OPERANDS[stage][i]))
                      for i, spec in enumerate((choice.quant, choice.quant_b)) if spec is not None]
        self.acc = choice.quant if choice.quant is not None and choice.quant.acc_bits is not None else None
        if self.specs:
            techniques.require(techniques.input_scale, f"{where}: quant")
            techniques.require(techniques.quantize_input, f"{where}: quant")
            self.input_scale, self.quantize_input = techniques.input_scale, techniques.quantize_input
        if self.acc is not None:
            techniques.require(techniques.accumulator, f"{where}: acc_bits")
            self.accumulator = techniques.accumulator
        self.static: dict[int, Any] = {}
        self.static_ranges: dict[int, Any] = {}
        for i, spec, site in self.specs:
            if spec.scale_mode == "static":
                self.static_ranges[i] = _ranges_for(ranges, site, f"{where}: a static scale for {site.operand}")
        self.weight_arg = WEIGHT_ARG.get(stage)
        if choice.weight is not None:
            techniques.require(techniques.prepare_weight, f"{where}: weight")
            self.prepare_weight = techniques.prepare_weight
        self.smooth_s: Any = None
        if choice.smooth is not None:
            techniques.require(techniques.smoothing_factors, f"{where}: smooth")
            self.smoothing_factors = techniques.smoothing_factors
            self.act_ranges = _ranges_for(ranges, Site(stage, layer, "x"), f"{where}: smoothing")
            self.weight_ranges = _ranges_for(ranges, Site(stage, layer, "w"), f"{where}: smoothing")
            if self.act_ranges.channel_absmax is None or self.weight_ranges.channel_absmax is None:
                raise ValueError(f"{where}: smoothing needs per-channel ranges for x and w in the calibration")
        self.prepared: dict[int, tuple[Any, int, Any]] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args = list(args)
        if self.choice.smooth is not None:
            s = self._smoothing(args[0])
            args[0] = args[0] / s
        if self.choice.weight is not None or self.choice.smooth is not None:
            idx = self.weight_arg[0]
            args[idx] = self._prepared(args[idx])
        for i, spec, site in self.specs:
            x = args[i]
            scale = self.static.get(i)
            if scale is None:
                scale = self.input_scale(x, spec, site, self.static_ranges.get(i))
                if spec.scale_mode != "dynamic":             # static or none: the same every call
                    self.static[i] = scale
            args[i] = self.quantize_input(x, spec, scale, site)
        y = self.fn(*args, **kwargs)
        if self.acc is not None:
            y = self.accumulator(y, self.acc, self.specs[0][2], tuple(args[i] for i, _, _ in self.specs))
        return y

    def _smoothing(self, x: Any) -> Any:
        if self.smooth_s is None:
            s = self.smoothing_factors(self.act_ranges.channel_absmax, self.weight_ranges.channel_absmax,
                                       self.choice.smooth, Site(self.stage, self.layer, "x"))
            self.smooth_s = s.to(device=x.device, dtype=x.dtype)
        return self.smooth_s

    def _prepared(self, w: Any) -> Any:
        """The weight as this stage multiplies by it, made once per weight tensor and version."""
        hit = self.prepared.get(id(w))
        if hit is not None and hit[0] is w and hit[1] == w._version:
            return hit[2]
        out = w
        if self.choice.smooth is not None:
            out = out * self._smoothing(w)                  # W[:, j] * s_j (columns = input channels)
        if self.choice.weight is not None:
            out = self.prepare_weight(out, self.choice.weight, Site(self.stage, self.layer, self.weight_arg[1]))
        self.prepared[id(w)] = (w, w._version, out)
        return out


class ObservedStage:
    """A stage function that first records its run-time inputs (and a linear stage's weight) with
    the Observer, then runs unchanged."""

    def __init__(self, fn: Callable, stage: str, layer: int | None, observer: "Observer") -> None:
        self.fn, self.observer = fn, observer
        self.sites = [Site(stage, layer, op) for op in OPERANDS[stage]]
        self.weight_site = Site(stage, layer, "w") if stage in LINEAR_STAGES else None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for i, site in enumerate(self.sites):
            self.observer.record(site, args[i])
        if self.weight_site is not None:
            self.observer.record_weight(self.weight_site, args[1])
        return self.fn(*args, **kwargs)


class KVStoreStage:
    """Layer i's KV cache storage, called by the attention wiring with the new keys and values
    (B, H, T, Dh) where they are written: records them for calibration, then returns what the cache
    gives back, kv_decode(kv_encode(x)), for K and V separately. k_before_rope tells the wiring to
    store keys before the rotation and rotate them on every read."""

    def __init__(self, layer: int, kv: KVStore, observer: Any, ranges: Any) -> None:
        self.layer, self.kv, self.observer = layer, kv, observer
        self.k_before_rope = kv.k_before_rope
        self.sites = (Site("kv_cache", layer, "k"), Site("kv_cache", layer, "v"))
        self.coded = [(j, spec, self.sites[j]) for j, spec in enumerate((kv.k, kv.v)) if spec is not None]
        self.static_ranges: dict[int, Any] = {}
        if self.coded:
            techniques.require(techniques.kv_encode, f"kv_cache layer {layer}")
            techniques.require(techniques.kv_decode, f"kv_cache layer {layer}")
            self.encode, self.decode = techniques.kv_encode, techniques.kv_decode
        for j, spec, site in self.coded:
            if spec.scale_mode == "static":
                self.static_ranges[j] = _ranges_for(ranges, site, f"kv_cache layer {layer}: a static scale")

    def __call__(self, k: Any, v: Any) -> tuple[Any, Any]:
        if self.observer is not None:
            self.observer.record(self.sites[0], k)
            self.observer.record(self.sites[1], v)
        kv = [k, v]
        for j, spec, site in self.coded:
            kv[j] = self.decode(self.encode(kv[j], spec, site, self.static_ranges.get(j)), spec, site)
        return kv[0], kv[1]


def _ranges_for(ranges: Any, site: Site, what: str) -> Any:
    if ranges is None or site.key() not in ranges:
        raise ValueError(f"{what} needs calibrated ranges for {site.key()}: run lut_host.calibrate on calibration "
                         f"text and set CALIBRATION in LUT/host.py (or pass ranges=)")
    return ranges[site.key()]


# ----------------------------------------------------------------------------------------------
# Describing the toggles
# ----------------------------------------------------------------------------------------------

def precision_label(stage: str, choice: Stage | None) -> str:
    """The precision slots of one stage line as one short string ("float" when none is set), for
    example "x:int8/tensor/p99.99,w:int8/channel" or "q:int8/tensor,k:int4/tensor"."""
    if choice is None:
        return "float"
    parts = []
    ops = OPERANDS[stage]
    for i, spec in enumerate((choice.quant, choice.quant_b)):
        if spec is not None:
            parts.append(f"{ops[i]}:{spec.label()}")
    if choice.weight is not None:
        w = choice.weight
        parts.append(f"{WEIGHT_ARG[stage][1]}:{w.label() if isinstance(w, QuantSpec) else getattr(w, 'name', type(w).__name__)}")
    if choice.smooth is not None:
        parts.append(f"smooth:{choice.smooth:g}")
    return ",".join(parts) or "float"


def describe(toggles: dict[str, Any], cfg: base.TransformerConfig) -> list[dict[str, str]]:
    """One row per stage (stages the toggles leave out run the baseline; a stage line with several
    entries gives one row per entry), for printing and for run metadata: what runs, the table shape
    and size, the layers, and the precision slots ("quant"). A stage the model doesn't have (rope
    without rotary positions, ffn_gate without a gated feed-forward) says "not in this model". A
    KV cache setting other than KV_BASE adds a "kv_cache" row."""
    rows = []
    for stage in base.STAGES:
        for choice in entries(toggles.get(stage)) or [None]:
            lut = choice.lut if choice else None
            alt = choice.alt if choice else None
            layers = choice.layers if choice and choice.layers is not None else None
            if not base.stage_used(stage, cfg):
                runs = "not in this model"       # e.g. rope without rotary positions; the toggle does nothing
            elif lut:
                runs = f"lut:{lut.name}"
            elif alt:
                runs = "alt:" + ",".join(f"{k}={nonlinear.name_of(v)}" for k, v in alt.items())
            else:
                runs = f"base:{base.function_name(stage, cfg)}"
            rows.append({
                "stage": stage,
                "runs": runs,
                "shape": f"{lut.inputs_per_table} x {lut.bits_per_input} -> {lut.out_bits}" if lut else "-",
                "table_bits": f"{lut.table_bits():,}" if lut else "-",
                "layers": ("-" if stage not in base.LAYER_STAGES
                           else "all" if layers is None else ",".join(map(str, layers))),
                "quant": precision_label(stage, choice),
            })
    kv = toggles.get("kv_cache", KV_BASE)
    if not kv.is_base():
        parts = [f"{name}:{spec.label()}" for name, spec in (("k", kv.k), ("v", kv.v)) if spec is not None]
        rows.append({"stage": "kv_cache", "runs": "store:" + ("keys before rope" if kv.k_before_rope else "rotated keys"),
                     "shape": "-", "table_bits": "-",
                     "layers": "all" if kv.layers is None else ",".join(map(str, kv.layers)),
                     "quant": ",".join(parts) or "float"})
    return rows


def _get(module: Any, name: str, stage: str) -> Any:
    fn = getattr(module, name, None)
    if fn is None:
        raise ValueError(f"stage {stage!r}: {module.__name__} has no function {name!r}")
    return fn
