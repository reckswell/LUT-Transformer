"""Baseline transformer host: the model definition and the train / evaluate / generate entry points.

This is the file to read to see how the model is put together, the way a torch.nn.Module's
forward() reads. It owns the configuration, the parameters, the list of stages, and the order in
which the stage functions are called. The math of each stage lives in train.py and inference.py,
written from base operations. There are no LUTs here: the LUT model is python/LUT/host.py, which
calls the same composition below with some stages switched to LUT functions.

Which device runs which task (train, prefill, decode) comes from a hardware config.

    cd python
    python -m baseline.host info     --model configs/model/tiny.yaml --hw configs/hw/gpu.yaml
    python -m baseline.host train    ...same flags...
    python -m baseline.host evaluate ...
    python -m baseline.host generate ... --prompt "Once upon a time"

Status: `info` works now. The model composition below is documented step by step and written
in window 1, after Alpha settles the architecture decision (D1).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from baseline import inference, train

PYTHON_DIR = Path(__file__).resolve().parents[1]   # python/, where configs/ lives
Mode = Literal["inference", "train"]


def _todo(name: str) -> NotImplementedError:
    return NotImplementedError(f"baseline.host.{name} is written in window 1")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Model shape and architecture choices (configs/model/*.yaml).

    Open decisions are marked DECISION in the yaml files and listed in the project plan.
    """

    name: str = "tiny"
    vocab_size: int | None = None      # set from the tokenizer once the dataset is chosen
    context_length: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    norm: str = "layernorm"            # layernorm | rmsnorm
    norm_placement: str = "pre"        # pre | post
    activation: str = "relu"           # relu | gelu
    positional_encoding: str = "sinusoidal"  # sinusoidal | learned
    tie_embeddings: bool = True
    bias: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.norm not in ("layernorm", "rmsnorm"):
            raise ValueError(f"norm must be layernorm or rmsnorm, got {self.norm!r}")
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be relu or gelu, got {self.activation!r}")
        if self.norm_placement not in ("pre", "post"):
            raise ValueError(f"norm_placement must be pre or post, got {self.norm_placement!r}")
        if self.positional_encoding not in ("sinusoidal", "learned"):
            raise ValueError(f"positional_encoding must be sinusoidal or learned, got {self.positional_encoding!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransformerConfig":
        data = load_yaml(path)
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        config = cls(**known)
        config.validate()
        return config


def load_configs(model_path: str | Path, hw_path: str | Path) -> tuple[TransformerConfig, dict]:
    """Read the model config and the hardware config (device per task)."""
    model = TransformerConfig.from_yaml(model_path)
    hw = load_yaml(hw_path)
    devices = hw.get("devices") or {}
    for task in ("train", "prefill", "decode"):
        if task not in devices:
            raise ValueError(f"{hw_path}: devices.{task} is missing")
    return model, hw


# ----------------------------------------------------------------------------------------------
# Stages: the named steps of the model, each one a function in inference.py / train.py
# ----------------------------------------------------------------------------------------------
#
# Stage name -> function name. "{norm}" and "{activation}" are filled from the model config
# (layernorm or rmsnorm, relu or gelu). In training mode the functions are <name>_forward and
# <name>_backward. The LUT host toggles these same stages, so a stage is the unit that can be
# swapped for a LUT version (and later, the unit that becomes an HLS kernel).

LAYER_STAGES: dict[str, str] = {        # inside every decoder layer, in the order they run
    "norm": "{norm}",                   # used twice per layer (before attention, before the FFN)
    "qkv_proj": "linear",               # Q, K, V projections
    "attn_scores": "attention_scores",  # QK^T / sqrt(Dh), causal mask
    "softmax": "softmax",
    "attn_values": "attention_values",  # PV
    "out_proj": "linear",
    "ffn_up": "linear",
    "ffn_act": "{activation}",
    "ffn_down": "linear",
}
MODEL_STAGES: dict[str, str] = {        # outside the layers
    "embedding": "embedding",
    "final_norm": "{norm}",
    "lm_head": "linear",
}
STAGES: dict[str, str] = {**LAYER_STAGES, **MODEL_STAGES}


def function_name(stage: str, cfg: TransformerConfig) -> str:
    """The function a stage calls, for example "norm" -> "layernorm" under the tiny config."""
    return STAGES[stage].format(norm=cfg.norm, activation=cfg.activation)


@dataclass
class ModelOps:
    """The function every stage calls, as the composition below uses them.

    layers[i][stage] is the function for `stage` inside decoder layer i; embedding, final_norm
    and lm_head sit outside the layers. In "inference" mode each entry is one function; in
    "train" mode each is a (forward, backward) pair. The baseline fills every entry from
    inference.py / train.py; the LUT host swaps individual entries for LUT functions.
    """

    embedding: Any
    layers: list[dict[str, Any]]
    final_norm: Any
    lm_head: Any


def base_function(stage: str, cfg: TransformerConfig, mode: Mode) -> Any:
    """The baseline function for one stage (or its (forward, backward) pair in train mode)."""
    name = function_name(stage, cfg)
    if mode == "inference":
        return _get(inference, name)
    return (_get(train, f"{name}_forward"), _get(train, f"{name}_backward"))


def base_ops(cfg: TransformerConfig, mode: Mode) -> ModelOps:
    """Every stage on its baseline function."""
    return ModelOps(
        embedding=base_function("embedding", cfg, mode),
        layers=[{stage: base_function(stage, cfg, mode) for stage in LAYER_STAGES} for _ in range(cfg.n_layers)],
        final_norm=base_function("final_norm", cfg, mode),
        lm_head=base_function("lm_head", cfg, mode),
    )


def _get(module: Any, name: str) -> Callable[..., Any]:
    fn = getattr(module, name, None)
    if fn is None:
        raise ValueError(f"{module.__name__} has no function {name!r}")
    return fn


# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------

def init_params(cfg: TransformerConfig, *, seed: int, device: str, dtype: Any = None) -> dict:
    """Create every parameter as a plain tensor in one flat dict, named like the reference model
    so weights can be copied in both directions (reference_torch.copy_params_*):

        tok_emb                      (V, D)
        pos_emb                      (context_length, D)   only if positional_encoding = learned
        layers.{i}.norm1.gamma/beta  (D,)                  beta only for layernorm
        layers.{i}.attn.wq/wk/wv/wo  (D, D)                plus .bq/.bk/.bv/.bo (D,) if bias
        layers.{i}.norm2.gamma/beta  (D,)
        layers.{i}.ffn.w1 (F, D), .b1 (F,), .w2 (D, F), .b2 (D,)
        norm_f.gamma/beta            (D,)
        lm_head                      (V, D)                only if not tie_embeddings

    DECISION (window 1): initialization scheme (for example normal(0, 0.02), scaled residual init).
    """
    raise _todo("init_params")


# ----------------------------------------------------------------------------------------------
# Model composition (the "transformer definition")
# ----------------------------------------------------------------------------------------------
#
# One decoder block i, pre-norm placement (post-norm moves each norm after its residual add),
# with L = ops.layers[i]:
#
#   h  = L["norm"](x)                                  # norm1
#   q, k, v = L["qkv_proj"](h, Wq), (h, Wk), (h, Wv)   # three projections, then split_heads
#   s  = L["attn_scores"](q, k, causal_mask)           # QK^T / sqrt(Dh), masked
#   p  = L["softmax"](s)
#   a  = L["attn_values"](p, v)                        # PV, then merge_heads
#   x  = x + L["out_proj"](a, Wo)                      # residual
#   h  = L["norm"](x)                                  # norm2
#   x  = x + L["ffn_down"](L["ffn_act"](L["ffn_up"](h, W1)), W2)   # residual
#
# Whole model: tokens -> ops.embedding (+ positions) -> n_layers blocks -> ops.final_norm ->
# ops.lm_head -> logits. Training runs this with the *_forward functions, keeps every cache, and
# then calls the *_backward functions in exactly the reverse order. Inference splits it into
# prefill (the whole prompt at once, filling the KV cache) and decode (one token at a time,
# reading the cache). Every function below takes `ops`, so the LUT host runs this same code.

def forward_train(params: dict, token_ids: Any, ops: ModelOps, cfg: TransformerConfig) -> tuple[Any, list]:
    """Full forward pass with caches for backward. Returns (logits (B, T, V), caches)."""
    raise _todo("forward_train")


def backward(d_logits: Any, caches: list, ops: ModelOps, cfg: TransformerConfig) -> dict:
    """Backpropagate d_logits through the model in reverse order. Returns {param name: gradient},
    with residual-branch gradients summed and tied-embedding gradients added together."""
    raise _todo("backward")


def train_step(params: dict, batch: Any, ops: ModelOps, opt_state: dict, cfg: TransformerConfig, *, lr: float) -> dict:
    """forward_train -> cross_entropy -> backward -> clip_grad_norm -> adamw_step.
    Returns {"loss", "grad_norm", "tokens"} for training_loop."""
    raise _todo("train_step")


def prefill(params: dict, token_ids: Any, cache: Any, ops: ModelOps, cfg: TransformerConfig) -> Any:
    """Run the prompt (B, T) through the model at once, filling the KV cache. Returns the last
    position's logits (B, V)."""
    raise _todo("prefill")


def decode_step(params: dict, token_ids: Any, cache: Any, ops: ModelOps, cfg: TransformerConfig) -> Any:
    """Run one new token per sequence (B, 1) against the cache, appending its key and value.
    Returns logits (B, V)."""
    raise _todo("decode_step")


def generate(params: dict, prompt_ids: Any, max_new_tokens: int, ops: ModelOps, cfg: TransformerConfig, **sampling: Any) -> Any:
    """prefill once, then decode_step + sample_next_token max_new_tokens times. Prefill and decode
    may run on different devices (hardware config), with the cache moved between them."""
    raise _todo("generate")


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the baseline and LUT hosts."""
    parser.add_argument("command", choices=["info", "train", "evaluate", "generate"])
    parser.add_argument("--model", default=str(PYTHON_DIR / "configs/model/tiny.yaml"))
    parser.add_argument("--hw", default=str(PYTHON_DIR / "configs/hw/gpu.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=None)


def _cmd_info(cfg: TransformerConfig, hw: dict) -> None:
    print(f"model   {cfg.name}: {asdict(cfg)}")
    print(f"hw      {hw.get('name', '?')}: {hw['devices']}")
    print()
    print(f"{'stage':<13}function")
    for stage in STAGES:
        print(f"{stage:<13}{function_name(stage, cfg)}")
    base_ops(cfg, "inference")   # raises if a function is missing
    base_ops(cfg, "train")
    print("\nall stage functions exist for inference and training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baseline (non-LUT) transformer host")
    add_common_args(parser)
    args = parser.parse_args(argv)

    cfg, hw = load_configs(args.model, args.hw)
    if args.command == "info":
        _cmd_info(cfg, hw)
        return
    raise SystemExit(f"'{args.command}' is written in window 1. Try 'info' for now.")


if __name__ == "__main__":
    main()
