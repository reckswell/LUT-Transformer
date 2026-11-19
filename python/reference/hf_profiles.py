"""Which Hugging Face architectures our from-scratch models can run: one profile per architecture,
each a config mapping, a weight export into our layout and a loader (reference/hf_gpt2.py,
reference/hf_llama.py). eval/harness.py and bench/sweep.py go through here, so adding a profile
means adding one line to PROFILES.

The Hugging Face model itself is imported and never modified; these functions only read its
config and state dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch
from torch import Tensor

from baseline import host
from reference import hf_gpt2, hf_llama
from reference.local_cache import CHECKPOINT_DIR


class Profile(NamedTuple):
    config_from_hf: Callable[[Any], host.TransformerConfig]
    params_from_hf: Callable[[dict[str, Tensor], host.TransformerConfig], dict[str, Tensor]]
    load: Callable[..., tuple[host.TransformerConfig, dict[str, Tensor], Any]]


# Hugging Face model_type -> profile.
PROFILES: dict[str, Profile] = {
    "gpt2": Profile(hf_gpt2.gpt2_config_from_hf, hf_gpt2.gpt2_params_from_hf, hf_gpt2.load_gpt2),
    "llama": Profile(hf_llama.llama_config_from_hf, hf_llama.llama_params_from_hf, hf_llama.load_llama),
}


def profile_for(model_type: str) -> Profile:
    if model_type not in PROFILES:
        raise ValueError(f"no profile for Hugging Face model_type {model_type!r}; profiles exist for {sorted(PROFILES)}")
    return PROFILES[model_type]


def export(hf_model: Any) -> tuple[host.TransformerConfig, dict[str, Tensor]]:
    """Our config and parameter dict for a loaded Hugging Face model, by its config.model_type."""
    profile = profile_for(hf_model.config.model_type)
    cfg = profile.config_from_hf(hf_model.config)
    return cfg, profile.params_from_hf(hf_model.state_dict(), cfg)


def load(model_type: str, model_name: str, *, device: str = "cpu", dtype: torch.dtype = host.DTYPE,
         cache_dir: str | Path = CHECKPOINT_DIR, local_files_only: bool = True) -> tuple[host.TransformerConfig, dict[str, Tensor], Any]:
    """Load model_name (a repo id or an installed folder) through its architecture's profile.
    Returns (cfg, params, hf_model), as hf_gpt2.load_gpt2 and hf_llama.load_llama do."""
    return profile_for(model_type).load(model_name, device=device, dtype=dtype, cache_dir=cache_dir,
                                        local_files_only=local_files_only)
