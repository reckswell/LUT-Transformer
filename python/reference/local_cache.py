"""Where downloaded models live: a folder beside the repository, never inside it and never in
~/.cache (Alpha, 2026-11-19: "keep all model installations in the local folder").

    LOCAL_DIR       ../LUT-Transformer-local, Alpha's folder for binaries next to the repository
                    ($LUT_LOCAL_DIR overrides it)
    CHECKPOINT_DIR  LOCAL_DIR/checkpoints: model weights, configs and tokenizer files, passed as
                    cache_dir= to every from_pretrained call ($LUT_CHECKPOINT_DIR overrides it)
    HF_HOME         LOCAL_DIR/hf-home: Hugging Face's own cache (hub metadata and the xet download
                    cache), which would otherwise go to ~/.cache/huggingface ($HF_HOME overrides it)

An environment variable Alpha sets always wins over these defaults. Hugging Face reads HF_HOME once,
when huggingface_hub is first imported, so use_local_hf_cache() must run before that: importing the
reference package runs it (reference/__init__.py), and tests/conftest.py imports the package before
any test imports transformers. If huggingface_hub was imported earlier with a different HF_HOME,
use_local_hf_cache() raises instead of letting files land in ~/.cache.

Nothing here imports torch or transformers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(os.environ.get("LUT_LOCAL_DIR", REPO_DIR.parent / "LUT-Transformer-local"))
CHECKPOINT_DIR = Path(os.environ.get("LUT_CHECKPOINT_DIR", LOCAL_DIR / "checkpoints"))
HF_HOME = Path(os.environ.get("HF_HOME", LOCAL_DIR / "hf-home"))


def use_local_hf_cache() -> Path:
    """Point Hugging Face's cache at HF_HOME (unless $HF_HOME is already set) and check that an
    already-imported huggingface_hub agrees. Returns the HF_HOME in effect."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    wanted = Path(os.environ["HF_HOME"]).resolve()
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and Path(constants.HF_HOME).resolve() != wanted:
        raise RuntimeError(
            f"huggingface_hub was imported before the reference package, so its cache is "
            f"{constants.HF_HOME} instead of {wanted}. Import reference (or set HF_HOME) before "
            f"transformers or huggingface_hub.")
    return wanted
