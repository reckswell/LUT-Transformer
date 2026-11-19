"""Make python/ importable (baseline, LUT, common, bench) wherever `pytest` is started from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the reference package before any test imports transformers, so Hugging Face's cache goes
# to ../LUT-Transformer-local/hf-home rather than ~/.cache/huggingface (reference/local_cache.py).
import reference  # noqa: E402,F401
