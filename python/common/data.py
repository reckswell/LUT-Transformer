"""Datasets, tokenization and batching.

DECISION D2: which dataset and tokenizer. Candidates are listed in the
plan; nothing is downloaded until Alpha picks one (and downloads go to data/, which is gitignored).
"""

from __future__ import annotations

from typing import Any, Iterator


def load_dataset(name: str, split: str) -> Any:
    """Load (and cache under data/) the token stream for a split ("train", "val", "test")."""
    raise NotImplementedError("data.load_dataset is written in window 1, after the dataset decision (D2)")


def batches(tokens: Any, *, batch_size: int, context_length: int, seed: int, device: str) -> Iterator[tuple[Any, Any]]:
    """Yield (inputs, targets) pairs of shape (batch_size, context_length), targets shifted by one
    token, sampled at random offsets with a fixed seed so runs are reproducible."""
    raise NotImplementedError("data.batches is written in window 1")
