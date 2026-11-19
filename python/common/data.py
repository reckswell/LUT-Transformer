"""Datasets: which ones the project knows, where they live, whether they're installed, and their text.

Every dataset lives in ../LUT-Transformer-local/datasets/<name>/, beside the repository and never in
it or in ~/.cache, the same rule as the models (Alpha, 2026-11-19: "all models installed locally
should stay in that folder"; reference/local_cache.py). $LUT_DATASET_DIR overrides the folder.
Nothing here downloads on its own: download() is called only when asked (the sweep's --download).

DATASETS lists the candidates from the plan (decision D2 is still Alpha's): WikiText-2 for
perplexity of pretrained models, TinyStories for small models. A dataset is a list of files in a
Hugging Face dataset repository, kept exactly as that repository ships them.

    from common import data
    spec = data.DATASETS["wikitext2"]
    data.is_installed(spec)          # its evaluation split is in the local folder
    text = data.load_text(spec)      # the evaluation split as one string, as the file has it
    span = data.eval_text(spec)      # the exact text perplexity is measured on, the same for every model

The same text for every model (Alpha, 2026-11-19: models must "cross check datasets" with Alpha's
own model). Different models have different tokenizers, so a token limit would cut a different
piece of text for each. The evaluated span is therefore fixed in characters: the first
eval_char_limit characters of the split (after the document separators are normalized, below),
cut back to the last whitespace so no word is split. Every model tokenizes exactly that text.
EvalText records the file's sha256, the span's characters, UTF-8 bytes and words, so a result can
be traced to the exact text and compared across tokenizers (bits per byte, word perplexity).

load_dataset and batches (token streams for training) come with the training data decision (D2, D3).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_DIR = Path(__file__).resolve().parents[2]
# The same local folder as reference/local_cache.LOCAL_DIR (tests/test_local_cache.py checks they agree).
LOCAL_DIR = Path(os.environ.get("LUT_LOCAL_DIR", REPO_DIR.parent / "LUT-Transformer-local"))
DATASET_DIR = Path(os.environ.get("LUT_DATASET_DIR", LOCAL_DIR / "datasets"))


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset.

    hf_repo          the Hugging Face dataset repository its files come from
    files            split -> file path inside that repository (kept at the same path locally)
    file_format      "txt" (plain text) or "parquet" (a table; reading it needs pyarrow)
    text_field       the parquet column with the text
    separator        joins the rows of a parquet file into one text stream
    eval_split       the split perplexity is measured on
    eval_char_limit  perplexity uses the first this-many characters of eval_split, cut back to the
                     last whitespace (None: all of it). In characters, not tokens, so every model
                     reads the same text. A choice to keep CPU runs short; the sweep's
                     --max-eval-chars overrides it.
    document_separator  a marker the file puts between documents (TinyStories: <|endoftext|>).
                     It is replaced by a blank line before anything else, so no tokenizer sees it:
                     GPT-2's tokenizer would read it as its own special token and Llama's as
                     several ordinary ones, which would make the texts differ between models.
    about            what it is and why it's here
    """

    name: str
    hf_repo: str
    files: dict[str, str]
    file_format: str
    eval_split: str
    about: str
    text_field: str | None = None
    separator: str = ""
    eval_char_limit: int | None = None
    document_separator: str | None = None

    def path(self, split: str | None = None) -> Path:
        """Where the split's file is (or would be) in the local folder."""
        return DATASET_DIR / self.name / self.files[split or self.eval_split]


DATASETS: dict[str, DatasetSpec] = {
    "wikitext2": DatasetSpec(
        name="wikitext2",
        hf_repo="Salesforce/wikitext",
        files={"test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
               "validation": "wikitext-2-raw-v1/validation-00000-of-00001.parquet"},
        file_format="parquet", text_field="text", separator="\n\n", eval_split="test",
        about="WikiText-2 raw (Wikipedia articles), the usual perplexity benchmark for quantized language "
              "models, so GPT-2 numbers compare with published ones. Test split 0.73 MB, validation 0.66 MB. "
              "Rows joined with blank lines, as in Hugging Face's perplexity guide."),
    "tinystories": DatasetSpec(
        name="tinystories",
        hf_repo="roneneldan/TinyStories",
        files={"validation": "TinyStoriesV2-GPT4-valid.txt"},
        file_format="txt", eval_split="validation", eval_char_limit=1_200_000, document_separator="<|endoftext|>",
        about="TinyStories V2 (short stories written by GPT-4) for small models; the validation file is "
              "22.5 MB of plain text, stories separated by <|endoftext|> (replaced by a blank line). Perplexity "
              "uses its first 1,200,000 characters, about 300,000 tokens and about the size of WikiText-2's "
              "test split."),
}


def is_installed(spec: DatasetSpec, split: str | None = None) -> bool:
    return spec.path(split).is_file()


def installed_datasets() -> list[str]:
    return [name for name, spec in DATASETS.items() if is_installed(spec)]


def missing_requirement(spec: DatasetSpec) -> str | None:
    """A Python package needed to read this dataset that isn't installed, or None."""
    if spec.file_format == "parquet" and importlib.util.find_spec("pyarrow") is None:
        return "pyarrow"
    return None


def download(spec: DatasetSpec, splits: list[str] | None = None) -> list[Path]:
    """Fetch the dataset's files from Hugging Face into DATASET_DIR/<name>/. Only call this when the
    person running asked for downloads. Needs huggingface_hub (installed with transformers)."""
    from huggingface_hub import hf_hub_download

    paths = []
    for split in splits or [spec.eval_split]:
        local = hf_hub_download(spec.hf_repo, spec.files[split], repo_type="dataset",
                                local_dir=str(DATASET_DIR / spec.name))
        paths.append(Path(local))
    return paths


def load_text(spec: DatasetSpec, split: str | None = None) -> str:
    """The split as one string: a text file as it is, a parquet file's text column joined with
    spec.separator."""
    path = spec.path(split)
    if spec.file_format == "txt":
        return path.read_text(encoding="utf-8")
    if spec.file_format == "parquet":
        import pyarrow.parquet as pq   # only for parquet datasets; see missing_requirement

        return spec.separator.join(pq.read_table(path, columns=[spec.text_field]).column(spec.text_field).to_pylist())
    raise ValueError(f"{spec.name}: unknown file format {spec.file_format!r}")


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, read in 1 MB blocks: identifies the exact file a result came from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvalText:
    """The exact text perplexity is measured on: one dataset, one split, one span, the same for
    every model whatever its tokenizer.

    file_sha256  the dataset file's sha256 (the whole file, before any cut)
    char_limit   the span's character limit (None: the whole split)
    text         the span itself, after the document separators were replaced by blank lines
    chars, utf8_bytes, words   its length three ways; words are whitespace-separated, as in
                 lm-evaluation-harness's word perplexity
    """

    dataset: str
    split: str
    file: Path
    file_sha256: str
    char_limit: int | None
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def utf8_bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def words(self) -> int:
        return len(self.text.split())


def normalize_text(text: str, document_separator: str | None) -> str:
    """Replace each document separator (and the whitespace around it) by one blank line, and strip
    the ends. Without a separator the text is only stripped."""
    if document_separator:
        text = re.sub(r"\s*" + re.escape(document_separator) + r"\s*", "\n\n", text)
    return text.strip()


def cut_text(text: str, char_limit: int | None) -> str:
    """The first char_limit characters, cut back to the last whitespace so no word is split
    (None or 0: all of it)."""
    if not char_limit or len(text) <= char_limit:
        return text
    cut = text[:char_limit]
    last_space = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
    return (cut[:last_space] if last_space > 0 else cut).rstrip()


def eval_text(spec: DatasetSpec, char_limit: int | None = None, split: str | None = None) -> EvalText:
    """The evaluation span of a dataset: its split, separators normalized, cut to char_limit
    characters (None: the dataset's eval_char_limit; 0: the whole split)."""
    limit = spec.eval_char_limit if char_limit is None else (char_limit or None)
    path = spec.path(split)
    text = cut_text(normalize_text(load_text(spec, split), spec.document_separator), limit)
    return EvalText(dataset=spec.name, split=split or spec.eval_split, file=path, file_sha256=file_sha256(path),
                    char_limit=limit, text=text)


def load_dataset(name: str, split: str) -> Any:
    """Load the token stream for a training split ("train", "val", "test")."""
    raise NotImplementedError("data.load_dataset (training data) comes after the dataset decision (D2, D3)")


def batches(tokens: Any, *, batch_size: int, context_length: int, seed: int, device: str) -> Iterator[tuple[Any, Any]]:
    """Yield (inputs, targets) pairs of shape (batch_size, context_length), targets shifted by one
    token, sampled at random offsets with a fixed seed so runs are reproducible."""
    raise NotImplementedError("data.batches comes with the training data (D2, D3)")
