"""Benchmark sweep: every model listed in a CSV, across the benchmarks and datasets listed with it,
one run per row, with a plan and a time estimate before anything runs and progress bars with the
time remaining while it runs (Alpha, 2026-11-19).

    cd python
    python -m bench.sweep --dry-run                    # print the plan and its ETA; no benchmark runs
    python -m bench.sweep                              # run every enabled row of configs/sweep/models.csv
    python -m bench.sweep --only core_cpu,gpt2_cpu     # just these rows
    python -m bench.sweep --csv configs/sweep/other.csv --no-power
    python -m bench.sweep --allow-gaps                 # run even if a row misses a cell the lut rows measure

The CSV: one row per thing to measure, columns in any order, lines starting with # are comments.
configs/sweep/models.csv is the example to copy. Required columns: enabled, name, arm, model, device.
One comment line may set sweep-wide defaults for blank benchmarks and datasets cells, so every row
measures the same things by construction:

    # defaults: benchmarks=speed;perplexity datasets=tinystories

A row with random weights takes only the speed part of the default (random weights are speed only).

    enabled          yes | no (also y/n, true/false, 1/0; blank means yes). "no" rows are listed
                     in the plan as skipped.
    name             unique row name (letters, digits, _ . -), used in the run id
    arm              hf | baseline | lut, the arms of eval/harness.py: the Hugging Face model as
                     transformers ships it, our from-scratch model, or our LUT model
    model            core      the core config (configs/model/core.yaml), GPT-2's 50,257-token vocabulary
                     gpt2      short for openai-community/gpt2
                     stories15m  short for Xenova/llama2.c-stories15M: Karpathy's 15M-parameter Llama
                               (RoPE, SwiGLU) trained on TinyStories
                     org/name  a Hugging Face repo id, installed in ../LUT-Transformer-local/checkpoints
                     folder    a folder in ../LUT-Transformer-local/checkpoints holding a saved model
                               (config.json, the weights and the tokenizer)
                     The hf arm runs what transformers runs (HF_ARCHITECTURES); our baseline and
                     lut arms run the architectures that have a profile (reference/hf_profiles.py:
                     gpt2 and llama, without grouped-query attention).
    weights          whether the model has been trained before (blank: pretrained for a Hugging Face
                     model, random for the core):
                     pretrained  already trained, by someone else or by us; benchmarked as it is.
                                 The sweep never trains or retrains it.
                     random      never trained: random weights (--seed). For speed only, since
                                 timing doesn't depend on the weights; quality benchmarks refuse it.
                     train       needs training before it means anything, because nobody has trained
                                 it (only the core: no one has published weights for it). Listed in
                                 the plan as "needs training" and not run. A model with published
                                 weights can't be marked train: training what others have already
                                 trained wastes time (Alpha, 2026-11-19).
    lut_preset       lut arm only: params (the PARAMETERS block of LUT/host.py, read when the row
                     runs) or a PRESETS name such as all_base. Blank for hf and baseline.
    device           cpu | cuda | cuda:N
    benchmarks       ";"-separated, blank means speed:
                     speed       prefill and decode latency, tokens/s and power on synthetic tokens
                     perplexity  next-token perplexity on each dataset in the datasets column
    datasets         ";"-separated names from common/data.py DATASETS (wikitext2, tinystories), read
                     from ../LUT-Transformer-local/datasets. Used by perplexity; speed ignores them.
    batch            speed: sequences per call, B (perplexity runs one sequence at a time)
    context_lengths  speed: prompt lengths T, separated by ";" (128;512;1024)
    decode_tokens    speed: tokens generated after each prefill, N (0 measures prefill only)
    repeats          speed: timed repeats per context length, after --warmup untimed ones
    notes            free text, saved with the run (quote it if it has commas)
The four speed columns are required when the row runs speed and ignored otherwise.

Models and datasets are read only from ../LUT-Transformer-local (checkpoints/ and datasets/;
reference/local_cache.py and common/data.py), never downloaded on the fly: a row naming a model or a
dataset that isn't there stops the sweep before anything runs, with the list of what is installed.
--download allows fetching Hugging Face models and the known datasets into that folder (nothing is
downloaded without it).

speed, for each context length T (float32, TF32 off: common/precision.py):
  - prompt: B x T random token ids (seeded), the same for every arm.
  - one repeat = prefill of the prompt, then N decode steps. Each is timed on its own with a device
    synchronize around it. Prefill time includes picking the first new token (argmax), so it is the
    time to first token; each decode step takes one token per sequence in and gives the next out
    (argmax included). Our arms run the KV cache from baseline/inference.py; the hf arm uses
    transformers' own cache, and asks for the last position's logits only (logits_to_keep=1), as
    our prefill does, so neither arm pays for an output layer over the whole prompt.
  - prefill_ms (median, p10, p90 over the repeats) and prefill tokens/s = B * T / median;
    decode_ms_per_token (median, p10, p90 over every decode step of every repeat; one step makes one
    token for each of the B sequences) and decode tokens/s = B / median.
  - power, per phase: the phase runs back to back for --power-warmup-seconds unsampled (so the
    device reaches its steady power state first: GPU power ramped for 0.8 s in run
    20261119T060430Z_matmul_cuda_4096_float32), then for --power-seconds while power is sampled.
    Average power and energy per token come from the device's own source: GPU board power from
    nvidia-smi for cuda rows, CPU package power (RAPL) for cpu rows. Sources are never summed. If a
    source can't be read (RAPL is root-only on alpha-pc), the row records why and skips its windows.
    Idle power is sampled once, before anything is loaded.
  - timing in PyTorch measures emulation for LUT and integer stages, not FPGA speed (agent.md 2.6).

perplexity, for each dataset: the same text for every model (common/data.py eval_text): the
dataset's evaluation split with its document separators replaced by blank lines, cut to the
dataset's character limit (or --max-eval-chars) at the last whitespace. It is tokenized with each
model's own tokenizer and read in windows of the model's context length W (1,024 for GPT-2, 256 for
stories15M) that start every W/2 tokens. Each window is one forward pass, and only the tokens a window
adds are scored, each with at least W/2 tokens of context (Hugging Face's method for fixed-length
models). The loss is in float64, the same for every arm, and every token after the first is scored
once. Recorded with each result: the dataset file's sha256, the window and stride, and
    nll_total        sum of -ln p(token | context) over every scored token, in nats
    perplexity       exp(nll_total / scored tokens)       per token: depends on the tokenizer
    bits_per_byte    nll_total / (ln 2 * UTF-8 bytes of the text)      compare across tokenizers
    word_perplexity  exp(nll_total / whitespace-separated words)       compare across tokenizers

Cross-checking (Alpha, 2026-11-19: every model must "cross check datasets" with Alpha's own model).
Alpha's model is the lut arm. Before anything runs, every row to run must cover the same (dataset,
benchmark) cells as the lut rows to run: speed on synthetic tokens, and perplexity on each of their
datasets. Rows with random weights are held to the speed cells only. A row that misses a cell
stops the sweep before anything runs, with the missing cells named; --allow-gaps runs it anyway.
The plan prints the coverage matrix (rows by cells) either way.

Time estimates. Before the sweep starts it prints a plan: each row, whether it was trained before,
its estimated time, where the estimate came from, and the total. An estimate comes from
  - history: earlier rows with the same arm, model, preset, device and batch, kept in
    ../LUT-Transformer-local/sweep-history.jsonl (appended after every row; interpolated or
    extrapolated in T when the context length is new), or
  - calibration: when there is no history, a short run of the row's model (one untimed and one timed
    prefill plus up to 4 decode steps per context length; one untimed and one timed perplexity window
    per dataset) before the plan is printed. --no-calibrate skips it and leaves those rows unknown.
    --dry-run calibrates too, unless told not.
While it runs: an outer bar over all rows (rows done, elapsed, remaining) and an inner bar for the
current row (warm-up, repeats, power windows, perplexity windows). Both are measured in estimated
seconds, so the remaining time is the plan's estimate corrected by the speed seen so far. Bars are
updated only between timed calls and outside power windows (bench/progress.py). The sweep folder's
status.json holds the same numbers (rows done, elapsed, remaining), for checking on a sweep from
elsewhere.

Results:
  results/runs/<UTC time>_sweep_<row name>/   one run per row (bench/runlog.py): meta.json,
                                              metrics.jsonl (every repeat's and window's times),
                                              power_*.json, summary.json; plus a row in
                                              results/summary.csv
  results/sweeps/<UTC time>_<csv name>/       models.csv (the CSV as run), plan.csv (estimates and
                                              the rows that need training), summary.csv (one line
                                              per row, benchmark and context length or dataset: the
                                              numbers above, the run id, estimated and actual time),
                                              results_long.csv (one line per row, dataset,
                                              benchmark, metric and context length, with its value
                                              and unit: the layout for a spreadsheet's pivot table),
                                              status.json
  results/all_results.csv                     results_long.csv of every sweep, appended after every
                                              row; a run id already in it is never added twice
Git ignores only the raw power samples (power_samples*.json); the rest can be committed. The CSVs
are plain UTF-8 with a header row and open directly in Excel.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import shutil
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from baseline import host as base
from bench import resources
from bench.progress import WeightedBar, format_seconds, write
from bench.resources import warm_up_until_steady
from bench.runlog import REPO_ROOT, Run
from bench.telemetry import PowerMonitor
from bench.timing import _synchronize, summarize
from common import data, precision
from LUT import host as lut_host
from reference import local_cache

PYTHON_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PYTHON_DIR / "configs/sweep/models.csv"
CORE_CONFIG = PYTHON_DIR / "configs/model/core.yaml"
# The core's vocabulary until the tokenizer is chosen (D2): GPT-2's, the plan's recommendation.
# It sets the size of the token table and the output layer, which dominate a small model's cost.
CORE_VOCAB = 50257

COLUMNS = ("enabled", "name", "arm", "model", "weights", "lut_preset", "device", "benchmarks", "datasets",
           "batch", "context_lengths", "decode_tokens", "repeats", "notes")
REQUIRED_COLUMNS = ("enabled", "name", "arm", "model", "device")
SPEED_COLUMNS = ("batch", "context_lengths", "decode_tokens", "repeats")
ARMS = ("hf", "baseline", "lut")
WEIGHTS = ("pretrained", "random", "train")
BENCHMARKS = ("speed", "perplexity")
MODEL_ALIASES = {"gpt2": "openai-community/gpt2",
                 # Karpathy's llama2.c stories15M (trained on TinyStories) in Hugging Face's format; its 56
                 # tensors match his original stories15M.pt (karpathy/tinyllamas) bit for bit (2026-11-19)
                 "stories15m": "Xenova/llama2.c-stories15M"}
# Architectures each arm can run. The hf arm runs whatever transformers runs: tested on gpt2 and
# llama; opt and qwen2 are listed for the research rows in models.csv and are untested until one is
# downloaded. Our from-scratch arms need a profile per architecture (reference/hf_profiles.py).
HF_ARCHITECTURES = ("gpt2", "llama", "opt", "qwen2")
OUR_ARCHITECTURES = ("gpt2", "llama")
PHASES = ("prefill", "decode")

CALIBRATION_DECODE_STEPS = 4      # decode steps per context length in a calibration run
OVERHEAD_S = 1.0                  # per row: run folder, system info, cache and prompt setup
UNKNOWN_ROW_S = 60.0              # bar weight for a row with no estimate (only with --no-calibrate)
SPEED_CELL = ("synthetic", "speed")   # the speed benchmark's (dataset, benchmark) cell
LN2 = math.log(2.0)


class SweepError(ValueError):
    """A problem found before the sweep runs; the message says what to fix."""


# ----------------------------------------------------------------------------------------------
# The CSV
# ----------------------------------------------------------------------------------------------

@dataclass
class Row:
    """One CSV row, parsed. lut_preset is "parameters" (the PARAMETERS block), a PRESETS name, or
    None for the hf and baseline arms. weights is None until validate() fills in the default."""

    line: int
    enabled: bool
    name: str
    arm: str
    model: str
    lut_preset: str | None
    device: str
    batch: int
    context_lengths: list[int]
    decode_tokens: int
    repeats: int
    notes: str = ""
    weights: str | None = None
    datasets: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=lambda: ["speed"])

    @property
    def preset_label(self) -> str:
        return {None: "-", "parameters": "params"}.get(self.lut_preset, self.lut_preset or "-")

    @property
    def runs_speed(self) -> bool:
        return "speed" in self.benchmarks

    @property
    def eval_datasets(self) -> list[str]:
        """The datasets perplexity runs on (none if the row doesn't run perplexity)."""
        return self.datasets if "perplexity" in self.benchmarks else []

    @property
    def benchmark_label(self) -> str:
        parts = ["speed"] if self.runs_speed else []
        if self.eval_datasets:
            parts.append("ppl:" + ",".join(self.eval_datasets))
        return ";".join(parts)

    def where(self, csv_name: str) -> str:
        return f"{csv_name} line {self.line} ({self.name or '?'})"


def _positive_int(text: str, column: str, minimum: int = 1) -> int:
    try:
        value = int(text)
    except ValueError:
        raise SweepError(f"{column} must be a whole number, got {text!r}") from None
    if value < minimum:
        raise SweepError(f"{column} must be at least {minimum}, got {value}")
    return value


def _list(text: str) -> list[str]:
    return [part.strip().lower() for part in text.split(";") if part.strip()]


@dataclass
class Defaults:
    """Sweep-wide values for blank benchmarks and datasets cells (a '# defaults:' line in the CSV)."""

    benchmarks: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    line: int | None = None

    def describe(self) -> str:
        parts = [f"benchmarks={';'.join(self.benchmarks)}"] if self.benchmarks else []
        parts += [f"datasets={';'.join(self.datasets)}"] if self.datasets else []
        return " ".join(parts)


DEFAULTS_KEYS = ("benchmarks", "datasets")


def parse_defaults(lines: list[str], csv_name: str) -> tuple[Defaults, list[str]]:
    """Find the '# defaults: benchmarks=speed;perplexity datasets=tinystories' comment line, if any.
    Returns (Defaults, errors). Keys are separated by spaces or commas; values by ';'."""
    found = [(number, line) for number, line in enumerate(lines, 1)
             if line.lstrip().startswith("#") and line.lstrip("# \t").lower().startswith("defaults:")]
    if not found:
        return Defaults(), []
    errors = [f"{csv_name} line {n}: a second '# defaults:' line (only one is allowed)" for n, _ in found[1:]]
    number, line = found[0]
    defaults = Defaults(line=number)
    body = line.lstrip("# \t")[len("defaults:"):]
    for item in [part for part in re.split(r"[\s,]+", body) if part]:
        key, _, value = item.partition("=")
        key = key.strip().lower()
        if key not in DEFAULTS_KEYS or not value:
            errors.append(f"{csv_name} line {number}: '# defaults:' takes {' and '.join(k + '=a;b' for k in DEFAULTS_KEYS)}, "
                          f"got {item!r}")
            continue
        setattr(defaults, key, _list(value))
    unknown = [b for b in defaults.benchmarks if b not in BENCHMARKS]
    unknown += [d for d in defaults.datasets if d not in data.DATASETS]
    if unknown:
        errors.append(f"{csv_name} line {number}: unknown default(s) {', '.join(unknown)}")
    return defaults, errors


def _parse_row(values: dict[str, str], line: int, defaults: Defaults | None = None) -> Row:
    """One row's cells -> Row, with blank benchmarks and datasets cells taken from the CSV's
    defaults. Raises SweepError listing every bad cell in the row."""
    defaults = defaults or Defaults()
    problems: list[str] = []

    def check(fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except SweepError as error:
            problems.append(str(error))
            return default

    enabled_text = values.get("enabled", "").lower()
    if enabled_text in ("", "yes", "y", "true", "1"):
        enabled = True
    elif enabled_text in ("no", "n", "false", "0"):
        enabled = False
    else:
        problems.append(f"enabled must be yes or no, got {values['enabled']!r}")
        enabled = False
    name = values.get("name", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        problems.append(f"name must be letters, digits, _ . or -, got {name!r}")
    arm = values.get("arm", "").lower()
    if arm not in ARMS:
        problems.append(f"arm must be one of {', '.join(ARMS)}, got {values.get('arm')!r}")
    model = values.get("model", "")
    if not model:
        problems.append("model is empty")
    weights = values.get("weights", "").lower() or None
    if weights is not None and weights not in WEIGHTS:
        problems.append(f"weights must be one of {', '.join(WEIGHTS)} (or blank), got {values['weights']!r}")
    preset_text = values.get("lut_preset", "")
    preset: str | None = None if preset_text in ("", "-") else preset_text
    if arm == "lut":
        preset = "parameters" if preset in (None, "params", "parameters") else preset
    elif preset is not None and arm in ARMS:
        problems.append(f"lut_preset applies to the lut arm only; leave it blank for {arm}")
        preset = None
    device = values.get("device", "").lower()
    if not re.fullmatch(r"cpu|cuda(:\d+)?", device):
        problems.append(f"device must be cpu, cuda or cuda:N, got {values.get('device')!r}")

    # Blank cells take the CSV's defaults. Random weights (the core, unless the row says otherwise)
    # take only the speed part: perplexity of random weights measures nothing.
    effective_weights = weights or ("random" if model == "core" else "pretrained")
    benchmarks = _list(values.get("benchmarks", ""))
    if not benchmarks:
        benchmarks = [b for b in defaults.benchmarks if b == "speed" or effective_weights == "pretrained"] or ["speed"]
    unknown = [b for b in benchmarks if b not in BENCHMARKS]
    if unknown:
        problems.append(f"benchmarks must be from {', '.join(BENCHMARKS)}, got {', '.join(unknown)}")
    datasets = _list(values.get("datasets", ""))
    if not datasets and "perplexity" in benchmarks:
        datasets = list(defaults.datasets)
    unknown = [d for d in datasets if d not in data.DATASETS]
    if unknown:
        problems.append(f"unknown dataset(s) {', '.join(unknown)}; known: {', '.join(data.DATASETS)} "
                        f"(common/data.py DATASETS)")
    for column, items in (("benchmarks", benchmarks), ("datasets", datasets)):
        if len(set(items)) != len(items):
            problems.append(f"{column} repeats a value: {values.get(column)!r}")
    if "perplexity" in benchmarks and not datasets:
        problems.append("perplexity needs at least one dataset in the datasets column")

    # The speed columns: required when the row runs speed, otherwise checked only if filled in.
    speed = "speed" in benchmarks
    if speed:
        missing = [c for c in SPEED_COLUMNS if not values.get(c)]
        if missing:
            problems.append(f"the speed benchmark needs {', '.join(missing)}")
    batch = check(lambda: _positive_int(values["batch"], "batch"), 1) if values.get("batch") else 1

    def contexts() -> list[int]:
        parts = [p.strip() for p in values.get("context_lengths", "").split(";") if p.strip()]
        if not parts:
            raise SweepError("context_lengths is empty (example: 128;512;1024)")
        lengths = [_positive_int(p, "each context length") for p in parts]
        if len(set(lengths)) != len(lengths):
            raise SweepError(f"context_lengths repeats a value: {values['context_lengths']!r}")
        return lengths

    context_lengths = check(contexts, []) if values.get("context_lengths") else []
    decode_tokens = (check(lambda: _positive_int(values["decode_tokens"], "decode_tokens", 0), 0)
                     if values.get("decode_tokens") else 0)
    repeats = check(lambda: _positive_int(values["repeats"], "repeats"), 1) if values.get("repeats") else 1
    if problems:
        raise SweepError("; ".join(problems))
    return Row(line=line, enabled=enabled, name=name, arm=arm, model=model, lut_preset=preset,
               device=device, batch=batch, context_lengths=context_lengths if speed else [],
               decode_tokens=decode_tokens, repeats=repeats, notes=values.get("notes", ""),
               weights=weights, datasets=datasets, benchmarks=benchmarks)


def read_csv(path: Path) -> tuple[list[Row], list[str]]:
    """Parse the sweep CSV. Returns (rows, errors); errors name the line and the column."""
    path = Path(path)
    if not path.is_file():
        return [], [f"{path}: no such file"]
    all_lines = path.read_text(encoding="utf-8").splitlines()
    defaults, default_errors = parse_defaults(all_lines, path.name)
    if default_errors:
        return [], default_errors
    numbered = [(number, line) for number, line in enumerate(all_lines, 1)
                if line.strip() and not line.lstrip().startswith("#")]
    if not numbered:
        return [], [f"{path.name}: no header row"]
    table = list(csv.reader([line for _, line in numbered], skipinitialspace=True))
    header = [cell.strip().lower() for cell in table[0]]
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    unknown = [c for c in header if c not in COLUMNS]
    if missing:
        errors.append(f"{path.name}: missing column(s) {', '.join(missing)}")
    if unknown:
        errors.append(f"{path.name}: unknown column(s) {', '.join(unknown)}; the columns are {', '.join(COLUMNS)}")
    if errors:
        return [], errors
    rows: list[Row] = []
    for (number, _), cells in zip(numbered[1:], table[1:]):
        if len(cells) > len(header):
            errors.append(f"{path.name} line {number}: {len(cells)} cells but {len(header)} columns "
                          f"(quote notes that contain commas)")
            continue
        values = {column: (cells[i].strip() if i < len(cells) else "") for i, column in enumerate(header)}
        try:
            rows.append(_parse_row(values, number, defaults))
        except SweepError as error:
            errors.append(f"{path.name} line {number} ({values.get('name') or '?'}): {error}")
    names = [row.name for row in rows]
    for name in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"{path.name}: row name {name!r} is used more than once")
    if not rows and not errors:
        errors.append(f"{path.name}: no rows under the header")
    return rows, errors


# ----------------------------------------------------------------------------------------------
# Models: only what is installed in ../LUT-Transformer-local/checkpoints
# ----------------------------------------------------------------------------------------------

@dataclass
class ModelSource:
    """Where a row's weights come from.

    key            the model's name for history and results: "core", a repo id, or a folder name
    kind           "core" (no published weights) or "pretrained" (loaded through reference/hf_profiles.py)
    load_name      what from_pretrained gets: the installed snapshot or folder, or a repo id to download
    installed      False only for a repo id that --download will fetch
    max_context    a hard limit from a learned position table (GPT-2's n_positions), or None (the
                   core's sinusoidal codes, Llama's rotary ones)
    has_tokenizer  its tokenizer files are installed with it (perplexity needs them)
    architecture   "core", or the Hugging Face model_type ("gpt2", "llama")
    eval_context   the context it was trained with, the perplexity window (None for the core: its config's)
    """

    key: str
    kind: str
    load_name: str | None
    installed: bool
    max_context: int | None
    vocab_size: int | None
    has_tokenizer: bool = False
    architecture: str = "core"
    eval_context: int | None = None


def _snapshot(repo_folder: Path) -> Path | None:
    """The snapshot folder of a Hugging Face cache entry (models--org--name), or None."""
    if not repo_folder.is_dir():
        return None
    ref = repo_folder / "refs" / "main"
    if ref.is_file():
        snapshot = repo_folder / "snapshots" / ref.read_text().strip()
        if (snapshot / "config.json").is_file():
            return snapshot
    for snapshot in sorted((repo_folder / "snapshots").glob("*")):
        if (snapshot / "config.json").is_file():
            return snapshot
    return None


def _has_weights(folder: Path) -> bool:
    return any(folder.glob("*.safetensors")) or (folder / "pytorch_model.bin").is_file() \
        or (folder / "model.safetensors.index.json").is_file()


def _has_tokenizer(folder: Path) -> bool:
    return (folder / "tokenizer.json").is_file() or \
        ((folder / "vocab.json").is_file() and (folder / "merges.txt").is_file())


def installed_models(checkpoint_dir: Path) -> list[str]:
    """Models installed in checkpoint_dir: Hugging Face repo ids (from its models--org--name
    folders) and plain folders holding a saved model."""
    found = []
    if not checkpoint_dir.is_dir():
        return found
    for child in sorted(checkpoint_dir.iterdir()):
        if child.name.startswith("models--"):
            snapshot = _snapshot(child)
            if snapshot is not None and _has_weights(snapshot):
                found.append(child.name[len("models--"):].replace("--", "/", 1))
        elif child.is_dir() and (child / "config.json").is_file() and _has_weights(child):
            found.append(child.name)
    return found


def resolve_model(spec: str, checkpoint_dir: Path, *, download: bool = False) -> ModelSource:
    """A CSV model name -> ModelSource, reading only checkpoint_dir. Raises SweepError if the model
    isn't installed (and can't or may not be downloaded) or isn't a GPT-2-architecture model."""
    if spec == "core":
        return ModelSource("core", "core", None, True, None, CORE_VOCAB)
    installed = installed_models(checkpoint_dir)
    listing = f"Installed in {checkpoint_dir}: {', '.join(installed) or 'none'}."
    if spec.startswith("models--"):
        spec = spec[len("models--"):].replace("--", "/", 1)
    repo = MODEL_ALIASES.get(spec, spec)
    if "/" in repo:
        snapshot = _snapshot(checkpoint_dir / ("models--" + repo.replace("/", "--")))
        if snapshot is None or not _has_weights(snapshot):
            if download:
                return ModelSource(repo, "pretrained", repo, False, None, None, has_tokenizer=True,
                                   architecture="unknown until downloaded")
            raise SweepError(f"model {repo!r} is not installed. {listing} Nothing is downloaded unless "
                             f"you pass --download (it fetches the repo into that folder).")
        folder, load_name = snapshot, str(snapshot)   # the folder itself: works offline even without refs/main
    else:
        folder = checkpoint_dir / spec
        if not ((folder / "config.json").is_file() and _has_weights(folder)):
            raise SweepError(f"model {spec!r} is neither 'core', 'gpt2', a Hugging Face repo id (org/name), "
                             f"nor a folder with config.json and weights in {checkpoint_dir}. {listing}")
        load_name = str(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    architecture = config.get("model_type")
    if architecture not in HF_ARCHITECTURES:
        raise SweepError(f"model {repo!r} is a {architecture!r} model; the sweep runs {', '.join(HF_ARCHITECTURES)} "
                         f"models for now")
    trained_context = config.get("n_positions") or config.get("n_ctx") or config.get("max_position_embeddings")
    return ModelSource(repo, "pretrained", load_name, True, trained_context if architecture == "gpt2" else None,
                       config.get("vocab_size"), has_tokenizer=_has_tokenizer(folder), architecture=architecture,
                       eval_context=trained_context)


@dataclass
class Settings:
    """Everything the command line sets. The defaults are the command's defaults."""

    csv_path: Path = DEFAULT_CSV
    only: list[str] | None = None
    warmup: int = 2
    power: bool = True
    power_seconds: float = 3.0
    power_warmup_s: float = 2.0
    idle_seconds: float = 3.0
    seed: int = 0
    download: bool = False
    calibrate: bool = True
    max_eval_chars: int | None = None      # None: each dataset's eval_char_limit; 0: the whole split
    allow_gaps: bool = False               # run rows that miss a cell the lut rows measure
    results_dir: Path = REPO_ROOT / "results"
    history_path: Path = field(default_factory=lambda: local_cache.LOCAL_DIR / "sweep-history.jsonl")
    checkpoint_dir: Path = field(default_factory=lambda: local_cache.CHECKPOINT_DIR)
    show_progress: bool | None = None


def select_rows(rows: list[Row], only: list[str] | None) -> tuple[list[Row], list[str]]:
    """The enabled rows (or the --only rows, enabled or not), in CSV order."""
    if not only:
        return [row for row in rows if row.enabled], []
    names = {row.name for row in rows}
    errors = [f"--only: no row named {name!r}" for name in only if name not in names]
    return [row for row in rows if row.name in only], errors


def datasets_to_download(rows: list[Row]) -> list[str]:
    """Datasets the runnable rows evaluate on that aren't in the local folder yet."""
    needed = {d for row in rows if row.weights != "train" for d in row.eval_datasets}
    return sorted(d for d in needed if not data.is_installed(data.DATASETS[d]))


def validate(rows: list[Row], settings: Settings) -> tuple[dict[str, ModelSource], list[str]]:
    """Check the rows against this machine and the trained-before rule: the model is installed, it
    is marked as trained or not correctly, the datasets are installed and readable, the device
    exists, the preset exists, the context fits. Fills in each row's default weights. Returns
    ({row name: ModelSource}, errors)."""
    sources: dict[str, ModelSource] = {}
    errors: list[str] = []
    csv_name = Path(settings.csv_path).name
    for row in rows:
        where = row.where(csv_name)
        try:
            source = resolve_model(row.model, settings.checkpoint_dir, download=settings.download)
        except SweepError as error:
            errors.append(f"{where}: {error}")
            continue
        sources[row.name] = source
        row.weights = row.weights or ("random" if source.kind == "core" else "pretrained")
        if row.weights == "train":
            if source.kind != "core":
                errors.append(f"{where}: {source.key} has already been trained (its weights are published); the "
                              f"sweep never trains a model someone else has trained. Use weights=pretrained.")
            continue                     # listed as "needs training" and not run: nothing else to check
        if row.weights == "pretrained" and source.kind == "core":
            errors.append(f"{where}: the core has no pretrained weights (nobody has trained it); weights must be "
                          f"random (speed only) or train")
        if row.weights == "random" and source.kind != "core":
            errors.append(f"{where}: random weights are for the core; {source.key} is benchmarked with its own "
                          f"trained weights (weights=pretrained)")
        if row.weights == "random" and row.eval_datasets:
            errors.append(f"{where}: perplexity of random weights measures nothing; use a pretrained model, or "
                          f"drop perplexity from benchmarks")
        if row.arm == "hf" and source.kind == "core":
            errors.append(f"{where}: the hf arm needs a Hugging Face model; the core has none")
        if row.arm != "hf" and source.kind == "pretrained" and source.installed \
                and source.architecture not in OUR_ARCHITECTURES:
            errors.append(f"{where}: {source.key} is a {source.architecture} model; our {row.arm} arm runs "
                          f"{', '.join(OUR_ARCHITECTURES)} models only until a {source.architecture} profile is "
                          f"built (reference/hf_profiles.py). Use arm=hf for now.")
        if row.arm == "lut" and row.lut_preset != "parameters" and row.lut_preset not in lut_host.PRESETS:
            errors.append(f"{where}: lut_preset {row.lut_preset!r} is not 'params' or one of "
                          f"LUT/host.py PRESETS ({', '.join(lut_host.PRESETS)})")
        if row.device.startswith("cuda"):
            index = int(row.device.split(":")[1]) if ":" in row.device else 0
            if not torch.cuda.is_available():
                errors.append(f"{where}: device {row.device} but PyTorch sees no CUDA GPU")
            elif index >= torch.cuda.device_count():
                errors.append(f"{where}: device {row.device} but there are {torch.cuda.device_count()} GPU(s)")
        if row.runs_speed and source.max_context is not None:
            too_long = [t for t in row.context_lengths if t + row.decode_tokens > source.max_context]
            if too_long:
                errors.append(f"{where}: context {too_long} + {row.decode_tokens} decode tokens "
                              f"exceeds the model's {source.max_context} learned positions")
        if row.eval_datasets and not source.has_tokenizer:
            errors.append(f"{where}: perplexity needs {source.key}'s tokenizer, which isn't installed with it")
        for name in row.eval_datasets:
            spec = data.DATASETS[name]
            requirement = data.missing_requirement(spec)
            if requirement:
                errors.append(f"{where}: reading {name} needs the {requirement} package, which isn't installed "
                              f"(installing a package needs Alpha's OK)")
            if not data.is_installed(spec) and not settings.download:
                errors.append(f"{where}: dataset {name!r} is not installed in {data.DATASET_DIR} (installed: "
                              f"{', '.join(data.installed_datasets()) or 'none'}). --download fetches it from "
                              f"{spec.hf_repo}: {spec.about}")
    return sources, errors


# ----------------------------------------------------------------------------------------------
# Loading and running a row's model
# ----------------------------------------------------------------------------------------------

@dataclass
class Loaded:
    """A loaded model: our config and parameters (None and {} for a model only the hf arm runs), and
    the Hugging Face model for the hf arm."""

    cfg: base.TransformerConfig | None
    params: dict[str, Tensor]
    hf_model: Any | None

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size if self.cfg is not None else self.hf_model.config.vocab_size

    def config_dict(self) -> dict[str, Any]:
        return vars(self.cfg) if self.cfg is not None else self.hf_model.config.to_dict()


def load_model(source: ModelSource, device: str, settings: Settings) -> Loaded:
    if source.kind == "core":
        cfg = base.TransformerConfig.from_yaml(CORE_CONFIG)
        cfg.vocab_size = CORE_VOCAB
        return Loaded(cfg, base.init_params(cfg, seed=settings.seed, device=device), None)
    if source.architecture not in OUR_ARCHITECTURES:
        # hf arm only: transformers' own model class, nothing in our layout
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(source.load_name, cache_dir=str(settings.checkpoint_dir),
                                                        dtype=base.DTYPE, local_files_only=not settings.download)
        return Loaded(None, {}, hf_model.eval())
    from reference import hf_profiles

    cfg, params, hf_model = hf_profiles.load(source.architecture, source.load_name, device=device,
                                             cache_dir=settings.checkpoint_dir, local_files_only=not settings.download)
    return Loaded(cfg, params, hf_model)


class ModelCache:
    """Keeps the most recently loaded model, so a calibration and the row after it, or rows in a
    row on the same model and device, load it once."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key: tuple[str, str] | None = None
        self.loaded: Loaded | None = None
        self.load_seconds: dict[tuple[str, str], float] = {}

    def get(self, source: ModelSource, device: str) -> tuple[Loaded, float]:
        """(the loaded model, seconds spent loading it now: 0 if it was already loaded)."""
        key = (source.key, device)
        if key == self.key and self.loaded is not None:
            return self.loaded, 0.0
        self.release()
        start = time.perf_counter()
        self.loaded = load_model(source, device, self.settings)
        _synchronize(device)
        seconds = time.perf_counter() - start
        self.key = key
        self.load_seconds[key] = seconds
        return self.loaded, seconds

    def release(self) -> None:
        self.key, self.loaded = None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OurRunner:
    """The baseline or LUT host's forward, prefill and decode_step, decoding on a preallocated KV cache."""

    def __init__(self, host: Any, params: dict[str, Tensor], ops: Any, cfg: base.TransformerConfig) -> None:
        self.host, self.params, self.ops, self.cfg = host, params, ops, cfg
        self.cache = None

    def forward(self, ids: Tensor) -> Tensor:
        """Logits (B, T, V) for every position, no cache (perplexity)."""
        return self.host.forward(self.params, ids, self.ops, self.cfg)

    def start(self, batch: int, max_len: int) -> None:
        self.cache = self.host.new_cache(self.params, self.cfg, batch, max_len)

    def prefill(self, ids: Tensor) -> Tensor:
        self.cache.length = 0                     # reuse the buffers; prefill fills them from 0
        return self.host.prefill(self.params, ids, self.cache, self.ops, self.cfg).argmax(-1, keepdim=True)

    def decode(self, ids: Tensor) -> Tensor:
        return self.host.decode_step(self.params, ids, self.cache, self.ops, self.cfg).argmax(-1, keepdim=True)

    def rewind(self, length: int) -> None:
        self.cache.length = length

    def stop(self) -> None:
        self.cache = None


class HFRunner:
    """The Hugging Face model's own forward with its own KV cache (past_key_values)."""

    def __init__(self, model: Any) -> None:
        import inspect

        self.model = model
        self.past = None
        keep = "logits_to_keep" in inspect.signature(model.forward).parameters
        self.extra = {"logits_to_keep": 1} if keep else {}

    def forward(self, ids: Tensor) -> Tensor:
        """Logits (B, T, V) for every position, no cache (perplexity)."""
        with torch.inference_mode():
            return self.model(input_ids=ids, use_cache=False).logits

    def start(self, batch: int, max_len: int) -> None:
        self.past = None

    def prefill(self, ids: Tensor) -> Tensor:
        with torch.inference_mode():
            out = self.model(input_ids=ids, use_cache=True, **self.extra)
        self.past = out.past_key_values
        return out.logits[:, -1].argmax(-1, keepdim=True)

    def decode(self, ids: Tensor) -> Tensor:
        with torch.inference_mode():
            out = self.model(input_ids=ids, past_key_values=self.past, use_cache=True, **self.extra)
        self.past = out.past_key_values
        return out.logits[:, -1].argmax(-1, keepdim=True)

    def rewind(self, length: int) -> None:
        # transformers 5 crops by a negative number of tokens to remove (a positive number, the
        # older "final length" form, raises).
        remove = self.past.get_seq_length() - length
        if remove > 0:
            self.past.crop(-remove)
        if self.past.get_seq_length() != length:
            raise RuntimeError(f"Hugging Face cache rewound to {self.past.get_seq_length()} tokens, not {length}")

    def stop(self) -> None:
        self.past = None


def make_runner(row: Row, loaded: Loaded) -> OurRunner | HFRunner:
    cfg = loaded.cfg
    if row.arm == "hf":
        return HFRunner(loaded.hf_model.to(row.device).eval())
    if row.arm == "baseline":
        return OurRunner(base, loaded.params, base.base_ops(cfg, "inference"), cfg)
    params = {name: t.to(lut_host.DTYPE) for name, t in loaded.params.items()}
    ops = lut_host.lut_ops(cfg, "inference", lut_host.preset_toggles(row.lut_preset))
    return OurRunner(lut_host, params, ops, cfg)


def row_dtype(row: Row, loaded: Loaded) -> str:
    if row.arm == "hf":
        return str(next(loaded.hf_model.parameters()).dtype)
    if row.arm == "lut":
        return str(lut_host.DTYPE)
    return str(next(iter(loaded.params.values())).dtype)


def make_prompt(vocab_size: int, batch: int, length: int, seed: int, device: str) -> Tensor:
    """B x T random token ids (int64), the same for every arm and row with this seed and T."""
    generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + length)
    return torch.randint(0, vocab_size, (batch, length), generator=generator, dtype=torch.int64).to(device)


def measure_context(runner: OurRunner | HFRunner, ids: Tensor, decode_tokens: int, *, warmup: int, repeats: int,
                    device: str, after_repeat: Callable[[int], Any] | None = None) -> tuple[list[float], list[float]]:
    """warmup + repeats times: prefill, then decode_tokens decode steps, each timed on its own with
    a synchronize around it. Returns (prefill ms per timed repeat, decode ms per timed step).
    after_repeat(i) runs after repeat i's last stop timestamp (the place for progress updates)."""
    prefill_ms: list[float] = []
    decode_ms: list[float] = []
    for i in range(warmup + repeats):
        _synchronize(device)
        start = time.perf_counter()
        token = runner.prefill(ids)
        _synchronize(device)
        stop = time.perf_counter()
        steps = []
        for _ in range(decode_tokens):
            step_start = time.perf_counter()
            token = runner.decode(token)
            _synchronize(device)
            steps.append((time.perf_counter() - step_start) * 1000.0)
        if i >= warmup:
            prefill_ms.append((stop - start) * 1000.0)
            decode_ms.extend(steps)
        if after_repeat is not None:
            after_repeat(i)
    return prefill_ms, decode_ms


# ----------------------------------------------------------------------------------------------
# Perplexity
# ----------------------------------------------------------------------------------------------

_TOKENS: dict[tuple[Any, ...], tuple[Tensor, float]] = {}


def char_limit(name: str, settings: Settings) -> int | None:
    """How many characters of the dataset's split perplexity reads, the same for every model:
    --max-eval-chars (0 = all), else the dataset's own limit. None means the whole split."""
    if settings.max_eval_chars is not None:
        return settings.max_eval_chars or None
    return data.DATASETS[name].eval_char_limit


_TEXTS: dict[tuple[str, int | None], data.EvalText] = {}


def eval_text(name: str, settings: Settings) -> data.EvalText:
    """The exact text perplexity reads for this dataset (common/data.py eval_text), cached for the
    sweep: every row, whatever its model, gets this same object."""
    limit = char_limit(name, settings)
    key = (name, limit)
    if key not in _TEXTS:
        _TEXTS[key] = data.eval_text(data.DATASETS[name], limit if limit is not None else 0)
    return _TEXTS[key]


def load_tokenizer(source: ModelSource, settings: Settings) -> Any:
    """The model's own tokenizer, from the local folder (and never the network without --download)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source.load_name, cache_dir=str(settings.checkpoint_dir),
                                              local_files_only=not settings.download)
    tokenizer.model_max_length = 10**12     # one long stream on purpose; windows are cut below
    return tokenizer


def eval_tokens(source: ModelSource, name: str, settings: Settings) -> tuple[Tensor, float]:
    """(the dataset's evaluation text, eval_text, as this model's token ids (N,), seconds it took to
    tokenize), cached for the rest of the sweep. Every model tokenizes the same text, so token
    counts differ between tokenizers while the text does not."""
    text = eval_text(name, settings)
    key = (source.key, name, text.char_limit)
    if key not in _TOKENS:
        start = time.perf_counter()
        ids = load_tokenizer(source, settings)(text.text, return_attention_mask=False)["input_ids"]
        if len(ids) < 2:
            raise SweepError(f"dataset {name} gave {len(ids)} token(s); perplexity needs at least 2")
        _TOKENS[key] = (torch.tensor(ids, dtype=torch.int64), time.perf_counter() - start)
    return _TOKENS[key]


def eval_window(source: ModelSource, cfg: base.TransformerConfig | None) -> tuple[int, int]:
    """(window W, stride W/2): the context length the model was trained with."""
    window = source.eval_context or cfg.context_length
    return window, max(1, window // 2)


def count_windows(n_tokens: int, window: int, stride: int) -> int:
    """Windows that start every `stride` tokens until one reaches the end."""
    return 1 if n_tokens <= window else math.ceil((n_tokens - window) / stride) + 1


def measure_perplexity(forward: Callable[[Tensor], Tensor], tokens: Tensor, *, window: int, stride: int,
                       device: str, after_window: Callable[[int], Any] | None = None) -> dict[str, Any]:
    """Sliding-window perplexity of tokens (N,): windows [b, b + W) for b = 0, S, 2S, ... until one
    reaches the end. In each window, the logits at position t - 1 score token t for every t the
    earlier windows didn't reach (t >= 1), so each token after the first is scored exactly once, with
    the window's earlier tokens as context. Loss in float64: logsumexp(z) - z[target]. Accuracy over
    the same scored tokens: top-1 = the true next token is the argmax, top-5 = it is among the 5 most
    likely (ties broken by torch.topk).
    after_window(i) runs after window i is scored (progress updates)."""
    n = tokens.numel()
    nll, scored, windows, prev_end, begin = 0.0, 0, 0, 0, 0
    top1 = top5 = 0                 # next-token accuracy: the true token is the argmax / in the top 5
    window_s: list[float] = []
    forward_ms: list[float] = []
    started = time.perf_counter()
    while True:
        end = min(begin + window, n)
        x = tokens[begin:end][None].to(device)
        _synchronize(device)
        step_start = time.perf_counter()
        logits = forward(x)
        _synchronize(device)
        forward_ms.append((time.perf_counter() - step_start) * 1000.0)
        first = max(prev_end, begin + 1)
        targets = tokens[first:end].to(device)
        z = logits[0, first - 1 - begin:end - 1 - begin].double()
        nll += float((torch.logsumexp(z, dim=-1) - z.gather(-1, targets[:, None])[:, 0]).sum())
        best5 = z.topk(5, dim=-1).indices                                  # (scored, 5), best first
        top1 += int((best5[:, 0] == targets).sum())
        top5 += int((best5 == targets[:, None]).any(dim=-1).sum())
        scored += end - first
        windows += 1
        window_s.append(time.perf_counter() - step_start)
        del logits, z
        if after_window is not None:
            after_window(windows - 1)
        if end == n:
            break
        prev_end, begin = end, begin + stride
    return {"perplexity": math.exp(nll / scored), "nll_per_token": nll / scored, "nll_total": nll, "eval_tokens": n,
            "top1_accuracy": top1 / scored, "top5_accuracy": top5 / scored, "top1_correct": top1, "top5_correct": top5,
            "scored_tokens": scored, "eval_window": window, "eval_stride": stride, "eval_windows": windows,
            "eval_s": time.perf_counter() - started, "forward_ms_median": statistics.median(forward_ms),
            "window_s_median": statistics.median(window_s)}


def text_metrics(nll_total: float, n_bytes: int, n_words: int) -> dict[str, float]:
    """Perplexity measures that don't depend on the tokenizer, from the total negative log-likelihood
    of one text (nats, summed over every scored token):

        bits_per_byte   = nll_total / (ln 2 * n_bytes)     n_bytes = UTF-8 bytes of the evaluated text
        word_perplexity = exp(nll_total / n_words)         n_words = its whitespace-separated words

    Token perplexity, exp(nll_total / scored tokens), divides by however many tokens the tokenizer
    cut the text into, so it only compares models that share a tokenizer. These two divide the same
    total by counts that belong to the text, so they compare across tokenizers
    (lm-evaluation-harness reports the same two). Dividing by ln 2 turns nats into bits. Caveat: a
    tokenizer that adds no BOS token (GPT-2's) leaves the text's first token unscored, so its total
    misses one token in hundreds of thousands.
    """
    if n_bytes <= 0 or n_words <= 0:
        raise ValueError(f"the evaluated text must have bytes and words, got {n_bytes} bytes and {n_words} words")
    return {"bits_per_byte": nll_total / (LN2 * n_bytes), "word_perplexity": math.exp(nll_total / n_words)}


# ----------------------------------------------------------------------------------------------
# Cross-checking: every row measures what Alpha's own model (the lut rows) measures
# ----------------------------------------------------------------------------------------------

def row_cells(row: Row) -> set[tuple[str, str]]:
    """The (dataset, benchmark) cells a row measures: speed on synthetic tokens, and perplexity on
    each of its datasets."""
    cells = {SPEED_CELL} if row.runs_speed else set()
    return cells | {(name, "perplexity") for name in row.eval_datasets}


def cell_label(cell: tuple[str, str]) -> str:
    return "speed" if cell == SPEED_CELL else f"ppl:{cell[0]}"


def _cell_order(cell: tuple[str, str]) -> tuple[int, str]:
    return (0, "") if cell == SPEED_CELL else (1, cell[0])


@dataclass
class Coverage:
    """reference: the cells the lut rows (Alpha's model) measure, together. gaps: row name -> the
    reference cells that row doesn't measure (random-weight rows only owe the speed cells)."""

    reference: set[tuple[str, str]]
    lut_rows: list[str]
    gaps: dict[str, list[tuple[str, str]]]


def check_coverage(rows: list[Row]) -> Coverage:
    """Compare every row that will run (rows marked weights=train never run) with the lut rows."""
    runnable = [row for row in rows if row.weights != "train"]
    lut = [row for row in runnable if row.arm == "lut"]
    reference: set[tuple[str, str]] = set().union(*(row_cells(row) for row in lut))
    gaps = {}
    for row in runnable:
        owed = reference if row.weights != "random" else {c for c in reference if c == SPEED_CELL}
        missing = sorted(owed - row_cells(row), key=_cell_order)
        if missing:
            gaps[row.name] = missing
    return Coverage(reference, [row.name for row in lut], gaps)


def format_coverage(rows: list[Row], coverage: Coverage) -> str:
    """The coverage matrix: one line per row (lut rows marked *), one column per cell.
        x    measured
        GAP  not measured, but the lut rows measure it
        .    not measured, and not owed (another row's extra, or perplexity for random weights)"""
    runnable = [row for row in rows if row.weights != "train"]
    cells = sorted(set(coverage.reference).union(*(row_cells(row) for row in runnable)), key=_cell_order)
    header = ["row", "arm", "model", "weights"] + [cell_label(c) for c in cells]
    table = [header]
    for row in runnable:
        have, missing = row_cells(row), set(coverage.gaps.get(row.name, []))
        marks = ["x" if c in have else "GAP" if c in missing else "." for c in cells]
        name = row.name + (" *" if row.arm == "lut" else "")
        table.append([name, row.arm, row.model, row.weights or "-"] + marks)
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    lines = ["Coverage (* = Alpha's model, the lut rows; every other row must measure what they measure):"]
    lines += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in table]
    if not coverage.lut_rows:
        lines.append("  No lut row runs, so there is nothing to cross-check against.")
    elif not coverage.gaps:
        lines.append("  Every row covers the lut rows' cells.")
    return "\n".join(lines)


def coverage_errors(coverage: Coverage) -> list[str]:
    return [f"{name} does not measure {', '.join(cell_label(c) for c in missing)}, which the lut rows "
            f"({', '.join(coverage.lut_rows)}) measure" for name, missing in coverage.gaps.items()]


# ----------------------------------------------------------------------------------------------
# Power
# ----------------------------------------------------------------------------------------------

def power_target(device: str) -> str:
    return "cpu" if device == "cpu" else "cuda"


def measure_idle(targets: set[str], seconds: float) -> dict[str, dict[str, Any]]:
    """Sample each power target with nothing running. Returns {target: {"available", "reason",
    "watts": {source: average W}}}. Also tells which targets can be read at all."""
    out = {}
    for target in sorted(targets):
        monitor = PowerMonitor(target=target, enabled=True, note="idle, before the sweep loads anything")
        monitor.start()
        time.sleep(seconds)
        report = monitor.stop()
        sources = report.get("sources", {})
        watts = {name: d["average_power_watts"] for name, d in sources.items()
                 if d.get("average_power_watts") is not None}
        reasons = [str(d.get("reason")) for d in sources.values() if d.get("reason")]
        out[target] = {"available": bool(watts), "watts": watts,
                       "reason": None if watts else ("; ".join(reasons) or "no power source found")}
    return out


GPU_SETTINGS: dict[str, Any] = {}  # resources.gpu_settings() for the sweep's GPU (main fills it)
POWER_SETTLE_TOLERANCE = 0.03      # warm-up continues until 3 power readings agree within 3% ...
POWER_MAX_WARMUP_S = 10.0          # ... or this many seconds (bench/resources.py warm_up_until_steady)
LAST_WARMUP: dict[str, Any] = {}   # the last power window's warm-up record (length, calls, settled)


def power_window(fn: Callable[[], Any], device: str, target: str, *, seconds: float,
                 warmup_s: float, note: str) -> tuple[dict[str, Any], int, int]:
    """Run fn() back to back unsampled for at least warmup_s and until GPU power is steady (three
    readings within POWER_SETTLE_TOLERANCE), then for `seconds` while power is sampled.
    Returns (PowerMonitor report, calls in the sampled part, calls in the warm-up)."""
    warm = warm_up_until_steady(fn, lambda: _synchronize(device), min_s=warmup_s, device=device,
                                tolerance=POWER_SETTLE_TOLERANCE, max_s=max(warmup_s, POWER_MAX_WARMUP_S))
    warm_calls = warm["warmup_calls"]
    LAST_WARMUP.clear()
    LAST_WARMUP.update(warm)
    monitor = PowerMonitor(target=target, enabled=True, note=note)
    monitor.start()
    calls = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        fn()
        _synchronize(device)
        calls += 1
    return monitor.stop(), calls, warm_calls


def power_numbers(report: dict[str, Any], tokens: int) -> dict[str, dict[str, Any]]:
    """{source: {"avg_power_w", "energy_j", "energy_per_token_mj", "scope"}} per readable source,
    {source: {"unavailable": reason}} otherwise. Sources are kept apart, never summed."""
    out = {}
    for source, d in report.get("sources", {}).items():
        if d.get("energy_joules") is None:
            out[source] = {"unavailable": d.get("reason")}
        else:
            out[source] = {"avg_power_w": d["average_power_watts"], "energy_j": d["energy_joules"],
                           "energy_per_token_mj": d["energy_joules"] / tokens * 1000.0 if tokens else None,
                           "scope": d.get("scope")}
    return out


# ----------------------------------------------------------------------------------------------
# Time estimates
# ----------------------------------------------------------------------------------------------

@dataclass
class Estimate:
    """A row's expected time.

    per_context  speed: T -> (prefill ms, decode ms per step)
    per_dataset  perplexity: dataset -> {"window_s", "windows", "tokenize_s"}
    """

    source: str                     # "history", "history (interpolated)", "calibration", "unknown"
    per_context: dict[int, tuple[float, float]] = field(default_factory=dict)
    load_s: float = 0.0
    overhead_s: float = OVERHEAD_S
    per_dataset: dict[str, dict[str, float]] = field(default_factory=dict)

    def repeat_s(self, length: int, decode_tokens: int) -> float | None:
        if length not in self.per_context:
            return None
        prefill_ms, decode_ms = self.per_context[length]
        return (prefill_ms + decode_tokens * decode_ms) / 1000.0

    def dataset_s(self, name: str) -> float | None:
        if name not in self.per_dataset:
            return None
        p = self.per_dataset[name]
        return p.get("tokenize_s", 0.0) + p["windows"] * p["window_s"]


def power_window_s(settings: Settings) -> float:
    return settings.power_warmup_s + settings.power_seconds


def row_phases(row: Row) -> tuple[str, ...]:
    """The phases with a power window: decode only if the row generates tokens."""
    return PHASES if row.decode_tokens > 0 else ("prefill",)


def row_seconds(row: Row, estimate: Estimate, settings: Settings, *, load_needed: bool,
                power_on: bool) -> float | None:
    """overhead + load (if needed) + speed: for each T, (warmup + repeats) repeats and the power
    windows + perplexity: for each dataset, tokenizing and every window."""
    if estimate.source == "unknown":
        return None
    total = estimate.overhead_s + (estimate.load_s if load_needed else 0.0)
    if row.runs_speed:
        for length in row.context_lengths:
            per_repeat = estimate.repeat_s(length, row.decode_tokens) or 0.0
            total += (settings.warmup + row.repeats) * per_repeat
            if power_on:
                total += len(row_phases(row)) * power_window_s(settings) + per_repeat
    for name in row.eval_datasets:
        total += estimate.dataset_s(name) or 0.0
    return total


def history_key(row: Row, source: ModelSource) -> dict[str, Any]:
    return {"arm": row.arm, "model": source.key, "preset": row.lut_preset, "device": row.device, "batch": row.batch}


def load_history(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _fit(points: dict[int, float], length: int, *, scale_alone: bool) -> float:
    """A value at `length` from known (T, value) points: exact, else linear between (or beyond)
    the two nearest known T, floored at a quarter of the nearest known value. With one point:
    proportional to T (scale_alone, for prefill) or unchanged (for decode)."""
    if length in points:
        return points[length]
    xs = sorted(points)
    nearest = min(xs, key=lambda x: abs(x - length))
    if len(xs) == 1:
        return points[nearest] * (length / nearest if scale_alone else 1.0)
    lower = [x for x in xs if x < length]
    upper = [x for x in xs if x > length]
    a, b = (lower[-1], upper[0]) if lower and upper else ((xs[0], xs[1]) if upper else (xs[-2], xs[-1]))
    value = points[a] + (points[b] - points[a]) * (length - a) / (b - a)
    return max(value, 0.25 * points[nearest])


def estimate_from_history(row: Row, source: ModelSource, history: list[dict[str, Any]],
                          settings: Settings) -> Estimate | None:
    """Combine every earlier row with the same key (later records win for a repeated T or dataset).
    None if something the row needs (a speed context or a perplexity dataset) has no history."""
    key = history_key(row, source)
    matching = [r for r in history if r.get("key") == key]
    if not matching:
        return None
    prefill: dict[int, float] = {}
    decode: dict[int, float] = {}
    perplexity: dict[str, dict[str, float]] = {}
    for record in matching:
        for length, values in record.get("contexts", {}).items():
            prefill[int(length)] = values["prefill_ms"]
            decode[int(length)] = values.get("decode_ms") or 0.0
        for name, values in record.get("perplexity", {}).items():
            if values.get("limit") == char_limit(name, settings):
                perplexity[name] = values
    if (row.runs_speed and not prefill) or any(name not in perplexity for name in row.eval_datasets):
        return None
    loads = [r["load_s"] for r in history
             if r.get("load_s") is not None and r.get("key", {}).get("model") == key["model"]
             and r.get("key", {}).get("device") == key["device"]]
    overheads = [r["overhead_s"] for r in matching if r.get("overhead_s") is not None]
    per_context = {t: (_fit(prefill, t, scale_alone=True), _fit(decode, t, scale_alone=False))
                   for t in row.context_lengths} if row.runs_speed else {}
    exact = all(t in prefill for t in per_context)
    return Estimate("history" if exact else "history (interpolated)", per_context,
                    load_s=loads[-1] if loads else 0.0,
                    overhead_s=max(0.0, overheads[-1]) if overheads else OVERHEAD_S,
                    per_dataset={name: perplexity[name] for name in row.eval_datasets})


def calibrate(row: Row, source: ModelSource, models: ModelCache, settings: Settings) -> Estimate:
    """Load the model (kept for the run), then for speed time one prefill plus a few decode steps
    per context length, and for perplexity tokenize each dataset and time one window; each after
    one untimed call. Short, and enough for an estimate."""
    loaded, load_s = models.get(source, row.device)
    load_s = load_s or models.load_seconds.get((source.key, row.device), 0.0)
    runner = make_runner(row, loaded)
    per_context = {}
    if row.runs_speed:
        steps = min(row.decode_tokens, CALIBRATION_DECODE_STEPS)
        for length in row.context_lengths:
            ids = make_prompt(loaded.vocab_size, row.batch, length, settings.seed, row.device)
            runner.start(row.batch, length + steps)
            prefill_ms, decode_ms = measure_context(runner, ids, steps, warmup=1, repeats=1, device=row.device)
            per_context[length] = (prefill_ms[0], statistics.median(decode_ms) if decode_ms else 0.0)
            runner.stop()
    per_dataset = {}
    for name in row.eval_datasets:
        tokens, tokenize_s = eval_tokens(source, name, settings)
        window, stride = eval_window(source, loaded.cfg)
        one = tokens[:window]
        measure_perplexity(runner.forward, one, window=window, stride=stride, device=row.device)   # untimed
        timed = measure_perplexity(runner.forward, one, window=window, stride=stride, device=row.device)
        per_dataset[name] = {"window_s": timed["window_s_median"], "tokenize_s": tokenize_s,
                             "windows": count_windows(tokens.numel(), window, stride)}
    return Estimate("calibration", per_context, load_s=load_s, per_dataset=per_dataset)


@dataclass
class PlanEntry:
    row: Row
    source: ModelSource | None
    estimate: Estimate | None = None       # None: not run (skip_reason says why)
    load_needed: bool = False
    seconds: float | None = None
    power_on: bool = False
    skip_reason: str | None = None


def make_plan(all_rows: list[Row], run_rows: list[Row], sources: dict[str, ModelSource], settings: Settings,
              models: ModelCache, power_available: dict[str, bool]) -> list[PlanEntry]:
    """One entry per CSV row; the rows to run get an estimate (history first, else calibration).
    Rows that need training are listed, not run."""
    history = load_history(settings.history_path)
    run_names = {row.name for row in run_rows}
    entries = []
    for row in all_rows:
        entry = PlanEntry(row, sources.get(row.name))
        if row.name not in run_names:
            entry.skip_reason = "enabled=no" if not row.enabled else "not in --only"
        elif row.weights == "train":
            entry.skip_reason = "needs training (nobody has trained it)"
        else:
            entry.power_on = settings.power and row.runs_speed and power_available.get(power_target(row.device), False)
            estimate = estimate_from_history(row, entry.source, history, settings)
            if estimate is None and settings.calibrate:
                write(f"calibrating {row.name} ({row.arm}, {entry.source.key}, {row.device}, {row.benchmark_label}): "
                      f"one short run per context length and dataset")
                estimate = calibrate(row, entry.source, models, settings)
            entry.estimate = estimate or Estimate("unknown")
        entries.append(entry)
    previous = models.key            # the model left loaded by the calibration, if any
    for entry in entries:
        if entry.estimate is None:
            continue
        key = (entry.source.key, entry.row.device)
        entry.load_needed = key != previous
        previous = key
        entry.seconds = row_seconds(entry.row, entry.estimate, settings, load_needed=entry.load_needed,
                                    power_on=entry.power_on)
    return entries


def format_plan(entries: list[PlanEntry], settings: Settings, power_info: dict[str, dict[str, Any]]) -> str:
    """The plan table: each row, whether it was trained before, what it measures, its estimate and
    where the estimate came from, and the total ETA."""
    header = ["#", "name", "arm", "model", "weights", "preset", "device", "benchmarks", "B", "contexts", "decode",
              "repeats", "estimate", "from"]
    table = [header]
    runnable = [e for e in entries if e.estimate is not None]
    for number, entry in enumerate(entries, 1):
        row = entry.row
        if entry.estimate is None:
            est, origin = "not run", entry.skip_reason or ""
        else:
            est, origin = format_seconds(entry.seconds), entry.estimate.source
            if entry.load_needed and entry.estimate.load_s >= 0.05:
                origin += f", load {entry.estimate.load_s:.1f} s"
        speed = row.runs_speed
        table.append([str(number), row.name, row.arm, entry.source.key if entry.source else row.model,
                      row.weights or "-", row.preset_label, row.device, row.benchmark_label,
                      str(row.batch) if speed else "-", ";".join(map(str, row.context_lengths)) if speed else "-",
                      str(row.decode_tokens) if speed else "-", str(row.repeats) if speed else "-", est, origin])
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    lines = [f"Sweep plan for {settings.csv_path}"]
    lines += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in table]
    known = [e.seconds for e in runnable if e.seconds is not None]
    unknown = len(runnable) - len(known)
    lines.append("")
    lines.append(f"Total ETA {format_seconds(sum(known))} for {len(runnable)} row(s)"
                 + (f", plus {unknown} row(s) with no estimate (--no-calibrate and no history)" if unknown else "")
                 + ".")
    if runnable:
        sources = sorted({e.estimate.source for e in runnable})
        lines.append(f"Estimates from: {', '.join(sources)}. Timing history: {settings.history_path}")
    training = [e.row.name for e in entries if e.row.weights == "train" and e.skip_reason
                and e.skip_reason.startswith("needs training")]
    if training:
        lines.append(f"Not run, need training first: {', '.join(training)}. The sweep never trains; models "
                     f"others have trained are benchmarked as they are (weights=pretrained).")
    pretrained = sorted({e.source.key for e in runnable if e.row.weights == "pretrained"})
    if pretrained:
        lines.append(f"Trained before, used as they are (no training): {', '.join(pretrained)}.")
    datasets = sorted({d for e in runnable for d in e.row.eval_datasets})
    for name in datasets:
        limit, spec = char_limit(name, settings), data.DATASETS[name]
        sha = f", file sha256 {data.file_sha256(spec.path())[:16]}" if data.is_installed(spec) else ""
        lines.append(f"Dataset {name}: {spec.eval_split} split, "
                     f"{'all of it' if limit is None else f'first {limit:,} characters'}, the same text for every "
                     f"model{sha}, from {data.DATASET_DIR / name}.")
    defaults, _ = parse_defaults(Path(settings.csv_path).read_text(encoding="utf-8").splitlines(),
                                 Path(settings.csv_path).name) if Path(settings.csv_path).is_file() else (Defaults(), [])
    if defaults.line:
        lines.append(f"Defaults for blank cells (line {defaults.line}): {defaults.describe()}.")
    if runnable:
        rows = [e.row for e in runnable]
        lines.append(format_coverage(rows, check_coverage(rows)))
    if settings.power and any(e.row.runs_speed for e in runnable):
        for target, info in sorted(power_info.items()):
            if info["available"]:
                watts = ", ".join(f"{name} {w:.1f} W" for name, w in info["watts"].items())
                lines.append(f"Power on {target}: {len(PHASES)} windows per context length, each "
                             f"{settings.power_warmup_s:g} s warm-up + {settings.power_seconds:g} s sampled; idle {watts}.")
            else:
                lines.append(f"Power on {target}: unavailable, windows skipped ({info['reason']}).")
    elif not settings.power:
        lines.append("Power: off (--no-power).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Running the sweep
# ----------------------------------------------------------------------------------------------

SUMMARY_COLUMNS = (
    "sweep_id", "row", "run_id", "status", "benchmark", "dataset", "arm", "model", "weights", "lut_preset",
    "device", "dtype", "batch", "context", "decode_tokens", "repeats", "prefill_ms_median", "prefill_ms_p10",
    "prefill_ms_p90", "prefill_tokens_per_s", "decode_ms_per_token_median", "decode_ms_p10", "decode_ms_p90",
    "decode_tokens_per_s", "power_source", "idle_power_w", "prefill_power_w", "prefill_energy_per_token_mj",
    "decode_power_w", "decode_energy_per_token_mj", "power_note", "perplexity", "nll_per_token", "eval_tokens",
    "nll_total", "eval_chars", "eval_bytes", "eval_words", "bits_per_byte", "word_perplexity", "dataset_sha256",
    "eval_window", "eval_stride", "eval_windows", "eval_s", "forward_ms_median", "estimated_row_s", "actual_row_s",
    "estimate_source",
)

# results_long.csv and all_results.csv: one line per row, dataset, benchmark, metric and context
# length, the layout a spreadsheet's pivot table wants. Units are in the unit column, or in the
# column name where it is one fixed unit (context_tokens, eval_window_tokens, eval_stride_tokens).
LONG_COLUMNS = (
    "timestamp_utc", "sweep_id", "run_id", "git_commit", "row", "model", "arm", "weights", "lut_preset",
    "stage_toggles", "device", "dtype", "benchmark", "dataset", "dataset_sha256", "context_tokens", "batch",
    "decode_tokens", "repeats", "eval_window_tokens", "eval_stride_tokens", "metric", "value", "unit", "note",
)
# metric (a summary.csv column) -> unit, in the order they are written
SPEED_METRICS = {
    "prefill_ms_median": "ms", "prefill_ms_p10": "ms", "prefill_ms_p90": "ms", "prefill_tokens_per_s": "tokens/s",
    "decode_ms_per_token_median": "ms/token", "decode_ms_p10": "ms/token", "decode_ms_p90": "ms/token",
    "decode_tokens_per_s": "tokens/s", "idle_power_w": "W", "prefill_power_w": "W",
    "prefill_energy_per_token_mj": "mJ/token", "decode_power_w": "W", "decode_energy_per_token_mj": "mJ/token",
}
PERPLEXITY_METRICS = {
    "perplexity": "per token", "nll_total": "nats", "nll_per_token": "nats/token", "eval_tokens": "tokens",
    "eval_chars": "chars", "eval_bytes": "bytes", "eval_words": "words", "bits_per_byte": "bits/byte",
    "word_perplexity": "per word", "eval_windows": "windows", "eval_s": "s", "forward_ms_median": "ms",
    "top1_accuracy": "fraction of scored tokens", "top5_accuracy": "fraction of scored tokens",
}
# Resource use per phase (bench/resources.py ResourceSampler.summary), benchmark = "resources" in the
# long format, dataset = the phase (speed, or the perplexity dataset's name)
RESOURCE_METRICS = {
    "window_s": "s", "power_draw_mean": "W", "power_draw_max": "W", "power_w_mean_steady": "W",
    "power_draw_instant_mean": "W", "power_draw_instant_max": "W", "energy_j": "J", "energy_per_token_mj": "mJ/token",
    "utilization_gpu_mean": "%", "utilization_gpu_max": "%", "utilization_memory_mean": "%",
    "utilization_memory_max": "%", "memory_used_max": "MiB", "torch_peak_allocated_mib": "MiB",
    "torch_peak_reserved_mib": "MiB", "clocks_sm_mean": "MHz", "clocks_sm_max": "MHz", "clocks_mem_mean": "MHz",
    "temperature_gpu_max": "C", "throttled": "yes/no", "clock_event_reasons_seen": "names", "pstates_seen": "P-states",
    "process_cpu_user_s": "s", "process_cpu_system_s": "s", "process_cpu_cores_mean": "cores",
    "host_cpu_busy_pct": "%", "process_rss_peak_mib": "MiB", "gpu_samples": "samples", "interval_ms": "ms",
}
METRIC_NOTES = {
    "perplexity": "token perplexity depends on the tokenizer; compare models with bits_per_byte or word_perplexity",
    "eval_tokens": "scored tokens of this model's tokenizer",
}


def context_stats(prefill_ms: list[float], decode_ms: list[float], row: Row, length: int) -> dict[str, Any]:
    p = summarize(prefill_ms, 0, row.device)
    stats: dict[str, Any] = {
        "prefill_ms_median": p.median_ms, "prefill_ms_p10": p.p10_ms, "prefill_ms_p90": p.p90_ms,
        "prefill_tokens_per_s": row.batch * length / (p.median_ms / 1000.0),
    }
    if decode_ms:
        d = summarize(decode_ms, 0, row.device)
        stats.update({"decode_ms_per_token_median": d.median_ms, "decode_ms_p10": d.p10_ms,
                      "decode_ms_p90": d.p90_ms, "decode_tokens_per_s": row.batch / (d.median_ms / 1000.0)})
    return stats


def _first_reading(numbers: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    for source, values in numbers.items():
        if "avg_power_w" in values:
            return source, values
    return None, None


def _write_csv(path: Path, columns: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _round(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def _measure_speed(row: Row, runner: Any, loaded: Loaded, entry: PlanEntry, settings: Settings, run: Run,
                   inner: WeightedBar, account: Callable[[float], None], fallback_s: float) -> dict[int, dict[str, Any]]:
    """The speed benchmark: every context length's repeats and power windows."""
    estimate, target = entry.estimate, power_target(row.device)
    per_context: dict[int, dict[str, Any]] = {}
    for length in row.context_lengths:
        ids = make_prompt(loaded.vocab_size, row.batch, length, settings.seed, row.device)
        runner.start(row.batch, length + row.decode_tokens)
        per_repeat = estimate.repeat_s(length, row.decode_tokens) or fallback_s

        def after_repeat(i: int, length: int = length, per_repeat: float = per_repeat) -> None:
            phase = "warm-up" if i < settings.warmup else f"repeat {i - settings.warmup + 1}/{row.repeats}"
            inner.set_note(f"speed T={length} {phase}")
            inner.advance(per_repeat, items=1)
            account(per_repeat)

        prefill_ms, decode_ms = measure_context(runner, ids, row.decode_tokens, warmup=settings.warmup,
                                                repeats=row.repeats, device=row.device, after_repeat=after_repeat)
        stats = context_stats(prefill_ms, decode_ms, row, length)
        run.log(kind="speed", context=length, prefill_ms=prefill_ms, decode_ms=decode_ms, **stats)
        if entry.power_on:
            stats["power"] = {}
            for phase in row_phases(row):
                inner.set_note(f"speed T={length} {phase} power window")
                if phase == "prefill":
                    fn, tokens_per_call = (lambda: runner.prefill(ids)), row.batch * length
                else:
                    state = {"token": runner.prefill(ids), "made": 0}

                    def fn(state: dict = state, length: int = length) -> None:
                        if state["made"] == row.decode_tokens:
                            runner.rewind(length)     # back to the prompt, so T stays near `length`
                            state["made"] = 0
                        state["token"] = runner.decode(state["token"])
                        state["made"] += 1
                    tokens_per_call = row.batch
                report, calls, warm_calls = power_window(
                    fn, row.device, target, seconds=settings.power_seconds, warmup_s=settings.power_warmup_s,
                    note=f"{row.name} {phase} T={length}")
                run.save_power(report, window=f"{phase}_T{length}")
                stats["power"][phase] = {"calls": calls, "warmup_calls": warm_calls,
                                         "warmup_s": LAST_WARMUP.get("warmup_s"),
                                         "warmup_settled": LAST_WARMUP.get("settled"),
                                         **power_numbers(report, calls * tokens_per_call)}
                inner.advance(power_window_s(settings), items=1)
                account(power_window_s(settings))
        runner.stop()
        per_context[length] = stats
        message = (f"{row.name} speed T={length}: prefill {stats['prefill_ms_median']:.3f} ms "
                   f"({stats['prefill_tokens_per_s']:,.0f} tok/s)")
        if "decode_ms_per_token_median" in stats:
            message += (f", decode {stats['decode_ms_per_token_median']:.3f} ms/token "
                        f"({stats['decode_tokens_per_s']:,.1f} tok/s)")
        for phase, numbers in stats.get("power", {}).items():
            source, values = _first_reading({k: v for k, v in numbers.items() if isinstance(v, dict)})
            if values:
                message += f", {phase} {values['avg_power_w']:.1f} W ({source})"
        write(message)
    return per_context


def _measure_perplexity(row: Row, runner: Any, loaded: Loaded, entry: PlanEntry, settings: Settings, run: Run,
                        inner: WeightedBar, account: Callable[[float], None]) -> dict[str, dict[str, Any]]:
    """The perplexity benchmark on every dataset of the row."""
    results: dict[str, dict[str, Any]] = {}
    for name in row.eval_datasets:
        inner.set_note(f"perplexity {name}: tokenizing")
        tokens, tokenize_s = eval_tokens(entry.source, name, settings)
        window, stride = eval_window(entry.source, loaded.cfg)
        planned = entry.estimate.per_dataset.get(name, {})
        per_window = planned.get("window_s") or 0.0
        account(planned.get("tokenize_s", 0.0))
        total = count_windows(tokens.numel(), window, stride)

        def after_window(i: int, name: str = name, total: int = total) -> None:
            inner.set_note(f"perplexity {name} window {i + 1}/{total}")
            inner.advance(per_window, items=1)
            account(per_window)

        stats = measure_perplexity(runner.forward, tokens, window=window, stride=stride, device=row.device,
                                   after_window=after_window)
        text = eval_text(name, settings)
        stats.update(text_metrics(stats["nll_total"], text.utf8_bytes, text.words))
        stats.update({"tokenize_s": tokenize_s, "split": text.split, "dataset_sha256": text.file_sha256,
                      "char_limit": text.char_limit, "eval_chars": text.chars, "eval_bytes": text.utf8_bytes,
                      "eval_words": text.words})
        run.log(kind="perplexity", dataset=name, **stats)
        results[name] = stats
        write(f"{row.name} perplexity {name}: {stats['perplexity']:.3f} per token over {stats['scored_tokens']:,} tokens, "
              f"{stats['bits_per_byte']:.4f} bits/byte, word perplexity {stats['word_perplexity']:.3f} "
              f"({stats['eval_windows']} windows of {window}, stride {stride}, {stats['eval_s']:.1f} s)")
    return results


def run_row(entry: PlanEntry, settings: Settings, models: ModelCache, idle: dict[str, dict[str, Any]],
            outer: WeightedBar, sweep_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure one row. Returns (one summary line per context length and dataset, the history record)."""
    row, estimate = entry.row, entry.estimate
    row_start = time.perf_counter()
    budget = entry.seconds if entry.seconds is not None else UNKNOWN_ROW_S
    spent = 0.0

    def account(seconds: float) -> None:
        nonlocal spent
        seconds = min(seconds, max(0.0, budget - spent))
        spent += seconds
        outer.advance(seconds)

    outer.set_note(f"{row.name}: loading {entry.source.key}" if entry.load_needed else row.name)
    loaded, load_s = models.get(entry.source, row.device)
    account(estimate.load_s if entry.load_needed else 0.0)
    runner = make_runner(row, loaded)
    dtype = row_dtype(row, loaded)
    target = power_target(row.device)
    run = Run(f"sweep_{row.name}", results_dir=settings.results_dir, notes=row.notes or None,
              devices={"prefill": row.device, "decode": row.device},
              configs={"sweep_id": sweep_id, "csv": str(settings.csv_path), "row": {
                  "name": row.name, "arm": row.arm, "model": entry.source.key, "weights": row.weights,
                  "lut_preset": row.lut_preset, "device": row.device, "benchmarks": row.benchmarks,
                  "datasets": row.eval_datasets, "batch": row.batch, "context_lengths": row.context_lengths,
                  "decode_tokens": row.decode_tokens, "repeats": row.repeats, "warmup": settings.warmup},
                  "model_config": loaded.config_dict(), "weights": "random, seed %d" % settings.seed
                  if entry.source.kind == "core" else entry.source.load_name, "dtype": dtype,
                  "power": {"on": entry.power_on, "seconds": settings.power_seconds,
                            "warmup_s": settings.power_warmup_s},
                  "perplexity": {name: {"split": data.DATASETS[name].eval_split, "char_limit": char_limit(name, settings),
                                        "file": str(data.DATASETS[name].path()),
                                        "file_sha256": eval_text(name, settings).file_sha256}
                                 for name in row.eval_datasets}})
    speed_steps = (settings.warmup + row.repeats + (len(row_phases(row)) if entry.power_on else 0)) \
        * len(row.context_lengths) if row.runs_speed else 0
    window_steps = sum(int(estimate.per_dataset.get(n, {}).get("windows", 0)) for n in row.eval_datasets)
    inner_total = budget - spent - estimate.overhead_s
    inner = WeightedBar(max(inner_total, 1e-3), speed_steps + window_steps, label=row.name,
                        unit="steps", show=settings.show_progress, position=1, leave=False)
    try:
        per_resources: dict[str, dict[str, Any]] = {}
        if row.runs_speed:
            with resources.ResourceSampler(row.device) as sampler:
                per_context = _measure_speed(row, runner, loaded, entry, settings, run, inner, account,
                                             inner_total / max(1, inner.count))
            per_resources["speed"] = sampler.summary()
        else:
            per_context = {}
        with resources.ResourceSampler(row.device) as sampler:
            per_dataset = _measure_perplexity(row, runner, loaded, entry, settings, run, inner, account)
        if per_dataset:
            scored = sum(s["scored_tokens"] for s in per_dataset.values())
            per_resources["perplexity"] = sampler.summary(tokens=scored)
            if len(per_dataset) == 1:      # one dataset: its own phase name
                per_resources[next(iter(per_dataset))] = per_resources.pop("perplexity")
        run.log(kind="resources", phases=per_resources)
    finally:
        inner.close()
    row_s = time.perf_counter() - row_start
    idle_info = idle.get(target, {})
    summary = {
        "sweep_id": sweep_id, "arm": row.arm, "model": entry.source.key, "weights": row.weights,
        "lut_preset": row.lut_preset, "device": row.device, "dtype": dtype, "benchmarks": ";".join(row.benchmarks),
        "datasets": ";".join(row.eval_datasets), "batch": row.batch, "decode_tokens": row.decode_tokens,
        "repeats": row.repeats, "warmup": settings.warmup,
        "context_lengths": ";".join(map(str, row.context_lengths)),
        "load_s": load_s if load_s else None, "row_s": row_s, "estimated_row_s": entry.seconds,
        "estimate_source": estimate.source,
        "idle_power_w": idle_info.get("watts") or None,
        "power_unavailable": None if entry.power_on or not settings.power or not row.runs_speed
        else idle_info.get("reason"),
        **{f"T{length}": stats for length, stats in per_context.items()},
        **{f"perplexity_{name}": stats for name, stats in per_dataset.items()},
        **{f"resources_{phase}": stats for phase, stats in per_resources.items()},
        "gpu_settings": {k: v for k, v in GPU_SETTINGS.items() if k != "raw_q"},
    }
    if GPU_SETTINGS.get("raw_q"):
        (run.dir / "gpu_settings.txt").write_text(GPU_SETTINGS["raw_q"], encoding="utf-8")
    run_dir = run.finish(summary)
    account(budget)                    # top up to the row's planned share
    outer.advance(0.0, items=1)

    common = {"sweep_id": sweep_id, "row": row.name, "run_id": run.run_id, "status": "ok", "arm": row.arm,
              "model": entry.source.key, "weights": row.weights, "lut_preset": row.lut_preset or "",
              "device": row.device, "dtype": dtype, "estimated_row_s": entry.seconds, "actual_row_s": row_s,
              "estimate_source": estimate.source}
    lines = []
    for length, stats in per_context.items():
        line = {**common, "benchmark": "speed", "dataset": "synthetic", "batch": row.batch, "context": length,
                "decode_tokens": row.decode_tokens, "repeats": row.repeats,
                **{k: v for k, v in stats.items() if k != "power"}}
        line["idle_power_w"] = next(iter(idle_info.get("watts", {}).values()), None)
        notes = []
        for phase in PHASES:
            numbers = {k: v for k, v in stats.get("power", {}).get(phase, {}).items() if isinstance(v, dict)}
            source, values = _first_reading(numbers)
            if values:
                line["power_source"] = source
                line[f"{phase}_power_w"] = values["avg_power_w"]
                line[f"{phase}_energy_per_token_mj"] = values["energy_per_token_mj"]
            if len([v for v in numbers.values() if "avg_power_w" in v]) > 1:
                notes.append(f"{phase}: more sources in {run.run_id}/summary.json")
        if not entry.power_on:
            notes.append("power off" if not settings.power else f"power unavailable: {idle_info.get('reason')}")
        line["power_note"] = "; ".join(notes)
        lines.append({k: _round(v) for k, v in line.items()})
    for name, stats in per_dataset.items():
        line = {**common, "benchmark": "perplexity", "dataset": name, "batch": 1,
                **{k: stats[k] for k in ("perplexity", "nll_per_token", "nll_total", "eval_chars", "eval_bytes",
                                         "eval_words", "bits_per_byte", "word_perplexity", "dataset_sha256",
                                         "eval_window", "eval_stride", "eval_windows", "eval_s", "forward_ms_median",
                                         "top1_accuracy", "top5_accuracy")},
                "eval_tokens": stats["scored_tokens"],
                "power_note": "GPU power sampled through the eval (resources lines, ResourceSampler)"
                if name in per_resources else "no power measured for perplexity"}
        lines.append({k: _round(v) for k, v in line.items()})
    resource_lines = []                    # GPU and host resource use per phase: long format and summary.json only
    for phase, stats in per_resources.items():
        line = {**common, "benchmark": "resources", "dataset": phase, "batch": row.batch if phase == "speed" else 1,
                **{k: stats.get(k) for k in RESOURCE_METRICS}}
        line["power_note"] = ("nvidia-smi power.draw (driver-averaged ~1 s) sampled through the phase; "
                              "power_w_mean_steady leaves out its first second")
        resource_lines.append({k: _round(v) for k, v in line.items()})

    timed_speed_s = sum((settings.warmup + row.repeats) * ((s["prefill_ms_median"] + row.decode_tokens *
                                                            s.get("decode_ms_per_token_median", 0.0)) / 1000.0)
                        for s in per_context.values())
    power_s = len(row_phases(row)) * power_window_s(settings) * len(per_context) if entry.power_on else 0.0
    perplexity_s = sum(s["eval_s"] + s["tokenize_s"] for s in per_dataset.values())
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "sweep_id": sweep_id, "run_id": run.run_id,
              "key": history_key(row, entry.source), "load_s": load_s if load_s else None,
              "overhead_s": row_s - (load_s or 0.0) - timed_speed_s - power_s - perplexity_s,
              "row_s": row_s, "run_dir": str(run_dir),
              "contexts": {str(t): {"prefill_ms": s["prefill_ms_median"],
                                    "decode_ms": s.get("decode_ms_per_token_median")} for t, s in per_context.items()},
              "perplexity": {name: {"window_s": s["window_s_median"], "windows": s["eval_windows"],
                                    "tokenize_s": s["tokenize_s"], "window": s["eval_window"],
                                    "limit": char_limit(name, settings)} for name, s in per_dataset.items()}}
    record["long"] = long_rows(lines + resource_lines, run, toggle_string(row, loaded))
    return lines, record


def toggle_string(row: Row, loaded: Loaded) -> str:
    """The lut arm's stage settings in one cell: every stage that isn't plain BASE, as
    stage=what[k x b -> out]@layers/quant, joined by ';'; "all BASE" when there is none. Empty for
    the hf and baseline arms."""
    if row.arm != "lut":
        return ""
    from LUT import ops_registry

    parts = []
    for item in ops_registry.describe(lut_host.preset_toggles(row.lut_preset), loaded.cfg):
        if item["runs"] == "not in this model" or (item["runs"].startswith("base:") and item["quant"] == "float"):
            continue
        part = f"{item['stage']}={item['runs']}"
        part += f"[{item['shape']}]" if item["shape"] != "-" else ""
        part += f"@{item['layers']}" if item["layers"] not in ("all", "-") else ""
        part += f"/{item['quant']}" if item["quant"] != "float" else ""
        parts.append(part)
    return ";".join(parts) or "all BASE"


def long_rows(lines: list[dict[str, Any]], run: Run, toggles: str) -> list[dict[str, Any]]:
    """The row's summary lines in long format (LONG_COLUMNS): one line per metric with a value, with
    its unit. Missing values (power that couldn't be read) are left out, not written as blanks."""
    git = run.git or {}
    commit = (git.get("commit") or "")[:10] + ("+dirty" if git.get("dirty") else "")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for line in lines:
        speed = line["benchmark"] == "speed"
        res = line["benchmark"] == "resources"
        common = {
            "timestamp_utc": stamp, "sweep_id": line["sweep_id"], "run_id": line["run_id"], "git_commit": commit,
            "row": line["row"], "model": line["model"], "arm": line["arm"], "weights": line["weights"],
            "lut_preset": line["lut_preset"], "stage_toggles": toggles, "device": line["device"],
            "dtype": line["dtype"], "benchmark": line["benchmark"], "dataset": line["dataset"],
            "dataset_sha256": "" if speed else line.get("dataset_sha256", ""),
            "context_tokens": line.get("context", "") if speed else "", "batch": line.get("batch", ""),
            "decode_tokens": line.get("decode_tokens", "") if speed else "",
            "repeats": line.get("repeats", "") if speed else "",
            "eval_window_tokens": "" if speed or res else line.get("eval_window", ""),
            "eval_stride_tokens": "" if speed or res else line.get("eval_stride", ""),
        }
        metrics = SPEED_METRICS if speed else RESOURCE_METRICS if res else PERPLEXITY_METRICS
        for metric, unit in metrics.items():
            value = line.get(metric)
            if value is None or value == "":
                continue
            note = METRIC_NOTES.get(metric, "")
            if res:
                note = line.get("power_note", "") if "power" in metric or "energy" in metric else ""
            elif metric.endswith("power_w") or metric.endswith("_mj"):
                note = line.get("power_source") or ""
            out.append({**common, "metric": metric, "value": value, "unit": unit, "note": note})
    return out


def append_cumulative(path: Path, rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Append long rows to the cumulative CSV (results/all_results.csv), skipping every run id that
    is already in it, so appending the same results twice changes nothing. Returns (lines added,
    run ids skipped). A file with other columns is left alone (SweepError): move it aside to start
    a new one."""
    existing: set[str] = set()
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(LONG_COLUMNS):
                raise SweepError(f"{path} has other columns than this sweep writes; move it aside and a new one "
                                 f"is started")
            existing = {line["run_id"] for line in reader}
    new = [line for line in rows if line["run_id"] not in existing]
    skipped = len({line["run_id"] for line in rows} & existing)
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.is_file()
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(LONG_COLUMNS), extrasaction="ignore")
            if fresh:
                writer.writeheader()
            writer.writerows(new)
    return len(new), skipped


def _append_history(path: Path, record: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as error:
        write(f"could not save the timing history to {path}: {error}")


def run_sweep(entries: list[PlanEntry], settings: Settings, models: ModelCache,
              idle: dict[str, dict[str, Any]]) -> tuple[Path, list[dict[str, Any]], int]:
    """Run every planned row. Returns (sweep folder, summary lines, number of failed rows)."""
    runnable = [e for e in entries if e.estimate is not None]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_id = f"{stamp}_{re.sub(r'[^A-Za-z0-9_.-]+', '-', Path(settings.csv_path).stem)}"
    sweep_dir = Path(settings.results_dir) / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(settings.csv_path, sweep_dir / "models.csv")
    _write_csv(sweep_dir / "plan.csv",
               ["row", "arm", "model", "weights", "device", "benchmarks", "estimated_s", "estimate_source",
                "load_needed", "not_run_because"],
               [{"row": e.row.name, "arm": e.row.arm, "model": e.source.key if e.source else e.row.model,
                 "weights": e.row.weights or "", "device": e.row.device, "benchmarks": e.row.benchmark_label,
                 "estimated_s": _round(e.seconds), "estimate_source": e.estimate.source if e.estimate else "",
                 "load_needed": e.load_needed if e.estimate else "", "not_run_because": e.skip_reason or ""}
                for e in entries])
    planned = [e.seconds if e.seconds is not None else UNKNOWN_ROW_S for e in runnable]
    total = sum(planned)
    started = time.perf_counter()

    def status(done: int, current: str | None, failed: list[str]) -> None:
        elapsed = time.perf_counter() - started
        planned_done = sum(planned[:done])
        speed = elapsed / planned_done if planned_done > 0 else 1.0
        info = {"sweep_id": sweep_id, "rows_done": done, "rows_total": len(runnable), "current_row": current,
                "failed_rows": failed, "elapsed_s": round(elapsed, 1),
                "remaining_s_estimate": round((total - planned_done) * speed, 1),
                "planned_total_s": round(total, 1), "updated_utc": datetime.now(timezone.utc).isoformat()}
        (sweep_dir / "status.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    lines: list[dict[str, Any]] = []
    long: list[dict[str, Any]] = []
    failed: list[str] = []
    outer = WeightedBar(total, len(runnable), label="sweep", unit="rows", show=settings.show_progress, position=0)
    try:
        for done, entry in enumerate(runnable):
            status(done, entry.row.name, failed)
            try:
                row_lines, record = run_row(entry, settings, models, idle, outer, sweep_id)
            except Exception as error:   # one bad row shouldn't lose the others; it is reported below
                failed.append(entry.row.name)
                write(f"{entry.row.name} FAILED: {type(error).__name__}: {error}")
                traceback.print_exc(file=sys.stderr)
                outer.advance(planned[done], items=1)
                lines.append({"sweep_id": sweep_id, "row": entry.row.name, "status": f"failed: {error}"[:300],
                              "arm": entry.row.arm, "model": entry.source.key, "device": entry.row.device})
                continue
            lines.extend(row_lines)
            long.extend(record.pop("long"))
            _append_history(settings.history_path, record)
            _write_csv(sweep_dir / "summary.csv", SUMMARY_COLUMNS, lines)   # rewritten after every row
            _write_csv(sweep_dir / "results_long.csv", LONG_COLUMNS, long)
            try:
                append_cumulative(Path(settings.results_dir) / "all_results.csv",
                                  [line for line in long if line["row"] == entry.row.name])
            except SweepError as error:
                write(f"all_results.csv not updated: {error}")
        status(len(runnable), None, failed)
    finally:
        outer.close()
        models.release()
        _write_csv(sweep_dir / "summary.csv", SUMMARY_COLUMNS, lines)
    actual = time.perf_counter() - started
    write(f"sweep {sweep_id}: {len(runnable) - len(failed)}/{len(runnable)} rows in {format_seconds(actual)} "
          f"(estimated {format_seconds(total)})" + (f"; failed: {', '.join(failed)}" if failed else ""))
    write(f"summary: {sweep_dir / 'summary.csv'}; long format for a spreadsheet: {sweep_dir / 'results_long.csv'}, "
          f"also appended to {Path(settings.results_dir) / 'all_results.csv'}")
    return sweep_dir, lines, len(failed)


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, Settings]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="the sweep CSV (default: %(default)s)")
    parser.add_argument("--only", default=None, help="comma-separated row names to run (even if enabled=no)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and ETA, then stop; runs no benchmark")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="estimate only from history; rows without history show no estimate")
    parser.add_argument("--download", action="store_true",
                        help="allow fetching Hugging Face models and the known datasets that aren't installed, "
                        "into ../LUT-Transformer-local (never without this flag)")
    parser.add_argument("--warmup", type=int, default=2, help="untimed repeats before the timed ones (default 2)")
    parser.add_argument("--no-power", action="store_true", help="skip idle and power measurement")
    parser.add_argument("--power-seconds", type=float, default=3.0, help="sampled length of each power window")
    parser.add_argument("--power-warmup-seconds", type=float, default=2.0,
                        help="unsampled run before each power window, so power reaches steady state")
    parser.add_argument("--idle-seconds", type=float, default=3.0, help="idle power sample at the start")
    parser.add_argument("--max-eval-chars", type=int, default=None,
                        help="perplexity reads the first this-many characters of each dataset's split, the same "
                        "text for every model (0: the whole split; default: each dataset's own limit in "
                        "common/data.py)")
    parser.add_argument("--allow-gaps", action="store_true",
                        help="run even if a row doesn't measure every (dataset, benchmark) the lut rows measure")
    parser.add_argument("--seed", type=int, default=0, help="seed for prompts and the core's random weights")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--history", type=Path, default=local_cache.LOCAL_DIR / "sweep-history.jsonl",
                        help="timing history used for estimates (default: %(default)s)")
    bars = parser.add_mutually_exclusive_group()
    bars.add_argument("--progress", dest="progress", action="store_true", default=None,
                      help="always show progress bars (default: only in a terminal)")
    bars.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be 0 or more")
    if args.max_eval_chars is not None and args.max_eval_chars < 0:
        parser.error("--max-eval-chars must be 0 or more")
    settings = Settings(csv_path=args.csv, only=[n.strip() for n in args.only.split(",") if n.strip()] if args.only else None,
                        warmup=args.warmup, power=not args.no_power, power_seconds=args.power_seconds,
                        power_warmup_s=args.power_warmup_seconds, idle_seconds=args.idle_seconds, seed=args.seed,
                        download=args.download, calibrate=not args.no_calibrate,
                        max_eval_chars=args.max_eval_chars, allow_gaps=args.allow_gaps, results_dir=args.results_dir,
                        history_path=args.history, show_progress=args.progress)
    return args, settings


def main(argv: list[str] | None = None) -> int:
    args, settings = parse_args(argv)
    precision.full_fp32()            # float32 means full float32: TF32 off before anything runs
    _TOKENS.clear()
    _TEXTS.clear()
    rows, errors = read_csv(settings.csv_path)
    run_rows, select_errors = select_rows(rows, settings.only)
    errors += select_errors
    sources, model_errors = validate(run_rows, settings)     # the rows that parsed; all problems at once
    errors += model_errors
    if not errors and not run_rows:
        errors.append(f"{Path(settings.csv_path).name}: no enabled rows to run")
    if errors:
        print("The sweep did not start; nothing ran. Fix these first:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    coverage = check_coverage(run_rows)                       # after validate filled in the weights
    if coverage.gaps and not settings.allow_gaps:
        print(format_coverage(run_rows, coverage), flush=True)
        print("The sweep did not start; nothing ran. These rows don't measure everything Alpha's model (the lut "
              "rows) measures, so their results wouldn't line up:", file=sys.stderr)
        for error in coverage_errors(coverage):
            print(f"  - {error}", file=sys.stderr)
        print("Add the missing benchmarks or datasets to those rows (a '# defaults:' line sets them for every "
              "row with blank cells), or pass --allow-gaps.", file=sys.stderr)
        return 2

    for name in datasets_to_download(run_rows):              # only with --download (validate checked)
        spec = data.DATASETS[name]
        write(f"downloading {name} from {spec.hf_repo} into {data.DATASET_DIR / name}")
        data.download(spec)

    # The GPU's power and clock settings, once per sweep, recorded in every row's summary (raw text in
    # the first run folder's gpu_settings.txt).
    GPU_SETTINGS.clear()
    if any(row.device.startswith("cuda") for row in run_rows):
        GPU_SETTINGS.update(resources.gpu_settings(next(r.device for r in run_rows if r.device.startswith("cuda"))))

    # Idle power first, before any model or CUDA context exists. A dry run only checks which power
    # sources can be read (a short sample, not recorded).
    targets = {power_target(row.device) for row in run_rows if row.runs_speed and row.weights != "train"}
    idle: dict[str, dict[str, Any]] = {}
    if settings.power and targets:
        seconds = min(settings.idle_seconds, 0.5) if args.dry_run else settings.idle_seconds
        if not args.dry_run:
            write(f"sampling idle power for {seconds:g} s before anything is loaded")
        idle = measure_idle(targets, seconds)
    models = ModelCache(settings)
    entries = make_plan(rows, run_rows, sources, settings, models,
                        {target: info["available"] for target, info in idle.items()})
    print(format_plan(entries, settings, idle), flush=True)
    if args.dry_run:
        models.release()
        print("Dry run: nothing was benchmarked and no results were written.")
        return 0
    if not any(e.estimate is not None for e in entries):
        models.release()
        print("Nothing to run: every selected row needs training first.")
        return 0
    _, _, failed = run_sweep(entries, settings, models, idle)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
