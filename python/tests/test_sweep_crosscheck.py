"""Cross-checked datasets and the spreadsheet output of the benchmark sweep (bench/sweep.py,
common/data.py), Alpha's 10:52 AM request: every model measures the same datasets and benchmarks
as Alpha's own model (the lut rows), on the same text, and every result lands in a long-format CSV
for Excel.

  - the coverage check stops the sweep on a gap, runs with --allow-gaps, and a '# defaults:' line
    makes rows match by construction
  - the evaluated text is the same for every model: same file sha256, bytes and words, with the
    document separators normalized
  - bits per byte and word perplexity, on a tiny example worked out by hand
  - results_long.csv has the documented columns and units; results/all_results.csv appends and
    never repeats a run id

Everything uses small text fixtures written into tmp_path; nothing is downloaded. The two tests
that run real models skip unless GPT-2 and stories15M are installed locally.
"""

import csv
import math

import pytest

torch = pytest.importorskip("torch")

from bench import sweep  # noqa: E402
from reference import local_cache  # noqa: E402
from test_sweep import _fake_checkpoints, _full_csv  # noqa: E402

# A TinyStories-like fixture: short stories between <|endoftext|> separators, as the V2 files have them.
STORIES = "".join(f"Once upon a time, there was a little cat number {i}. It liked to play in the sun with a red "
                  f"ball. One day the ball rolled away, and the cat ran after it.\n<|endoftext|>\n"
                  for i in range(60))


def _fixture_dataset(tmp_path, monkeypatch, name="fixture", text=STORIES, char_limit=None,
                     separator="<|endoftext|>"):
    """A dataset in tmp_path/datasets/<name>/valid.txt, registered in common/data.py DATASETS for the test."""
    folder = tmp_path / "datasets" / name
    folder.mkdir(parents=True)
    (folder / "valid.txt").write_text(text, encoding="utf-8")
    spec = sweep.data.DatasetSpec(name=name, hf_repo="local/none", files={"validation": "valid.txt"},
                                  file_format="txt", eval_split="validation", about="test fixture",
                                  eval_char_limit=char_limit, document_separator=separator)
    monkeypatch.setattr(sweep.data, "DATASET_DIR", tmp_path / "datasets")
    monkeypatch.setitem(sweep.data.DATASETS, name, spec)
    return spec


def _fake_llama(tmp_path, monkeypatch):
    """An installed-looking Llama (config, empty weights, a tokenizer file): enough for the checks
    that run before any model is loaded."""
    root = _fake_checkpoints(tmp_path / "ckpt", "org/llama-ish", model_type="llama")
    (root / "models--org--llama-ish" / "snapshots" / "abc123" / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(local_cache, "CHECKPOINT_DIR", root)
    return root


# ----------------------------------------------------------------------------------------------
# The coverage check
# ----------------------------------------------------------------------------------------------

def test_coverage_finds_the_cells_a_row_misses(tmp_path, monkeypatch):
    _fixture_dataset(tmp_path, monkeypatch)
    _fake_llama(tmp_path, monkeypatch)
    path = _full_csv(tmp_path, "yes,mine,lut,org/llama-ish,,all_base,cpu,speed;perplexity,fixture,1,8,2,1,\n"
                               "yes,theirs,hf,org/llama-ish,,,cpu,speed,,1,8,2,1,\n"
                               "yes,full,baseline,org/llama-ish,,,cpu,speed;perplexity,fixture,1,8,2,1,\n"
                               "yes,core_speed,baseline,core,,,cpu,speed,,1,8,2,1,\n"
                               "yes,slow_only,hf,org/llama-ish,,,cpu,perplexity,fixture,,,,,\n")
    rows, errors = sweep.read_csv(path)
    _, errors = sweep.validate(rows, sweep.Settings(csv_path=path, checkpoint_dir=local_cache.CHECKPOINT_DIR))
    assert errors == []
    coverage = sweep.check_coverage(rows)
    assert coverage.lut_rows == ["mine"]
    assert coverage.reference == {("synthetic", "speed"), ("fixture", "perplexity")}
    # The core has random weights, so it owes only the speed cell.
    assert coverage.gaps == {"theirs": [("fixture", "perplexity")], "slow_only": [("synthetic", "speed")]}
    matrix = sweep.format_coverage(rows, coverage)
    assert "mine *" in matrix and "ppl:fixture" in matrix
    lines = {line.split()[0]: line.split() for line in matrix.splitlines()[2:] if line.strip()}
    assert lines["theirs"][-2:] == ["x", "GAP"] and lines["core_speed"][-2:] == ["x", "."]
    assert lines["full"][-2:] == ["x", "x"] and lines["slow_only"][-2:] == ["GAP", "x"]


def test_a_gap_stops_the_sweep_and_allow_gaps_runs_it(tmp_path, monkeypatch, capsys):
    _fixture_dataset(tmp_path, monkeypatch)
    _fake_llama(tmp_path, monkeypatch)
    path = _full_csv(tmp_path, "yes,mine,lut,org/llama-ish,,all_base,cpu,speed;perplexity,fixture,1,8,2,1,\n"
                               "yes,theirs,hf,org/llama-ish,,,cpu,speed,,1,8,2,1,\n")
    results = tmp_path / "results"
    common = ["--csv", str(path), "--results-dir", str(results), "--history", str(tmp_path / "h.jsonl"),
              "--no-power", "--no-calibrate", "--no-progress"]
    assert sweep.main(common) == 2                                 # stopped before anything ran
    captured = capsys.readouterr()
    assert "did not start" in captured.err and "theirs does not measure ppl:fixture" in captured.err
    assert "GAP" in captured.out and not results.exists()
    assert sweep.main(common + ["--dry-run", "--allow-gaps"]) == 0
    captured = capsys.readouterr()
    assert "Coverage" in captured.out and "GAP" in captured.out and "did not start" not in captured.err


def test_defaults_line_makes_rows_match_by_construction(tmp_path, monkeypatch, capsys):
    _fixture_dataset(tmp_path, monkeypatch)
    _fake_llama(tmp_path, monkeypatch)
    path = tmp_path / "defaults.csv"
    path.write_text("# defaults: benchmarks=speed;perplexity datasets=fixture\n" + sweep_header()
                    + "yes,mine,lut,org/llama-ish,,all_base,cpu,,,1,8,2,1,\n"
                    + "yes,theirs,hf,org/llama-ish,,,cpu,,,1,8,2,1,\n"
                    + "yes,core_speed,baseline,core,,,cpu,,,1,8,2,1,\n", encoding="utf-8")
    rows, errors = sweep.read_csv(path)
    assert errors == []
    by_name = {row.name: row for row in rows}
    assert by_name["theirs"].benchmarks == ["speed", "perplexity"] and by_name["theirs"].datasets == ["fixture"]
    assert by_name["core_speed"].benchmarks == ["speed"] and by_name["core_speed"].eval_datasets == []
    sweep.validate(rows, sweep.Settings(csv_path=path, checkpoint_dir=local_cache.CHECKPOINT_DIR))
    assert sweep.check_coverage(rows).gaps == {}
    assert sweep.main(["--csv", str(path), "--dry-run", "--no-calibrate", "--no-power", "--no-progress",
                       "--history", str(tmp_path / "h.jsonl"), "--results-dir", str(tmp_path / "r")]) == 0
    out = capsys.readouterr().out
    assert "Defaults for blank cells (line 1): benchmarks=speed;perplexity datasets=fixture." in out
    assert "Every row covers the lut rows' cells." in out


def sweep_header() -> str:
    return ("enabled,name,arm,model,weights,lut_preset,device,benchmarks,datasets,batch,context_lengths,"
            "decode_tokens,repeats,notes\n")


def test_bad_defaults_lines_are_reported(tmp_path):
    for line, message in (("# defaults: benchmark=speed", "takes benchmarks=a;b and datasets=a;b"),
                          ("# defaults: benchmarks=speed;loss", "unknown default(s) loss")):
        path = tmp_path / "bad.csv"
        path.write_text(line + "\n" + sweep_header() + "yes,a,baseline,core,,,cpu,speed,,1,8,2,1,\n")
        rows, errors = sweep.read_csv(path)
        assert rows == [] and len(errors) == 1 and message in errors[0], errors


# ----------------------------------------------------------------------------------------------
# The same text for every model, and the tokenizer-independent measures
# ----------------------------------------------------------------------------------------------

def test_evaluated_text_is_normalized_cut_and_counted(tmp_path, monkeypatch):
    spec = _fixture_dataset(tmp_path, monkeypatch, char_limit=300)
    text = sweep.data.eval_text(spec)
    assert "<|endoftext|>" not in text.text and "\n\nOnce upon a time" in text.text
    assert text.chars <= 300 and not text.text.endswith(" ") and STORIES.startswith(text.text[:50])
    assert text.text.split()[-1] in STORIES.split()                # cut at a space, no split word
    assert text.file_sha256 == sweep.data.file_sha256(spec.path()) and len(text.file_sha256) == 64
    assert sweep.data.eval_text(spec, 0).chars > 5000              # 0: the whole split
    counted = sweep.data.EvalText("x", "s", spec.path(), "0" * 64, None, "héllo wörld")
    assert (counted.chars, counted.utf8_bytes, counted.words) == (11, 13, 2)   # é and ö are 2 bytes each


def test_bits_per_byte_and_word_perplexity_by_hand():
    # 3 scored tokens, each given probability 1/2: nll_total = 3 ln 2 nats = 3 bits. The text has
    # 5 bytes and 2 words ("ab cd"):
    #   bits per byte   = 3 bits / 5 bytes = 0.6
    #   word perplexity = exp(3 ln 2 / 2) = 2^1.5
    #   token perplexity = exp(3 ln 2 / 3) = 2
    nll_total = 3 * math.log(2)
    metrics = sweep.text_metrics(nll_total, n_bytes=len("ab cd".encode()), n_words=len("ab cd".split()))
    assert metrics["bits_per_byte"] == pytest.approx(0.6, abs=1e-15)
    assert metrics["word_perplexity"] == pytest.approx(2 ** 1.5, abs=1e-12)
    with pytest.raises(ValueError):
        sweep.text_metrics(1.0, 0, 1)


def test_perplexity_totals_match_the_hand_formula_on_uniform_logits():
    # A model that gives every one of V tokens the same logit scores each token at -ln(1/V) = ln V
    # nats, so 9 scored tokens (10 tokens, the first unscored) total 9 ln V, whatever the windows.
    vocab = 7
    forward = lambda ids: torch.zeros(1, ids.shape[1], vocab, dtype=torch.float64)   # noqa: E731
    stats = sweep.measure_perplexity(forward, torch.arange(10) % vocab, window=4, stride=2, device="cpu")
    assert stats["scored_tokens"] == 9 and stats["nll_total"] == pytest.approx(9 * math.log(vocab), abs=1e-12)
    assert stats["perplexity"] == pytest.approx(vocab, abs=1e-12)
    bpb = sweep.text_metrics(stats["nll_total"], n_bytes=18, n_words=4)["bits_per_byte"]
    assert bpb == pytest.approx(9 * math.log2(vocab) / 18, abs=1e-12)


def _installed_or_skip(name: str) -> None:
    try:
        sweep.resolve_model(name, local_cache.CHECKPOINT_DIR)
    except sweep.SweepError:
        pytest.skip(f"{name} isn't installed in ../LUT-Transformer-local/checkpoints")
    pytest.importorskip("transformers")


def test_two_tokenizers_read_the_same_text_and_land_in_the_long_csv(tmp_path, monkeypatch):
    """GPT-2 and stories15M on one fixture: different token counts, the same text (sha256, bytes,
    words), and every number in results_long.csv with a unit."""
    _installed_or_skip("gpt2")
    _installed_or_skip("stories15m")
    _fixture_dataset(tmp_path, monkeypatch, char_limit=3000)
    path = _full_csv(tmp_path, "yes,s15_lut,lut,stories15m,,all_base,cpu,speed;perplexity,fixture,1,16,2,1,\n"
                               "yes,s15_hf,hf,stories15m,,,cpu,speed;perplexity,fixture,1,16,2,1,\n"
                               "yes,gpt2_hf,hf,gpt2,,,cpu,speed;perplexity,fixture,1,16,2,1,\n")
    results = tmp_path / "results"
    assert sweep.main(["--csv", str(path), "--results-dir", str(results), "--history", str(tmp_path / "h.jsonl"),
                       "--no-power", "--no-progress", "--warmup", "1"]) == 0
    (sweep_dir,) = (results / "sweeps").iterdir()
    with open(sweep_dir / "summary.csv", newline="", encoding="utf-8") as handle:
        ppl = {line["row"]: line for line in csv.DictReader(handle) if line["benchmark"] == "perplexity"}
    same = {(line["dataset_sha256"], line["eval_chars"], line["eval_bytes"], line["eval_words"]) for line in ppl.values()}
    assert len(same) == 1 and ppl["s15_hf"]["eval_tokens"] != ppl["gpt2_hf"]["eval_tokens"]
    assert float(ppl["s15_lut"]["bits_per_byte"]) == pytest.approx(float(ppl["s15_hf"]["bits_per_byte"]), rel=1e-4)

    with open(sweep_dir / "results_long.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(sweep.LONG_COLUMNS)
        long = list(reader)
    for line in long:
        float(line["value"])                                           # every value is a number
        units = sweep.SPEED_METRICS if line["benchmark"] == "speed" else sweep.PERPLEXITY_METRICS
        assert line["unit"] == units[line["metric"]] and line["run_id"] and line["git_commit"]
        assert line["timestamp_utc"].endswith("+00:00") and line["dtype"] == "torch.float32"
        if line["benchmark"] == "speed":
            assert line["context_tokens"] == "16" and line["dataset"] == "synthetic" and line["dataset_sha256"] == ""
        else:
            assert line["eval_window_tokens"] in ("256", "1024") and line["dataset_sha256"] == ppl["s15_hf"]["dataset_sha256"]
    toggles = {line["row"]: line["stage_toggles"] for line in long}
    assert toggles == {"s15_lut": "all BASE", "s15_hf": "", "gpt2_hf": ""}
    metrics = {(line["row"], line["metric"]) for line in long}
    assert {("gpt2_hf", "bits_per_byte"), ("gpt2_hf", "word_perplexity"), ("s15_lut", "decode_tokens_per_s")} <= metrics
    # all_results.csv got the same lines
    with open(results / "all_results.csv", newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == long


# ----------------------------------------------------------------------------------------------
# The long CSV of a tiny CPU sweep, and the cumulative file
# ----------------------------------------------------------------------------------------------

def test_long_csv_schema_on_a_tiny_cpu_sweep(tmp_path):
    path = _full_csv(tmp_path, "yes,tiny_lut,lut,core,,all_base,cpu,speed,,1,8,2,2,\n"
                               "yes,tiny_base,baseline,core,,,cpu,speed,,1,8;16,2,2,\n")
    results = tmp_path / "results"
    assert sweep.main(["--csv", str(path), "--results-dir", str(results), "--history", str(tmp_path / "h.jsonl"),
                       "--no-power", "--no-progress", "--warmup", "1"]) == 0
    (sweep_dir,) = (results / "sweeps").iterdir()
    with open(sweep_dir / "results_long.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(sweep.LONG_COLUMNS)
        long = list(reader)
    keys = [(line["row"], line["benchmark"], line["context_tokens"], line["metric"]) for line in long]
    assert len(keys) == len(set(keys))                    # one line per row, benchmark, context and metric
    assert {line["row"] for line in long} == {"tiny_lut", "tiny_base"}
    res = [line for line in long if line["benchmark"] == "resources"]
    assert {(line["row"], line["dataset"]) for line in res} == {("tiny_lut", "speed"), ("tiny_base", "speed")}
    for line in res:                                      # per-phase resource use, each with its unit
        assert line["unit"] == sweep.RESOURCE_METRICS[line["metric"]] and line["context_tokens"] == ""
    assert {"process_cpu_user_s", "window_s"} <= {line["metric"] for line in res}
    long = [line for line in long if line["benchmark"] == "speed"]
    assert {(line["row"], line["context_tokens"]) for line in long} == {("tiny_lut", "8"), ("tiny_base", "8"),
                                                                        ("tiny_base", "16")}
    for line in long:
        assert line["unit"] == sweep.SPEED_METRICS[line["metric"]] and float(line["value"]) > 0
        assert line["weights"] == "random" and line["repeats"] == "2" and line["device"] == "cpu"
    assert {line["stage_toggles"] for line in long if line["row"] == "tiny_lut"} == {"all BASE"}
    # No power on this run, so no power metrics, rather than blank values
    assert not any(line["metric"].endswith("power_w") for line in long)
    with open(results / "all_results.csv", newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(long) + len(res)


def test_cumulative_csv_appends_and_never_repeats_a_run_id(tmp_path):
    path = tmp_path / "all_results.csv"
    base = {column: "" for column in sweep.LONG_COLUMNS}
    first = [{**base, "run_id": "run_a", "metric": "m1", "value": 1.0},
             {**base, "run_id": "run_a", "metric": "m2", "value": 2.0}]
    second = [{**base, "run_id": "run_b", "metric": "m1", "value": 3.0}]
    assert sweep.append_cumulative(path, first) == (2, 0)
    assert sweep.append_cumulative(path, first) == (0, 1)               # same run again: nothing added
    assert sweep.append_cumulative(path, first + second) == (1, 1)      # only the new run
    with open(path, newline="", encoding="utf-8") as handle:
        lines = list(csv.DictReader(handle))
    assert [(line["run_id"], line["metric"]) for line in lines] == [("run_a", "m1"), ("run_a", "m2"), ("run_b", "m1")]
    other = tmp_path / "other.csv"
    other.write_text("run_id,value\nx,1\n", encoding="utf-8")
    with pytest.raises(sweep.SweepError, match="other columns"):
        sweep.append_cumulative(other, second)
