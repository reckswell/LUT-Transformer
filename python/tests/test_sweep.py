"""The benchmark sweep (bench/sweep.py) and the progress bars (bench/progress.py):
  - the CSV parses, and every bad cell is reported with its line and column
  - a model that isn't installed in the local folder stops the sweep before anything runs or is
    written, and nothing is downloaded without --download
  - --dry-run prints the plan and ETA (from history, calibration, or unknown) and writes nothing
  - a tiny end-to-end sweep on the core model with random weights, on the CPU
  - estimates: interpolation in context length, the per-row arithmetic
  - progress updates happen between timed calls, never inside one
CPU only and a few seconds in all. The GPT-2 check skips when GPT-2 isn't installed locally.
"""

import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bench import sweep  # noqa: E402
from bench.progress import WeightedBar, format_seconds  # noqa: E402
from bench.timing import time_fn  # noqa: E402
from reference import local_cache  # noqa: E402

HEADER = "enabled,name,arm,model,lut_preset,device,batch,context_lengths,decode_tokens,repeats,notes\n"


def _csv(tmp_path: Path, body: str, name: str = "sweep.csv") -> Path:
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def _fake_checkpoints(root: Path, repo: str = "org/tiny-gpt2", model_type: str = "gpt2") -> Path:
    """A Hugging Face cache entry (models--org--name) with a config and an empty weights file."""
    folder = root / ("models--" + repo.replace("/", "--"))
    snapshot = folder / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text("abc123")
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type, "n_positions": 64, "vocab_size": 100}))
    (snapshot / "model.safetensors").write_bytes(b"")
    return root


# ----------------------------------------------------------------------------------------------
# CSV parsing
# ----------------------------------------------------------------------------------------------

def test_csv_parses_rows_comments_and_defaults(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text(
        "# a comment line\n"
        "enabled, name, arm, model, lut_preset, device, batch, context_lengths, decode_tokens, repeats, notes\n"
        "\n"
        "yes, a, baseline, core, , cpu, 2, 128;512, 16, 3, \"notes, with a comma\"\n"
        ", b, lut, core, params, cuda:1, 1, 64, 0, 1,\n"
        "no, c, lut, core, all_base, CUDA, 1, 8, 2, 1\n", encoding="utf-8")
    rows, errors = sweep.read_csv(path)
    assert errors == []
    a, b, c = rows
    assert (a.enabled, a.arm, a.batch, a.context_lengths, a.decode_tokens, a.repeats, a.notes, a.line) == \
        (True, "baseline", 2, [128, 512], 16, 3, "notes, with a comma", 4)
    assert a.lut_preset is None and a.preset_label == "-"
    assert b.enabled and b.lut_preset == "parameters" and b.preset_label == "params" and b.device == "cuda:1"
    assert not c.enabled and c.lut_preset == "all_base" and c.device == "cuda"


def test_csv_reports_every_bad_cell_with_line_and_column(tmp_path):
    path = _csv(tmp_path,
                "maybe,x y,gpu,,,tpu,0,1;x,-1,0,\n"
                "yes,ok,baseline,core,all_base,cpu,1,8;8,2,1,\n"
                "yes,dup,baseline,core,,cpu,1,8,2,1,\n"
                "yes,dup,baseline,core,,cpu,1,8,2,1,\n"
                "yes,long,baseline,core,,cpu,1,8,2,1,notes,extra\n")
    rows, errors = sweep.read_csv(path)
    text = "\n".join(errors)
    for expected in ("line 2 (x y)", "enabled must be yes or no", "name must be letters", "arm must be one of",
                     "model is empty", "device must be cpu, cuda or cuda:N", "batch must be at least 1",
                     "each context length must be a whole number", "decode_tokens must be at least 0",
                     "repeats must be at least 1", "line 3 (ok)", "lut_preset applies to the lut arm only",
                     "context_lengths repeats a value", "'dup' is used more than once",
                     "line 6: 12 cells but 11 columns"):
        assert expected in text, expected
    assert [r.name for r in rows] == ["dup", "dup"]


def test_csv_header_problems(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("enabled,name,arm,model,batch,context_lengths,decode_tokens,repeats,colour\n")
    _, errors = sweep.read_csv(path)
    assert any("missing column(s) device" in e for e in errors)
    assert any("unknown column(s) colour" in e for e in errors)
    assert sweep.read_csv(tmp_path / "nope.csv")[1] == [f"{tmp_path / 'nope.csv'}: no such file"]


def test_the_example_csv_parses():
    rows, errors = sweep.read_csv(sweep.DEFAULT_CSV)
    assert errors == [] and rows
    assert all(not row.enabled for row in rows if row.device.startswith("cuda")), \
        "GPU rows ship disabled, so a default sweep never takes the GPU by surprise"


# ----------------------------------------------------------------------------------------------
# Models: installed locally or the sweep doesn't start
# ----------------------------------------------------------------------------------------------

def test_resolve_model_reads_only_the_checkpoint_folder(tmp_path):
    root = _fake_checkpoints(tmp_path / "ckpt")
    assert sweep.installed_models(root) == ["org/tiny-gpt2"]
    source = sweep.resolve_model("org/tiny-gpt2", root)
    assert (source.key, source.installed, source.max_context, source.vocab_size) == ("org/tiny-gpt2", True, 64, 100)
    assert sweep.resolve_model("models--org--tiny-gpt2", root).key == "org/tiny-gpt2"
    assert sweep.resolve_model("core", root).kind == "core"

    with pytest.raises(sweep.SweepError, match="not installed.*org/tiny-gpt2.*--download"):
        sweep.resolve_model("gpt2", root)                 # openai-community/gpt2 isn't in this folder
    with pytest.raises(sweep.SweepError, match="neither 'core'"):
        sweep.resolve_model("my-folder", root)
    pending = sweep.resolve_model("org/other", root, download=True)
    assert not pending.installed and pending.load_name == "org/other"

    llama = sweep.resolve_model("org/llama-ish", _fake_checkpoints(root, "org/llama-ish", model_type="llama"))
    assert llama.architecture == "llama" and llama.max_context is None and llama.eval_context == 64
    assert llama.load_name.endswith("snapshots/abc123")         # the installed folder: loads offline
    _fake_checkpoints(root, "org/bert-ish", model_type="bert")
    with pytest.raises(sweep.SweepError, match="the sweep runs gpt2, llama, opt, qwen2 models"):
        sweep.resolve_model("org/bert-ish", root)


def test_uninstalled_model_stops_the_sweep_before_anything_runs(tmp_path, monkeypatch, capsys):
    root = _fake_checkpoints(tmp_path / "ckpt")
    monkeypatch.setattr(local_cache, "CHECKPOINT_DIR", root)
    path = _csv(tmp_path,
                "yes,fine,baseline,core,,cpu,1,8,2,1,\n"
                "yes,medium,hf,openai-community/gpt2-medium,,cpu,1,8,2,1,\n"
                "yes,mine,baseline,my-finetune,,cpu,1,8,2,1,\n"
                "yes,hfcore,hf,core,,cpu,1,8,2,1,\n"
                "yes,toolong,baseline,org/tiny-gpt2,,cpu,1,60,8,1,\n"
                "yes,nopreset,lut,core,no_such_preset,cpu,1,8,2,1,\n")
    results, history = tmp_path / "results", tmp_path / "history.jsonl"
    code = sweep.main(["--csv", str(path), "--results-dir", str(results), "--history", str(history), "--no-power"])
    err = capsys.readouterr().err
    assert code == 2
    assert "nothing ran" in err
    assert "'openai-community/gpt2-medium' is not installed" in err and "--download" in err
    assert f"Installed in {root}: org/tiny-gpt2" in err
    assert "'my-finetune' is neither" in err
    assert "the hf arm needs a Hugging Face model" in err
    assert "exceeds the model's 64 learned positions" in err
    assert "'no_such_preset' is not 'params'" in err
    assert not results.exists() and not history.exists()


def test_cuda_rows_need_a_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rows, _ = sweep.read_csv(_csv(tmp_path, "yes,g,baseline,core,,cuda,1,8,2,1,\n"))
    _, errors = sweep.validate(rows, sweep.Settings(csv_path=tmp_path / "sweep.csv"))
    assert errors and "PyTorch sees no CUDA GPU" in errors[0]


# ----------------------------------------------------------------------------------------------
# Dry run and estimates
# ----------------------------------------------------------------------------------------------

TINY = "yes,tiny_base,baseline,core,,cpu,1,8;16,2,2,\nno,off,baseline,core,,cpu,1,8,2,2,\n"


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    path = _csv(tmp_path, TINY)
    results, history = tmp_path / "results", tmp_path / "history.jsonl"
    common = ["--csv", str(path), "--results-dir", str(results), "--history", str(history), "--no-power", "--dry-run"]

    assert sweep.main(common + ["--no-calibrate"]) == 0
    out = capsys.readouterr().out
    assert "Sweep plan for" in out and "tiny_base" in out and "8;16" in out
    assert "unknown" in out and "1 row(s) with no estimate" in out
    assert "off" in out and "not run" in out and "enabled=no" in out
    assert "Dry run: nothing was benchmarked" in out

    history.write_text(json.dumps({
        "key": {"arm": "baseline", "model": "core", "preset": None, "device": "cpu", "batch": 1},
        "load_s": 2.0, "overhead_s": 1.0,
        "contexts": {"8": {"prefill_ms": 100.0, "decode_ms": 50.0}, "16": {"prefill_ms": 200.0, "decode_ms": 50.0}},
    }) + "\n")
    assert sweep.main(common) == 0
    out = capsys.readouterr().out
    # 1 s overhead + 2 s load + (2 warm-up + 2 timed) x [(0.1 + 2 x 0.05) + (0.2 + 2 x 0.05)] s = 5 s
    assert "0:00:05" in out and "history, load 2.0 s" in out and "Total ETA 0:00:05" in out

    history.unlink()
    assert sweep.main(common) == 0                    # no history: a short calibration run
    captured = capsys.readouterr()
    assert "calibrating tiny_base" in captured.err + captured.out
    assert "calibration" in captured.out
    assert not results.exists() and not history.exists()


def test_fit_interpolates_and_extrapolates_in_context_length():
    points = {128: 10.0, 512: 40.0}
    assert sweep._fit(points, 128, scale_alone=True) == 10.0
    assert sweep._fit(points, 320, scale_alone=True) == pytest.approx(25.0)
    assert sweep._fit(points, 1024, scale_alone=True) == pytest.approx(80.0)
    assert sweep._fit(points, 32, scale_alone=True) == pytest.approx(2.5)        # floor: a quarter of 10
    assert sweep._fit({100: 4.0}, 400, scale_alone=True) == pytest.approx(16.0)  # prefill grows with T
    assert sweep._fit({100: 4.0}, 400, scale_alone=False) == pytest.approx(4.0)  # decode kept


def test_row_seconds_counts_repeats_power_and_load():
    row = sweep.Row(1, True, "r", "baseline", "core", None, "cpu", 1, [8, 16], 4, 3)
    est = sweep.Estimate("history", {8: (10.0, 5.0), 16: (20.0, 5.0)}, load_s=2.0, overhead_s=1.0)
    settings = sweep.Settings(warmup=1, power_seconds=3.0, power_warmup_s=2.0)
    # per repeat: T=8 0.03 s, T=16 0.04 s; 4 repeats each -> 0.28 s
    assert sweep.row_seconds(row, est, settings, load_needed=False, power_on=False) == pytest.approx(1.28)
    assert sweep.row_seconds(row, est, settings, load_needed=True, power_on=False) == pytest.approx(3.28)
    # power: 2 windows of 5 s and one extra prefill-plus-decode per context length
    assert sweep.row_seconds(row, est, settings, load_needed=False, power_on=True) == pytest.approx(1.28 + 20 + 0.07)
    assert sweep.row_seconds(row, sweep.Estimate("unknown"), settings, load_needed=False, power_on=False) is None


# ----------------------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------------------

def test_tiny_sweep_end_to_end_on_the_cpu(tmp_path, capsys):
    path = _csv(tmp_path,
                "yes,tiny_base,baseline,core,,cpu,1,8;16,2,2,first row\n"
                "yes,tiny_lut,lut,core,all_base,cpu,2,8,0,1,\n"
                "no,off,baseline,core,,cpu,1,8,2,2,\n")
    results, history = tmp_path / "results", tmp_path / "history.jsonl"
    code = sweep.main(["--csv", str(path), "--results-dir", str(results), "--history", str(history),
                       "--idle-seconds", "0.1", "--power-seconds", "0.2", "--power-warmup-seconds", "0.1",
                       "--warmup", "1", "--no-progress"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Total ETA" in out

    (sweep_dir,) = (results / "sweeps").iterdir()
    assert sorted(p.name for p in sweep_dir.iterdir()) == ["models.csv", "plan.csv", "results_long.csv", "status.json",
                                                           "summary.csv"]
    assert (results / "all_results.csv").is_file()
    status = json.loads((sweep_dir / "status.json").read_text())
    assert status["rows_done"] == status["rows_total"] == 2 and status["failed_rows"] == []
    with open(sweep_dir / "summary.csv", newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert [(line["row"], line["context"]) for line in lines] == [("tiny_base", "8"), ("tiny_base", "16"),
                                                                   ("tiny_lut", "8")]
    for line in lines:
        assert line["status"] == "ok" and line["dtype"] == "torch.float32" and float(line["prefill_ms_median"]) > 0
    assert float(lines[0]["decode_ms_per_token_median"]) > 0 and lines[2]["decode_ms_per_token_median"] == ""
    assert float(lines[2]["prefill_tokens_per_s"]) == pytest.approx(2 * 8 / (float(lines[2]["prefill_ms_median"]) / 1000))
    # CPU power: RAPL is root-only on alpha-pc; if it can't be read the row says why instead of a number
    if not lines[0]["prefill_power_w"]:
        assert lines[0]["power_note"].startswith("power unavailable")

    runs = sorted((results / "runs").iterdir())
    assert [r.name.split("_", 1)[1] for r in runs] == ["sweep_tiny_base", "sweep_tiny_lut"]
    summary = json.loads((runs[0] / "summary.json").read_text())
    assert summary["arm"] == "baseline" and summary["model"] == "core" and summary["dtype"] == "torch.float32"
    assert set(summary["T8"]) >= {"prefill_ms_median", "decode_ms_per_token_median", "decode_tokens_per_s"}
    meta = json.loads((runs[0] / "meta.json").read_text())
    assert meta["notes"] == "first row" and meta["configs"]["row"]["context_lengths"] == [8, 16]
    records_jsonl = [json.loads(line) for line in (runs[0] / "metrics.jsonl").read_text().splitlines()]
    contexts = [r for r in records_jsonl if r.get("kind") == "speed"]
    assert [len(c["prefill_ms"]) for c in contexts] == [2, 2] and [len(c["decode_ms"]) for c in contexts] == [4, 4]
    # resource use per phase (bench/resources.py): host CPU time always, GPU readings only on cuda rows
    (res,) = [r for r in records_jsonl if r.get("kind") == "resources"]
    assert res["phases"]["speed"]["process_cpu_user_s"] >= 0 and res["phases"]["speed"]["window_s"] > 0
    assert "resources_speed" in summary and "power_draw_mean" not in summary["resources_speed"]

    records = [json.loads(line) for line in history.read_text().splitlines()]
    assert [r["key"]["arm"] for r in records] == ["baseline", "lut"]
    assert set(records[0]["contexts"]) == {"8", "16"}

    # The next plan for the same rows comes from that history, with no calibration
    assert sweep.main(["--csv", str(path), "--history", str(history), "--no-power", "--dry-run",
                       "--results-dir", str(results)]) == 0
    captured = capsys.readouterr()
    assert "calibrating" not in captured.err + captured.out and "history" in captured.out


def test_a_failing_row_is_reported_and_the_others_still_run(tmp_path, monkeypatch, capsys):
    path = _csv(tmp_path, "yes,first,baseline,core,,cpu,1,8,1,1,\nyes,second,baseline,core,,cpu,1,8,1,1,\n")
    original = sweep.measure_context
    state = {"raised": False}

    def flaky(*args, **kwargs):
        # calibrations use warmup=1; the first real measurement (warmup=2, the default) fails once
        if kwargs.get("warmup") == 2 and not state["raised"]:
            state["raised"] = True
            raise RuntimeError("boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "measure_context", flaky)
    code = sweep.main(["--csv", str(path), "--results-dir", str(tmp_path / "r"), "--history",
                       str(tmp_path / "h.jsonl"), "--no-power", "--no-progress"])
    assert code == 1
    (sweep_dir,) = (tmp_path / "r" / "sweeps").iterdir()
    with open(sweep_dir / "summary.csv", newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert lines[0]["row"] == "first" and lines[0]["status"].startswith("failed: boom")
    assert lines[1]["row"] == "second" and lines[1]["status"] == "ok"
    assert json.loads((sweep_dir / "status.json").read_text())["failed_rows"] == ["first"]


def test_hf_and_baseline_runners_agree_on_gpt2():
    """The hf arm (transformers' cache, logits_to_keep=1) and our baseline make the same greedy
    tokens from GPT-2's weights, and rewinding the hf cache gives the same next token again."""
    try:
        source = sweep.resolve_model("gpt2", local_cache.CHECKPOINT_DIR)
    except sweep.SweepError:
        pytest.skip("GPT-2 isn't installed in ../LUT-Transformer-local/checkpoints")
    pytest.importorskip("transformers")
    loaded = sweep.load_model(source, "cpu", sweep.Settings())
    ids = sweep.make_prompt(loaded.cfg.vocab_size, 1, 16, 0, "cpu")
    tokens = {}
    for arm in ("hf", "baseline"):
        runner = sweep.make_runner(sweep.Row(1, True, arm, arm, "gpt2", None, "cpu", 1, [16], 4, 1), loaded)
        runner.start(1, 20)
        out = [runner.prefill(ids)]
        for _ in range(4):
            out.append(runner.decode(out[-1]))
        tokens[arm] = [int(t) for t in out]
        runner.rewind(16)
        assert int(runner.decode(out[0])) == tokens[arm][1]
    assert tokens["hf"] == tokens["baseline"]


# ----------------------------------------------------------------------------------------------
# Measurement plumbing and progress bars
# ----------------------------------------------------------------------------------------------

class _CountingRunner:
    def __init__(self, events):
        self.events = events

    def prefill(self, ids):
        self.events.append("prefill")
        return ids[:, -1:]

    def decode(self, token):
        self.events.append("decode")
        return token


def test_measure_context_updates_progress_only_between_repeats():
    events = []
    prefill_ms, decode_ms = sweep.measure_context(
        _CountingRunner(events), torch.zeros(1, 4, dtype=torch.int64), 3, warmup=1, repeats=2, device="cpu",
        after_repeat=lambda i: events.append(f"progress {i}"))
    assert len(prefill_ms) == 2 and len(decode_ms) == 6        # warm-up not recorded
    assert events == ["prefill", "decode", "decode", "decode", "progress 0"] + \
        ["prefill", "decode", "decode", "decode", "progress 1"] + \
        ["prefill", "decode", "decode", "decode", "progress 2"]


def test_time_fn_calls_after_call_between_calls():
    events = []
    time_fn(lambda: events.append("call"), device="cpu", warmup=1, repeats=2, after_call=lambda: events.append("bar"))
    assert events == ["call", "bar", "call", "bar", "call", "bar"]


def test_power_numbers_keep_sources_apart():
    report = {"sources": {"nvidia_gpu:0": {"energy_joules": 6.0, "average_power_watts": 120.0, "scope": "board"},
                          "cpu_package_rapl": {"energy_joules": None, "reason": "root-only"}}}
    numbers = sweep.power_numbers(report, tokens=300)
    assert numbers["nvidia_gpu:0"]["energy_per_token_mj"] == pytest.approx(20.0)
    assert numbers["cpu_package_rapl"] == {"unavailable": "root-only"}


def test_weighted_bar_never_shows_done_before_close():
    bar = WeightedBar(10.0, 2, label="sweep", unit="rows", show=False)
    bar.advance(4.0, items=1)
    bar.advance(100.0)                                         # an underestimate: capped below 100%
    assert bar.done == 1 and bar.spent_s < 10.0
    bar.close()
    assert bar.spent_s == 10.0
    assert format_seconds(3725) == "1:02:05" and format_seconds(None) == "?"


def test_training_loop_shows_progress_and_reports_time():
    from baseline.train import training_loop

    seen = []
    summary = training_loop(lambda step, batch: seen.append(step) or {"loss": 1.0 / (step + 1), "tokens": 8},
                            iter(range(5)), num_steps=5, progress=False)
    assert seen == [0, 1, 2, 3, 4]
    assert summary["steps"] == 5 and summary["tokens_seen"] == 40 and summary["train_time_s"] >= 0


# ----------------------------------------------------------------------------------------------
# Benchmarks, datasets and the trained-before rule
# ----------------------------------------------------------------------------------------------

FULL = ("enabled,name,arm,model,weights,lut_preset,device,benchmarks,datasets,batch,context_lengths,"
        "decode_tokens,repeats,notes\n")


def _full_csv(tmp_path: Path, body: str, name: str = "full.csv") -> Path:
    path = tmp_path / name
    path.write_text(FULL + body, encoding="utf-8")
    return path


def _gpt2_or_skip() -> None:
    try:
        sweep.resolve_model("gpt2", local_cache.CHECKPOINT_DIR)
    except sweep.SweepError:
        pytest.skip("GPT-2 isn't installed in ../LUT-Transformer-local/checkpoints")
    pytest.importorskip("transformers")


def test_csv_parses_weights_benchmarks_and_datasets(tmp_path):
    rows, errors = sweep.read_csv(_full_csv(tmp_path,
                                            "yes,a,hf,gpt2,pretrained,,cpu,speed;perplexity,wikitext2;tinystories,1,128,4,2,\n"
                                            "yes,b,baseline,gpt2,,,cpu,perplexity,wikitext2,,,,,\n"
                                            "yes,c,baseline,core,,,cpu,,,1,8,2,1,\n"))
    assert errors == []
    a, b, c = rows
    assert a.weights == "pretrained" and a.benchmarks == ["speed", "perplexity"]
    assert a.eval_datasets == ["wikitext2", "tinystories"] and a.benchmark_label == "speed;ppl:wikitext2,tinystories"
    assert b.weights is None and not b.runs_speed and b.context_lengths == [] and b.benchmark_label == "ppl:wikitext2"
    assert c.benchmarks == ["speed"] and c.datasets == [] and c.eval_datasets == []


def test_csv_reports_bad_weights_benchmarks_and_datasets(tmp_path):
    _, errors = sweep.read_csv(_full_csv(tmp_path,
                                         "yes,a,hf,gpt2,trained,,cpu,speed;accuracy,imagenet,1,128,4,2,\n"
                                         "yes,b,hf,gpt2,,,cpu,perplexity,,,,,,\n"
                                         "yes,c,hf,gpt2,,,cpu,speed,,,,,,\n"
                                         "yes,d,hf,gpt2,,,cpu,perplexity;perplexity,wikitext2,,,,,\n"))
    text = "\n".join(errors)
    for expected in ("weights must be one of pretrained, random, train",
                     "benchmarks must be from speed, perplexity, got accuracy", "unknown dataset(s) imagenet",
                     "line 3 (b): perplexity needs at least one dataset",
                     "line 4 (c): the speed benchmark needs batch, context_lengths, decode_tokens, repeats",
                     "benchmarks repeats a value"):
        assert expected in text, expected


def test_trained_before_rule(tmp_path, monkeypatch):
    """A model someone has trained is never trained again; the core has nothing to load; random
    weights are for speed only."""
    root = _fake_checkpoints(tmp_path / "ckpt")
    monkeypatch.setattr(local_cache, "CHECKPOINT_DIR", root)
    rows, errors = sweep.read_csv(_full_csv(tmp_path,
                                            "yes,retrain,baseline,org/tiny-gpt2,train,,cpu,perplexity,tinystories,,,,,\n"
                                            "yes,core_pre,baseline,core,pretrained,,cpu,speed,,1,8,2,1,\n"
                                            "yes,gpt_random,baseline,org/tiny-gpt2,random,,cpu,speed,,1,8,2,1,\n"
                                            "yes,random_ppl,baseline,core,random,,cpu,perplexity,tinystories,,,,,\n"
                                            "yes,core_train,baseline,core,train,,cpu,perplexity,tinystories,,,,,\n"
                                            "yes,defaults_core,baseline,core,,,cpu,speed,,1,8,2,1,\n"
                                            "yes,defaults_gpt,baseline,org/tiny-gpt2,,,cpu,speed,,1,8,2,1,\n"))
    assert errors == []
    _, errors = sweep.validate(rows, sweep.Settings(csv_path=tmp_path / "full.csv"))
    text = "\n".join(errors)
    assert "(retrain): org/tiny-gpt2 has already been trained" in text
    assert "never trains a model someone else has trained" in text
    assert "(core_pre): the core has no pretrained weights" in text
    assert "(gpt_random): random weights are for the core" in text
    assert "(random_ppl): perplexity of random weights measures nothing" in text
    assert "core_train" not in text                         # needs training: listed in the plan, not an error
    weights = {row.name: row.weights for row in rows}
    assert weights["defaults_core"] == "random" and weights["defaults_gpt"] == "pretrained"


def test_rows_that_need_training_are_listed_not_run(tmp_path, capsys):
    path = _full_csv(tmp_path, "yes,core_train,baseline,core,train,,cpu,perplexity,tinystories,,,,,\n")
    results = tmp_path / "results"
    code = sweep.main(["--csv", str(path), "--results-dir", str(results), "--history", str(tmp_path / "h.jsonl"),
                       "--no-power"])
    out = capsys.readouterr().out
    assert code == 0
    assert "needs training" in out and "Not run, need training first: core_train" in out
    assert "Nothing to run" in out and not results.exists()


def test_datasets_load_only_from_the_local_folder(tmp_path, monkeypatch, capsys):
    _gpt2_or_skip()
    monkeypatch.setattr(sweep.data, "DATASET_DIR", tmp_path / "datasets")
    monkeypatch.setattr(sweep.data, "missing_requirement", lambda spec: None)    # as if pyarrow were installed
    downloads = []
    monkeypatch.setattr(sweep.data, "download", lambda spec, splits=None: downloads.append(spec.name))
    path = _full_csv(tmp_path, "yes,ppl,hf,gpt2,pretrained,,cpu,perplexity,wikitext2,,,,,\n")
    common = ["--csv", str(path), "--results-dir", str(tmp_path / "r"), "--history", str(tmp_path / "h.jsonl"),
              "--no-power", "--dry-run", "--no-calibrate"]
    assert sweep.main(common) == 2
    err = capsys.readouterr().err
    assert "dataset 'wikitext2' is not installed" in err and "--download fetches it from Salesforce/wikitext" in err
    assert downloads == []
    assert sweep.main(common + ["--download"]) == 0           # asked for: fetched before anything else runs
    assert downloads == ["wikitext2"]


def test_missing_reader_package_is_reported(tmp_path, monkeypatch, capsys):
    _gpt2_or_skip()
    monkeypatch.setattr(sweep.data, "missing_requirement", lambda spec: "pyarrow")
    path = _full_csv(tmp_path, "yes,ppl,hf,gpt2,pretrained,,cpu,perplexity,wikitext2,,,,,\n")
    assert sweep.main(["--csv", str(path), "--no-power", "--dry-run", "--history", str(tmp_path / "h.jsonl")]) == 2
    assert "needs the pyarrow package, which isn't installed" in capsys.readouterr().err


def test_count_windows():
    assert sweep.count_windows(100, 128, 64) == 1
    assert sweep.count_windows(128, 128, 64) == 1
    assert sweep.count_windows(129, 128, 64) == 2
    assert sweep.count_windows(1500, 1024, 512) == 2
    assert sweep.count_windows(2049, 1024, 512) == 4


def test_perplexity_scores_every_token_once_with_the_right_target():
    vocab = 7
    tokens = torch.arange(300) % vocab
    lengths = []

    def uniform(x):
        lengths.append(x.shape[1])
        return torch.zeros(1, x.shape[1], vocab)

    stats = sweep.measure_perplexity(uniform, tokens, window=128, stride=64, device="cpu")
    assert stats["perplexity"] == pytest.approx(7.0)            # uniform over 7 tokens
    assert stats["scored_tokens"] == 299 and stats["eval_windows"] == sweep.count_windows(300, 128, 64) == len(lengths)
    assert lengths[:-1] == [128] * (len(lengths) - 1)

    def knows_the_next_token(x):     # position t predicts token t + 1 = (x_t + 1) mod 7
        out = torch.full((1, x.shape[1], vocab), -100.0)
        out[0, torch.arange(x.shape[1]), (x[0] + 1) % vocab] = 100.0
        return out

    assert sweep.measure_perplexity(knows_the_next_token, tokens, window=128, stride=64,
                                    device="cpu")["perplexity"] == pytest.approx(1.0)


def test_perplexity_in_one_window_equals_the_full_forward_loss():
    from baseline import host as base
    from eval.harness import next_token_loss

    cfg = base.TransformerConfig(vocab_size=50, context_length=64, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    params = base.init_params(cfg, seed=0)
    ops = base.base_ops(cfg, "inference")
    tokens = torch.randint(0, 50, (40,), generator=torch.Generator().manual_seed(1))
    stats = sweep.measure_perplexity(lambda x: base.forward(params, x, ops, cfg), tokens, window=64, stride=32,
                                     device="cpu")
    expected = next_token_loss(base.forward(params, tokens[None], ops, cfg), tokens[None])
    assert stats["nll_per_token"] == pytest.approx(expected, rel=1e-12)


def test_perplexity_end_to_end_hf_and_baseline_agree_on_gpt2(tmp_path, monkeypatch, capsys):
    """Both arms on GPT-2's weights, on a small local text file (no download), through the sweep:
    the same text (sha256, bytes, words) for both, the same perplexity and bits per byte to float32
    precision, two windows of 1,024 tokens, and the next plan comes from history."""
    _gpt2_or_skip()
    folder = tmp_path / "datasets" / "tinytext"
    folder.mkdir(parents=True)
    (folder / "story.txt").write_text(" ".join(f"Sentence {i} is about a small model that reads a story."
                                               for i in range(200)))
    spec = sweep.data.DatasetSpec(name="tinytext", hf_repo="local/none", files={"test": "story.txt"},
                                  file_format="txt", eval_split="test", about="test text", eval_char_limit=6000)
    monkeypatch.setattr(sweep.data, "DATASET_DIR", tmp_path / "datasets")
    monkeypatch.setitem(sweep.data.DATASETS, "tinytext", spec)
    path = _full_csv(tmp_path, "yes,hf_ppl,hf,gpt2,pretrained,,cpu,perplexity,tinytext,,,,,\n"
                               "yes,base_ppl,baseline,gpt2,,,cpu,perplexity,tinytext,,,,,\n")
    results, history = tmp_path / "results", tmp_path / "history.jsonl"
    assert sweep.main(["--csv", str(path), "--results-dir", str(results), "--history", str(history),
                       "--no-power", "--no-progress"]) == 0
    (sweep_dir,) = (results / "sweeps").iterdir()
    with open(sweep_dir / "summary.csv", newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert [(line["row"], line["benchmark"], line["dataset"]) for line in lines] == \
        [("hf_ppl", "perplexity", "tinytext"), ("base_ppl", "perplexity", "tinytext")]
    sha = sweep.data.file_sha256(folder / "story.txt")
    for line in lines:
        # 6,000 characters cut back to a space: 5,993 characters, 1,199 words, 1,417 GPT-2 tokens
        assert (line["eval_windows"], line["eval_tokens"], line["eval_window"], line["eval_stride"]) == \
            ("2", "1416", "1024", "512")
        assert (line["eval_chars"], line["eval_bytes"], line["eval_words"], line["dataset_sha256"]) == \
            ("5993", "5993", "1199", sha)
        assert line["weights"] == "pretrained"
    hf, ours = (float(line["perplexity"]) for line in lines)
    assert ours == pytest.approx(hf, rel=1e-4)
    hf_bpb, our_bpb = (float(line["bits_per_byte"]) for line in lines)
    assert our_bpb == pytest.approx(hf_bpb, rel=1e-4) and 0 < hf_bpb < 2
    assert 1.0 < hf < 50.0                                       # repetitive text: low perplexity

    capsys.readouterr()
    assert sweep.main(["--csv", str(path), "--history", str(history), "--no-power", "--dry-run",
                       "--results-dir", str(results)]) == 0
    captured = capsys.readouterr()
    assert "calibrating" not in captured.out + captured.err and "history" in captured.out


def test_datasets_and_models_share_the_local_folder():
    assert sweep.data.LOCAL_DIR == local_cache.LOCAL_DIR
    assert sweep.data.DATASET_DIR.parent == local_cache.LOCAL_DIR or "LUT_DATASET_DIR" in __import__("os").environ


# ----------------------------------------------------------------------------------------------
# stories15M: a Llama model every arm runs (our arms through the Llama profile, reference/hf_llama.py)
# ----------------------------------------------------------------------------------------------

def _stories15m_or_skip() -> sweep.ModelSource:
    try:
        source = sweep.resolve_model("stories15m", local_cache.CHECKPOINT_DIR)
    except sweep.SweepError:
        pytest.skip("stories15M isn't installed in ../LUT-Transformer-local/checkpoints")
    pytest.importorskip("transformers")
    return source


def test_every_arm_takes_a_llama_model_and_ours_refuse_one_without_a_profile(tmp_path, monkeypatch):
    root = tmp_path / "ckpt"
    _fake_checkpoints(root, "org/llama-ish", model_type="llama")
    _fake_checkpoints(root, "org/opt-ish", model_type="opt")
    monkeypatch.setattr(local_cache, "CHECKPOINT_DIR", root)
    rows, _ = sweep.read_csv(_full_csv(tmp_path, "yes,ours,baseline,org/llama-ish,,,cpu,speed,,1,8,2,1,\n"
                                                 "yes,lut,lut,org/llama-ish,,params,cpu,speed,,1,8,2,1,\n"
                                                 "yes,theirs,hf,org/llama-ish,,,cpu,speed,,1,8,2,1,\n"
                                                 "yes,opt_ours,baseline,org/opt-ish,,,cpu,speed,,1,8,2,1,\n"
                                                 "yes,opt_lut,lut,org/opt-ish,,params,cpu,speed,,1,8,2,1,\n"
                                                 "yes,opt_theirs,hf,org/opt-ish,,,cpu,speed,,1,8,2,1,\n"))
    _, errors = sweep.validate(rows, sweep.Settings(csv_path=tmp_path / "full.csv"))
    assert len(errors) == 2 and all("opt_" in e and "is a opt model" in e and "Use arm=hf for now" in e for e in errors)


def test_stories15m_hf_runner_matches_generate_and_rewinds():
    source = _stories15m_or_skip()
    assert source.architecture == "llama" and source.vocab_size == 32000 and source.eval_context == 256
    loaded = sweep.load_model(source, "cpu", sweep.Settings())
    assert loaded.cfg.name == "stories15m" and loaded.vocab_size == 32000
    assert sum(p.numel() for p in loaded.hf_model.parameters()) == 15_191_712
    assert sweep.base.count_params(loaded.cfg)["total"] == 15_191_712
    ids = sweep.make_prompt(loaded.vocab_size, 1, 16, 0, "cpu")
    runner = sweep.make_runner(sweep.Row(1, True, "s", "hf", "stories15m", None, "cpu", 1, [16], 4, 1), loaded)
    runner.start(1, 20)
    tokens = [runner.prefill(ids)]
    for _ in range(4):
        tokens.append(runner.decode(tokens[-1]))
    with torch.inference_mode():
        expected = loaded.hf_model.generate(ids, max_new_tokens=5, do_sample=False)[0, 16:]
    assert [int(t) for t in tokens] == expected.tolist()
    runner.rewind(16)
    assert int(runner.decode(tokens[0])) == int(tokens[1])


def test_stories15m_speed_row_through_the_sweep(tmp_path):
    _stories15m_or_skip()
    path = _full_csv(tmp_path, "yes,s15,hf,stories15m,,,cpu,speed,,1,8;16,2,1,\n")
    assert sweep.main(["--csv", str(path), "--results-dir", str(tmp_path / "r"), "--history", str(tmp_path / "h.jsonl"),
                       "--no-power", "--no-progress", "--warmup", "1"]) == 0
    (sweep_dir,) = (tmp_path / "r" / "sweeps").iterdir()
    with open(sweep_dir / "summary.csv", newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert [(l["model"], l["weights"], l["context"]) for l in lines] == \
        [("Xenova/llama2.c-stories15M", "pretrained", "8"), ("Xenova/llama2.c-stories15M", "pretrained", "16")]
    assert all(float(l["decode_ms_per_token_median"]) > 0 for l in lines)


def test_stories15m_our_runners_decode_the_same_tokens_as_the_hf_runner():
    # Greedy prefill + decode through the sweep's own runners: hf, baseline and lut (all_base) agree
    # on every token, and the two from-scratch arms agree bit for bit on the logits path (same tokens
    # here; tests/test_llama_profile.py checks the logits).
    source = _stories15m_or_skip()
    loaded = sweep.load_model(source, "cpu", sweep.Settings())
    ids = sweep.make_prompt(loaded.vocab_size, 1, 16, 0, "cpu")
    tokens = {}
    for arm, preset in (("hf", None), ("baseline", None), ("lut", "all_base")):
        runner = sweep.make_runner(sweep.Row(1, True, arm, arm, "stories15m", preset, "cpu", 1, [16], 6, 1), loaded)
        runner.start(1, 24)
        out = [runner.prefill(ids)]
        for _ in range(6):
            out.append(runner.decode(out[-1]))
        runner.rewind(16)
        assert int(runner.decode(out[0])) == int(out[1])        # the cache rewinds like the hf arm's
        tokens[arm] = [int(t) for t in out]
    assert tokens["baseline"] == tokens["hf"] == tokens["lut"]


def test_stories15m_baseline_and_lut_speed_rows_through_the_sweep(tmp_path):
    _stories15m_or_skip()
    path = _full_csv(tmp_path, "yes,s15_base,baseline,stories15m,,,cpu,speed,,1,8;16,2,1,\n"
                               "yes,s15_lut,lut,stories15m,,all_base,cpu,speed,,1,8,2,1,\n")
    assert sweep.main(["--csv", str(path), "--results-dir", str(tmp_path / "r"), "--history", str(tmp_path / "h.jsonl"),
                       "--no-power", "--no-progress", "--warmup", "1"]) == 0
    (sweep_dir,) = (tmp_path / "r" / "sweeps").iterdir()
    with open(sweep_dir / "summary.csv", newline="") as handle:
        lines = list(csv.DictReader(handle))
    assert [(l["arm"], l["model"], l["context"]) for l in lines] == \
        [("baseline", "Xenova/llama2.c-stories15M", "8"), ("baseline", "Xenova/llama2.c-stories15M", "16"),
         ("lut", "Xenova/llama2.c-stories15M", "8")]
    assert all(float(l["decode_ms_per_token_median"]) > 0 for l in lines)


def test_top1_and_top5_accuracy_over_the_scored_tokens():
    """Accuracy uses the same windows and scored tokens as the NLL: top-1 = the true next token is
    the argmax, top-5 = it is among the 5 largest logits."""
    vocab = 7
    tokens = torch.arange(300) % vocab

    def ranked(rank):                 # the true next token gets the rank-th largest logit (0 = best)
        def forward(x):
            out = torch.zeros(1, x.shape[1], vocab)
            for i in range(vocab):    # logits 6, 5, ..., 0 for tokens shifted by rank
                out[0, torch.arange(x.shape[1]), (x[0] + 1 + i - rank) % vocab] = float(vocab - 1 - i)
            return out
        return forward

    best = sweep.measure_perplexity(ranked(0), tokens, window=128, stride=64, device="cpu")
    assert best["top1_accuracy"] == 1.0 and best["top5_accuracy"] == 1.0 and best["top1_correct"] == 299
    fourth = sweep.measure_perplexity(ranked(3), tokens, window=128, stride=64, device="cpu")
    assert fourth["top1_accuracy"] == 0.0 and fourth["top5_accuracy"] == 1.0
    sixth = sweep.measure_perplexity(ranked(5), tokens, window=128, stride=64, device="cpu")
    assert sixth["top1_accuracy"] == 0.0 and sixth["top5_accuracy"] == 0.0 and sixth["scored_tokens"] == 299
