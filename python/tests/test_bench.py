"""Timing, run logging and telemetry plumbing. Runs without PyTorch."""

import csv
import json

from bench.runlog import Run
from bench.telemetry import PowerMonitor
from bench.timing import Stopwatch, summarize, time_fn


def test_summarize_statistics():
    stats = summarize([4.0, 1.0, 3.0, 2.0, 5.0], warmup=2, device="cpu")
    assert stats.median_ms == 3.0
    assert stats.min_ms == 1.0 and stats.max_ms == 5.0
    assert abs(stats.p10_ms - 1.4) < 1e-9 and abs(stats.p90_ms - 4.6) < 1e-9
    assert "samples_ms" not in stats.to_dict()


def test_time_fn_counts_calls():
    calls = []
    stats = time_fn(lambda: calls.append(1), device="cpu", warmup=2, repeats=5)
    assert len(calls) == 7
    assert stats.repeats == 5 and stats.median_ms >= 0


def test_stopwatch_measures_elapsed_time():
    with Stopwatch("cpu") as watch:
        sum(range(10000))
    assert watch.elapsed_s >= 0


def test_run_writes_files_and_summary(tmp_path):
    first = Run("unit test", configs={"a": 1}, devices={"train": "cpu"}, results_dir=tmp_path)
    first.log(step=0, loss=1.5)
    first.save_power({"sources": {"cpu": {"energy_joules": 2.0, "samples": [{"watts": 1.0}]}}})
    run_dir = first.finish({"latency_ms": 1.25, "nested": {"x": 2}})

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["devices"] == {"train": "cpu"} and "system" in meta and "finished_utc" in meta
    assert json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[0])["loss"] == 1.5
    assert "samples" not in json.loads((run_dir / "power.json").read_text())["sources"]["cpu"]
    assert (run_dir / "power_samples.json").exists()

    second = Run("second", results_dir=tmp_path)
    second.finish({"throughput": 10})
    with open(tmp_path / "summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["name"] for row in rows] == ["unit test", "second"]
    assert rows[0]["nested.x"] == "2" and rows[1]["throughput"] == "10"


def test_power_monitor_disabled_returns_report():
    monitor = PowerMonitor(target="cpu", enabled=False)
    monitor.start()
    assert monitor.stop()["status"] == "disabled"


def test_idle_power_is_measured_before_the_workload(tmp_path, monkeypatch):
    """Idle measured after the timed loop caught the GPU still in its high-power state, so the
    benchmark samples idle before it builds the workload (or creates a CUDA context)."""
    from bench import run_benchmark

    order = []
    monkeypatch.setattr(run_benchmark, "Run", lambda *a, **k: Run(*a, results_dir=tmp_path, **k))
    monkeypatch.setattr(run_benchmark, "_idle_power", lambda target, seconds: order.append("idle") or {})
    monkeypatch.setattr(run_benchmark, "_matmul_workload",
                        lambda *args: order.append("workload") or ((lambda: None), {"flops_per_call": 1}))
    monkeypatch.setattr(run_benchmark, "_power_window", lambda *args: order.append("power") or ({}, 1, {
        "warmup_s": 0.0, "warmup_calls": 0, "settled": True}))
    run_benchmark.main(["--device", "cpu", "--power", "--repeats", "1", "--warmup", "0"])
    assert order == ["idle", "workload", "power"]


def test_power_window_warms_up_before_sampling(monkeypatch):
    """The power window used to start cold, so GPU power was still ramping (57 W to 330 W in the
    first 0.8 s, run 20261119T060430Z_matmul_cuda_4096_float32): the workload must run for the
    warm-up before the power monitor starts, and the warm-up length and call count are returned."""
    from bench import resources, run_benchmark

    events = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append("monitor_start")

        def stop(self):
            events.append("monitor_stop")
            return {"sources": {}}

    monkeypatch.setattr(run_benchmark, "PowerMonitor", FakeMonitor)
    monkeypatch.setattr(resources.shutil, "which", lambda name: None)    # no GPU readings: fixed warm-up
    report, calls, warm = run_benchmark._power_window(lambda: events.append("call"), "cpu", "cpu", 0.05,
                                                      warmup_s=0.05)
    first_start = events.index("monitor_start")
    assert events[:first_start] and set(events[:first_start]) == {"call"}     # warm-up calls came first
    assert warm["warmup_calls"] == first_start and warm["warmup_s"] >= 0.05
    assert calls == events.count("call") - warm["warmup_calls"] and calls > 0


def test_warm_up_waits_for_steady_power(monkeypatch):
    """With power readings, the warm-up continues past min_s until three readings agree."""
    from bench import resources

    readings = iter([57.0, 120.0, 250.0, 330.0, 358.0, 360.0, 361.0, 360.0, 360.0, 360.0])

    class Done:
        def __init__(self, value):
            self.stdout = f"{value}\n"

    monkeypatch.setattr(resources.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: Done(next(readings)))
    clock = iter(range(1000))
    monkeypatch.setattr(resources.time, "perf_counter", lambda: next(clock) * 0.3)
    warm = resources.warm_up_until_steady(lambda: None, lambda: None, min_s=0.0, device="cuda", tolerance=0.03,
                                          max_s=100.0)
    assert warm["settled"] and warm["readings_w"][-3:] == [358.0, 360.0, 361.0]
