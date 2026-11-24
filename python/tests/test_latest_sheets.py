"""The latest-only view (bench/latest.py) and the two-target Google Sheets sync (bench/gsheets.py),
tested on a small synthetic results folder and a local fake of the Apps Script web app (localhost only;
nothing is sent to Google)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench import excel, gsheets, latest, opcount


# ----------------------------------------------------------------------------------------------
# A synthetic results folder: two base-metrics runs of the baseline arm, an HLS build overlapping
# the newer one, and a per-stage timing run
# ----------------------------------------------------------------------------------------------

def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)


def _sweep_run(results, run_id, start, end):
    _write(results / "runs" / run_id / "meta.json", {
        "name": "sweep_s15m_base_gpu", "started_utc": start, "finished_utc": end,
        "devices": {"prefill": "cuda", "decode": "cuda"},
        "configs": {"sweep_id": "S", "row": {"arm": "baseline", "model": "stories15m", "lut_preset": None,
                                              "benchmarks": ["speed", "perplexity"], "context_lengths": [224],
                                              "warmup": 2, "repeats": 10, "decode_tokens": 32},
                    "perplexity": {"tinystories": {"char_limit": 1200000}}}})
    _write(results / "runs" / run_id / "summary.json", {"row_s": 10.0, "model": "stories15m"})


def _long(run_id, benchmark, metric, value, context="", dataset="tinystories"):
    return {"timestamp_utc": "", "sweep_id": "S", "run_id": run_id, "git_commit": "", "row": "r", "model": "stories15m",
            "arm": "baseline", "weights": "pretrained", "lut_preset": "", "stage_toggles": "", "device": "cuda",
            "dtype": "torch.float32", "benchmark": benchmark, "dataset": dataset if benchmark != "speed" else "synthetic",
            "dataset_sha256": "", "context_tokens": context, "batch": "1", "decode_tokens": "32", "repeats": "10",
            "eval_window_tokens": "", "eval_stride_tokens": "", "metric": metric, "value": str(value), "unit": "",
            "note": ""}


@pytest.fixture
def results(tmp_path, monkeypatch):
    import csv

    monkeypatch.setattr(excel, "HLS_BUILD_ROOT", tmp_path / "no-builds")     # no live HLS builds in the test
    res = tmp_path / "results"
    old, new = "20261124T010000Z_sweep_s15m_base_gpu", "20261124T020000Z_sweep_s15m_base_gpu"
    _sweep_run(res, old, "2026-11-24T01:00:00+00:00", "2026-11-24T01:00:10+00:00")
    _sweep_run(res, new, "2026-11-24T02:00:00+00:00", "2026-11-24T02:00:10+00:00")
    rows = []
    for rid, ppl in ((old, 3.9), (new, 3.8)):
        rows += [_long(rid, "perplexity", "perplexity", ppl), _long(rid, "perplexity", "top1_accuracy", 0.66),
                 _long(rid, "perplexity", "eval_tokens", 321526), _long(rid, "speed", "decode_ms_per_token_median", 1.5, "224")]
    with open(res / "all_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # an HLS synthesis that overlaps the newer run only
    _write(res / "hls" / "runs" / "hls_overlap" / "summary.json", {
        "kernel": "k", "variant": "baseline", "run_id": "hls_overlap", "started_utc": "20261124T015900Z",
        "finished_utc": "20261124T020200Z",
        "stages": {"synth": {"status": "pass", "seconds": 120.0, "started_utc": "2026-11-24T02:00:00Z",
                             "finished_utc": "2026-11-24T02:02:00Z"}},
        "results": {"synth": {"latency_cycles_max": 10, "target_clock_ns": 3.33, "resources": {"LUT": 100}}}})
    # a per-stage timing run (decode at T = 256, the opcount definition)
    cfg = {"name": "stories15m", "vocab_size": 32000, "context_length": 256, "d_model": 288, "n_heads": 6,
           "n_layers": 6, "d_ff": 768, "norm": "rmsnorm", "norm_eps": 1e-5, "norm_placement": "pre",
           "activation": "silu", "ffn": "glu", "positional_encoding": "rope", "rope_theta": 10000.0,
           "tie_embeddings": True, "bias": False, "dropout": 0.0}
    lt = "20261124T030000Z_layer_timing_stories15m_baseline"
    _write(res / "runs" / lt / "meta.json", {
        "name": "layer_timing_stories15m_baseline", "started_utc": "2026-11-24T03:00:00+00:00",
        "finished_utc": "2026-11-24T03:00:01+00:00", "devices": {"prefill": "cuda", "decode": "cuda"},
        "configs": {"arm": "baseline", "variant": "baseline", "decode_context": "T keys", "model_config": cfg}})
    _write(res / "runs" / lt / "summary.json", {})
    stages = {f"{layer}:{stage}": [0.1, 0.1, 0.1] for layer in range(6)
              for stage in opcount.FORWARD_ORDER if stage not in ("embedding", "final_norm", "lm_head")}
    stages.update({f"model:{s}": [0.5, 0.5, 0.5] for s in ("embedding", "final_norm", "lm_head", "unattributed")})
    _write(res / "runs" / lt / "metrics.jsonl", json.dumps({"kind": "layer_timing", "phase": "decode", "context": 256,
                                                            "plain_ms": [1.0] * 3, "timed_ms": [2.0] * 3,
                                                            "stages": stages}) + "\n")
    return res, old, new


def test_pick_shows_the_newest_and_a_clean_one_when_the_newest_is_contended():
    rows = [{"run_id": "a", "ts": "1", "contended": ""}, {"run_id": "b", "ts": "2", "contended": "hls x synth"}]
    picked = latest.pick(rows)
    assert [r["run_id"] for r, _ in picked] == ["b", "a"]
    assert picked[0][1].startswith("newest, contended") and picked[1][1] == "newest clean"
    assert [(r["run_id"], s) for r, s in latest.pick(rows[:1])] == [("a", "newest")]


def test_latest_summary_selects_the_newest_run_per_arm_and_marks_contention(results):
    res, old, new = results
    sheets = latest.build(res)
    assert [s.name for s in sheets] == [latest.PAPER, latest.SUMMARY, latest.PER_STAGE, latest.COMPARISON, latest.TIME]
    summary = sheets[1]
    head = [c.header for c in summary.columns]
    full = [line for line in summary.lines if line[head.index("section")].startswith("PyTorch base metrics")]
    assert [line[head.index("run_id")] for line in full] == [new, old]          # newest first, then newest clean
    assert full[0][head.index("status")].startswith("newest, contended") and "hls_overlap" in full[0][head.index("status")]
    assert full[1][head.index("status")] == "newest clean"
    assert full[0][head.index("perplexity")] == 3.8
    assert summary.preamble and "newest" in summary.preamble[0]


def test_per_stage_latest_is_in_forward_order_with_the_caveat(results):
    res, _, _ = results
    sheet = latest.build(res)[2]
    head = [c.header for c in sheet.columns]
    block = [line[head.index("stage")] for line in sheet.lines
             if line[head.index("phase")] == "decode" and line[head.index("context T")] == 256]
    stages = [s for s in block if "(not a stage)" not in s]
    assert stages == list(opcount.FORWARD_ORDER)                                # embedding ... lm_head, never by size
    assert block[-1] == "unattributed (not a stage)"
    assert sheet.preamble == [excel.PER_STAGE_CAVEAT] and sheet.charts


# ----------------------------------------------------------------------------------------------
# Two targets against a fake web app
# ----------------------------------------------------------------------------------------------

class _Fake(BaseHTTPRequestHandler):
    posts: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.posts.append((self.path, body))
        self.send_response(302)
        self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/echo?t={body.get('token')}")
        self.end_headers()

    def do_GET(self):
        ok = "t=tok-" in self.path
        data = json.dumps({"ok": ok} if ok else {"ok": False, "error": "bad token"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Fake.posts = []
    monkeypatch.setattr(gsheets, "CONFIG_PATH", tmp_path / "cfg" / "sheets.json")
    monkeypatch.setattr(gsheets, "RETRIES", (0.01,))
    (tmp_path / "cfg").mkdir()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _config(url_full, url_latest):
    gsheets.CONFIG_PATH.write_text(json.dumps({
        "spreadsheet_id": "full", "token": "tok-full", "web_app_url": url_full,
        "targets": {"latest": {"spreadsheet_id": "latest", "token": "tok-latest", "url": url_latest}}}))


def test_a_target_without_a_url_is_skipped_quietly(fake, results):
    res, _, _ = results
    _config(f"{fake}/full/exec", None)
    assert gsheets.configured_targets("all") == ["full"]
    report = gsheets.sync(latest.build(res), target="latest")
    assert "skipped" in report and not _Fake.posts


def test_targets_keep_separate_state_and_only_latest_prunes(fake, results):
    res, _, _ = results
    _add_training_and_fpga(res)                       # the Paper table needs something to show
    _config(f"{fake}/full/exec", f"{fake}/latest/exec")
    full_report = gsheets.sync(excel.build(res), target="full")
    latest_report = gsheets.sync(latest.build(res), target="latest")
    assert full_report["changed"] and latest_report["changed"]
    assert gsheets._state_path("full") != gsheets._state_path("latest")
    assert set(gsheets.load_state("latest")["tab_hashes"]) == {latest.PAPER, latest.SUMMARY, latest.PER_STAGE,
                                                               latest.COMPARISON, latest.TIME}
    assert "All data" in gsheets.load_state("full")["tab_hashes"]
    finishes = {path: body for path, body in _Fake.posts if body.get("finish")}
    assert finishes["/latest/exec"]["prune"] is True and finishes["/latest/exec"]["meta"]["stamp_tab"] == latest.PAPER
    paper_post = next(b for p, b in _Fake.posts if p.startswith("/latest") and b.get("tab") == latest.PAPER)
    assert paper_post["frozen_rows"] == 0 and paper_post["bold_rows"]          # free-form tab: bold titles, no frozen header
    assert "prune" not in finishes["/full/exec"] and "stamp_tab" not in finishes["/full/exec"]["meta"]
    assert {b["token"] for p, b in _Fake.posts if p.startswith("/latest")} == {"tok-latest"}
    # nothing changed: neither target uploads again
    count = len(_Fake.posts)
    assert not gsheets.sync(latest.build(res), target="latest")["changed"]
    assert not gsheets.sync(excel.build(res), target="full")["changed"]
    assert len(_Fake.posts) == count


# ----------------------------------------------------------------------------------------------
# Paper table
# ----------------------------------------------------------------------------------------------

def _add_training_and_fpga(res, board=False):
    _write(res / "runs" / "20261124T040000Z_train_throughput_stories15m" / "meta.json", {
        "name": "train_throughput_stories15m", "started_utc": "2026-11-24T04:00:00+00:00",
        "finished_utc": "2026-11-24T04:00:30+00:00", "devices": {"train": "cuda"}, "configs": {"timed_steps": 10},
        "system": {"gpus": [{"name": "NVIDIA GeForce RTX 5080"}]}})
    _write(res / "runs" / "20261124T040000Z_train_throughput_stories15m" / "summary.json", {
        "ours": {"batch": 64, "T": 256, "ms_step": 143.0, "tok_s": 114000.0, "power": {"power_w_mean_steady": 294.0}},
        "hf_fp32": {"batch": 64, "T": 256, "ms_step": 97.0, "tok_s": 169000.0}})
    _write(res / "hls" / "runs" / "build1" / "summary.json", {
        "kernel": "transformer", "variant": "baseline", "run_id": "build1", "part": "xcu50-fsvh2104-2-e",
        "started_utc": "20261124T030000Z", "finished_utc": "20261124T031000Z",
        "results": {"impl": {"post_route_fmax_mhz": 333.7, "resources": {"LUT": 29023, "FF": 36697, "DSP": 50, "BRAM": 73},
                             "utilization_pct": {"LUT": 3.33, "FF": 2.1, "DSP": 0.84, "BRAM": 2.72}},
                    "power": {"total_w": 3.651, "dynamic_w": 1.414}}})
    _write(res / "hls" / "inference" / "inf1" / "summary.json", {
        "run_id": "inf1", "backend": "c_model", "variant": "baseline", "started_utc": "20261124T050000Z",
        "finished_utc": "20261124T050100Z",
        "host": {"windows": 392, "scored_tokens": 50211, "top1": 0.6555734, "mean_ms_per_step": 11.7},
        "timing": {"fpga_estimate": {"source_run_id": "build1", "ms_per_step": 63.512}}})
    if board:
        _write(res / "hls" / "inference" / "inf2" / "summary.json", {
            "run_id": "inf2", "backend": "hw", "variant": "baseline", "started_utc": "20261124T060000Z",
            "finished_utc": "20261124T060100Z",
            "host": {"windows": 392, "scored_tokens": 50211, "top1": 0.6555734, "mean_ms_per_step": 41.25},
            "timing": {"fpga_estimate": {"source_run_id": "build1", "ms_per_step": 63.512}}})


def _tables(sheet):
    """{sub-table title: (header, rows, footnote)} parsed back from the free-form layout."""
    out, lines, i = {}, sheet.lines, 0
    while i < len(lines):
        line = lines[i]
        if len(line) == 1 and str(line[0]).split(":")[0] in ("Training", "Inference") and ":" in str(line[0]):
            header, rows, j = lines[i + 1], [], i + 2
            while j < len(lines) and len(lines[j]) > 1:
                rows.append(lines[j])
                j += 1
            out[line[0]] = (header, rows, lines[j][0] if j < len(lines) and len(lines[j]) == 1 else "")
            i = j
        else:
            i += 1
    return out


def _eval_run(res, kind, scored, start, end, *, complete=True, top1=0.6556, time_s=1.0, power=140.0):
    """A bench/eval_power.py (window-forward) or bench/eval_decode.py (decode-step) run on the GPU."""
    chars = 187000 if scored == 50211 else 1200000
    rid = f"{start[:4]}{start[5:7]}{start[8:10]}T{start[11:13]}{start[14:16]}{start[17:19]}Z_{kind}_stories15m_chars{chars}"
    _write(res / "runs" / rid / "meta.json", {
        "name": f"{kind}_stories15m_chars{chars}", "started_utc": start, "finished_utc": end,
        "devices": {"inference": "cuda"}, "configs": {"model": "stories15m", "lut_preset": "all_base"},
        "system": {"gpus": [{"name": "NVIDIA GeForce RTX 5080"}]}})
    windows, steps = (392, 99868) if scored == 50211 else (2511, 640296)
    summary = {"scored_tokens": scored, "windows": windows}
    if kind == "eval_decode":
        summary.update({"protocol": "decode-step", "decode_steps": steps, "complete": complete})
    for arm in ("hf", "baseline", "lut"):
        info = {"scored_tokens": scored, "top1_accuracy": top1, "power_w_mean_steady": power, "warmup": {"warmup_s": 3.0},
                "resources": {"memory_used_max": 2900.0, "torch_peak_allocated_mib": 282.2}}
        if kind == "eval_decode":
            info.update({"decode_s": time_s, "ms_per_step": 1000.0 * time_s / steps,
                         "reference_check": {"baseline": {"nll_abs_diff_max": 2e-5, "top1_flag_disagreements": 0},
                                             "hf": {"nll_abs_diff_max": 2e-5, "top1_flag_disagreements": 0}}})
        else:
            info.update({"pass_s_median": time_s, "passes": 9})
        summary[arm] = info
    _write(res / "runs" / rid / "summary.json", summary)
    return rid


def _add_full_fpga(res):
    _write(res / "hls" / "inference" / "inf_full" / "summary.json", {
        "run_id": "inf_full", "backend": "c_model", "variant": "baseline", "started_utc": "20261124T070000Z",
        "finished_utc": "20261124T071000Z", "selection": {"first_window": 0, "max_windows": -1, "max_steps": -1},
        "host": {"windows": 2511, "scored_tokens": 321526, "decode_steps": 640296, "top1": 0.66167, "mean_ms_per_step": 11.7},
        "timing": {"fpga_estimate": {"source_run_id": "build1", "ms_per_step": 63.6}}})


def _inference_title(sheet):
    return next(line[0] for line in sheet.lines if len(line) == 1 and str(line[0]).startswith("Inference"))


def test_paper_table_sections_columns_and_dashes(results):
    res, _, _ = results
    _add_training_and_fpga(res)
    clean = _eval_run(res, "eval_power", 50211, "2026-11-24T01:30:00+00:00", "2026-11-24T01:30:30+00:00")
    contended = _eval_run(res, "eval_power", 50211, "2026-11-24T02:00:30+00:00", "2026-11-24T02:01:00+00:00")
    _eval_run(res, "eval_power", 321526, "2026-11-24T03:30:00+00:00", "2026-11-24T03:31:00+00:00", time_s=6.0)
    sheet = latest.build(res)[0]
    assert sheet.name == latest.PAPER and sheet.freeform
    titles = [line[0] for line in sheet.lines if len(line) == 1 and (line[0] == "Training" or
                                                                    str(line[0]).startswith("Inference ("))]
    assert titles == ["Training", "Inference (50,211 tokens, the same tokens on GPU and FPGA)"]
    first_inference = sheet.lines.index([titles[1]])
    assert sheet.lines[first_inference - 1] == []                                 # a blank row between sections
    tables = _tables(sheet)
    assert set(tables) == {"Training: GPU", "Inference: GPU", "Inference: FPGA"}  # no empty or invented tables
    assert tables["Training: GPU"][0] == latest.GPU_COLUMNS and tables["Inference: GPU"][0] == latest.GPU_COLUMNS
    assert tables["Inference: FPGA"][0] == latest.FPGA_COLUMNS
    train = tables["Training: GPU"][1]
    assert train[0][4] == latest.DASH and train[1][5] == latest.DASH               # no accuracy, no power for HF: dashes
    fpga = tables["Inference: FPGA"][1][0]
    assert fpga[3] == "29,023 (3.3%)" and fpga[7] == latest.DASH and fpga[8] == "65.56" and fpga[10] == "334"
    assert "C simulation" in tables["Inference: FPGA"][2] and "No board run yet" in tables["Inference: FPGA"][2]
    gpu = tables["Inference: GPU"][1]
    assert [r[0] for r in gpu] == ["stories15M (HF FP32)", "stories15M baseline", "stories15M LUT"]
    assert all("50,211 tokens" in r[1] for r in gpu + tables["Inference: FPGA"][1])   # the same tokens on both
    assert gpu[1][2] == "GPU (RTX 5080)" and gpu[1][3] == "1.00" and gpu[1][5] == "140.0"
    assert "Window-forward protocol" in tables["Inference: GPU"][2] and "not like for like" in tables["Inference: GPU"][2]
    assert ["321,526-token comparison running: waiting on the FPGA run."] in sheet.lines
    sources = sheet.lines[-2][0]
    assert sources.startswith("Source runs: ") and clean in sources and contended not in sources  # contended newest skipped
    assert all(i < len(sheet.lines) for i in sheet.bold_rows)


def test_paper_table_uses_the_largest_shared_token_set_and_the_decode_protocol(results):
    res, _, _ = results
    _add_training_and_fpga(res)
    _add_full_fpga(res)
    _eval_run(res, "eval_power", 50211, "2026-11-24T01:30:00+00:00", "2026-11-24T01:30:30+00:00")
    _eval_run(res, "eval_power", 321526, "2026-11-24T03:30:00+00:00", "2026-11-24T03:31:00+00:00", time_s=6.0)
    decode = _eval_run(res, "eval_decode", 321526, "2026-11-24T04:30:00+00:00", "2026-11-24T05:20:00+00:00",
                       time_s=960.4, power=120.0)
    partial = _eval_run(res, "eval_decode", 321526, "2026-11-24T06:00:00+00:00", "2026-11-24T06:01:00+00:00",
                        complete=False, time_s=5.0)
    sheet = latest.build(res)[0]
    assert _inference_title(sheet) == "Inference (321,526 tokens, the same tokens on GPU and FPGA)"
    tables = _tables(sheet)
    gpu, note = tables["Inference: GPU"][1], tables["Inference: GPU"][2]
    assert all("321,526 tokens" in r[1] for r in gpu + tables["Inference: FPGA"][1])
    assert gpu[1][3] == "960.4" and gpu[1][5] == "120.0"                          # decode-step beats window-forward
    assert "GPU and FPGA both decode the same 321,526 tokens one step at a time" in note and "640,296" in note
    assert not any("comparison running" in str(line[0]) for line in sheet.lines if len(line) == 1)
    assert ["One decode step = one token through the model with a KV cache; with 256-token windows at stride 128, "
            "the 321,526 scored tokens take 640,296 steps."] in sheet.lines
    sources = sheet.lines[-2][0]
    assert decode in sources and partial not in sources                           # an incomplete decode run is ignored


def test_paper_table_prefers_a_board_run_when_one_appears(results):
    res, _, _ = results
    _add_training_and_fpga(res, board=True)
    _eval_run(res, "eval_power", 50211, "2026-11-24T01:30:00+00:00", "2026-11-24T01:30:30+00:00")
    tables = _tables(latest.build(res)[0])
    fpga_row, note = tables["Inference: FPGA"][1][0], tables["Inference: FPGA"][2]
    assert fpga_row[9] == "41.25" and "Measured on the board" in note and "No board run yet" not in note


def _memory_run(res, start="2026-11-24T08:00:00+00:00"):
    """A bench/memory_pass.py run: one arm with a LUT table on the GPU, the others without LUT ops."""
    rid = "20261124T080000Z_memory_stories15m_chars1200000"
    _write(res / "runs" / rid / "meta.json", {
        "name": "memory_stories15m_chars1200000", "started_utc": start, "finished_utc": "2026-11-24T08:02:00+00:00",
        "devices": {"inference": "cuda", "train": "cuda"}, "configs": {"lut_preset": "all_base", "window": 256}})
    MiB = 2 ** 20

    def phase(peak, kv=3 * MiB):
        return {"torch_peak_allocated_bytes": peak, "torch_peak_reserved_bytes": peak + 20 * MiB,
                "process_gpu_peak_bytes": peak + 520 * MiB, "torch_allocated_steady_bytes": peak - 10 * MiB,
                "host_rss_peak_bytes": 1500 * MiB, "weights_bytes": 58 * MiB, "kv_cache_bytes": kv,
                "lut_table_bytes": 0, "activations_workspace_bytes": peak - 61 * MiB,
                "context_and_allocator_bytes": 500 * MiB, "allocator_cached_bytes": 20 * MiB, "harness_bytes": 0,
                "windows": 8, "steps": 2040}
    table = {"stage": "softmax", "layer": 0, "op": "LUT.inference_lut.softmax", "arch": "exp_table", "inputs_per_table": 1,
             "bits_per_input": 8, "out_bits": 16, "entries_per_table": 256, "planned_table_bits": 4096,
             "name": "tables['exp']", "shape": [256], "dtype": "int16", "bytes": 512, "device": "cuda:0",
             "place": "GPU VRAM", "pinned": False, "mmap_file": None}
    none = {"rows": [], "totals": {"lut_tables": 0, "lut_bytes_gpu_vram": 0}, "lut_stages": 0}
    arms = {"hf": {"phases": {"decode": phase(100 * MiB), "window_forward": phase(140 * MiB, 0)},
                   "lut_inventory": {**none, "label": "none (library model, no LUT ops)"}, "weights_bytes_by_place": {}},
            "baseline": {"phases": {"decode": phase(110 * MiB), "window_forward": phase(150 * MiB, 0)},
                         "lut_inventory": {**none, "label": "none (all base ops)"}, "weights_bytes_by_place": {},
                         "htod_decode_window": {"per_step_copies": 0.0}},
            "lut": {"phases": {"decode": phase(111 * MiB), "window_forward": phase(151 * MiB, 0)},
                    "lut_inventory": {"rows": [table], "totals": {"lut_tables": 1, "lut_bytes_gpu_vram": 512},
                                      "lut_stages": 1, "label": "GPU VRAM, 0.000512 MB"}, "weights_bytes_by_place": {}}}
    training = {"ours": {**phase(11000 * MiB), "batch": 64, "T": 256, "lut_label": "none (all base ops)"}}
    _write(res / "runs" / rid / "summary.json", {"arms": arms, "training": training})
    return rid


def test_paper_table_reports_peak_memory_and_where_the_luts_live(results):
    res, _, _ = results
    _add_training_and_fpga(res)
    _eval_run(res, "eval_power", 50211, "2026-11-24T01:30:00+00:00", "2026-11-24T01:30:30+00:00")
    mem = _memory_run(res)
    sheet = latest.build(res)[0]
    tables = _tables(sheet)
    gpu, note = tables["Inference: GPU"][1], tables["Inference: GPU"][2]
    i_mem = latest.GPU_COLUMNS.index("Peak GPU memory, PyTorch allocations (MB)")
    i_lut = latest.GPU_COLUMNS.index("LUT storage")
    assert [r[i_mem] for r in gpu] == [f"{140 * 2 ** 20 / 1e6:,.1f}", f"{150 * 2 ** 20 / 1e6:,.1f}", f"{151 * 2 ** 20 / 1e6:,.1f}"]
    assert [r[i_lut] for r in gpu] == ["none (library model, no LUT ops)", "none (all base ops)", "GPU VRAM, 0.000512 MB"]
    assert "Peak GPU memory, PyTorch allocations: the peak of PyTorch's allocated GPU memory with only that model loaded" in note
    assert "Per-process total including the CUDA context: about 630 MiB (nvidia-smi)." in note
    assert "2,900" not in note and "2900" not in " ".join(str(c) for r in gpu for c in r)   # never the device-wide figure
    assert latest.FPGA_COLUMNS[9] == "Latency (ms/step)"
    train = tables["Training: GPU"][1]
    assert train[0][i_mem] == f"{11000 * 2 ** 20 / 1e6:,.1f}" and train[0][i_lut] == "none (all base ops)"
    assert mem in sheet.lines[-2][0]


def test_memory_and_lut_storage_tabs(results):
    res, _, _ = results
    mem = _memory_run(res)
    sheets = {s.name: s for s in excel.build(res)}
    m, lut = sheets["PyTorch Memory"], sheets["LUT storage"]
    h = [c.header for c in m.columns]
    rows = [r for r in m.lines if r[h.index("run_id")] == mem]
    assert len(rows) == 3 * 2 + 1                         # 3 arms x (decode, window-forward) + 1 training setup
    dec = next(r for r in rows if r[h.index("arm")] == "lut" and r[h.index("phase")] == "inference decode")
    assert dec[h.index("KV cache (MiB)")] == 3.0 and dec[h.index("CUDA context + overhead (MiB)")] == 500.0
    assert dec[h.index("LUT storage")] == "GPU VRAM, 0.000512 MB"
    lh = [c.header for c in lut.columns]
    by_arm = {r[lh.index("arm")]: r for r in lut.lines}
    assert by_arm["lut"][lh.index("stored in")] == "GPU VRAM" and by_arm["lut"][lh.index("bytes")] == 512
    assert by_arm["baseline"][lh.index("stored in")] == "none (all base ops)"
    assert by_arm["baseline"][lh.index("HtoD copies per decode step")] == 0.0


def test_a_reused_hls_build_costs_no_build_time_and_is_labelled(results):
    res, _, _ = results
    stages = {"synth": {"status": "pass", "seconds": 600.0, "started_utc": "2026-11-24T09:00:00Z",
                        "finished_utc": "2026-11-24T09:10:00Z"}}
    _write(res / "hls" / "runs" / "orig" / "summary.json", {
        "kernel": "transformer", "variant": "baseline", "run_id": "orig", "started_utc": "20261124T090000Z",
        "finished_utc": "20261124T091000Z", "stages": stages,
        "results": {"synth": {"latency_cycles_max": 10, "target_clock_ns": 3.0, "resources": {"LUT": 100}}}})
    _write(res / "hls" / "runs" / "rerun" / "summary.json", {      # copies the original's synth stage, then runs csim
        "kernel": "transformer", "variant": "baseline", "run_id": "rerun", "started_utc": "20261124T100000Z",
        "finished_utc": "20261124T100010Z", "build_reused": True, "source_build_id": "orig",
        "stages": {**stages, "csim": {"status": "pass", "seconds": 5.0, "started_utc": "2026-11-24T10:00:00Z",
                                      "finished_utc": "2026-11-24T10:00:05Z"}},
        "results": {"synth": {"latency_cycles_max": 10, "target_clock_ns": 3.0, "resources": {"LUT": 100}}}})
    clash = _eval_run(res, "eval_power", 50211, "2026-11-24T09:02:00+00:00", "2026-11-24T09:03:00+00:00")
    results_obj = excel.load(res)
    timing = excel.timing_rows(results_obj)
    excel.mark_contention(results_obj, timing)
    rerun = {r["component"]: r for r in timing if r["run_id"] == "rerun"}
    assert rerun["synth"]["seconds"] == 0.0 and rerun["synth"]["basis"] == "reused"
    assert rerun["synth"]["activity"] == "HLS synth (reused build)" and "reused orig" in rerun["synth"]["note"]
    assert rerun["csim"]["seconds"] == 5.0 and rerun["csim"]["basis"] == "measured"
    orig = next(r for r in timing if r["run_id"] == "orig" and r["component"] == "synth")
    assert orig["seconds"] == 600.0                                       # the build time stays with the original
    hits = next(r for r in timing if r["run_id"] == clash and r["component"] == "run total (meta start to finish)")
    assert "orig" in hits["contended"] and "rerun" not in hits["contended"]   # the reused copy did no work
    kernels = next(s for s in excel.build(res) if s.name == "HLS Kernels")
    builds = {line[[c.header for c in kernels.columns].index("run_id")]: line[-1] for line in kernels.lines}
    assert builds["rerun"] == "reused orig" and builds["orig"] == "built"


def test_run_py_build_cache_records_are_read_per_stage(results):
    """The exact shape the pipeline's run.py build cache writes (2026-11-24): build_reused, source_build_id,
    build_cache{...}, and per reused stage reused/reused_from/original_seconds with seconds 0."""
    res, _, _ = results
    _write(res / "hls" / "runs" / "cached" / "summary.json", {
        "kernel": "transformer", "variant": "baseline", "run_id": "cached", "started_utc": "20261124T110000Z",
        "finished_utc": "20261124T110030Z", "build_reused": True, "source_build_id": "20261124T012757Z_impl16t",
        "build_cache": {"key": "abc123", "stages_reused": ["synth", "impl"], "pytorch_fingerprint": "4146e3d4"},
        "stages": {"synth": {"status": "pass", "seconds": 0, "reused": True, "reused_from": "20261124T012757Z_impl16t",
                             "original_seconds": 640.0},
                   "impl": {"status": "pass", "seconds": 0, "reused": True, "reused_from": "20261124T012757Z_impl16t",
                            "original_seconds": 2400.0},
                   "hw_emu": {"status": "pass", "seconds": 0, "reused": True, "reused_from": "20261124T020000Z_hwemu",
                              "original_seconds": 900.0},
                   "csim": {"status": "pass", "seconds": 12.0}},
        "results": {"synth": {"latency_cycles_max": 10, "target_clock_ns": 3.0, "resources": {"LUT": 100}}}})
    results_obj = excel.load(res)
    timing = [r for r in excel.timing_rows(results_obj) if r["run_id"] == "cached"]
    stage = {r["component"]: r for r in timing}
    assert stage["synth"]["seconds"] == 0.0 and stage["impl"]["seconds"] == 0.0 and stage["hw_emu"]["seconds"] == 0.0
    assert stage["hw_emu"]["status"] == "reused 20261124T020000Z_hwemu"          # the stage's own source
    assert stage["csim"]["seconds"] == 12.0 and stage["csim"]["basis"] == "measured"
    assert not any(r["seconds"] in (640.0, 2400.0, 900.0) for r in timing)       # original_seconds never counted
    budget = next(s for s in excel.build(res) if s.name == "Time budget")
    h = [c.header for c in budget.columns]
    reused_lines = [l for l in budget.lines if "cached" in str(l[h.index("run ids")]) and "reused" in str(l[h.index("activity")])]
    assert reused_lines and all(l[h.index("max (s)")] == 0.0 for l in reused_lines)
    kernels = next(s for s in excel.build(res) if s.name == "HLS Kernels")
    builds = {line[[c.header for c in kernels.columns].index("run_id")]: line[-1] for line in kernels.lines}
    assert builds["cached"] == "reused 20261124T012757Z_impl16t"


def test_a_ledger_with_a_list_of_stage_names_is_not_a_reuse(results):
    res, _, _ = results
    _write(res / "hls" / "runs" / "ledgered" / "summary.json", {
        "kernel": "k", "variant": "baseline", "run_id": "ledgered", "stages": {"csim": {"status": "pass", "seconds": 3.0}}})
    _write(res / "hls" / "runs" / "ledgered" / "ledger_record.json", {"run_id": "ledgered", "variant": "baseline",
                                                                      "stages": ["ffn_up", "ffn_down"]})
    assert excel.reused_build({"stages": ["ffn_up"]}) is None
    timing = [r for r in excel.timing_rows(excel.load(res)) if r["run_id"] == "ledgered"]
    assert timing and not any(r.get("reused") for r in timing)


def _decode_runs(res, times, *, start_hour=4, contended_idx=(), top1s=None, peaks=None):
    """Repeated full-set eval_decode runs (one per value in times); runs in contended_idx get an overnight
    run_condition 'busy'."""
    import csv

    rids = []
    for i, t in enumerate(times):
        start = f"2026-11-24T{start_hour + i // 2:02d}:{(i % 2) * 30:02d}:00+00:00"
        end = f"2026-11-24T{start_hour + i // 2:02d}:{(i % 2) * 30 + 20:02d}:00+00:00"
        rid = _eval_run(res, "eval_decode", 321526, start, end, time_s=t, power=60.0 + i,
                        top1=(top1s or [0.66167] * len(times))[i])
        if peaks:
            meta = json.loads((res / "runs" / rid / "meta.json").read_text())
            meta["configs"]["loaded_only_arm"] = True
            _write(res / "runs" / rid / "meta.json", meta)
            summ = json.loads((res / "runs" / rid / "summary.json").read_text())
            for arm in ("hf", "baseline", "lut"):
                summ[arm]["resources"] = {"torch_peak_allocated_mib": peaks[i]}
            _write(res / "runs" / rid / "summary.json", summ)
        rids.append(rid)
    rows = [{"timestamp_utc": "", "run_id": rids[i], "flow": "pytorch", "variant": "", "arm": "", "model": "stories15m",
             "platform": "gpu", "phase": "condition", "context": "", "layer": "", "stage": "run condition",
             "metric": "run_condition", "value": "busy: xsimk(99%, pid 1)" if i in contended_idx else "clean", "unit": "",
             "git_commit": "", "note": ""} for i in range(len(times))]
    with open(res / "extra_results.csv", "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    return rids


def test_paper_table_shows_mean_and_std_over_clean_runs(results):
    res, _, _ = results
    _add_training_and_fpga(res)
    _add_full_fpga(res)
    rids = _decode_runs(res, [960.0, 970.0, 980.0, 5000.0], contended_idx=(3,), peaks=[100.0, 102.0, 104.0, 999.0])
    sheet = latest.build(res)[0]
    tables = _tables(sheet)
    gpu, note = tables["Inference: GPU"][1], tables["Inference: GPU"][2]
    i_mem = latest.GPU_COLUMNS.index("Peak GPU memory, PyTorch allocations (MB)")
    assert gpu[1][3] == "970.0 ± 10.0" and gpu[1][5] == "61.0 ± 1.0"           # the contended 5000 s run is left out
    assert gpu[1][i_mem] == f"{102 * 2 ** 20 / 1e6:,.1f} ± {2 * 2 ** 20 / 1e6:,.1f}"
    assert "mean ± std over n clean runs (HF n=3, baseline n=3, LUT n=3); accuracy identical in every run." in note
    sources = sheet.lines[-2][0]
    assert all(r in sources for r in rids[:3]) and rids[3] not in sources


def test_the_averages_tab_and_an_accuracy_change_is_flagged(results):
    res, _, _ = results
    rids = _decode_runs(res, [960.0, 970.0, 980.0], top1s=[0.66167, 0.66167, 0.66170])
    sheet = next(s for s in excel.build(res) if s.name == "Averages")
    h = [c.header for c in sheet.columns]
    row = next(l for l in sheet.lines if l[h.index("arm / setup")] == "baseline" and l[h.index("metric")] == "time for all decode steps")
    assert row[h.index("n (clean)")] == 3 and row[h.index("mean")] == 970.0 and row[h.index("std")] == 10.0
    assert abs(row[h.index("95% CI high")] - (970.0 + 4.303 * 10.0 / 3 ** 0.5)) < 1e-6
    acc = next(l for l in sheet.lines if l[h.index("arm / setup")] == "baseline" and l[h.index("metric")] == "top-1 accuracy")
    assert acc[h.index("note")] == "ACCURACY DIFFERS BETWEEN RUNS"
    note = _tables(latest.build(res)[0])
    assert excel.describe([1.0, 2.0, 3.0])["std"] == 1.0 and excel.t95(40) == 1.96


def test_runs_of_different_code_versions_are_not_averaged_together(results):
    res, _, _ = results
    _add_training_and_fpga(res)
    _add_full_fpga(res)
    _decode_runs(res, [1000.0, 1010.0], start_hour=4)               # before the rope_tables fix (05:36 UTC)
    post = _decode_runs(res, [900.0, 910.0], start_hour=6)          # after it
    gpu = _tables(latest.build(res)[0])["Inference: GPU"][1]
    assert gpu[1][3] == "905.0 ± 7.1"                                # baseline: the newest code version only
    assert gpu[0][3] == "905.0 ± 7.1"                                # hf too: one measurement campaign per average
    sheet = next(s for s in excel.build(res) if s.name == "Averages")
    h = [c.header for c in sheet.columns]
    versions = {l[h.index("code version")] for l in sheet.lines if l[h.index("arm / setup")] == "baseline"}
    assert versions == {"before the rope_tables fix", "after the rope_tables fix (2026-11-24)"}
    assert excel.code_version(excel.load(res), post[0], "hf") == "after the rope_tables fix (2026-11-24)"


def test_a_run_without_a_summary_is_left_out_everywhere(results):
    res, _, _ = results
    _write(res / "runs" / "20261124T065442Z_eval_decode_stories15m_chars1200000" / "meta.json", {
        "name": "eval_decode_stories15m_chars1200000", "started_utc": "2026-11-24T06:54:42+00:00",
        "devices": {"inference": "cuda"}, "configs": {}})
    loaded = excel.load(res)
    assert "20261124T065442Z_eval_decode_stories15m_chars1200000" not in loaded.runs
    assert not any("065442Z" in str(c) for s in excel.build(res) for line in s.lines for c in line)
