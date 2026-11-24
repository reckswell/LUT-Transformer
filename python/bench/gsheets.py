"""Upload the benchmark workbook's sheets to Alpha's Google Sheet, so results can be watched live
(2026-11-24; the Apps Script web-app route: no installs, no Cloud project).

    cd python
    python -m bench.gsheets status [--target full|latest|all]    # config, last sync, which tabs changed
    python -m bench.gsheets sync [--target ...] [--force]        # build the tabs from results/, upload changed ones
    python -m bench.gsheets verify [--target ...]                # read the sheets back (tabs, rows, columns, charts)
    python -m bench.gsheets tick                                 # the systemd timer: both targets, only on changes

Targets: "full" is the whole workbook (bench/excel.py) in Alpha's first sheet; "latest" is the
latest-only view (bench/latest.py) in the second sheet. Each has its own url, token and state; a
target without a url is skipped quietly. The latest target also asks its script to delete tabs it
created earlier that are no longer sent (prune), so its sheet holds only the current view.

How it works:
  - The sheets are the ones bench/excel.py builds (same tabs, order, values and charts), built in
    process from results/ so numbers stay numbers.
  - Config: ~/.config/lut-transformer/sheets.json (chmod 600; $LUT_SHEETS_CONFIG overrides):
    spreadsheet_id, token (a shared secret the Apps Script checks), web_app_url (the deployment's
    /exec URL; empty = sync is off). State (per-tab content hashes, the results fingerprint, the last
    sync) sits next to it in sheets_state.json.
  - A tab is uploaded only when its content hash changed since the last successful upload. Big tabs
    go in chunks (at most CHUNK_ROWS rows and about CHUNK_BYTES of JSON per request); the first chunk
    clears the tab, the last trims it and freezes the header. A last request sends the tab order, the
    "Last synced" stamp and the Per-stage charts.
  - Apps Script answers a POST with a 302 to script.googleusercontent.com; the JSON result is read
    with a GET of that location. Each request is retried with backoff.
  - Nothing here may break a benchmark: sync_quietly() logs a warning and returns on any error.
  - Only urllib, json and hashlib (standard library).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(os.environ.get("LUT_SHEETS_CONFIG", Path.home() / ".config" / "lut-transformer" / "sheets.json"))
CHUNK_ROWS = 3000
CHUNK_BYTES = 1_500_000
RETRIES = (2.0, 5.0, 15.0)        # seconds to wait before each retry
TIMEOUT_S = 120


TARGETS = ("full", "latest")
# Where the "Last synced" stamp goes: the full sheet keeps it on About at D1:E3 (its deployed Code.gs
# does that by default); the latest-only sheet puts it right of the Latest summary table.
STAMP_TAB = {"latest": "Paper table"}


def _state_path(target: str = "full") -> Path:
    """Per-target state: sheets_state.json for the full sheet (as before), sheets_state-<target>.json else."""
    return CONFIG_PATH.with_name("sheets_state.json" if target == "full" else f"sheets_state-{target}.json")


def load_config() -> dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def target_config(target: str = "full", config: dict[str, Any] | None = None) -> dict[str, Any]:
    """{"url", "token", "spreadsheet_id"} of one target. The full sheet's settings stay where they
    were (top level: web_app_url, token, spreadsheet_id; targets.full is read if present); other
    targets live under targets.<name> as {url, token, spreadsheet_id}. A missing url means skip."""
    config = load_config() if config is None else config
    entry = (config.get("targets") or {}).get(target) or {}
    if target == "full":
        return {"url": config.get("web_app_url") or entry.get("url") or "",
                "token": config.get("token") or entry.get("token") or "",
                "spreadsheet_id": config.get("spreadsheet_id") or entry.get("spreadsheet_id")}
    return {"url": entry.get("url") or "", "token": entry.get("token") or "", "spreadsheet_id": entry.get("spreadsheet_id")}


def load_state(target: str = "full") -> dict[str, Any]:
    try:
        return json.loads(_state_path(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any], target: str = "full") -> None:
    path = _state_path(target)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    os.replace(tmp, path)


def build_sheets(target: str, results_dir: Path | None = None) -> list[Any]:
    """The tabs of one target: the full workbook (bench/excel.py) or the latest-only view (bench/latest.py)."""
    from bench import excel

    if target == "latest":
        from bench import latest

        return latest.build(results_dir or excel.RESULTS_DIR)
    return excel.build(results_dir or excel.RESULTS_DIR)


def warn(message: str) -> None:
    print(f"gsheets: {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------
# Payloads
# ----------------------------------------------------------------------------------------------

def cell(value: Any) -> Any:
    """A cell for Sheets: numbers stay numbers, None is empty, text that Sheets would read as a
    formula gets a leading apostrophe."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return value if value == value and value not in (float("inf"), float("-inf")) else str(value)
    text = str(value)
    return "'" + text if text[:1] in ("=", "+") else text


def sheet_rows(sheet: Any) -> tuple[list[list[Any]], int]:
    """(every row of the tab, top to bottom: preamble, header, data; the number of frozen rows)."""
    width = len(sheet.columns)
    if getattr(sheet, "freeform", False):     # laid out row by row: no header row, nothing frozen
        rows = []
        for line in sheet.lines:
            values = [cell(v) for v in line][:width]
            rows.append(values + [""] * (width - len(values)))
        return rows, 0
    rows = [[cell(text)] + [""] * (width - 1) for text in sheet.preamble]
    rows.append([cell(c.header) for c in sheet.columns])
    for line in sheet.lines:
        values = [cell(v) for v in line][:width]
        rows.append(values + [""] * (width - len(values)))
    return rows, len(sheet.preamble) + 1


def tab_hash(rows: list[list[Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def chunks(rows: list[list[Any]]) -> list[tuple[int, list[list[Any]]]]:
    """[(0-based first row, rows)] with at most CHUNK_ROWS rows and about CHUNK_BYTES of JSON each."""
    out, start, current, size = [], 0, [], 0
    for i, row in enumerate(rows):
        row_bytes = len(json.dumps(row, separators=(",", ":"), default=str)) + 1
        if current and (len(current) >= CHUNK_ROWS or size + row_bytes > CHUNK_BYTES):
            out.append((start, current))
            start, current, size = i, [], 0
        current.append(row)
        size += row_bytes
    if current or not out:
        out.append((start, current))
    return out


def chart_specs(sheet: Any) -> list[dict[str, Any]]:
    """The xlsx charts of a tab, as A1 ranges for the Apps Script (same rows as the workbook)."""
    from bench.excel import column_letter

    specs = []
    for chart in sheet.charts:
        r1, r2 = sheet.data_row(chart.first), sheet.data_row(chart.stop - 1)
        specs.append({"title": chart.title, "categories": f"{column_letter(chart.cat_col)}{r1}:{column_letter(chart.cat_col)}{r2}",
                      "values": f"{column_letter(chart.val_col)}{r1}:{column_letter(chart.val_col)}{r2}",
                      "row": chart.anchor_row + 1, "col": chart.anchor_col + 1})
    return specs


# ----------------------------------------------------------------------------------------------
# HTTP: POST, then follow Apps Script's redirect with a GET
# ----------------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):   # noqa: D401 - urllib hook
        return None


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST payload as JSON; on a 301/302/303/307/308 GET the Location and read the JSON there.
    Returns the decoded JSON answer, or raises."""
    body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "lut-transformer-sync"})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=TIMEOUT_S) as response:
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        if error.code not in (301, 302, 303, 307, 308) or not error.headers.get("Location"):
            raise
        follow = urllib.request.Request(error.headers["Location"], method="GET",
                                        headers={"User-Agent": "lut-transformer-sync"})
        with urllib.request.urlopen(follow, timeout=TIMEOUT_S) as response:
            text = response.read().decode("utf-8", "replace")
    try:
        return json.loads(text)
    except ValueError:
        raise RuntimeError(f"the endpoint did not answer with JSON: {text[:200]!r}")


def send(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """post_json with retries; raises after the last failure."""
    last: Exception | None = None
    for attempt, delay in enumerate((0.0,) + RETRIES):
        if delay:
            time.sleep(delay)
        try:
            answer = post_json(url, payload)
            if answer.get("ok"):
                return answer
            last = RuntimeError(f"endpoint error: {answer.get('error')}")
            if "token" in str(answer.get("error", "")).lower():
                break                    # a wrong token won't fix itself
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as error:
            last = error
    raise RuntimeError(f"upload failed after {attempt + 1} attempt(s): {last}")


# ----------------------------------------------------------------------------------------------
# Sync
# ----------------------------------------------------------------------------------------------

def git_commit() -> str:
    from bench.excel import REPO_ROOT

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def sync(sheets: list[Any] | None = None, *, target: str = "full", force: bool = False,
         dry_run: bool = False) -> dict[str, Any]:
    """Upload every tab of one target whose content changed. Returns a report; raises on upload failure.
    A target without a url is skipped (quietly, with a reason in the report)."""
    cfg = target_config(target)
    url, token = cfg["url"], cfg["token"]
    if not url and not dry_run:
        return {"target": target, "skipped": f"no url for target {target!r} in {CONFIG_PATH}"}
    if not token:
        return {"target": target, "skipped": f"no token for target {target!r} in {CONFIG_PATH}"}
    if sheets is None:
        sheets = build_sheets(target)
    state = load_state(target)
    hashes = state.get("tab_hashes", {})
    order = [s.name for s in sheets]
    changed, requests, sent_rows = [], 0, 0
    payloads = []
    for sheet in sheets:
        rows, frozen = sheet_rows(sheet)
        digest = tab_hash(rows)
        if not force and hashes.get(sheet.name) == digest:
            continue
        parts = chunks(rows)          # chunk 0 clears the tab and the last trims it: no stale rows survive
        for index, (start, block) in enumerate(parts):
            payloads.append({"token": token, "tab": sheet.name, "chunk_index": index, "chunk_count": len(parts),
                             "start_row": start + 1, "rows": block, "total_rows": len(rows),
                             "total_cols": len(sheet.columns), "frozen_rows": frozen,
                             "bold_rows": [i + 1 for i in getattr(sheet, "bold_rows", [])],
                             "charts": chart_specs(sheet) if index == len(parts) - 1 else []})
            sent_rows += len(block)
        changed.append((sheet.name, digest, len(rows), len(parts)))
    meta = {"synced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
            "changed_tabs": [c[0] for c in changed]}
    stamp_tab = STAMP_TAB.get(target)
    if stamp_tab and stamp_tab in order:
        meta["stamp_tab"] = stamp_tab
        meta["stamp_col"] = len(next(s for s in sheets if s.name == stamp_tab).columns) + 2   # right of the table
    finish = {"token": token, "finish": True, "order": order, "meta": meta}
    if target != "full":
        finish["prune"] = True        # drop tabs this target's script created that are no longer sent
    if dry_run:
        return {"target": target, "dry_run": True, "changed": [(n, r, p) for n, _, r, p in changed],
                "requests": len(payloads) + 1, "rows": sent_rows}
    for payload in payloads:
        send(url, payload)
        requests += 1
        if payload["chunk_index"] == payload["chunk_count"] - 1:     # tab complete: remember its hash
            hashes[payload["tab"]] = next(d for n, d, _, _ in changed if n == payload["tab"])
            state["tab_hashes"] = hashes
            save_state(state, target)
    if changed or force or state.get("order") != order:
        send(url, finish)
        requests += 1
        state["order"] = order
    state["last_sync_utc"] = meta["synced_utc"]
    state["last_sync_tabs"] = [c[0] for c in changed]
    save_state(state, target)
    return {"target": target, "changed": [(n, r, p) for n, _, r, p in changed], "requests": requests,
            "rows": sent_rows}


def configured_targets(which: str = "all") -> list[str]:
    names = TARGETS if which == "all" else (which,)
    return [t for t in names if target_config(t)["url"]]


def sync_quietly(sheets: list[Any] | None = None, target: str = "full") -> None:
    """For bench.excel: upload one target if it has a url; never raise."""
    try:
        if not target_config(target)["url"]:
            return
        with _lock() as got:
            if not got:
                warn("another sync is running; skipped")
                return
            report = sync(sheets, target=target)
        if report.get("changed"):
            print(f"gsheets: {target}: uploaded {len(report['changed'])} tab(s) in {report['requests']} request(s)",
                  flush=True)
    except Exception as error:            # a failed upload must never break a benchmark or a rebuild
        warn(f"{target} sync failed, results are only local for now: {type(error).__name__}: {error}")


class _lock:
    """A non-blocking file lock, so the timer and a manual sync never upload at the same time."""

    def __enter__(self) -> bool:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(CONFIG_PATH.with_name("sync.lock"), "w")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def __exit__(self, *exc: Any) -> None:
        self.handle.close()


# ----------------------------------------------------------------------------------------------
# The timer's tick: rebuild and sync only when results changed
# ----------------------------------------------------------------------------------------------

def fingerprint(results_dir: Path) -> dict[str, Any]:
    """File count and newest modification time of everything under results/ (all of results/hls:
    runs, inference records, the pipeline's status files), leaving out what the rebuild itself writes
    (benchmarks.xlsx and its temporary files)."""
    count, newest = 0, 0.0
    for root, _dirs, files in os.walk(results_dir):
        for name in files:
            if (name.startswith("benchmarks") or name.startswith(".benchmarks")) and ".xlsx" in name:
                continue                 # what the rebuild writes (benchmarks.xlsx, benchmarks-latest.xlsx, temp files)
            try:
                mtime = os.stat(os.path.join(root, name)).st_mtime
            except OSError:              # a file that vanished mid-walk
                continue
            count += 1
            newest = max(newest, mtime)
    return {"files": count, "newest_mtime": round(newest, 3)}


def tick(results_dir: Path | None = None, *, force: bool = False) -> str:
    """Rebuild the workbook and sync when the results fingerprint changed; otherwise do nothing."""
    from bench import excel

    results_dir = results_dir or excel.RESULTS_DIR
    with _lock() as got:
        if not got:
            return "another sync is running; skipped"
        state = load_state()
        now = fingerprint(results_dir)
        if not force and state.get("fingerprint") == now:
            return "no change"
        if results_dir == excel.RESULTS_DIR and (results_dir / "hls" / "inference").is_dir():
            try:                                   # new HLS inference records: check them token by token first
                from bench import check_hls_inference

                check_hls_inference.main()
            except Exception as error:
                warn(f"HLS inference check skipped: {type(error).__name__}: {error}")
            now = fingerprint(results_dir)         # include the check's own rows, so the next tick doesn't loop
        from bench import latest

        messages = []
        for target, out_name in (("full", excel.WORKBOOK_PATH.name), ("latest", latest.LATEST_PATH.name)):
            sheets = build_sheets(target, results_dir)   # files still being written parse as empty: next tick
            excel.write_atomically(results_dir / out_name, excel.xlsx_bytes(sheets))
            message = f"rebuilt {out_name}"
            if target_config(target)["url"]:
                try:
                    report = sync(sheets, target=target)
                    message += f"; uploaded {len(report.get('changed', []))} tab(s) in {report.get('requests', 0)} request(s)"
                except Exception as error:         # one target failing must not stop the other
                    message += f"; upload failed: {type(error).__name__}: {error}"
            messages.append(message)
        state = load_state()                       # sync() updated it
        state["fingerprint"] = now
        save_state(state)
        return " | ".join(messages)


def remote_tabs(target: str = "full") -> list[dict[str, Any]]:
    """The target sheet's tabs as its Apps Script sees them (name, last row, last column, charts)."""
    cfg = target_config(target)
    answer = post_json(cfg["url"], {"token": cfg["token"], "status": True})
    if not answer.get("ok"):
        raise RuntimeError(f"status refused: {answer.get('error')}")
    return answer.get("tabs", [])


def verify(sheets: list[Any] | None = None, target: str = "full") -> list[dict[str, Any]]:
    """Compare a target sheet's tabs (names, row and column counts, charts) with the local sheets."""
    sheets = sheets if sheets is not None else build_sheets(target)
    remote = {t["name"]: t for t in remote_tabs(target)}
    stamp_tab = STAMP_TAB.get(target, "About")
    out = []
    for sheet in sheets:
        rows, _ = sheet_rows(sheet)
        got = remote.get(sheet.name, {})
        width = len(sheet.columns)
        stamp_width = max(width, 5) if target == "full" else width + 3      # the sync stamp beside the table
        out.append({"target": target, "tab": sheet.name, "local_rows": len(rows), "sheet_rows": got.get("rows"),
                    "local_cols": width, "sheet_cols": got.get("cols"), "charts": got.get("charts"),
                    "match": got.get("rows") == len(rows) and (got.get("cols") == width or (
                        sheet.name == stamp_tab and got.get("cols") == stamp_width))})
    return out


def status(target: str = "full") -> dict[str, Any]:
    cfg, state = target_config(target), load_state(target)
    out = {"target": target, "config": str(CONFIG_PATH), "spreadsheet_id": cfg.get("spreadsheet_id"),
           "url_set": bool(cfg["url"]), "token_set": bool(cfg["token"]), "state": str(_state_path(target)),
           "last_sync_utc": state.get("last_sync_utc"), "last_sync_tabs": state.get("last_sync_tabs"),
           "tabs_known": len(state.get("tab_hashes", {}))}
    if target == "full":
        out["fingerprint"] = state.get("fingerprint")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sync", "status", "tick", "verify"])
    parser.add_argument("--target", choices=["full", "latest", "all"], default="all",
                        help="which sheet: the full workbook, the latest-only view, or both (default)")
    parser.add_argument("--force", action="store_true", help="upload every tab / rebuild even without changes")
    parser.add_argument("--dry-run", action="store_true", help="sync: show what would be uploaded, send nothing")
    args = parser.parse_args(argv)
    names = list(TARGETS) if args.target == "all" else [args.target]
    if args.command == "status":
        print(json.dumps([status(t) for t in names], indent=2))
        return 0
    if args.command == "verify":
        code, reports = 0, []
        for t in names:
            if not target_config(t)["url"]:
                reports.append({"target": t, "skipped": "no url"})
                continue
            try:
                report = verify(target=t)
            except Exception as error:
                warn(f"{t}: verify failed: {type(error).__name__}: {error}")
                code = 1
                continue
            reports.append({"target": t, "all_match": all(r["match"] for r in report), "tabs": report})
            code = code or (0 if all(r["match"] for r in report) else 1)
        print(json.dumps(reports, indent=2))
        return code
    if args.command == "tick":
        try:
            print(f"gsheets tick: {tick(force=args.force)}", flush=True)
        except Exception as error:
            warn(f"tick failed: {type(error).__name__}: {error}")
        return 0
    with _lock() as got:
        if not got:
            warn("another sync is running; try again shortly")
            return 1
        code, reports = 0, []
        for t in names:
            try:
                reports.append(sync(target=t, force=args.force, dry_run=args.dry_run))
            except Exception as error:
                warn(f"{t}: sync failed: {type(error).__name__}: {error}")
                code = 1
    print(json.dumps(reports, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
