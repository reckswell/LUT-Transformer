"""results/benchmarks.xlsx: every benchmark result, PyTorch and HLS, in one Excel workbook, rebuilt
from the long-format result files (Alpha, 2026-11-19: an Excel sheet to open and see; 2026-11-23:
a tab with the time spent in each layer and each calculation within a layer, and one sheet that
combines every flow).

    cd python
    python -m bench.excel                         # rebuild results/benchmarks.xlsx now
    python -m bench.excel --csv-dir DIR           # also write each sheet as DIR/<sheet>.csv
    python -m bench.excel --no-sync               # don't upload to the Google Sheet (bench/gsheets.py)
After a rebuild of results/benchmarks.xlsx the changed tabs are uploaded to Alpha's Google Sheet when
bench/gsheets.py is configured (a web_app_url in ~/.config/lut-transformer/sheets.json).

Nobody edits the workbook by hand and nothing calls this during a run: rebuild it after new
PyTorch or HLS runs. Inputs (all under results/, read only):
    all_results.csv        the sweep's long format (bench/sweep.py): speed and perplexity
    layer_timing.csv       per-layer, per-stage timing (bench/layer_timing.py)
    extra_results.csv      other PyTorch runs in All-data form (training throughput, modeled cost,
                           reconstructed process wall times with phase = time)
    runs/*/meta.json, summary.json   every PyTorch run's provenance (Runs sheet, char limits, cost)
    hls/runs/*/summary.json, ledger_record.json   the Vitis/Vivado pipeline's records (written by the
                           pipeline threads; never written here). ledger_record.json is preferred for
                           variant (the PyTorch name), hls_variant, stages, platform and metrics;
                           summary.json gives per-stage tool times, threads and parallel jobs.

Sheets
    All data          every measurement from every flow in one long, filterable table
    Time budget       how long each activity takes (PyTorch runs, HLS stages per kernel) and projections
    PyTorch Summary   one row per PyTorch arm: perplexity and accuracy (full span and slice), speed,
                      power, training
    PyTorch Power & resources   per run and phase: GPU power and energy, utilization, memory, clocks,
                      temperature, throttle reasons, CPU time, RSS, the GPU's power settings
    PyTorch Perplexity  one row per perplexity run (full span and screening slice)
    PyTorch Speed     prefill and decode latency against context, and training tokens/s
    PyTorch Per-stage one row per stage in forward order (embedding ... lm_head), prefill and decode at
                      T = 64, 128, 256: modeled cost per token over the layers (MACs, nonlinear, INT8 and
                      FP32 bytes), measured GPU time over the layers and its share of the step, with bar
                      charts. Per-layer rows stay in All data.
    Averages          repeated runs: mean, std, min, max, n and 95 % CI over the clean runs, per metric
    PyTorch Memory    per run and measured phase: torch peak allocated / reserved, process and device GPU
                      memory, host RSS, pinned memory, and (memory pass) the split into weights, LUT tables,
                      KV cache, optimizer state, gradients, activations/workspace, CUDA context
    LUT storage       every LUT table per arm: stage, shape, dtype, bytes, and where it lives (GPU VRAM,
                      host DRAM, pinned, memory-mapped from disk), plus host-to-device copies per step
    PyTorch Hardware cost  LUT/ops_registry.describe and bench/opcount.py per stage
    PyTorch Runs      every PyTorch run: commit, versions, character limit and token counts
    HLS Kernels       one row per HLS record with a variant (headers only until there is one)
    HLS Inference     the HLS kernel run on the slice tokens (C model or emulation): perplexity, top-1/top-5,
                      per-token agreement with the FP32 GPU reference (results/hls/inference)
    Final comparison (PT vs HLS)  variant x stage, PyTorch next to HLS, empty until both sides share a variant
    About             what each sheet and metric means
Stage names are base.STAGES on both sides (embedding, norm, qkv_proj, rope, attn_scores, softmax,
attn_values, out_proj, ffn_up, ffn_act, ffn_gate, ffn_down, final_norm, lm_head), so PyTorch stage
times line up with HLS kernels. No power columns until the power-ramp fix (agent/STATE.md item 6).

Every sheet has a bold, frozen header row and an autofilter over its columns. The .xlsx is written
with the Python standard library (an .xlsx is a zip of SpreadsheetML parts), deterministically and
atomically (a temporary file renamed over the old one).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
# The workbook's path, the one constant to change if the results layout moves (for example to
# results/benchmarking/, pending Alpha).
WORKBOOK_PATH = RESULTS_DIR / "benchmarks.xlsx"
TRAINED_CONTEXT_NOTE = "beyond trained context (256), timing only"
NOT_MEASURED_POWER = "not measured (pre-fix)"      # runs before the power-ramp fix and the resource sampler
NOT_MEASURED_ACC = "not measured (run before accuracy was added)"


# ----------------------------------------------------------------------------------------------
# A sheet: columns (header, kind of value, number format) and lines of values
# ----------------------------------------------------------------------------------------------

# Number formats: name -> (Excel numFmtId, format code). Ids below 164 are Excel's built-in
# formats (they need no code); 164 and up are defined in the workbook's styles.
FORMATS = {
    "general": (0, None),
    "int": (3, None),                       # 1,234
    "0.0": (164, "0.0"),
    "0.00": (2, None),
    "0.000": (165, "0.000"),
    "0.0000": (166, "0.0000"),
    "0.000000": (167, "#,##0.000000"),
    "#,##0.0": (168, "#,##0.0"),
    "time": (169, "yyyy-mm-dd hh:mm:ss"),
    "pct": (10, None),                     # 12.34% (built-in)
}
# Cell style index (cellXfs) of each format; 0 is the default style and 1 the header style.
STYLE_OF = {name: index + 2 for index, name in enumerate(FORMATS)}
HEADER_STYLE = 1


@dataclass
class Column:
    """One column. kind says how a CSV string becomes a cell: text, int, num (a float) or time
    (an ISO 8601 time, stored as an Excel date). fmt names one of FORMATS."""

    header: str
    kind: str = "text"
    fmt: str = "general"


@dataclass
class Chart:
    """A bar chart drawn on its sheet from one block of rows: categories from one column, values
    from another (1-based columns, rows counted among sheet.lines, 0-based, end exclusive).
    Categories run top to bottom in row order (horizontal bars), so a block in forward order reads
    embedding at the top and lm_head at the bottom."""

    title: str
    cat_col: int
    val_col: int
    first: int
    stop: int
    series: str
    anchor_col: int          # top-left cell of the chart (0-based column and row)
    anchor_row: int
    width_cols: int = 9
    height_rows: int = 18


@dataclass
class Sheet:
    name: str
    columns: list[Column]
    lines: list[list[Any]] = field(default_factory=list)
    preamble: list[str] = field(default_factory=list)      # text rows above the header (caveats)
    charts: list[Chart] = field(default_factory=list)
    freeform: bool = False          # no header row, filter or frozen pane: lines are laid out as they are
    bold_rows: list[int] = field(default_factory=list)      # freeform: indices into lines shown in bold

    @property
    def header_row(self) -> int:
        return len(self.preamble) + 1

    def data_row(self, index: int) -> int:
        """The worksheet row of sheet.lines[index]."""
        return self.header_row + 1 + index


EXCEL_EPOCH = datetime(1899, 12, 30, tzinfo=timezone.utc)     # Excel's day 0 (1900 date system)


def cell_value(text: Any, kind: str) -> Any:
    """A CSV string as a typed cell value: str, int, float, or None for an empty cell. Anything that
    doesn't parse as its kind stays text, so nothing is lost."""
    if text is None or text == "":
        return None
    if kind == "text":
        return str(text)              # a commit like 1234567890 stays text
    if not isinstance(text, str):
        return text                   # already a number
    try:
        if kind == "int":
            number = float(text)
            return int(number) if number.is_integer() else number
        if kind == "num":
            number = float(text)
            return number if math.isfinite(number) else text
        if kind == "time":
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return (moment - EXCEL_EPOCH).total_seconds() / 86400.0     # days since day 0
    except ValueError:
        return text
    return text


def _display_length(value: Any, fmt: str) -> int:
    """Roughly how many characters Excel shows for a value (for column widths)."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if fmt == "time":
        return 19
    decimals = {"0.0": 1, "0.00": 2, "0.000": 3, "0.0000": 4, "0.000000": 6, "#,##0.0": 1, "int": 0}.get(fmt)
    if decimals is None:
        return len(repr(value))
    return len(f"{value:,.{decimals}f}")


# ----------------------------------------------------------------------------------------------
# Writing the .xlsx (SpreadsheetML)
# ----------------------------------------------------------------------------------------------

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_BAD_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def column_letter(number: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA."""
    letters = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _text_cell(ref: str, text: str, style: int = 0) -> str:
    text = _BAD_XML_CHARS.sub("", str(text))[:32767]           # Excel's limit per cell
    space = ' xml:space="preserve"' if text != text.strip() else ""
    style_attr = f' s="{style}"' if style else ""
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t{space}>{escape(text)}</t></is></c>'


def _cell(ref: str, value: Any, fmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _text_cell(ref, value)
    number = str(value) if isinstance(value, int) else repr(float(value))
    return f'<c r="{ref}" s="{STYLE_OF[fmt]}"><v>{number}</v></c>'


def _last_cell(sheet: Sheet) -> str:
    return f"{column_letter(len(sheet.columns))}{sheet.header_row + len(sheet.lines)}"


def _filter_ref(sheet: Sheet) -> str:
    return f"A{sheet.header_row}:{_last_cell(sheet)}"


def sheet_xml(sheet: Sheet, selected: bool) -> str:
    """One worksheet: frozen header row, column widths, the cells, and an autofilter over them.
    (The element order is fixed by the SpreadsheetML schema.)"""
    widths = []
    for i, column in enumerate(sheet.columns):
        cells = [_display_length(line[i] if i < len(line) else None, column.fmt) for line in sheet.lines
                 if not (sheet.freeform and len(line) == 1 and i == 0)]      # long title or footnote lines don't widen A
        longest = max(([] if sheet.freeform else [len(column.header)]) + cells + [0])
        widths.append(min(max(longest + 2, 6), 60))
    if sheet.freeform:
        return _freeform_xml(sheet, widths, selected)
    head = sheet.header_row
    rows = [f'<row r="{r}">' + _text_cell(f"A{r}", text, HEADER_STYLE) + "</row>"
            for r, text in enumerate(sheet.preamble, start=1)]
    rows.append(f'<row r="{head}">' + "".join(_text_cell(f"{column_letter(i + 1)}{head}", column.header, HEADER_STYLE)
                                              for i, column in enumerate(sheet.columns)) + "</row>")
    for number, line in enumerate(sheet.lines, start=head + 1):
        cells = "".join(_cell(f"{column_letter(i + 1)}{number}", value, column.fmt)
                        for i, (value, column) in enumerate(zip(line, sheet.columns)))
        rows.append(f'<row r="{number}">{cells}</row>')
    last = _last_cell(sheet)
    return (XML_HEAD + f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
            + f'<dimension ref="A1:{last}"/>'
            + '<sheetViews><sheetView workbookViewId="0"' + (' tabSelected="1"' if selected else "") + ">"
            + f'<pane ySplit="{head}" topLeftCell="A{head + 1}" activePane="bottomLeft" state="frozen"/>'
            + f'<selection pane="bottomLeft" activeCell="A{head + 1}" sqref="A{head + 1}"/></sheetView></sheetViews>'
            + '<sheetFormatPr defaultRowHeight="15"/>'
            + "<cols>" + "".join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
                                 for i, w in enumerate(widths, start=1)) + "</cols>"
            + "<sheetData>" + "".join(rows) + "</sheetData>"
            + f'<autoFilter ref="{_filter_ref(sheet)}"/>'
            + ('<drawing r:id="rId1"/>' if sheet.charts else "")
            + "</worksheet>")


def _freeform_xml(sheet: Sheet, widths: list[int], selected: bool) -> str:
    """A sheet laid out row by row (section titles, small tables, footnotes): no header row, no
    autofilter, no frozen pane; rows in sheet.bold_rows are bold."""
    bold = set(sheet.bold_rows)
    rows = []
    for number, line in enumerate(sheet.lines, start=1):
        cells = []
        for i, value in enumerate(line):
            ref = f"{column_letter(i + 1)}{number}"
            if value is None or value == "":
                continue
            if (number - 1) in bold or not isinstance(value, (int, float)) or isinstance(value, bool):
                cells.append(_text_cell(ref, value, HEADER_STYLE if (number - 1) in bold else 0))
            else:
                cells.append(_cell(ref, value, "general"))
        rows.append(f'<row r="{number}">{"".join(cells)}</row>')
    last = f"{column_letter(max(1, len(sheet.columns)))}{max(1, len(sheet.lines))}"
    return (XML_HEAD + f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
            + f'<dimension ref="A1:{last}"/>'
            + '<sheetViews><sheetView workbookViewId="0"' + (' tabSelected="1"' if selected else "") + "/></sheetViews>"
            + '<sheetFormatPr defaultRowHeight="15"/>'
            + "<cols>" + "".join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
                                 for i, w in enumerate(widths, start=1)) + "</cols>"
            + "<sheetData>" + "".join(rows) + "</sheetData></worksheet>")


def styles_xml() -> str:
    """Two fonts (regular, bold), a light fill for the header, and one cell style per number format."""
    custom = [(fid, code) for fid, code in FORMATS.values() if code is not None]
    num_fmts = "".join(f'<numFmt numFmtId="{fid}" formatCode="{escape(code)}"/>' for fid, code in custom)
    xfs = ['<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>',
           '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>']
    xfs += [f'<xf numFmtId="{fid}" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
            for fid, _ in FORMATS.values()]
    return (XML_HEAD + f'<styleSheet xmlns="{MAIN_NS}">'
            + f'<numFmts count="{len(custom)}">{num_fmts}</numFmts>'
            + '<fonts count="2"><font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
            + '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font></fonts>'
            + '<fills count="3"><fill><patternFill patternType="none"/></fill>'
            + '<fill><patternFill patternType="gray125"/></fill>'
            + '<fill><patternFill patternType="solid"><fgColor rgb="FFD9E1F2"/><bgColor indexed="64"/>'
            + "</patternFill></fill></fills>"
            + '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            + '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            + f'<cellXfs count="{len(xfs)}">{"".join(xfs)}</cellXfs>'
            + '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            + "</styleSheet>")


def xlsx_bytes(sheets: list[Sheet]) -> bytes:
    """The whole workbook as .xlsx bytes. Fixed zip timestamps, so the same sheets give the same bytes."""
    names = [sheet.name for sheet in sheets]
    if len(set(names)) != len(names) or any(len(n) > 31 or re.search(r"[\[\]:*?/\\]", n) for n in names):
        raise ValueError(f"sheet names must be unique, at most 31 characters, without []:*?/\\: {names}")
    n = len(sheets)
    content_types = (XML_HEAD + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
                       'relationships+xml"/>'
                     + '<Default Extension="xml" ContentType="application/xml"/>'
                     + '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
                       'officedocument.spreadsheetml.sheet.main+xml"/>'
                     + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/'
                               f'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                               for i in range(1, n + 1))
                     + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-'
                       'officedocument.spreadsheetml.styles+xml"/>'
                     + "</Types>")
    root_rels = (XML_HEAD + f'<Relationships xmlns="{PKG_REL_NS}">'
                 + f'<Relationship Id="rId1" Type="{REL_NS}/officeDocument" Target="xl/workbook.xml"/>'
                 + "</Relationships>")
    # The autofilter of each sheet is also named in the workbook (_xlnm._FilterDatabase), as Excel does.
    filters = "".join(f'<definedName name="_xlnm._FilterDatabase" localSheetId="{i}" hidden="1">'
                      f"{escape(quoted(sheet.name))}!$A${sheet.header_row}:${column_letter(len(sheet.columns))}"
                      f"${sheet.header_row + len(sheet.lines)}</definedName>" for i, sheet in enumerate(sheets)
                      if not sheet.freeform)
    workbook = (XML_HEAD + f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
                + '<bookViews><workbookView activeTab="0"/></bookViews><sheets>'
                + "".join(f'<sheet name="{escape(sheet.name, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"/>'
                          for i, sheet in enumerate(sheets, start=1))
                + f"</sheets>" + (f"<definedNames>{filters}</definedNames>" if filters else "") + "</workbook>")
    workbook_rels = (XML_HEAD + f'<Relationships xmlns="{PKG_REL_NS}">'
                     + "".join(f'<Relationship Id="rId{i}" Type="{REL_NS}/worksheet" Target="worksheets/sheet{i}.xml"/>'
                               for i in range(1, n + 1))
                     + f'<Relationship Id="rId{n + 1}" Type="{REL_NS}/styles" Target="styles.xml"/>'
                     + "</Relationships>")
    parts = [("[Content_Types].xml", content_types), ("_rels/.rels", root_rels), ("xl/workbook.xml", workbook),
             ("xl/_rels/workbook.xml.rels", workbook_rels), ("xl/styles.xml", styles_xml())]
    parts += [(f"xl/worksheets/sheet{i}.xml", sheet_xml(sheet, selected=i == 1))
              for i, sheet in enumerate(sheets, start=1)]
    chart_no, drawing_no, overrides = 0, 0, []
    for i, sheet in enumerate(sheets, start=1):
        if not sheet.charts:
            continue
        drawing_no += 1
        anchors, drawing_rels = [], []
        for k, chart in enumerate(sheet.charts, start=1):
            chart_no += 1
            parts.append((f"xl/charts/chart{chart_no}.xml", chart_xml(sheet, chart)))
            overrides.append(f'<Override PartName="/xl/charts/chart{chart_no}.xml" ContentType="application/'
                             'vnd.openxmlformats-officedocument.drawingml.chart+xml"/>')
            drawing_rels.append(f'<Relationship Id="rId{k}" Type="{REL_NS}/chart" Target="../charts/chart{chart_no}.xml"/>')
            anchors.append(_anchor_xml(chart, k))
        parts.append((f"xl/drawings/drawing{drawing_no}.xml", XML_HEAD + f'<xdr:wsDr xmlns:xdr="{XDR_NS}" xmlns:a="{A_NS}">'
                      + "".join(anchors) + "</xdr:wsDr>"))
        parts.append((f"xl/drawings/_rels/drawing{drawing_no}.xml.rels", XML_HEAD + f'<Relationships xmlns="{PKG_REL_NS}">'
                      + "".join(drawing_rels) + "</Relationships>"))
        parts.append((f"xl/worksheets/_rels/sheet{i}.xml.rels", XML_HEAD + f'<Relationships xmlns="{PKG_REL_NS}">'
                      + f'<Relationship Id="rId1" Type="{REL_NS}/drawing" Target="../drawings/drawing{drawing_no}.xml"/>'
                      + "</Relationships>"))
        overrides.append(f'<Override PartName="/xl/drawings/drawing{drawing_no}.xml" ContentType="application/'
                         'vnd.openxmlformats-officedocument.drawing+xml"/>')
    if overrides:
        parts[0] = ("[Content_Types].xml", content_types.replace("</Types>", "".join(overrides) + "</Types>"))
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, text.encode("utf-8"))
    return buffer.getvalue()


XDR_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"


def _anchor_xml(chart: Chart, rel: int) -> str:
    """Where a chart sits on the sheet (a two-cell anchor) and which chart part it shows."""
    c0, r0 = chart.anchor_col, chart.anchor_row
    return ("<xdr:twoCellAnchor>"
            f"<xdr:from><xdr:col>{c0}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{r0}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>"
            f"<xdr:to><xdr:col>{c0 + chart.width_cols}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{r0 + chart.height_rows}"
            "</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>"
            f'<xdr:graphicFrame macro=""><xdr:nvGraphicFramePr><xdr:cNvPr id="{rel + 1}" name="Chart {rel}"/>'
            "<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr>"
            '<xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>'
            f'<a:graphic><a:graphicData uri="{C_NS}"><c:chart xmlns:c="{C_NS}" xmlns:r="{REL_NS}" r:id="rId{rel}"/>'
            "</a:graphicData></a:graphic></xdr:graphicFrame><xdr:clientData/></xdr:twoCellAnchor>")


def chart_xml(sheet: Sheet, chart: Chart) -> str:
    """A clustered horizontal bar chart over sheet cells (DrawingML, ECMA-376 element order).
    Categories top to bottom in row order (category axis reversed)."""
    r1, r2 = sheet.data_row(chart.first), sheet.data_row(chart.stop - 1)
    name = quoted(sheet.name)
    cat = f"{name}!${column_letter(chart.cat_col)}${r1}:${column_letter(chart.cat_col)}${r2}"
    val = f"{name}!${column_letter(chart.val_col)}${r1}:${column_letter(chart.val_col)}${r2}"
    title = (f"<c:title><c:tx><c:rich><a:bodyPr/><a:p><a:r><a:t>{escape(chart.title)}</a:t></a:r></a:p></c:rich></c:tx>"
             '<c:overlay val="0"/></c:title>')
    return (XML_HEAD + f'<c:chartSpace xmlns:c="{C_NS}" xmlns:a="{A_NS}" xmlns:r="{REL_NS}">'
            '<c:roundedCorners val="0"/><c:chart>' + title + '<c:autoTitleDeleted val="0"/>'
            "<c:plotArea><c:layout/><c:barChart>"
            '<c:barDir val="bar"/><c:grouping val="clustered"/><c:varyColors val="0"/>'
            f'<c:ser><c:idx val="0"/><c:order val="0"/><c:tx><c:v>{escape(chart.series)}</c:v></c:tx>'
            '<c:invertIfNegative val="0"/>'
            f"<c:cat><c:strRef><c:f>{escape(cat)}</c:f></c:strRef></c:cat>"
            f"<c:val><c:numRef><c:f>{escape(val)}</c:f></c:numRef></c:val></c:ser>"
            '<c:gapWidth val="60"/><c:axId val="10"/><c:axId val="20"/></c:barChart>'
            '<c:catAx><c:axId val="10"/><c:scaling><c:orientation val="maxMin"/></c:scaling><c:delete val="0"/>'
            '<c:axPos val="l"/><c:numFmt formatCode="General" sourceLinked="0"/><c:majorTickMark val="out"/>'
            '<c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/><c:crossAx val="20"/><c:crosses val="autoZero"/>'
            '<c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/><c:noMultiLvlLbl val="0"/></c:catAx>'
            '<c:valAx><c:axId val="20"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/>'
            '<c:axPos val="b"/><c:majorGridlines/><c:numFmt formatCode="General" sourceLinked="1"/>'
            '<c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>'
            '<c:crossAx val="10"/><c:crosses val="max"/><c:crossBetween val="between"/></c:valAx>'
            "</c:plotArea>"
            '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart></c:chartSpace>')


def quoted(sheet_name: str) -> str:
    """A sheet name as a formula reference writes it: 'All results'."""
    return "'" + sheet_name.replace("'", "''") + "'"


def write_atomically(path: Path, data: bytes) -> None:
    """Write to a temporary file next to path, then rename it over path: a reader sees the old
    workbook or the new one, never half of one. On any error the temporary file is removed and the
    old workbook is left as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise



# ----------------------------------------------------------------------------------------------
# Reading the result files
# ----------------------------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def num(text: Any) -> Any:
    """A CSV value as a number when it is one, else the text (None for empty)."""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return text
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        return text
    return int(value) if value.is_integer() and "." not in str(text) and "e" not in str(text).lower() else value


def platform_of(device: str) -> str:
    return "gpu" if str(device).startswith("cuda") else (device or "")


def variant_of(arm: str, lut_preset: str) -> str:
    """PyTorch variant names, the join key with the HLS records: baseline, hf, or the LUT preset."""
    if arm == "lut":
        return lut_preset or "params"
    return arm


@dataclass
class Results:
    sweep: list[dict[str, str]]
    layers: list[dict[str, str]]
    extra: list[dict[str, str]]
    runs: dict[str, dict[str, Any]]           # run id -> {"meta": ..., "summary": ...}
    hls: list[dict[str, Any]]                 # HLS records with a variant (ledger merged over summary)
    hls_all: list[dict[str, Any]] = field(default_factory=list)   # every HLS run: {"summary", "ledger", "dir"}
    sweeps: dict[str, dict[str, Any]] = field(default_factory=dict)  # sweep id -> status.json
    hls_inference: list[dict[str, Any]] = field(default_factory=list)  # results/hls/inference/*: summary or testbench
    results_dir: Path = RESULTS_DIR


REUSE_KEYS = ("build_reused", "source_build_id", "reused_build_id", "build_reused_from", "reused_from", "source_build")
BUILD_STAGES = ("synth", "impl", "link", "hw_link", "xclbin", "export", "hw_build")


def reused_build(*records: dict[str, Any] | None) -> str | None:
    """The build id an HLS record reused instead of building ("build_reused": true plus the source
    build's id under one of REUSE_KEYS, or a stage marked reused with reused_from), "unknown build"
    if the id is missing, or None for a fresh build."""
    for record in records:
        if not record:
            continue
        if not record.get("build_reused"):
            stage_map = record.get("stages")             # the runner's {stage: info}; a ledger's is a list of names
            stages = [i for i in stage_map.values() if isinstance(i, dict) and i.get("reused") is True] \
                if isinstance(stage_map, dict) else []
            if stages:
                return str(stages[0].get("reused_from") or "unknown build")
            continue
        if record.get("build_reused"):
            for key in REUSE_KEYS[1:]:
                value = record.get(key)
                if isinstance(value, dict):
                    value = value.get("run_id") or value.get("id")
                if value:
                    return str(value)
            return "unknown build"
    return None


def merged_hls(summary: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """One HLS record: the runner's summary.json with the ladder's ledger_record.json on top for the
    fields the ledger normalizes: variant (the PyTorch variant name), hls_variant, stages (base.STAGES
    names), platform, profile and metrics. A record with only one of the two files uses that one."""
    record = dict(summary)
    if ledger:
        for key in ("variant", "hls_variant", "stages_list", "platform", "profile", "metrics", "knobs", "kernel",
                    "run_id", "family", "reference", "valid", "seconds", "times", *REUSE_KEYS):
            source = "stages" if key == "stages_list" else key
            if ledger.get(source) is not None:
                record[key] = ledger[source]
        if summary.get("variant") and summary.get("variant") != ledger.get("variant"):
            record.setdefault("hls_variant", summary.get("variant"))
    return record


def load(results_dir: Path) -> Results:
    runs = {}
    for meta_path in sorted((results_dir / "runs").glob("*/meta.json")):
        if not (meta_path.parent / "summary.json").is_file():
            continue          # still running, or cut off before it finished (e.g. 20261124T065442Z): excluded everywhere
        runs[meta_path.parent.name] = {"meta": read_json(meta_path), "summary": read_json(meta_path.parent / "summary.json")}
    hls, hls_all = [], []
    for folder in sorted(p for p in (results_dir / "hls" / "runs").glob("*") if p.is_dir()):
        summary, ledger = read_json(folder / "summary.json"), read_json(folder / "ledger_record.json")
        if not summary and not ledger:
            continue
        hls_all.append({"dir": folder.name, "summary": summary, "ledger": ledger})
        record = merged_hls(summary, ledger)
        if isinstance(record.get("variant"), str) and record["variant"].strip():
            hls.append(record)
    sweeps = {p.parent.name: read_json(p) for p in sorted((results_dir / "sweeps").glob("*/status.json"))}
    hls_inference = []
    for folder in sorted(p for p in (results_dir / "hls" / "inference").glob("*") if p.is_dir()):
        summary, bench = read_json(folder / "summary.json"), read_json(folder / "testbench.json")
        if summary or bench:          # a record still being written parses as {} and is picked up next time
            hls_inference.append({"dir": folder.name, "summary": summary, "testbench": bench})
    return Results(read_csv(results_dir / "all_results.csv"), read_csv(results_dir / "layer_timing.csv"),
                   read_csv(results_dir / "extra_results.csv"), runs, hls, hls_all=hls_all, sweeps=sweeps,
                   hls_inference=hls_inference, results_dir=results_dir)


def char_limit_of(results: Results, run_id: str, dataset: str) -> Any:
    meta = results.runs.get(run_id, {}).get("meta", {})
    return (meta.get("configs", {}).get("perplexity", {}).get(dataset, {}) or {}).get("char_limit")


def span_label(limit: Any) -> str:
    """full = the dataset's default evaluation span (common/data.py, 1,200,000 characters for TinyStories)."""
    if limit in (None, ""):
        return "whole split"
    return "full (1,200,000 chars)" if int(limit) == 1_200_000 else f"slice ({int(limit):,} chars)"


# ----------------------------------------------------------------------------------------------
# HLS records (results/hls/runs/*/summary.json, written by the Vitis/Vivado pipeline)
# ----------------------------------------------------------------------------------------------

def hls_view(record: dict[str, Any]) -> dict[str, Any]:
    """The numbers the HLS sheets use, from one pipeline record: the best available source for each
    (cosim latency over synthesis estimates, post-route clock and resources over synthesis)."""
    results = record.get("results", {}) or {}
    synth, cosim, impl = results.get("synth", {}) or {}, results.get("cosim", {}) or {}, results.get("impl", {}) or {}
    stages = record.get("stages", {}) or {}
    compare = record.get("compare", {}) or {}
    platforms = record.get("platforms")
    if not platforms:
        platforms = [name for name, info in stages.items() if (info or {}).get("status") == "pass"]
    mismatches = [(name, c.get("mismatches")) for name, c in compare.items() if isinstance(c, dict)]
    bit_exact = all(c.get("pass") for c in compare.values() if isinstance(c, dict)) if compare else None
    tols = [c.get("tolerance") or {} for c in compare.values() if isinstance(c, dict)]
    exact = all(not t.get("atol") and not t.get("rtol") for t in tols)
    tol_label = "exact" if exact else "; ".join(sorted({f"atol {t.get('atol')}, rtol {t.get('rtol')}" for t in tols}))
    lat_src = cosim if cosim.get("latency_cycles_max") is not None else synth
    if impl.get("post_route_clock_ns"):
        clock_ns, clock_src = impl["post_route_clock_ns"], "post-route (impl)"
    elif synth.get("target_clock_ns"):
        clock_ns, clock_src = synth["target_clock_ns"], "target (impl not run)"
    else:
        clock_ns = 1000.0 / record["clock_target_mhz"] if record.get("clock_target_mhz") else None
        clock_src = "target"
    res = impl.get("resources") or synth.get("resources") or {}
    lat_max = lat_src.get("latency_cycles_max")
    power = results.get("power", {}) or {}
    stage = record.get("stage") if isinstance(record.get("stage"), str) else record.get("pytorch_stage", "")
    if isinstance(record.get("stages_list"), list):             # ledger_record.json: base.STAGES names
        stage = ";".join(record["stages_list"])
    m = record.get("metrics") or {}                               # ledger_record.json's normalized metrics
    if m:
        lat_max = m.get("latency_cycles", lat_max)
        clock_ns = m.get("clock_ns") or clock_ns
        if not impl.get("post_route_clock_ns"):
            clock_src = "target" if m.get("clock_ns") == m.get("target_clock_ns") else "ledger clock_ns"
        res = {"LUT": m.get("lut", res.get("LUT")), "FF": m.get("ff", res.get("FF")), "DSP": m.get("dsp", res.get("DSP")),
               "BRAM": m.get("bram18k", res.get("BRAM", res.get("BRAM_18K"))), "URAM": m.get("uram", res.get("URAM"))}
    return {
        "hls_variant": record.get("hls_variant"), "profile": record.get("profile"), "platform": record.get("platform"),
        "latency_ns": m.get("latency_ns"), "latency_note": m.get("latency_source"),
        "lut_pct": m.get("lut_pct"), "dsp_pct": m.get("dsp_pct"), "threads": record.get("threads"),
        "parallel_jobs": record.get("parallel_jobs"), "wall_s": record.get("wall_seconds") or record.get("seconds"),
        "kernel": record.get("kernel"), "variant": record.get("variant"), "stage": stage or "",
        "platforms": ";".join(platforms) if isinstance(platforms, list) else str(platforms),
        "bit_exact": None if bit_exact is None else (("yes" if bit_exact else "no") + f" ({tol_label})"),
        "mismatches": "; ".join(f"{n}: {m}" for n, m in mismatches) or None,
        "latency_cycles_min": lat_src.get("latency_cycles_min"), "latency_cycles_max": lat_max,
        "latency_source": "cosim" if lat_src is cosim and cosim else ("synth estimate" if lat_src else ""),
        "ii": synth.get("interval_min") if synth.get("interval_min") == synth.get("interval_max")
        else f"{synth.get('interval_min')}-{synth.get('interval_max')}" if synth else None,
        "clock_ns": clock_ns, "clock_source": clock_src,
        "clock_mhz": round(1000.0 / clock_ns, 1) if clock_ns else None,
        "latency_us": round(lat_max * clock_ns / 1000.0, 4) if (lat_max is not None and clock_ns) else None,
        "LUT": res.get("LUT"), "FF": res.get("FF"), "DSP": res.get("DSP"),
        "BRAM": res.get("BRAM", res.get("BRAM_18K")), "URAM": res.get("URAM"),
        "dynamic_power_w": power.get("dynamic_w"), "power_kind": power.get("kind"),
        "run_id": record.get("run_id"), "git_commit": (record.get("provenance", {}) or {}).get("git_commit", "")[:10],
        "timestamp": record.get("finished_utc") or record.get("started_utc"),
    }


# ----------------------------------------------------------------------------------------------
# Sheets
# ----------------------------------------------------------------------------------------------

ALL_COLUMNS = [Column("run_id"), Column("flow"), Column("variant"), Column("arm"), Column("model"),
               Column("platform"), Column("phase"), Column("context", "int", "int"), Column("layer"), Column("stage"),
               Column("dataset"), Column("metric"), Column("value", "num", "general"), Column("unit"),
               Column("git_commit"), Column("timestamp_utc"), Column("note")]
SPEED_PHASE = {"prefill": "prefill", "decode": "decode", "idle": "idle"}
LAYER_METRICS = [("ms_median", "ms"), ("ms_p10", "ms"), ("ms_p90", "ms"), ("share_of_step", "fraction of step"),
                 ("mul_adds_per_token", "mul/token"), ("nonlinear_per_token", "evals/token"),
                 ("bytes_per_token", "bytes/token")]
STEP_METRICS = [("step_ms_instrumented", "ms"), ("step_ms_uninstrumented", "ms"), ("instrumentation_overhead_pct", "%")]


def all_data(results: Results, timing: list[dict[str, Any]] | None = None) -> Sheet:
    rows: list[list[Any]] = []

    def add(**v: Any) -> None:
        rows.append([v.get(c.header) if c.header != "value" else num(v.get("value")) for c in ALL_COLUMNS])

    for line in results.sweep:
        speed = line["benchmark"] == "speed"
        phase = SPEED_PHASE.get(line["metric"].split("_")[0], "speed") if speed else \
            f"resources: {line['dataset']}" if line["benchmark"] == "resources" else "perplexity"
        note = line.get("note", "")
        if not speed:
            note = "; ".join(x for x in (span_label(char_limit_of(results, line["run_id"], line["dataset"])), note) if x)
        add(run_id=line["run_id"], flow="pytorch", variant=variant_of(line["arm"], line.get("lut_preset", "")),
            arm=line["arm"], model=line["model"], platform=platform_of(line["device"]), phase=phase,
            context=num(line.get("context_tokens")), layer="", stage="", dataset=line.get("dataset"),
            metric=line["metric"], value=line["value"], unit=line["unit"], git_commit=line.get("git_commit"),
            timestamp_utc=line.get("timestamp_utc"), note=note)
    seen_steps = set()
    for line in results.layers:
        common = dict(run_id=line["run_id"], flow=line.get("flow", "pytorch"), variant=line["variant"], arm=line["arm"],
                      model=line["model"], platform=line["platform"], phase=line["phase"], context=num(line["context"]),
                      dataset="synthetic", git_commit=line.get("git_commit"), timestamp_utc=line.get("timestamp_utc"))
        note = line.get("note", "")
        for metric, unit in LAYER_METRICS:
            if line.get(metric) not in (None, ""):
                add(**common, layer=line["layer"], stage=line["stage"], metric=metric, value=line[metric], unit=unit,
                    note=note)
        key = (line["run_id"], line["phase"], line["context"])
        if key not in seen_steps:
            seen_steps.add(key)
            for metric, unit in STEP_METRICS:
                add(**common, layer="model", stage="step", metric=metric, value=line[metric], unit=unit,
                    note=("beyond trained context, timing only" if line.get("beyond_trained_context") == "yes" else ""))
    for line in results.extra:
        metric, unit = line["metric"], line.get("unit")
        if metric == "energy_per_token_mj" and ("_eval_decode_" in line["run_id"] or "_eval_power_" in line["run_id"]):
            metric, unit = "energy_per_scored_token_mj", "mJ per scored token (secondary; primary unit is per decode step)"
        add(run_id=line["run_id"], flow=line.get("flow", "pytorch"), variant=line.get("variant"), arm=line.get("arm"),
            model=line.get("model"), platform=line.get("platform"), phase=line.get("phase"),
            context=num(line.get("context")), layer=line.get("layer"), stage=line.get("stage"), dataset="",
            metric=metric, value=line["value"], unit=unit, git_commit=line.get("git_commit"),
            timestamp_utc=line.get("timestamp_utc"), note=line.get("note"))
    for record in results.hls:
        view = hls_view(record)
        base = dict(run_id=view["run_id"], flow="hls", variant=view["variant"], arm=record.get("arm", ""),
                    model=record.get("model", ""), layer=record.get("layer", ""), stage=view["stage"] or view["kernel"],
                    dataset="", git_commit=view["git_commit"], timestamp_utc=view["timestamp"])
        for platform, info in (record.get("stages") or {}).items():
            add(**base, platform=platform, phase="build", metric="status", value=(info or {}).get("status"), unit="")
        for platform, c in (record.get("compare") or {}).items():
            if isinstance(c, dict):
                add(**base, platform=platform, phase="check", metric="mismatches", value=c.get("mismatches"),
                    unit="elements", note=f"compared {c.get('compared')}")
        for platform, res in (record.get("results") or {}).items():
            for metric, value in (res or {}).items():
                if isinstance(value, dict):
                    for sub, v in value.items():
                        if not isinstance(v, (dict, list)):
                            add(**base, platform=platform, phase="result", metric=f"{metric}.{sub}", value=v, unit="")
                elif not isinstance(value, list):
                    add(**base, platform=platform, phase="result", metric=metric, value=value, unit="")
    for r in power_rows(results):                # per-layer timing runs' resource use (sweep phases come from all_results)
        if r.get("_source") != "layer":
            continue
        for header, kind, _fmt in POWER_COLUMNS[8:-1]:
            value = r.get(header)
            if value is None or header in ("power limit (W)", "enforced limit (W)", "min/max limit (W)", "persistence mode"):
                continue
            add(run_id=r["run_id"], flow="pytorch", variant=r.get("variant"), arm=r.get("arm"), model="stories15m",
                platform=r.get("platform"), phase=f"resources: {r['phase']}", context=None, layer="", stage="",
                dataset="synthetic", metric=header, value=value, unit="", git_commit="", timestamp_utc="",
                note=r.get("note", ""))
    for r in hls_inference_rows(results):        # the HLS kernel's inference on the slice
        if r.get("kind") != "inference":
            continue
        for metric, unit in (("perplexity", "per token"), ("NLL sum (nats)", "nats"), ("top-1", "fraction"),
                             ("top-5", "fraction"), ("top-1 agreement (pipeline)", "fraction"),
                             ("top-5 agreement (pipeline)", "fraction"), ("max |dNLL| per token (pipeline)", "nats"),
                             ("top-1 flags agree (check)", "fraction"), ("top-5 flags agree (check)", "fraction"),
                             ("wall (s)", "s"), ("C model ms per decode step", "ms"), ("FPGA estimate ms per step", "ms"),
                             ("FPGA estimate mJ per step (total power)", "mJ")):
            if r.get(metric) is not None:
                add(run_id=r["run_id"], flow="hls", variant=r["variant"], arm="hls", model="stories15m",
                    platform=r["platform"], phase="inference", context=None, layer="", stage="whole model",
                    dataset=r["dataset"], metric=metric, value=r[metric], unit=unit, git_commit=r.get("git_commit"),
                    timestamp_utc=r.get("timestamp"), note=f"{r['scored_tokens']} scored tokens; {r['match']}")
    for m in memory_rows(results):               # torch peaks etc. of runs other than the memory pass (it writes its own rows)
        if m["source"] == "memory pass":
            continue
        for header in ("torch peak allocated (MiB)", "torch peak reserved (MiB)", "device memory used peak (MiB, all processes)",
                       "host RSS peak (MiB)"):
            if m.get(header) is not None:
                add(run_id=m["run_id"], flow="pytorch", variant=m["variant"], arm=m["arm"], model="stories15m",
                    platform=m["platform"], phase="memory", context=None, layer="", stage=m["phase"], dataset="",
                    metric=header, value=m[header], unit="MiB", git_commit="", timestamp_utc=m["date (UTC)"], note=m["note"])
    for t in timing or []:                       # every timed piece of every run (the Time budget's raw rows)
        note = "; ".join(x for x in (t["basis"], t.get("contended"), t.get("condition"), t.get("note")) if x)
        add(run_id=t["run_id"], flow=t["flow"], variant=t.get("variant"), arm=t.get("arm"), model=t.get("model"),
            platform=t.get("platform"), phase="time", context=None, layer="", stage=t["activity"], dataset="",
            metric=f"seconds ({t['component']})", value=t["seconds"], unit="s", git_commit="",
            timestamp_utc=t.get("timestamp"), note=note)
    rows.sort(key=lambda r: (str(r[1]), str(r[15] or ""), str(r[0])))
    return Sheet("All data", ALL_COLUMNS, rows)


def _latest(lines: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(lines, key=lambda r: (r.get("timestamp_utc", ""), r["run_id"]))


def perplexity_sheet(results: Results) -> Sheet:
    columns = [Column("run_id"), Column("arm"), Column("variant"), Column("platform"), Column("model"),
               Column("dataset"), Column("span"), Column("char limit", "int", "int"), Column("chars", "int", "int"),
               Column("UTF-8 bytes", "int", "int"), Column("words", "int", "int"), Column("scored tokens", "int", "int"),
               Column("window", "int", "int"), Column("stride", "int", "int"), Column("windows", "int", "int"),
               Column("total NLL (nats)", "num", "0.000000"), Column("NLL/token (nats)", "num", "0.000000"),
               Column("perplexity", "num", "0.000000"), Column("bits/byte", "num", "0.000000"),
               Column("word perplexity", "num", "0.0000"), Column("top-1 accuracy", "num", "pct"),
               Column("top-5 accuracy", "num", "pct"), Column("eval time (s)", "num", "0.000"),
               Column("forward ms (median)", "num", "0.000"), Column("dataset sha256"), Column("git_commit"),
               Column("timestamp_utc")]
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for line in results.sweep:
        if line["benchmark"] != "perplexity":
            continue
        g = groups.setdefault((line["run_id"], line["dataset"]), {"line": line, "v": {}})
        g["v"][line["metric"]] = line["value"]
    sheet = Sheet("PyTorch Perplexity", columns)
    for (run_id, dataset), g in sorted(groups.items(), key=lambda kv: (kv[1]["line"]["timestamp_utc"], kv[0])):
        line, v = g["line"], g["v"]
        limit = char_limit_of(results, run_id, dataset)
        sheet.lines.append([run_id, line["arm"], variant_of(line["arm"], line.get("lut_preset", "")),
                            platform_of(line["device"]), line["model"], dataset, span_label(limit), limit,
                            num(v.get("eval_chars")), num(v.get("eval_bytes")), num(v.get("eval_words")),
                            num(v.get("eval_tokens")), num(line.get("eval_window_tokens")),
                            num(line.get("eval_stride_tokens")), num(v.get("eval_windows")), num(v.get("nll_total")),
                            num(v.get("nll_per_token")), num(v.get("perplexity")), num(v.get("bits_per_byte")),
                            num(v.get("word_perplexity")), num(v.get("top1_accuracy")) if "top1_accuracy" in v else NOT_MEASURED_ACC,
                            num(v.get("top5_accuracy")) if "top5_accuracy" in v else NOT_MEASURED_ACC,
                            num(v.get("eval_s")), num(v.get("forward_ms_median")),
                            line.get("dataset_sha256"), line.get("git_commit"), line.get("timestamp_utc")])
    return sheet


def speed_sheet(results: Results) -> Sheet:
    columns = [Column("run_id"), Column("arm"), Column("variant"), Column("platform"), Column("model"),
               Column("phase"), Column("context T", "int", "int"), Column("batch", "int", "int"),
               Column("decode tokens", "int", "int"), Column("repeats", "int", "int"),
               Column("ms median", "num", "0.000"), Column("ms p10", "num", "0.000"), Column("ms p90", "num", "0.000"),
               Column("ms unit"), Column("tokens/s", "num", "#,##0.0"), Column("note"), Column("git_commit"),
               Column("timestamp_utc")]
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for line in results.sweep:
        if line["benchmark"] == "speed":
            g = groups.setdefault((line["run_id"], line["context_tokens"]), {"line": line, "v": {}})
            g["v"][line["metric"]] = line["value"]
    sheet = Sheet("PyTorch Speed", columns)
    for (run_id, ctx), g in sorted(groups.items(), key=lambda kv: (kv[1]["line"]["timestamp_utc"], kv[0][0],
                                                                    int(kv[0][1]))):
        line, v = g["line"], g["v"]
        head = [run_id, line["arm"], variant_of(line["arm"], line.get("lut_preset", "")), platform_of(line["device"]),
                line["model"]]
        for phase, prefix, unit, tps in (("prefill", "prefill_ms", "ms per prefill (time to first token)",
                                          "prefill_tokens_per_s"),
                                         ("decode", "decode_ms", "ms per decoded token", "decode_tokens_per_s")):
            median = v.get(f"{prefix}_median") or v.get("decode_ms_per_token_median" if phase == "decode" else "")
            if median in (None, ""):
                continue
            sheet.lines.append(head + [phase, num(ctx), num(line.get("batch")), num(line.get("decode_tokens")),
                                       num(line.get("repeats")), num(median), num(v.get(f"{prefix}_p10")),
                                       num(v.get(f"{prefix}_p90")), unit, num(v.get(tps)),
                                       "decode context grows from T to T + decode tokens - 1" if phase == "decode" else "",
                                       line.get("git_commit"), line.get("timestamp_utc")])
    train: dict[tuple[str, str], dict[str, Any]] = {}
    for line in results.extra:
        if line.get("phase") == "train":
            g = train.setdefault((line["run_id"], line["variant"]), {"line": line, "v": {}})
            g["v"][line["metric"]] = line["value"]
    for (run_id, variant), g in sorted(train.items(), key=lambda kv: (kv[1]["line"]["timestamp_utc"], kv[0])):
        line, v = g["line"], g["v"]
        sheet.lines.append([run_id, line["arm"], variant, line["platform"], line["model"], "train", num(line["context"]),
                            64, None, 10, num(v.get("train_ms_per_step")), None, None, "ms per training step",
                            num(v.get("train_tokens_per_s")), line.get("note"), line.get("git_commit"),
                            line.get("timestamp_utc")])
    return sheet


LAYER_SHEET = [("run_id", "text", "general"), ("platform", "text", "general"), ("arm", "text", "general"),
               ("variant", "text", "general"), ("phase", "text", "general"), ("context", "int", "int"),
               ("beyond_trained_context", "text", "general"), ("rank in step", "int", "int"), ("layer", "text", "general"),
               ("stage", "text", "general"), ("calls", "int", "int"), ("ms_median", "num", "0.0000"),
               ("ms_p10", "num", "0.0000"), ("ms_p90", "num", "0.0000"), ("share_of_step", "num", "pct"),
               ("step_ms_instrumented", "num", "0.000"), ("step_ms_uninstrumented", "num", "0.000"),
               ("instrumentation_overhead_pct", "num", "0.0"), ("mul_adds_per_token", "num", "#,##0.0"),
               ("nonlinear_per_token", "num", "#,##0.0"), ("bytes_per_token", "num", "#,##0.0"),
               ("tokens_in_step", "int", "int"), ("repeats", "int", "int"), ("method", "text", "general"),
               ("note", "text", "general"), ("git_commit", "text", "general"), ("timestamp_utc", "text", "general")]


def layer_sheet(results: Results) -> Sheet:
    columns = [Column(h, k, f) for h, k, f in LAYER_SHEET]
    lines = list(results.layers)
    order = {"gpu": 0, "cpu": 1}
    lines.sort(key=lambda r: (order.get(r["platform"], 9), r["arm"], r["phase"], int(r["context"]), r["run_id"],
                              -float(r["ms_median"])))
    sheet = Sheet("PyTorch Per-layer timing", columns)
    rank, last = 0, None
    for line in lines:
        key = (line["run_id"], line["phase"], line["context"])
        rank = rank + 1 if key == last else 1
        last = key
        values = {**line, "rank in step": rank}
        sheet.lines.append([cell_value(values.get(h), k) if k != "text" else values.get(h) for h, k, _ in LAYER_SHEET])
    return sheet


def hardware_sheet(results: Results) -> Sheet:
    """LUT/ops_registry.describe for all_base plus bench/opcount.py's modeled cost, one row per stage
    in forward order (embedding first, lm_head last). Layer stages are per layer."""
    points = [("prefill", 256), ("decode", 256)]
    columns = [Column("run_id"), Column("preset"), Column("stage"), Column("runs"), Column("k x b -> out"),
               Column("table bits", "int", "int"), Column("layers"), Column("quant"), Column("calls per step", "int", "int"),
               Column("weight params", "int", "int")]
    for phase, t in points:
        columns += [Column(f"{phase} T{t} mul/token", "num", "#,##0.0"),
                    Column(f"{phase} T{t} nonlinear/token", "num", "#,##0.0"),
                    Column(f"{phase} T{t} INT8 bytes/token", "num", "#,##0.0"),
                    Column(f"{phase} T{t} FP32 bytes/token", "num", "#,##0.0")]
    sheet = Sheet("PyTorch Hardware cost", columns)
    cost_runs = [rid for rid in sorted(results.runs) if "_hw_cost_" in rid]
    if not cost_runs:
        return sheet
    rid = cost_runs[-1]
    summary = results.runs[rid]["summary"]
    order = {name: i for i, name in enumerate(FORWARD_ORDER)}
    for s in sorted(summary.get("stages", []), key=lambda x: order.get(x["stage"], 99)):
        table_bits = 0 if s.get("table_bits") in ("-", None) else int(str(s["table_bits"]).replace(",", ""))
        line = [rid, summary.get("preset"), s["stage"], s.get("runs"), s.get("shape"), table_bits, s.get("layers"),
                s.get("quant"), s.get("calls"), s.get("weight_params_per_layer_or_model")]
        for phase, t in points:
            line += [s.get(f"{phase}_T{t}_{k}") for k in ("mul_adds_per_token", "nonlinear_per_token",
                                                          "bytes_int8_per_token", "bytes_per_token")]
        sheet.lines.append(line)
    totals = summary.get("step_totals", {})
    line = [rid, summary.get("preset"), "step_total (all layers + model stages)", "", "", 0, "", "", None,
            (summary.get("params") or {}).get("total")]
    for phase, t in points:
        tot = totals.get(f"{phase}_T{t}", {})
        n = tot.get("tokens") or 1
        line += [tot.get("mul_adds", 0) / n, tot.get("nonlinear", 0) / n, tot.get("bytes_int8", 0) / n,
                 tot.get("bytes", 0) / n]
    sheet.lines.append(line)
    return sheet


# ----------------------------------------------------------------------------------------------
# Per-stage: modeled cost and measured GPU time per stage, summed over the layers, in forward order
# ----------------------------------------------------------------------------------------------

FORWARD_ORDER = ("embedding", "norm", "qkv_proj", "rope", "attn_scores", "softmax", "attn_values", "out_proj",
                 "ffn_up", "ffn_act", "ffn_gate", "ffn_down", "final_norm", "lm_head")   # = bench/opcount.py
HF_INSIDE_SDPA = ("rope", "attn_scores", "softmax", "attn_values")
PER_STAGE_CONTEXTS = (64, 128, 256)
PER_STAGE_CAVEAT = ("GPU stage time at batch 1 is mostly kernel-launch overhead. It describes the GPU baseline, not "
                    "FPGA cost; use the modeled-cost columns for hardware decisions.")
PER_STAGE_COLUMNS = [
    ("arm", "text", "general"), ("variant", "text", "general"), ("phase", "text", "general"), ("context T", "int", "int"),
    ("stage", "text", "general"), ("calls per step (all layers)", "int", "int"), ("MACs per token", "num", "#,##0.0"),
    ("nonlinear evals per token", "num", "#,##0.0"), ("INT8 bytes per token", "num", "#,##0.0"),
    ("FP32 bytes per token", "num", "#,##0.0"), ("GPU ms median (sum over layers)", "num", "0.0000"),
    ("GPU ms p10", "num", "0.0000"), ("GPU ms p90", "num", "0.0000"), ("share of step", "num", "pct"),
    ("step ms instrumented", "num", "0.000"), ("step ms plain", "num", "0.000"),
    ("accuracy sensitivity", "text", "general"), ("run_id", "text", "general"), ("note", "text", "general")]


def _model_cfg(results: Results) -> Any:
    """The model's shape from the newest run that recorded it (layer timing or modeled-cost run)."""
    from types import SimpleNamespace

    for rid in sorted(results.runs, reverse=True):
        cfg = (results.runs[rid]["meta"].get("configs") or {}).get("model_config")
        if cfg and cfg.get("d_model") and cfg.get("vocab_size"):
            return SimpleNamespace(**cfg, head_dim=cfg["d_model"] // cfg["n_heads"])
    return None


def _per_stage_measured(results: Results) -> dict[tuple[str, str, int], dict[str, Any]]:
    """{(arm, phase, context): {"run_id", "step", "plain", "stages": {stage: [per-repeat ms summed over
    layers]}}} from the newest GPU per-layer timing run of each arm made with the decode-context
    definition of bench/opcount.py (T keys); older runs held T tokens in the cache and are left out."""
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for rid in sorted(results.runs):
        meta = results.runs[rid]["meta"]
        cfg = meta.get("configs") or {}
        if not meta.get("name", "").startswith("layer_timing") or "decode_context" not in cfg:
            continue
        if not str(next(iter((meta.get("devices") or {}).values()), "")).startswith("cuda"):
            continue
        path = results.results_dir / "runs" / rid / "metrics.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("kind") != "layer_timing":
                continue
            sums: dict[str, list[float]] = {}
            for key, values in (record.get("stages") or {}).items():
                stage = key.split(":", 1)[1]
                acc = sums.setdefault(stage, [0.0] * len(values))
                for i, v in enumerate(values):
                    acc[i] += v
            out[(cfg.get("arm"), record["phase"], int(record["context"]))] = {
                "run_id": rid, "step": record.get("timed_ms") or [], "plain": record.get("plain_ms") or [],
                "stages": sums, "variant": cfg.get("variant")}
    return out


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def per_stage_sheet(results: Results) -> Sheet:
    """One row per stage in forward order (embedding first, lm_head last; never sorted by size), for
    prefill and decode at T = 64, 128 and 256, per arm: modeled cost per token summed over the layers
    (bench/opcount.py: MACs, nonlinear evaluations, bytes at INT8 and FP32) and the measured GPU time
    summed over the layers (median, p10, p90 of the per-step sums) with its share of the step. The hf
    arm's attention core is one SDPA call, so it has one attn_core row in place of rope ... attn_values.
    accuracy sensitivity is left for the quantization screen. Bar charts of decode GPU time and decode
    MACs per stage at T = 256 (and prefill) sit to the right of the table."""
    sheet = Sheet("PyTorch Per-stage", [Column(h, k, f) for h, k, f in PER_STAGE_COLUMNS], preamble=[PER_STAGE_CAVEAT])
    cfg = _model_cfg(results)
    if cfg is None:
        return sheet
    from bench import opcount       # standard library only; no torch

    measured = _per_stage_measured(results)
    arms = [a for a in ("baseline", "lut", "hf") if any(k[0] == a for k in measured)] or ["model only"]
    sens = "to be filled from the quantization screen"
    blocks: dict[tuple[str, str, int], tuple[int, int]] = {}
    for arm in arms:
        for phase in ("prefill", "decode"):
            for ctx in PER_STAGE_CONTEXTS:
                costs = opcount.model_costs(cfg, phase, ctx)
                m = measured.get((arm, phase, ctx))
                step = statistics.median(m["step"]) if m and m["step"] else None
                plain = statistics.median(m["plain"]) if m and m["plain"] else None
                stages = [s for s in opcount.FORWARD_ORDER if s in costs]
                if arm == "hf":
                    i = stages.index("rope") if "rope" in stages else stages.index("attn_scores")
                    stages = [s for s in stages if s not in HF_INSIDE_SDPA]
                    stages.insert(i, "attn_core")
                first = len(sheet.lines)
                for stage in stages:
                    if stage == "attn_core":
                        parts = [costs[s] for s in HF_INSIDE_SDPA if s in costs]
                        c = opcount.StageCost("attn_core", sum(p.calls for p in parts), parts[0].tokens,
                                              sum(p.mul_adds for p in parts), sum(p.nonlinear for p in parts),
                                              sum(p.elements for p in parts))
                        note = "hf: rope, scores, softmax and PV in one SDPA call (module minus its projections)"
                    else:
                        c, note = costs[stage], ""
                    series = (m or {}).get("stages", {}).get(stage, []) if m else []
                    med = statistics.median(series) if series else None
                    if not m:
                        note = (note + "; " if note else "") + "not measured yet (GPU pass pending)"
                    sheet.lines.append([arm, (m or {}).get("variant") or ("all_base" if arm == "lut" else arm), phase, ctx,
                                        stage, c.calls, c.mul_adds / c.tokens, c.nonlinear / c.tokens,
                                        c.bytes_int8 / c.tokens, c.bytes / c.tokens, med, _pct(series, 0.1),
                                        _pct(series, 0.9), (med / step) if (med is not None and step) else None, step,
                                        plain, sens, (m or {}).get("run_id"), note or None])
                blocks[(arm, phase, ctx)] = (first, len(sheet.lines))
                for extra in sorted(set((m or {}).get("stages", {})) - set(stages) - set(HF_INSIDE_SDPA)):
                    series = m["stages"][extra]          # measured but not a stage: unattributed, hf's rope_tables
                    med = statistics.median(series) if series else None
                    note = ("step minus stages: heads split/merge, KV append, residual adds, rope table, mask, "
                            "argmax, launch gaps" if extra == "unattributed" else "hf rotary-table module (wiring)")
                    sheet.lines.append([arm, m.get("variant"), phase, ctx, f"{extra} (not a stage)", None, None, None,
                                        None, None, med, _pct(series, 0.1), _pct(series, 0.9),
                                        med / step if (med is not None and step) else None, step, plain, None,
                                        m["run_id"], note])
    chart_arm = next((a for a in ("baseline", "lut", "hf") if a in arms), arms[0])
    ncols = len(PER_STAGE_COLUMNS)
    col = {h: i + 1 for i, (h, _, _) in enumerate(PER_STAGE_COLUMNS)}
    place = 0
    for phase in ("decode", "prefill"):
        block = blocks.get((chart_arm, phase, 256))
        if not block:
            continue
        for value, label in (("GPU ms median (sum over layers)", "GPU time per stage (ms, median)"),
                             ("MACs per token", "MACs per token")):
            sheet.charts.append(Chart(f"{phase}, T = 256, {chart_arm}: {label}", col["stage"], col[value], block[0],
                                      block[1], label, anchor_col=ncols + 1, anchor_row=1 + place * 20))
            place += 1
    return sheet


def runs_sheet(results: Results) -> Sheet:
    columns = [Column("run_id"), Column("name"), Column("started_utc"), Column("finished_utc"), Column("git_commit"),
               Column("git_dirty"), Column("branch"), Column("devices"), Column("torch"), Column("CUDA build"),
               Column("cuDNN", "int", "int"), Column("TF32"), Column("GPU"), Column("driver"), Column("python"),
               Column("OS"), Column("CPU"), Column("model"), Column("char limit", "int", "int"),
               Column("eval tokens (scored)", "int", "int"), Column("notes")]
    sheet = Sheet("PyTorch Runs", columns)
    for rid, run in sorted(results.runs.items()):
        meta, summary = run["meta"], run["summary"]
        system, configs = meta.get("system", {}), meta.get("configs", {})
        torch_info = system.get("torch", {})
        gpu = (system.get("gpus") or [{}])[0]
        row_cfg = configs.get("row", {})
        limit = None
        for info in (configs.get("perplexity") or {}).values():
            limit = info.get("char_limit")
        if limit is None:
            limit = configs.get("char_limit")
        tokens = None
        for key, value in summary.items():
            if key.startswith("perplexity_") and isinstance(value, dict):
                tokens = value.get("scored_tokens")
        tokens = tokens if tokens is not None else summary.get("scored_tokens")
        tf32 = torch_info.get("cuda_matmul_allow_tf32")
        sheet.lines.append([
            rid, meta.get("name"), meta.get("started_utc"), meta.get("finished_utc"),
            (meta.get("git", {}).get("commit") or "")[:10], str(meta.get("git", {}).get("dirty")),
            meta.get("git", {}).get("branch"), json.dumps(meta.get("devices", {}), sort_keys=True),
            torch_info.get("version"), torch_info.get("cuda_build"), torch_info.get("cudnn"),
            None if tf32 is None else ("on" if tf32 else "off"), gpu.get("name"), gpu.get("driver"),
            system.get("python"), system.get("os"), system.get("cpu"),
            row_cfg.get("model") or configs.get("model"), limit, tokens, meta.get("notes")])
    return sheet


HLS_COLUMNS = [Column("kernel"), Column("variant (PyTorch name)"), Column("hls_variant"), Column("profile"),
               Column("stages (base.STAGES)"), Column("platform reached"), Column("platforms passed"), Column("matches reference (tolerance)"),
               Column("mismatches"), Column("latency cycles min", "int", "int"),
               Column("latency cycles max", "int", "int"), Column("latency source"), Column("latency (ns)", "num", "0.00"),
               Column("II"), Column("clock (ns)", "num", "0.000"), Column("clock (MHz)", "num", "0.0"),
               Column("clock source"), Column("latency (us)", "num", "0.0000"), Column("LUT", "int", "int"),
               Column("FF", "int", "int"), Column("DSP", "int", "int"), Column("BRAM", "int", "int"),
               Column("URAM", "int", "int"), Column("LUT %", "num", "0.00"), Column("DSP %", "num", "0.00"),
               Column("dynamic power estimate (W)", "num", "0.000"), Column("power estimate kind"),
               Column("threads"), Column("parallel jobs"), Column("wall (s)", "num", "0.0"), Column("run_id"),
               Column("git_commit"), Column("build")]


def hls_sheet(results: Results) -> Sheet:
    sheet = Sheet("HLS Kernels", HLS_COLUMNS)
    keys = ["kernel", "variant", "hls_variant", "profile", "stage", "platform", "platforms", "bit_exact", "mismatches",
            "latency_cycles_min", "latency_cycles_max", "latency_source", "latency_ns", "ii", "clock_ns", "clock_mhz",
            "clock_source", "latency_us", "LUT", "FF", "DSP", "BRAM", "URAM", "lut_pct", "dsp_pct", "dynamic_power_w",
            "power_kind", "threads", "parallel_jobs", "wall_s", "run_id", "git_commit"]
    for record in results.hls:
        view = hls_view(record)
        reused = reused_build(record)
        sheet.lines.append([view.get(k) for k in keys] + [f"reused {reused}" if reused else "built"])
    return sheet


def comparison_sheet(results: Results) -> Sheet:
    """variant x stage: PyTorch per-stage time (gpu, decode at T = 224, mean over layers) and the
    variant's slice perplexity, next to the HLS kernel for the same variant and stage."""
    columns = [Column("variant"), Column("stage"), Column("PyTorch gpu decode T256 ms per layer", "num", "0.0000"),
               Column("PyTorch cpu decode T256 ms per layer", "num", "0.0000"),
               Column("PyTorch slice perplexity", "num", "0.000000"), Column("HLS kernel"),
               Column("HLS latency (us)", "num", "0.0000"), Column("HLS latency cycles max", "int", "int"),
               Column("HLS clock (MHz)", "num", "0.0"), Column("HLS matches reference (tolerance)"), Column("LUT", "int", "int"),
               Column("FF", "int", "int"), Column("DSP", "int", "int"), Column("BRAM", "int", "int"),
               Column("URAM", "int", "int"), Column("HLS profile"), Column("HLS run_id"), Column("note"),
               Column("PyTorch slice top-1", "num", "pct"), Column("PyTorch slice top-5", "num", "pct"),
               Column("HLS perplexity", "num", "0.000000"), Column("HLS top-1", "num", "pct"), Column("HLS top-5", "num", "pct"),
               Column("scored tokens", "int", "int"), Column("per-token match"),
               Column("PyTorch GPU ms per decode step (same tokens)", "num", "0.0000"),
               Column("HLS FPGA estimate ms per step", "num", "0.00"),
               Column("PyTorch GPU mJ per decode step (same tokens)", "num", "0.0"),
               Column("HLS FPGA estimate mJ per step (total power)", "num", "0.0"),
               Column("HLS FPGA estimate mJ per step (dynamic power)", "num", "0.0"),
               Column("PyTorch GPU mJ per scored token (secondary)", "num", "0.0"),
               Column("HLS FPGA estimate mJ per scored token (total power, secondary)", "num", "0.0"),
               Column("PyTorch decode run_id")]
    sheet = Sheet("Final comparison (PT vs HLS)", columns)
    stage_ms: dict[tuple[str, str, str], list[float]] = {}
    for line in results.layers:
        if line["phase"] == "decode" and line["context"] == "256":
            stage_ms.setdefault((line["variant"], line["stage"], line["platform"]), []).append(float(line["ms_median"]))
    slice_ppl: dict[str, Any] = {}
    for line in results.sweep:
        if line["benchmark"] == "perplexity" and line["metric"] == "perplexity":
            limit = char_limit_of(results, line["run_id"], line["dataset"])
            if limit and int(limit) != 1_200_000:
                slice_ppl[variant_of(line["arm"], line.get("lut_preset", ""))] = num(line["value"])
    def per_layer(variant: str, stage: str, platform: str) -> float | None:
        """Mean over layers of a stage's time; a fused kernel names several stages joined by ';'
        (ffn_up;ffn_act;ffn_gate;ffn_down), whose per-layer means are added."""
        parts = [stage_ms.get((variant, s.strip(), platform)) for s in stage.split(";") if s.strip()]
        if not parts or any(p is None for p in parts):
            return None
        return sum(statistics.fmean(p) for p in parts)

    for record in results.hls:
        view = hls_view(record)
        variant, stage = view["variant"], view["stage"]
        if not stage:
            continue                     # the record doesn't say which base.STAGES it implements
        gpu, cpu = per_layer(variant, stage, "gpu"), per_layer(variant, stage, "cpu")
        if gpu is None and cpu is None and variant not in slice_ppl:
            continue                     # no PyTorch side for this variant yet
        note = ("HLS test-size kernel (profile test): its shapes are not stories15M's, so time and latency don't "
                "compare one to one" if view.get("profile") == "test" else "")
        sheet.lines.append([variant, stage, gpu, cpu, slice_ppl.get(variant), view["kernel"], view["latency_us"],
                            view["latency_cycles_max"], view["clock_mhz"], view["bit_exact"], view["LUT"], view["FF"],
                            view["DSP"], view["BRAM"], view["URAM"], view.get("profile"), view["run_id"], note]
                           + [None] * 15)
    # accuracy: the HLS kernel's inference on the slice next to PyTorch's, same tokens and windows
    slice_acc: dict[str, dict[str, Any]] = {}
    for line in _latest(results.sweep):
        if line["benchmark"] == "perplexity" and line["metric"] in ("top1_accuracy", "top5_accuracy"):
            limit = char_limit_of(results, line["run_id"], line["dataset"])
            if limit and int(limit) != 1_200_000:
                slice_acc.setdefault(variant_of(line["arm"], line.get("lut_preset", "")), {})[line["metric"]] = num(line["value"])
    decode = decode_runs(results)
    for row in hls_inference_rows(results):
        if row.get("kind") != "inference":
            continue
        variant = row["variant"]
        same = row.get("reference_same_tokens") or {}
        n = row.get("scored_tokens") or 0
        if same.get("top1") is not None:      # the FP32 reference on exactly the same tokens (independent check)
            pt_ppl, pt1, pt5 = same.get("perplexity"), same.get("top1"), same.get("top5")
            scope = f"PyTorch FP32 reference on the same {n:,} tokens"
        elif row.get("windows") == 392 and row.get("whole"):
            pt_ppl, pt1, pt5 = slice_ppl.get(variant), slice_acc.get(variant, {}).get("top1_accuracy"), \
                slice_acc.get(variant, {}).get("top5_accuracy")
            scope = "PyTorch over the whole slice"
        else:
            pt_ppl = pt1 = pt5 = None
            scope = "no PyTorch result on these tokens yet"
        gpu = decode.get(("gpu", n, variant)) if row.get("whole") else None
        sheet.lines.append([variant, f"whole model: accuracy on {n:,} tokens", None, None, pt_ppl,
                            row["kernel"], None, None, None, None, None, None, None, None, None, None, row["run_id"],
                            f"{row['platform']}; {row['note']}; {scope}",
                            pt1, pt5, row["perplexity"], row["top1"], row["top5"], n, row["match"],
                            gpu["ms_per_step"] if gpu else None, row.get("FPGA estimate ms per step"),
                            gpu.get("energy_per_step_mj") if gpu else None, row.get("FPGA estimate mJ per step (total power)"),
                            row.get("FPGA estimate mJ per step (dynamic power)"),
                            gpu.get("energy_per_token_mj") if gpu else None,
                            row.get("FPGA estimate mJ per scored token (total power, secondary)"),
                            gpu["run_id"] if gpu else None])
    return sheet


def decode_runs(results: Results) -> dict[tuple[str, int, str], dict[str, Any]]:
    """(platform, scored tokens, variant) -> the newest complete bench/eval_decode.py result: the
    decode-step protocol (one token per step on a KV cache), the FPGA kernel's own."""
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    for rid in sorted(results.runs):
        run = results.runs[rid]
        meta, summary = run["meta"], run["summary"] or {}
        if not meta.get("name", "").startswith("eval_decode") or not summary.get("complete"):
            continue
        device = (meta.get("devices") or {}).get("inference", "")
        for arm in ("hf", "baseline", "lut"):
            info = summary.get(arm)
            if isinstance(info, dict) and info.get("scored_tokens"):
                variant = variant_of(arm, (meta.get("configs") or {}).get("lut_preset") or "")
                out[(platform_of(device), int(info["scored_tokens"]), variant)] = {"run_id": rid, **info}
    return out


HLS_INFERENCE_COLUMNS = [
    ("run_id", "text", "general"), ("kind", "text", "general"), ("variant", "text", "general"), ("kernel", "text", "general"),
    ("platform", "text", "general"), ("dataset", "text", "general"), ("char limit", "int", "int"), ("windows", "int", "int"),
    ("scored tokens", "int", "int"), ("NLL sum (nats)", "num", "0.000000"), ("NLL/token", "num", "0.000000"),
    ("perplexity", "num", "0.000000"), ("top-1", "num", "pct"), ("top-5", "num", "pct"),
    ("top-1 agreement (pipeline)", "num", "pct"), ("top-5 agreement (pipeline)", "num", "pct"),
    ("max |dNLL| per token (pipeline)", "num", "general"), ("top-1 flags agree (check)", "num", "pct"),
    ("top-5 flags agree (check)", "num", "pct"), ("max |dNLL| per token (check)", "num", "general"),
    ("match", "text", "general"), ("wall (s)", "num", "0.0"), ("C model ms per decode step", "num", "0.00"),
    ("FPGA estimate ms per step", "num", "0.00"), ("FPGA estimate mJ per step (total power)", "num", "0.0"),
    ("FPGA estimate mJ per step (dynamic power)", "num", "0.0"),
    ("FPGA estimate mJ per scored token (total power, secondary)", "num", "0.0"),
    ("threads", "int", "int"), ("build", "text", "general"), ("note", "text", "general")]


def hls_inference_rows(results: Results) -> list[dict[str, Any]]:
    """One row per HLS inference record (results/hls/inference, written by the pipeline thread):
    the kernel run on the slice tokens, with the pipeline's comparison against the FP32 GPU reference
    and this side's independent per-token check (bench/check_hls_inference.py, in extra_results.csv)."""
    checks: dict[str, dict[str, Any]] = {}
    for line in results.extra:
        if line.get("flow") == "hls" and line.get("metric", "").startswith("check_"):
            checks.setdefault(line["run_id"], {})[line["metric"]] = num(line["value"])
    out = []
    for item in results.hls_inference:
        s, bench = item["summary"], item["testbench"]
        if bench and not s:
            checks_tb = bench.get("checks") or {}
            out.append({"run_id": item["dir"], "kind": "testbench", "variant": "baseline", "kernel": "transformer",
                        "platform": "testbench (golden, csim)", "match": "all pass" if bench.get("all_pass") else "FAIL",
                        "note": "; ".join(f"{k}: {'pass' if (v or {}).get('pass') else 'fail'}" for k, v in checks_tb.items())})
            continue
        host, comp, timing = s.get("host") or {}, s.get("compare") or {}, s.get("timing") or {}
        est = timing.get("fpga_estimate") or {}
        diff = comp.get("per_token_nll_abs_diff") or {}
        dataset = s.get("dataset", "")
        limit = re.search(r"chars(\d+)", dataset)
        source = est.get("source_run_id", "")
        variant = s.get("variant") or ("baseline" if "baseline" in source else "")
        chk = checks.get(s.get("run_id") or item["dir"], {})
        n = host.get("scored_tokens")
        if chk:
            ok = chk.get("check_top1_flag_disagreements") == 0 and chk.get("check_top5_flag_disagreements") == 0 \
                and chk.get("check_targets_match") == "yes"
            match = (f"{'match' if ok else 'MISMATCH'}: {n - int(chk.get('check_top1_flag_disagreements', 0)):,}/{n:,} "
                     f"top-1 and {n - int(chk.get('check_top5_flag_disagreements', 0)):,}/{n:,} top-5 flags agree "
                     f"(independent check)")
        elif comp:
            match = ("match" if comp.get("top1_disagreements") == 0 and comp.get("top5_disagreements") == 0 else "MISMATCH") \
                + " (pipeline's comparison)"
        else:
            match = "not compared"
        math_note = {"hls": "HLS math library", None: "glibc math"}.get(s.get("c_model_math"), str(s.get("c_model_math")))
        windows = host.get("windows")
        sel = s.get("selection") or {}
        whole = (sel.get("max_windows", -1) in (-1, None) and sel.get("max_steps", -1) in (-1, None)
                 and not sel.get("first_window")) if sel else windows == 392
        out.append({
            "run_id": s.get("run_id") or item["dir"], "kind": "inference", "variant": variant,
            "kernel": s.get("kernel") or "transformer",
            "platform": s.get("platform") or (f"C model ({math_note})" if s.get("backend") == "c_model" else s.get("backend")),
            "dataset": dataset, "char limit": int(limit.group(1)) if limit else None, "windows": windows,
            "scored tokens": n, "scored_tokens": n, "NLL sum (nats)": host.get("nll_sum"), "NLL/token": host.get("nll_per_token"),
            "perplexity": host.get("perplexity"), "top1": host.get("top1"), "top5": host.get("top5"),
            "top-1": host.get("top1"), "top-5": host.get("top5"),
            "top-1 agreement (pipeline)": comp.get("top1_agreement"), "top-5 agreement (pipeline)": comp.get("top5_agreement"),
            "max |dNLL| per token (pipeline)": diff.get("max"),
            "top-1 flags agree (check)": chk.get("check_top1_flag_agreement"),
            "top-5 flags agree (check)": chk.get("check_top5_flag_agreement"),
            "max |dNLL| per token (check)": chk.get("check_nll_abs_diff_max"), "match": match,
            "wall (s)": s.get("wall_seconds"), "C model ms per decode step": host.get("mean_ms_per_step"),
            "FPGA estimate ms per step": est.get("ms_per_step"),
            "FPGA estimate mJ per step (total power)": est.get("energy_mj_per_step_total_power"),
            "FPGA estimate mJ per step (dynamic power)": est.get("energy_mj_per_step_dynamic_power"),
            "FPGA estimate mJ per scored token (total power, secondary)":
                est["energy_mj_per_step_total_power"] * host["decode_steps"] / n
                if est.get("energy_mj_per_step_total_power") and host.get("decode_steps") and n else None,
            "threads": s.get("threads"),
            "build": (f"reused {reused_build(s)}" if reused_build(s) else
                      f"FPGA numbers from build {source}" if source else ""),
            "whole": whole,
            "note": (f"all {windows:,} windows" if whole else f"windows {sel.get('first_window') or 0}.."
                     f"{(sel.get('first_window') or 0) + (windows or 0) - 1} only") +
                    (f"; FPGA numbers are estimates ({est.get('basis')})" if est else ""),
            "timestamp": s.get("finished_utc") or s.get("started_utc"),
            "git_commit": ((s.get("provenance") or {}).get("git_commit") or "")[:10],
            "reference_same_tokens": {
                "perplexity": math.exp(chk["check_reference_nll_sum_same_tokens"] / n)
                if chk.get("check_reference_nll_sum_same_tokens") is not None and n else None,
                "top1": chk.get("check_reference_top1_same_tokens"), "top5": chk.get("check_reference_top5_same_tokens")}})
    return out


def hls_inference_sheet(results: Results) -> Sheet:
    sheet = Sheet("HLS Inference", [Column(h, k, f) for h, k, f in HLS_INFERENCE_COLUMNS])
    for r in hls_inference_rows(results):
        sheet.lines.append([cell_value(r.get(h), k) if k != "text" else r.get(h) for h, k, _ in HLS_INFERENCE_COLUMNS])
    return sheet


# ----------------------------------------------------------------------------------------------
# Time budget: how long each activity takes, from every run's own timing records
# ----------------------------------------------------------------------------------------------

TRAIN_PASS_TOKENS = 0.59e9     # about one pass over TinyStories V2 train (2.23 GB / 3.73 bytes per stories15M token)
SCREEN_CONFIGS = 276
EMULATION_FACTOR = (1.3, 2.0)
HLS_STAGE_ORDER = {"csim": 0, "synth": 1, "cosim": 2, "impl": 3, "hw_emu": 4}


def _iso_seconds(start: Any, end: Any) -> float | None:
    """Seconds between two timestamps (ISO 8601, or the HLS runner's 20261123T230313Z form)."""
    def parse(text: Any) -> datetime | None:
        if not text:
            return None
        text = str(text)
        for fmt in ("%Y%m%dT%H%M%SZ",):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    a, b = parse(start), parse(end)
    return (b - a).total_seconds() if a and b else None


def _speed_parts(results_dir: Path, run_id: str, warmup: int, decode_tokens: int) -> tuple[float, float] | None:
    """(seconds in the timed repeats, reconstructed seconds in the warm-up repeats) of a sweep row,
    from its metrics.jsonl: the timed part is the sum of every recorded prefill and decode time; the
    warm-up is warmup x (median prefill + decode_tokens x median decode step) per context length."""
    path = results_dir / "runs" / run_id / "metrics.jsonl"
    if not path.is_file():
        return None
    timed = warm = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "speed":
            continue
        prefill, decode = record.get("prefill_ms") or [], record.get("decode_ms") or []
        timed += (sum(prefill) + sum(decode)) / 1000.0
        if prefill:
            step = statistics.median(prefill) + decode_tokens * (statistics.median(decode) if decode else 0.0)
            warm += warmup * step / 1000.0
    return timed, warm


def hls_condition(summary: dict[str, Any]) -> tuple[Any, Any, str]:
    """(threads, parallel jobs, condition label) of one HLS run, so builds that ran at Vivado's
    default 8 threads or next to other jobs are not mixed with 16-thread solo builds."""
    threads, jobs = summary.get("threads"), summary.get("parallel_jobs")
    vti = (summary.get("provenance") or {}).get("vivado_threads_init")
    parts = [f"threads {threads}" if threads else "threads unknown"]
    parts.append("Vivado 16" if vti is True else "Vivado 8 (init not installed)" if vti is False else "Vivado threads unknown")
    parts.append("solo" if jobs == 1 else f"{jobs} jobs at once" if jobs else "parallel jobs unknown")
    solo16 = threads == 16 and jobs == 1 and vti is True
    return threads, jobs, ("16-thread solo: " if solo16 else "") + ", ".join(parts)


def timing_rows(results: Results) -> list[dict[str, Any]]:
    """One row per timed piece of every run, PyTorch and HLS (seconds). basis: measured (a duration
    the run recorded), derived (summed from recorded samples) or reconstructed (approximate, from
    logs or medians)."""
    rows: list[dict[str, Any]] = []
    for run_id, run in sorted(results.runs.items()):
        meta, summary = run["meta"], run["summary"]
        name, cfg = meta.get("name", ""), meta.get("configs", {})
        device = next(iter((meta.get("devices") or {}).values()), "")
        base = {"flow": "pytorch", "run_id": run_id, "platform": platform_of(device) if device else "none",
                "threads": (meta.get("system", {}).get("torch", {}) or {}).get("num_threads"), "parallel_jobs": 1,
                "condition": "", "timestamp": meta.get("finished_utc") or meta.get("started_utc"),
                "model": cfg.get("model") or (cfg.get("row") or {}).get("model") or "stories15m"}
        total = _iso_seconds(meta.get("started_utc"), meta.get("finished_utc"))
        if name.startswith("sweep_"):
            row = cfg.get("row", {})
            variant = variant_of(row.get("arm", ""), row.get("lut_preset") or "")
            common = {**base, "variant": variant, "arm": row.get("arm"), "model": summary.get("model", base["model"])}
            limits = {ds: (info or {}).get("char_limit") for ds, info in (cfg.get("perplexity") or {}).items()}
            slice_row = any(lim and int(lim) != 1_200_000 for lim in limits.values())
            activity = "sweep row (slice perplexity only)" if slice_row else "sweep row (speed + full-span perplexity)"
            rows.append({**common, "activity": activity, "component": "row total", "seconds": summary.get("row_s"),
                         "basis": "measured"})
            if summary.get("load_s"):
                rows.append({**common, "activity": "model load", "component": "load", "seconds": summary["load_s"],
                             "basis": "measured"})
            for key, value in summary.items():
                if key.startswith("perplexity_") and isinstance(value, dict):
                    span = "slice config (187k chars)" if slice_row else "full span (1.2M chars, 321,526 tokens)"
                    rows.append({**common, "activity": f"perplexity, {span}", "component": "windows (eval)",
                                 "seconds": value.get("eval_s"), "basis": "measured"})
                    rows.append({**common, "activity": "tokenize the evaluation text", "component": "tokenize",
                                 "seconds": value.get("tokenize_s"), "basis": "measured",
                                 "note": "paid once per sweep (cached), repeated in every row's record"})
            if "speed" in (row.get("benchmarks") or []):
                parts = _speed_parts(results.results_dir, run_id, int(row.get("warmup") or 0),
                                     int(row.get("decode_tokens") or 0))
                if parts:
                    label = (f"speed row ({len(row.get('context_lengths') or [])} contexts x "
                             f"{(row.get('warmup') or 0) + (row.get('repeats') or 0)} repeats x "
                             f"{1 + (row.get('decode_tokens') or 0)} steps)")
                    rows.append({**common, "activity": label, "component": "timed repeats", "seconds": parts[0],
                                 "basis": "derived"})
                    rows.append({**common, "activity": label, "component": "warm-up repeats", "seconds": parts[1],
                                 "basis": "reconstructed", "note": "warm-up count x median repeat"})
                    rows.append({**common, "activity": label, "component": "timed + warm-up",
                                 "seconds": parts[0] + parts[1], "basis": "reconstructed"})
            continue
        variant = cfg.get("variant") or cfg.get("preset") or ""
        if name.startswith("reference_nll"):
            n = summary.get("scored_tokens")
            activity = f"reference NLL ({n:,} tokens, arms x 2 passes, saved)" if n else "reference NLL (3 arms x 2 passes, saved)"
        elif name.startswith(("eval_power", "eval_decode")):
            decode = name.startswith("eval_decode")
            arms = [a for a in ("hf", "baseline", "lut") if isinstance(summary.get(a), dict)]
            if decode:
                activity = (f"GPU inference, decode-step protocol ({summary.get('scored_tokens') or 0:,} tokens, "
                            f"{summary.get('decode_steps') or 0:,} steps per arm)")
            else:
                activity = f"GPU inference, window-forward protocol ({summary.get('scored_tokens') or 0:,} tokens, repeated passes)"
            for arm in arms:
                info = summary[arm]
                var = variant_of(arm, cfg.get("lut_preset") or "")
                if decode:
                    rows.append({**base, "variant": var, "arm": arm, "activity": activity, "component": "decode pass (all steps)",
                                 "seconds": info.get("decode_s"), "basis": "measured",
                                 "note": f"{info.get('ms_per_step', 0):.4f} ms per step"})
                else:
                    rows.append({**base, "variant": var, "arm": arm, "activity": activity, "component": "one pass (median)",
                                 "seconds": info.get("pass_s_median"), "basis": "measured",
                                 "note": f"{info.get('passes')} passes after the warm-up"})
        elif name.startswith("layer_timing"):
            activity = "per-layer timing (6 points x 35 repeats x 2 steps)"
            variant = cfg.get("variant", "")
        elif name.startswith("train_throughput"):
            activity = "training throughput run (3 setups x 13 steps)"
            for setup, arm, var in (("ours", "baseline", "baseline"), ("hf_fp32", "hf", "hf"),
                                    ("hf_bf16", "hf", "hf_bf16_tf32")):
                info = summary.get(setup) or {}
                if info.get("ms_step"):
                    common = {**base, "variant": var, "arm": arm}
                    rows.append({**common, "activity": "training step (64 x 256 tokens)", "component": setup,
                                 "seconds": info["ms_step"] / 1000.0, "basis": "measured"})
                    rows.append({**common, "activity": "training per 1M tokens", "component": setup,
                                 "seconds": 1e6 / info["tok_s"], "basis": "derived",
                                 "note": "1e6 / tokens per second (batch 64 x 256, random tokens)"})
        elif name.startswith("hw_cost"):
            activity = "modeled cost (no GPU)"
        else:
            activity = name
        rows.append({**base, "variant": variant, "arm": cfg.get("arm", ""), "activity": activity,
                     "component": "run total (meta start to finish)", "seconds": total, "basis": "measured",
                     "note": "excludes python start, imports and the model load before the run opened"})
        if isinstance(summary.get("load_s"), (int, float)):          # scripts that time their own model load
            rows.append({**base, "variant": variant, "arm": cfg.get("arm", ""), "activity": "model load",
                         "component": "load", "seconds": summary["load_s"], "basis": "measured"})
        for key in ("main_wall_s", "main_wall_s_so_far"):
            if isinstance(summary.get(key), (int, float)):
                rows.append({**base, "variant": variant, "arm": cfg.get("arm", ""), "activity": f"{activity}: main() wall",
                             "component": key, "seconds": summary[key], "basis": "measured",
                             "note": "from main() start; python start and imports excluded"})
    for sweep_id, status in results.sweeps.items():
        rows.append({"flow": "pytorch", "run_id": sweep_id, "platform": "", "variant": "", "arm": "all rows",
                     "model": "", "activity": "sweep rows (status.json elapsed)", "component": "rows",
                     "seconds": status.get("elapsed_s"), "basis": "measured", "threads": None, "parallel_jobs": 1,
                     "condition": "", "timestamp": status.get("updated_utc"),
                     "note": f"planned {status.get('planned_total_s')} s"})
    for line in results.extra:
        if line.get("phase") == "time":
            rows.append({"flow": line.get("flow", "pytorch"), "run_id": line["run_id"], "platform": line.get("platform"),
                         "variant": line.get("variant"), "arm": line.get("arm"), "model": line.get("model"),
                         "activity": line.get("stage"), "component": line.get("metric"), "seconds": num(line["value"]),
                         "basis": "reconstructed", "threads": None, "parallel_jobs": 1, "condition": "",
                         "timestamp": line.get("timestamp_utc"), "note": line.get("note")})
    for item in results.hls_all:
        summary, ledger = item["summary"], item["ledger"]
        record = merged_hls(summary, ledger)
        threads, jobs, condition = hls_condition(summary)
        base = {"flow": "hls", "run_id": record.get("run_id") or item["dir"], "model": record.get("kernel"),
                "variant": record.get("variant") or "", "arm": record.get("hls_variant") or "",
                "threads": threads, "parallel_jobs": jobs, "condition": condition,
                "timestamp": summary.get("finished_utc") or summary.get("started_utc"),
                "note": f"profile {record.get('profile')}" if record.get("profile") else ""}
        reused = reused_build(summary, ledger)
        reused_rows = 0
        for stage, info in (summary.get("stages") or {}).items():
            info = info or {}
            # a stage the build cache served (run.py marks it reused: true, reused_from, original_seconds, seconds 0),
            # or a build stage of a record marked build_reused as a whole: no work ran, so 0 s here; the original
            # run keeps its time (original_seconds is never added)
            stage_reused = info.get("reused") is True or (reused and stage in BUILD_STAGES and info.get("reused") is not False)
            if stage_reused:
                source = info.get("reused_from") or reused or "unknown build"
                rows.append({**base, "platform": stage, "activity": f"HLS {stage} (reused build)", "component": stage,
                             "seconds": 0.0, "basis": "reused", "status": f"reused {source}", "reused": True,
                             "note": f"reused {source}; the build time is counted once, on that run"})
                reused_rows += 1
                continue
            seconds = info.get("seconds")
            if seconds is None:
                seconds = _iso_seconds(info.get("started_utc"), info.get("finished_utc"))
            rows.append({**base, "platform": stage, "activity": f"HLS {stage}", "component": stage, "seconds": seconds,
                         "basis": "measured", "status": info.get("status")})
            steps = info.get("steps")
            for step_name, step in (steps.items() if isinstance(steps, dict) else
                                    ((s.get("name", str(i)), s) for i, s in enumerate(steps or []))):
                step = step if isinstance(step, dict) else {"seconds": step}
                rows.append({**base, "platform": stage, "activity": f"HLS {stage}: {step_name}", "component": step_name,
                             "seconds": step.get("seconds") or _iso_seconds(step.get("started_utc"), step.get("finished_utc")),
                             "basis": "measured"})
        if reused and not reused_rows:
            rows.append({**base, "platform": "build", "activity": "HLS build (reused build)", "component": "build",
                         "seconds": 0.0, "basis": "reused", "status": f"reused {reused}", "reused": True,
                         "note": f"reused {reused}; the build time is counted once, on that run"})
        if summary.get("wall_seconds"):
            rows.append({**base, "platform": "all", "activity": "HLS runner, whole run (wall)", "component": "wall",
                         "seconds": summary["wall_seconds"], "basis": "measured"})
        elif summary.get("started_utc") and summary.get("finished_utc"):
            rows.append({**base, "platform": "all", "activity": "HLS runner, whole run (wall)", "component": "wall",
                         "seconds": _iso_seconds(summary["started_utc"], summary["finished_utc"]),
                         "basis": "reconstructed", "note": (base["note"] + "; " if base["note"] else "")
                         + "from started_utc and finished_utc (whole seconds)"})
        if ledger.get("seconds"):
            rows.append({**base, "platform": "all", "activity": "HLS ladder call (convert + C model + runner)",
                         "component": "ladder", "seconds": ledger["seconds"], "basis": "measured"})
        for key, label in (("convert_s", "hls4ml convert"), ("cmodel_s", "C model check")):
            value = (ledger.get("times") or {}).get(key)
            if value is not None:
                rows.append({**base, "platform": "all", "activity": f"HLS {label}", "component": key,
                             "seconds": value, "basis": "measured"})
    return [r for r in rows if isinstance(r.get("seconds"), (int, float))]


def _fmt_span(lo: float, hi: float) -> str:
    def one(s: float) -> str:
        return f"{s:,.1f} s" if s < 120 else f"{s / 60:,.1f} min" if s < 7200 else f"{s / 3600:,.2f} h"
    return one(lo) if abs(hi - lo) < 1e-9 else f"{one(lo)} to {one(hi)}"


def _parse_time(text: Any) -> datetime | None:
    if not text:
        return None
    text = str(text)
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


HLS_BUILD_ROOT = Path(os.environ.get("HLS_BUILD_ROOT", REPO_ROOT.parent / "LUT-Transformer-local" / "hls-builds"))
IN_PROGRESS_WINDOW_S = 600      # an unrecorded build whose logs changed this recently counts as still running


def build_folder_intervals(recorded: set[str], now: datetime) -> list[dict[str, Any]]:
    """HLS builds that have no results/hls/runs record yet (still running, or never recorded), from
    the build folders' names (UTC start) and the newest modification time of any file in them (read
    only). A build with a live process naming its folder, or a file changed within IN_PROGRESS_WINDOW_S
    of now, is taken to run until now (the moment the workbook is built)."""
    out = []
    if not HLS_BUILD_ROOT.is_dir():
        return out
    cmdlines = []                                  # live processes (Linux /proc): a build still running names its folder
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            cmdlines.append(proc.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace"))
        except OSError:
            pass
    for folder in sorted(HLS_BUILD_ROOT.glob("*/runs/*")):
        if not folder.is_dir() or folder.name in recorded:
            continue
        start = _parse_time(folder.name[:16])
        if start is None:
            continue
        newest = folder.stat().st_mtime           # newest change anywhere in the build (Vivado's own run logs too)
        for root, _dirs, files in os.walk(folder):
            for name in files:
                try:
                    newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
                except OSError:
                    pass
        last = datetime.fromtimestamp(newest, tz=timezone.utc)
        running = (now - last).total_seconds() < IN_PROGRESS_WINDOW_S or any(str(folder) in c for c in cmdlines)
        out.append({"job": folder.name, "start": start, "end": now if running else last, "flow": "hls",
                    "label": f"{folder.parent.parent.name} {folder.name}" + (" (running, no record yet)" if running
                                                                            else " (no record)")})
    return out


def attach_intervals(results: Results, rows: list[dict[str, Any]]) -> None:
    """Give every timed row a (start, end) in UTC: a PyTorch run's meta start and finish (a process
    wall ends at its run's or sweep's end and starts its length earlier); an HLS stage its own
    started_utc and finished_utc, else the run's start plus the stages before it (reconstructed);
    other HLS rows the run's start and finish."""
    run_span = {rid: (_parse_time(r["meta"].get("started_utc")), _parse_time(r["meta"].get("finished_utc")))
                for rid, r in results.runs.items()}
    sweep_span: dict[str, list[datetime]] = {}
    for rid, r in results.runs.items():
        sid = (r["meta"].get("configs") or {}).get("sweep_id")
        a, b = run_span[rid]
        if sid and a and b:
            sweep_span.setdefault(sid, []).extend([a, b])
    hls_span: dict[str, dict[str, tuple[datetime | None, datetime | None]]] = {}
    for item in results.hls_all:
        summary = item["summary"]
        rid = (item["ledger"] or {}).get("run_id") or summary.get("run_id") or item["dir"]
        start, end = _parse_time(summary.get("started_utc")), _parse_time(summary.get("finished_utc"))
        spans = {"_run": (start, end)}
        cursor = start
        for stage, info in (summary.get("stages") or {}).items():
            info = info or {}
            a, b = _parse_time(info.get("started_utc")), _parse_time(info.get("finished_utc"))
            if a is None and cursor is not None and info.get("seconds") is not None:
                a = cursor
                b = cursor + timedelta(seconds=float(info["seconds"]))
            spans[stage] = (a, b)
            cursor = b or cursor
        hls_span[rid] = spans
    for r in rows:
        if r["flow"] == "hls":
            spans = hls_span.get(r["run_id"], {})
            r["start"], r["end"] = spans.get(r.get("platform") if r.get("platform") in spans else "_run", (None, None))
        elif r["run_id"] in run_span:
            r["start"], r["end"] = run_span[r["run_id"]]
        elif r["run_id"] in sweep_span:
            r["start"], r["end"] = min(sweep_span[r["run_id"]]), max(sweep_span[r["run_id"]])
        else:
            r["start"] = r["end"] = None
        if r.get("component") == "process_wall_s" and r.get("end"):
            r["start"] = r["end"] - timedelta(seconds=float(r["seconds"]))


def mark_contention(results: Results, rows: list[dict[str, Any]], now: datetime | None = None) -> None:
    """contended = another heavy job overlapped the row in time: any other HLS run's stage (recorded,
    or an unrecorded build folder), or, for HLS rows, any of our PyTorch benchmark runs. PyTorch rows
    are only checked against HLS work: our own runs ran one after another in one session."""
    now = now or datetime.now(timezone.utc)
    attach_intervals(results, rows)
    heavy = []
    for r in rows:
        if r.get("start") is None or r.get("end") is None:
            continue
        if r.get("reused"):
            continue                     # a reused build did no work
        if r["flow"] == "hls" and r["component"] in HLS_STAGE_ORDER:
            heavy.append({"job": r["run_id"], "start": r["start"], "end": r["end"], "flow": "hls",
                          "label": f"{r['model']} {r['run_id']} {r['component']}"})
        elif r["flow"] == "pytorch" and r["component"] in ("run total (meta start to finish)", "row total"):
            heavy.append({"job": r["run_id"], "start": r["start"], "end": r["end"], "flow": "pytorch",
                          "label": f"pytorch {r['run_id']}"})
    recorded = {item["dir"] for item in results.hls_all} | {str((item["ledger"] or {}).get("run_id")) for item in results.hls_all}
    heavy += build_folder_intervals(recorded, now)
    for r in rows:
        if r.get("start") is None or r.get("end") is None:
            r["contended"] = "unknown (no start/finish)"
            continue
        hits = []
        for h in heavy:
            if h["job"] == r["run_id"] or (r["flow"] == "pytorch" and h["flow"] == "pytorch"):
                continue
            overlap = (min(r["end"], h["end"]) - max(r["start"], h["start"])).total_seconds()
            if overlap > 1.0:
                hits.append(h["label"])
        r["contended"] = ("contended: " + "; ".join(sorted(set(hits)))[:200]) if hits else "uncontended"


def time_budget_sheet(results: Results, rows: list[dict[str, Any]]) -> Sheet:
    columns = [Column("flow"), Column("activity"), Column("kernel / model"), Column("variant / arm"),
               Column("platform / device"), Column("threads"), Column("parallel jobs"), Column("condition"),
               Column("contention"), Column("runs", "int", "int"), Column("median (s)", "num", "0.000"),
               Column("min (s)", "num", "0.000"),
               Column("max (s)", "num", "0.000"), Column("projection"), Column("projection formula"),
               Column("basis"), Column("run ids"), Column("note")]
    sheet = Sheet("Time budget", columns)
    skip = {"warm-up repeats", "timed repeats", "tokenize"}       # their totals are listed; details stay in All data
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r["flow"] == "pytorch" and r["component"] in skip:
            continue
        flag = "uncontended" if r.get("contended") == "uncontended" else "contended" \
            if str(r.get("contended", "")).startswith("contended") else "unknown"
        if r["flow"] == "hls":
            key = ("hls", r["activity"], r["model"], "", r["platform"], r["condition"], flag)
        else:
            key = ("pytorch", r["activity"], "stories15m", r.get("variant") or r.get("arm") or "", r["platform"], "", flag)
        groups.setdefault(key, []).append(r)
    lines = []
    for (flow, activity, model, variant, platform, condition, flag), items in groups.items():
        values = [i["seconds"] for i in items]
        threads = sorted({str(i.get("threads")) for i in items if i.get("threads") is not None}) or ["unknown"]
        jobs = sorted({str(i.get("parallel_jobs")) for i in items if i.get("parallel_jobs") is not None}) or ["unknown"]
        ids = [i["run_id"] for i in items]
        bases = sorted({i["basis"] for i in items})
        notes = sorted({i.get("note") or "" for i in items} - {""})
        if flag == "contended":
            notes = sorted({i["contended"][len("contended: "):] for i in items}) + notes
        if flow == "hls":
            variant = ";".join(sorted({i.get("variant") or "-" for i in items}))
        lines.append([flow, activity, model, variant, platform, ";".join(threads), ";".join(jobs), condition, flag, len(values),
                      statistics.median(values), min(values), max(values), None, None, "; ".join(bases),
                      "; ".join(ids[:3]) + (f"; ... ({len(ids)} runs)" if len(ids) > 3 else ""),
                      " | ".join(notes)[:300]])

    def median_of(flow: str, activity: str, variant: str | None = None, component: str | None = None) -> float | None:
        """Median of the uncontended rows, or of all rows when none is uncontended."""
        match = [r for r in rows if r["flow"] == flow and r["activity"] == activity
                 and (variant is None or r.get("variant") == variant) and (component is None or r["component"] == component)]
        clean = [r["seconds"] for r in match if r.get("contended") == "uncontended"]
        vals = clean or [r["seconds"] for r in match]
        return statistics.median(vals) if vals else None

    projections = []
    slice_lut = median_of("pytorch", "perplexity, slice config (187k chars)", "all_base", "windows (eval)")
    slice_fp32 = median_of("pytorch", "perplexity, slice config (187k chars)", "baseline", "windows (eval)")
    if slice_lut:
        lo, hi = (SCREEN_CONFIGS * slice_lut * f for f in EMULATION_FACTOR)
        projections.append(["pytorch", f"{SCREEN_CONFIGS}-config quantization screen (slice, NLL only)", "stories15m",
                            "all_base -> quantized configs", "gpu", None, "1", "", "uncontended medians", None,
                            slice_lut, None, None,
                            _fmt_span(lo, hi), f"{SCREEN_CONFIGS} x {slice_lut:.3f} s x {EMULATION_FACTOR[0]} to "
                            f"{EMULATION_FACTOR[1]} (emulation factor, assumed)", "projection", "", ""])
        if slice_fp32:
            projections.append(["pytorch", f"{SCREEN_CONFIGS}-config screen with KL (FP32 recomputed per config)",
                                "stories15m", "all_base + baseline FP32", "gpu", None, "1", "", "uncontended medians", None,
                                slice_lut + slice_fp32, None, None, _fmt_span(lo + SCREEN_CONFIGS * slice_fp32,
                                                                              hi + SCREEN_CONFIGS * slice_fp32),
                                f"previous + {SCREEN_CONFIGS} x {slice_fp32:.3f} s (one FP32 baseline pass per config)",
                                "projection", "", "sharing one FP32 pass across configs brings it back to the NLL-only line"])
    for setup, variant in (("ours", "baseline"), ("hf_fp32", "hf"), ("hf_bf16", "hf_bf16_tf32")):
        per_m = median_of("pytorch", "training per 1M tokens", variant, setup)
        if per_m:
            projections.append(["pytorch", f"training, one TinyStories-sized pass ({TRAIN_PASS_TOKENS / 1e9:.2f}B tokens)",
                                "stories15m", f"{variant} ({setup})", "gpu", None, "1", "", "uncontended medians", None,
                                per_m, None, None,
                                _fmt_span(per_m * TRAIN_PASS_TOKENS / 1e6, per_m * TRAIN_PASS_TOKENS / 1e6),
                                f"{TRAIN_PASS_TOKENS / 1e6:,.0f} x {per_m:.3f} s per 1M tokens (batch 64 x 256)",
                                "projection", "", "TinyStories V2 train = 2,227,753,162 bytes at about 3.73 bytes per "
                                "stories15M token (validation ratio); not downloaded"])
    hls_kernels = sorted({r["model"] for r in rows if r["flow"] == "hls" and r["model"]})
    for kernel in hls_kernels:
        med, used_clean = {}, {}
        for s_name in HLS_STAGE_ORDER:
            match = [r for r in rows if r["flow"] == "hls" and r["model"] == kernel and r["activity"] == f"HLS {s_name}"]
            if not match:
                continue
            clean = [r["seconds"] for r in match if r.get("contended") == "uncontended"]
            med[s_name] = statistics.median(clean or [r["seconds"] for r in match])
            used_clean[s_name] = bool(clean)
        chain, total = [], 0.0
        for stage in ("csim", "synth", "cosim", "impl"):
            if stage not in med:
                break
            chain.append(stage)
            total += med[stage]
            dirty = [s_name for s_name in chain if not used_clean[s_name]]
            projections.append(["hls", f"HLS design point up to {stage}", kernel, "", "+".join(chain), None, None,
                                "medians over every thread setting (see the per-stage rows)",
                                "uncontended medians" if not dirty else f"contended only for {', '.join(dirty)}", None,
                                total, None, None, _fmt_span(total, total),
                                " + ".join(f"{s_name} {med[s_name]:.1f} s" for s_name in chain), "projection", "", ""])
    order = {"pytorch": 0, "hls": 1}
    lines.sort(key=lambda l: (order[l[0]], l[2] or "", HLS_STAGE_ORDER.get(str(l[4]), 9) if l[0] == "hls" else 0,
                              l[1], str(l[3]), str(l[7]), str(l[8]) != "uncontended"))
    sheet.lines = lines + projections
    return sheet


POWER_COLUMNS = [
    ("run_id", "text", "general"), ("kind", "text", "general"), ("arm", "text", "general"), ("variant", "text", "general"),
    ("platform", "text", "general"), ("phase", "text", "general"), ("context T", "int", "int"),
    ("power source", "text", "general"), ("avg power (W)", "num", "0.0"), ("steady avg power (W)", "num", "0.0"),
    ("peak power (W)", "num", "0.0"), ("energy per token (mJ)", "num", "0.000"), ("idle power (W)", "num", "0.0"),
    ("warm-up (s)", "num", "0.00"), ("warm-up settled", "text", "general"), ("sample interval (ms)", "int", "int"),
    ("window (s)", "num", "0.00"), ("GPU util mean (%)", "num", "0.0"), ("GPU util max (%)", "num", "0.0"),
    ("mem-ctrl util mean (%)", "num", "0.0"), ("mem-ctrl util max (%)", "num", "0.0"),
    ("device memory used max (MiB, all processes)", "num", "int"), ("torch peak allocated (MiB)", "num", "0.0"),
    ("torch peak reserved (MiB)", "num", "0.0"), ("SM clock mean (MHz)", "num", "int"), ("mem clock mean (MHz)", "num", "int"),
    ("temperature max (C)", "num", "int"), ("throttled", "text", "general"), ("clock event reasons", "text", "general"),
    ("P-states", "text", "general"), ("process CPU user (s)", "num", "0.00"), ("process CPU system (s)", "num", "0.00"),
    ("process CPU cores mean", "num", "0.00"), ("host CPU busy (%)", "num", "0.0"), ("process RSS peak (MiB)", "num", "int"),
    ("power limit (W)", "text", "general"), ("enforced limit (W)", "text", "general"),
    ("min/max limit (W)", "text", "general"), ("persistence mode", "text", "general"), ("note", "text", "general")]


def _res_values(res: dict[str, Any]) -> dict[str, Any]:
    """bench/resources.py summary -> the Power & resources columns."""
    return {"avg power (W)": res.get("power_draw_mean"), "steady avg power (W)": res.get("power_w_mean_steady"),
            "peak power (W)": res.get("power_draw_max"), "energy per token (mJ)": res.get("energy_per_token_mj"),
            "sample interval (ms)": res.get("interval_ms"), "window (s)": res.get("window_s"),
            "GPU util mean (%)": res.get("utilization_gpu_mean"), "GPU util max (%)": res.get("utilization_gpu_max"),
            "mem-ctrl util mean (%)": res.get("utilization_memory_mean"),
            "mem-ctrl util max (%)": res.get("utilization_memory_max"),
            "device memory used max (MiB, all processes)": res.get("memory_used_max"),
            "torch peak allocated (MiB)": res.get("torch_peak_allocated_mib"),
            "torch peak reserved (MiB)": res.get("torch_peak_reserved_mib"), "SM clock mean (MHz)": res.get("clocks_sm_mean"),
            "mem clock mean (MHz)": res.get("clocks_mem_mean"), "temperature max (C)": res.get("temperature_gpu_max"),
            "throttled": None if "throttled" not in res else ("yes" if res["throttled"] else "no"),
            "clock event reasons": res.get("clock_event_reasons_seen"), "P-states": res.get("pstates_seen"),
            "process CPU user (s)": res.get("process_cpu_user_s"), "process CPU system (s)": res.get("process_cpu_system_s"),
            "process CPU cores mean": res.get("process_cpu_cores_mean"), "host CPU busy (%)": res.get("host_cpu_busy_pct"),
            "process RSS peak (MiB)": res.get("process_rss_peak_mib"),
            "power source": "nvidia-smi power.draw (ResourceSampler)" if res.get("power_draw_mean") is not None else None}


def _settings_values(settings: dict[str, Any]) -> dict[str, Any]:
    if not settings:
        return {}
    return {"power limit (W)": settings.get("power.limit"), "enforced limit (W)": settings.get("enforced.power.limit"),
            "min/max limit (W)": f"{settings.get('power.min_limit')}/{settings.get('power.max_limit')}",
            "persistence mode": settings.get("persistence_mode")}


def power_rows(results: Results) -> list[dict[str, Any]]:
    """One row per run and phase: power, energy and resource use, from the runs' summary.json. Runs
    made before the power-ramp fix and the resource sampler get one row saying so."""
    out = []
    for rid, run in sorted(results.runs.items()):
        meta, summary = run["meta"], run["summary"]
        name, cfg = meta.get("name", ""), meta.get("configs", {})
        device = next(iter((meta.get("devices") or {}).values()), "")
        settings = _settings_values(summary.get("gpu_settings") or {})
        base = {"run_id": rid, "platform": platform_of(device) if device else "", "_source": name.split("_")[0]}
        rows_here = []
        if name.startswith("sweep_"):
            row = cfg.get("row", {})
            base.update(kind="sweep", arm=row.get("arm"), variant=variant_of(row.get("arm", ""), row.get("lut_preset") or ""))
            idle = next(iter((summary.get("idle_power_w") or {}).values()), None) if isinstance(summary.get("idle_power_w"), dict) \
                else summary.get("idle_power_w")
            for key, value in summary.items():
                if key.startswith("resources_") and isinstance(value, dict):
                    phase = key[len("resources_"):]
                    label = "speed (all contexts)" if phase == "speed" else f"perplexity eval ({phase})"
                    rows_here.append({**base, "phase": label, **_res_values(value), "idle power (W)": idle, **settings})
                if key.startswith("T") and key[1:].isdigit() and isinstance(value, dict):
                    for phase, sources in (value.get("power") or {}).items():
                        for source, numbers in sources.items():
                            if not isinstance(numbers, dict) or "avg_power_w" not in numbers:
                                continue
                            rows_here.append({**base, "phase": f"{phase} power window", "context T": int(key[1:]),
                                              "power source": f"{source} (PowerMonitor, trapezoid)",
                                              "avg power (W)": numbers["avg_power_w"],
                                              "energy per token (mJ)": numbers.get("energy_per_token_mj"),
                                              "idle power (W)": idle, "warm-up (s)": sources.get("warmup_s"),
                                              "warm-up settled": str(sources.get("warmup_settled")),
                                              "sample interval (ms)": 100, **settings})
        elif name.startswith("train_throughput"):
            base.update(kind="training")
            idle = next(iter((summary.get("idle_power_w") or {}).values()), None)
            for setup, arm, variant in (("ours", "baseline", "baseline"), ("hf_fp32", "hf", "hf"),
                                        ("hf_bf16", "hf", "hf_bf16_tf32")):
                power = (summary.get(setup) or {}).get("power")
                if power:
                    warm = power.get("warmup") or {}
                    rows_here.append({**base, "arm": arm, "variant": variant, "phase": f"training ({setup})",
                                      "context T": 256, **_res_values(power), "idle power (W)": idle,
                                      "warm-up (s)": warm.get("warmup_s"), "warm-up settled": str(warm.get("settled")),
                                      **settings, "note": f"{power.get('steps')} steps of 64 x 256 tokens"})
        elif name.startswith("layer_timing"):
            base.update(kind="per-layer timing", arm=cfg.get("arm"), variant=cfg.get("variant"))
            if summary.get("resources"):
                rows_here.append({**base, "phase": "whole arm (instrumented steps)", **_res_values(summary["resources"]),
                                  **settings, "note": "utilization over the timing loop; power not windowed here"})
        elif name.startswith("reference_nll"):
            base.update(kind="reference NLL")
        else:
            continue
        if not rows_here:
            rows_here.append({**base, "phase": "-", "note": NOT_MEASURED_POWER})
        out.extend(rows_here)
    return out


def power_sheet(results: Results, rows: list[dict[str, Any]]) -> Sheet:
    sheet = Sheet("PyTorch Power & resources", [Column(h, k, f) for h, k, f in POWER_COLUMNS])
    for r in rows:
        sheet.lines.append([cell_value(r.get(h), k) if k != "text" else r.get(h) for h, k, _ in POWER_COLUMNS])
    return sheet


def summary_sheet(results: Results) -> Sheet:
    columns = [Column("arm"), Column("variant"), Column("model"), Column("platform"),
               Column("perplexity, full span", "num", "0.000000"), Column("bits/byte, full span", "num", "0.000000"),
               Column("total NLL, full span (nats)", "num", "0.000000"), Column("scored tokens, full", "int", "int"),
               Column("perplexity, slice", "num", "0.000000"), Column("total NLL, slice (nats)", "num", "0.000000"),
               Column("scored tokens, slice", "int", "int"), Column("prefill ms, T=224", "num", "0.000"),
               Column("decode ms/token, T=224", "num", "0.000"), Column("decode tokens/s, T=224", "num", "#,##0.0"),
               Column("top-1 accuracy, full", "num", "pct"), Column("top-5 accuracy, full", "num", "pct"),
               Column("top-1 accuracy, slice", "num", "pct"), Column("top-5 accuracy, slice", "num", "pct"),
               Column("decode GPU power W, T=224", "num", "0.0"), Column("decode mJ/token, T=224", "num", "0.00"),
               Column("perplexity eval GPU power W (steady)", "num", "0.0"),
               Column("train tokens/s (64 x 256)", "num", "int"), Column("train GPU power W (steady)", "num", "0.0"),
               Column("train mJ/token", "num", "0.000"), Column("run ids (speed+full; slice; train)")]
    sheet = Sheet("PyTorch Summary", columns)
    arms: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for line in _latest(results.sweep):
        key = (line["arm"], variant_of(line["arm"], line.get("lut_preset", "")), line["model"], platform_of(line["device"]))
        a = arms.setdefault(key, {"runs": {}})
        if line["benchmark"] == "perplexity":
            limit = char_limit_of(results, line["run_id"], line["dataset"])
            span = "full" if limit and int(limit) == 1_200_000 else "slice"
            a[(span, line["metric"])] = line["value"]
            a["runs"][span] = line["run_id"]
        elif line.get("context_tokens") == "224":
            a[("speed", line["metric"])] = line["value"]
            a["runs"]["full" if "full" not in a["runs"] else "speed"] = line["run_id"]
        if line["benchmark"] == "resources" and line["metric"] == "power_w_mean_steady" and line["dataset"] != "speed":
            limit = char_limit_of(results, line["run_id"], line["dataset"])
            if limit and int(limit) == 1_200_000:
                a[("full", "eval_power")] = line["value"]
    train: dict[str, tuple[str, str]] = {}
    train_power: dict[str, dict[str, str]] = {}
    for line in _latest(results.extra):
        if line.get("metric") == "train_tokens_per_s" and line.get("variant") in ("baseline", "hf"):
            train[line["variant"]] = (line["value"], line["run_id"])
        if line.get("phase") == "train" and line.get("variant") in ("baseline", "hf") and \
                line.get("metric") in ("train_power_w_mean_steady", "train_energy_per_token_mj"):
            train_power.setdefault(line["variant"], {})[line["metric"]] = line["value"]
    for (arm, variant, model, platform), a in sorted(arms.items()):
        t = train.get(variant) if platform == "gpu" else None
        runs = "; ".join(x for x in (a["runs"].get("full", ""), a["runs"].get("slice", ""), t[1] if t else "") if x)
        sheet.lines.append([arm, variant, model, platform, num(a.get(("full", "perplexity"))),
                            num(a.get(("full", "bits_per_byte"))), num(a.get(("full", "nll_total"))),
                            num(a.get(("full", "eval_tokens"))), num(a.get(("slice", "perplexity"))),
                            num(a.get(("slice", "nll_total"))), num(a.get(("slice", "eval_tokens"))),
                            num(a.get(("speed", "prefill_ms_median"))), num(a.get(("speed", "decode_ms_per_token_median"))),
                            num(a.get(("speed", "decode_tokens_per_s"))),
                            *[num(a[(span, m)]) if (span, m) in a else NOT_MEASURED_ACC
                              for span in ("full", "slice") for m in ("top1_accuracy", "top5_accuracy")],
                            num(a[("speed", "decode_power_w")]) if ("speed", "decode_power_w") in a else NOT_MEASURED_POWER,
                            num(a[("speed", "decode_energy_per_token_mj")]) if ("speed", "decode_energy_per_token_mj") in a
                            else NOT_MEASURED_POWER,
                            num(a[("full", "eval_power")]) if ("full", "eval_power") in a else NOT_MEASURED_POWER,
                            num(t[0]) if t else None,
                            num(train_power.get(variant, {}).get("train_power_w_mean_steady")) if t and variant in train_power
                            else (NOT_MEASURED_POWER if t else None),
                            num(train_power.get(variant, {}).get("train_energy_per_token_mj")) if t and variant in train_power
                            else (NOT_MEASURED_POWER if t else None), runs])
    return sheet


ABOUT = [
    ("PyTorch Memory", "torch peak allocated = the largest total of live tensors in the phase (reset at its start); "
                       "reserved adds the caching allocator's spare blocks; process GPU = nvidia-smi's figure for this "
                       "process (adds the CUDA context); device memory used = the whole GPU, the desktop included. "
                       "Memory pass rows load one arm at a time and split the peak into its parts."),
    ("LUT storage", "From bench/memory.py lut_inventory: a stage is a LUT op when its function carries a LUTArch; "
                    "its tables are the tensors it holds. 'none (all base ops)' = no stage runs a LUT op. GPU VRAM = "
                    "a cuda tensor; host DRAM = cpu (pinned or pageable); disk = memory-mapped file."),
    ("All data", "Every measurement from every flow (pytorch, hls), one per line: metric, value, unit. Filter on "
                 "flow, variant, platform, phase, layer and stage; a pivot table on it gives any comparison."),
    ("variant", "PyTorch: baseline (our from-scratch model), hf (Hugging Face reference), or the lut arm's preset "
                "(all_base = every stage on its baseline function). HLS: the pipeline's --variant. The join key."),
    ("stage", "base.STAGES names on both flows: embedding, norm, qkv_proj, rope, attn_scores, softmax, attn_values, "
              "out_proj, ffn_up, ffn_act, ffn_gate, ffn_down, final_norm, lm_head. hf only: attn_core (rope + scores + "
              "softmax + PV inside one SDPA call) and rope_tables. unattributed = step minus the stages."),
    ("perplexity", "Same text for every arm: TinyStories V2 validation, separators replaced by blank lines, cut at "
                   "a space. full = 1,200,000 chars (321,526 scored stories15M tokens); slice = 187,000 chars "
                   "(50,211 scored tokens), the screening slice. Windows of 256 tokens every 128; loss in float64."),
    ("bits/byte", "total NLL / (ln 2 x UTF-8 bytes): compares models with different tokenizers."),
    ("prefill ms", "Time to first token for a B x T random prompt (argmax included), median of the timed repeats."),
    ("decode ms/token", "One decode step against the KV cache (argmax included); in the Speed sheet the context "
                        "grows from T to T + 31 over 32 steps; in the per-layer sheet it is held at T."),
    ("share_of_step", "Stage time / instrumented step time. On the GPU at batch 1 most stages are launch-bound: "
                      "the time tracks kernel count and Python overhead more than arithmetic. Read it next to the "
                      "modeled cost columns. The instrumented step is slower than the plain one; both are listed."),
    ("mul/token, nonlinear/token, bytes/token", "Estimated by bench/opcount.py from the shapes, per token of the "
                                                "step: multiplies; exp, sqrt and divide evaluations; FP32 bytes read "
                                                "and written as the code materializes them (weights, activations, KV)."),
    ("beyond trained context", "stories15M was trained with 256 tokens; T = 1024 points are timing only."),
    ("HLS Kernels", "From results/hls/runs/*/summary.json records that carry a variant. Latency from cosim when it "
                    "ran, else the synthesis estimate; clock post-route when impl ran, else the target."),
    ("Final comparison", "Rows appear when an HLS record's variant and stages match PyTorch results; HLS records "
                         "come from ledger_record.json (variant = the PyTorch name, stages = base.STAGES) over summary.json."),
    ("Time budget", "How long each activity takes: median, min and max seconds over its runs, with threads and parallel "
                    "jobs, and projections (276-config screen, a TinyStories-sized training pass, an HLS design point up "
                    "to each stage). basis: measured (the run recorded the duration), derived (summed from recorded "
                    "samples), reconstructed (from console timing or medians), projection. HLS rows are grouped by "
                    "condition so builds at Vivado's default 8 threads or next to other jobs are not mixed with "
                    "16-thread solo builds. contention: a row is contended when another heavy job overlapped it in time (any "
                    "other HLS run's stage, recorded or still running in a build folder, or one of our PyTorch "
                    "benchmark runs); medians and projections use uncontended rows when there are any. HLS stage "
                    "times without their own start and finish are placed from the run's start plus the stages before. "
                    "Raw rows are in All data with phase = time."),
    ("power", "No power or energy numbers until the GPU power-ramp fix (agent/STATE.md, Next item 6)."),
    ("accuracy", "Next-token accuracy over the same scored tokens as the NLL: top-1 = the true next token is the "
                 "argmax, top-5 = it is among the 5 most likely. Runs before 2026-11-24 say not measured."),
    ("PyTorch Power & resources", "Per run and phase: GPU board power from nvidia-smi (power.draw is the driver's "
                                  "~1 s average; 'steady' leaves out the first second of the window), energy per "
                                  "token, idle power sampled before the CUDA context exists, the warm-up before "
                                  "sampling (at least 2 s and until 3 readings agree within 3%), GPU and memory-"
                                  "controller utilization, memory, clocks, temperature, clock event (throttle) "
                                  "reasons, the process's CPU time and peak RSS, host CPU busy %, and the GPU's "
                                  "power limits. GPU board power only; never added to CPU power. Runs before the "
                                  "power-ramp fix say not measured (pre-fix)."),
    ("caveat", "PyTorch emulation speed says nothing about FPGA speed (agent.md 2.6). GPU timings are not "
               "clock-locked; the RTX 5080 boosts."),
]


def about_sheet() -> Sheet:
    return Sheet("About", [Column("item"), Column("meaning")], [[a, b] for a, b in ABOUT])


def build(results_dir: Path) -> list[Sheet]:
    results = load(results_dir)
    timing = timing_rows(results)
    mark_contention(results, timing)
    return [all_data(results, timing), averages_sheet(results, timing), time_budget_sheet(results, timing), summary_sheet(results),
            power_sheet(results, power_rows(results)), perplexity_sheet(results), speed_sheet(results),
            per_stage_sheet(results), memory_sheet(results), lut_storage_sheet(results), hardware_sheet(results),
            runs_sheet(results), hls_sheet(results),
            hls_inference_sheet(results),
            comparison_sheet(results), about_sheet()]


# ----------------------------------------------------------------------------------------------
# Memory: where each PyTorch run's memory goes, and where the LUT tables live (Alpha, 2026-11-24)
# ----------------------------------------------------------------------------------------------

MEMORY_COLUMNS = [
    ("run_id", "text", "general"), ("source", "text", "general"), ("variant", "text", "general"), ("arm", "text", "general"),
    ("platform", "text", "general"), ("phase", "text", "general"), ("shape / tokens", "text", "general"),
    ("torch peak allocated (MiB)", "num", "0.0"), ("torch peak reserved (MiB)", "num", "0.0"),
    ("torch allocated after (MiB)", "num", "0.0"), ("process GPU peak (MiB)", "num", "0.0"),
    ("device memory used peak (MiB, all processes)", "num", "0.0"), ("host RSS peak (MiB)", "num", "0.0"),
    ("host RSS after (MiB)", "num", "0.0"), ("pinned host (MiB)", "num", "0.000"), ("weights (MiB)", "num", "0.00"),
    ("LUT tables on GPU (MiB)", "num", "0.000"), ("KV cache (MiB)", "num", "0.000"), ("optimizer state (MiB)", "num", "0.0"),
    ("gradients (MiB)", "num", "0.0"), ("activations / workspace (MiB)", "num", "0.0"), ("allocator cached (MiB)", "num", "0.0"),
    ("CUDA context + overhead (MiB)", "num", "0.0"), ("harness buffers (MiB)", "num", "0.00"),
    ("HtoD copies per step / window", "num", "0.00"), ("LUT storage", "text", "general"), ("date (UTC)", "text", "general"),
    ("note", "text", "general")]
MEMORY_NOTE_SHARED = ("one process held both the HF model and our parameters (about 58 MiB each), so the peak includes "
                      "the other arm's weights; the memory pass (source = memory pass) loads one arm at a time")


def _mib(value: Any) -> float | None:
    return None if value is None else float(value) / MiB_BYTES


MiB_BYTES = 2 ** 20


def memory_rows(results: Results) -> list[dict[str, Any]]:
    """One row per (run, arm or setup, measured phase): bench/memory_pass.py runs give the full
    breakdown; every other run with ResourceSampler records gives its torch peaks, the device-wide
    nvidia-smi memory.used and host RSS."""
    rows: list[dict[str, Any]] = []
    for rid in sorted(results.runs):
        run = results.runs[rid]
        meta, summary = run["meta"], run["summary"] or {}
        name, cfg = meta.get("name", ""), meta.get("configs") or {}
        ts = meta.get("finished_utc") or meta.get("started_utc") or ""
        device = next(iter((meta.get("devices") or {}).values()), "")
        platform = platform_of(device) if device else ""
        if name.startswith("memory_"):
            for arm, out in (summary.get("arms") or {}).items():
                variant = variant_of(arm, cfg.get("lut_preset") or "")
                inv = out.get("lut_inventory") or {}
                for phase, p in (out.get("phases") or {}).items():
                    copies = (out.get("htod_decode_window") or {}).get("per_step_copies") if phase == "decode" else \
                        (out.get("htod_window_forward") or {}).get("per_window_copies")
                    shape = (f"{p.get('windows')} windows, {p.get('steps'):,} decode steps, batch 1" if phase == "decode"
                             else f"{p.get('windows'):,} windows of {cfg.get('window')} tokens, batch 1")
                    rows.append({"run_id": rid, "source": "memory pass", "variant": variant, "arm": arm, "platform": platform,
                                 "phase": f"inference {phase.replace('_', '-')}", "shape / tokens": shape,
                                 **_memory_fields(p), "HtoD copies per step / window": copies,
                                 "LUT storage": inv.get("label"), "date (UTC)": ts,
                                 "note": f"weights by place: {', '.join(f'{k} {v / MiB_BYTES:.1f} MiB' for k, v in (out.get('weights_bytes_by_place') or {}).items())}"})
            for setup, t in (summary.get("training") or {}).items():
                rows.append({"run_id": rid, "source": "memory pass", "variant": "baseline" if setup == "ours" else setup,
                             "arm": "baseline" if setup == "ours" else "hf", "platform": platform, "phase": "training step",
                             "shape / tokens": f"batch {t.get('batch')} x {t.get('T')}, {setup}", **_memory_fields(t),
                             "LUT storage": t.get("lut_label"), "date (UTC)": ts, "note": t.get("note", "")})
            continue
        sources: list[tuple[str, str, str, dict[str, Any], str]] = []      # (arm, variant, phase, resources, shape)
        if name.startswith(("eval_decode", "eval_power")):
            phase = "inference decode-step" if name.startswith("eval_decode") else "inference window-forward"
            for arm in ("hf", "baseline", "lut"):
                info = summary.get(arm)
                if isinstance(info, dict) and info.get("resources"):
                    sources.append((arm, variant_of(arm, cfg.get("lut_preset") or ""), phase, info["resources"],
                                    f"{info.get('scored_tokens') or 0:,} tokens"))
        elif name.startswith("sweep_"):
            row = cfg.get("row") or {}
            for key, res in summary.items():
                if key.startswith("resources_") and isinstance(res, dict):
                    sources.append((row.get("arm", ""), variant_of(row.get("arm", ""), row.get("lut_preset") or ""),
                                    f"sweep {key[len('resources_'):]}", res, ""))
        elif name.startswith("layer_timing") and isinstance(summary.get("resources"), dict):
            sources.append((cfg.get("arm", ""), cfg.get("variant", ""), "per-stage timing", summary["resources"], ""))
        elif name.startswith("train_throughput"):
            for setup in ("ours", "hf_fp32", "hf_bf16"):
                s = summary.get(setup) or {}
                if isinstance(s.get("power"), dict):
                    sources.append(("baseline" if setup == "ours" else "hf", "baseline" if setup == "ours" else setup,
                                    "training step", s["power"], f"batch {s.get('batch')} x {s.get('T')}, {setup}"))
        for arm, variant, phase, res, shape in sources:
            rows.append({"run_id": rid, "source": name.split("_stories")[0].split("_s15m")[0], "variant": variant, "arm": arm,
                         "platform": platform, "phase": phase, "shape / tokens": shape,
                         "torch peak allocated (MiB)": res.get("torch_peak_allocated_mib"),
                         "torch peak reserved (MiB)": res.get("torch_peak_reserved_mib"),
                         "device memory used peak (MiB, all processes)": res.get("memory_used_max"),
                         "host RSS peak (MiB)": res.get("process_rss_peak_mib"), "date (UTC)": ts,
                         "note": MEMORY_NOTE_SHARED if name.startswith(("eval_", "sweep_")) else ""})
    return rows


def _memory_fields(p: dict[str, Any]) -> dict[str, Any]:
    return {"torch peak allocated (MiB)": _mib(p.get("torch_peak_allocated_bytes")),
            "torch peak reserved (MiB)": _mib(p.get("torch_peak_reserved_bytes")),
            "torch allocated after (MiB)": _mib(p.get("torch_allocated_steady_bytes")),
            "process GPU peak (MiB)": _mib(p.get("process_gpu_peak_bytes")),
            "host RSS peak (MiB)": _mib(p.get("host_rss_peak_bytes")), "host RSS after (MiB)": _mib(p.get("host_rss_steady_bytes")),
            "pinned host (MiB)": _mib(p.get("pinned_host_bytes")), "weights (MiB)": _mib(p.get("weights_bytes")),
            "LUT tables on GPU (MiB)": _mib(p.get("lut_table_bytes")), "KV cache (MiB)": _mib(p.get("kv_cache_bytes")),
            "optimizer state (MiB)": _mib(p.get("optimizer_state_bytes")), "gradients (MiB)": _mib(p.get("gradient_bytes")),
            "activations / workspace (MiB)": _mib(p.get("activations_workspace_bytes")),
            "allocator cached (MiB)": _mib(p.get("allocator_cached_bytes")),
            "CUDA context + overhead (MiB)": _mib(p.get("context_and_allocator_bytes")),
            "harness buffers (MiB)": _mib(p.get("harness_bytes"))}


def memory_sheet(results: Results) -> Sheet:
    preamble = ["Memory per run and measured phase. torch peak allocated = the largest total of live tensors (weights, "
                "KV cache, LUT tables, activations); reserved adds the caching allocator's spare blocks; process GPU = this "
                "process's own GPU memory from nvidia-smi (adds the CUDA context); device memory used = the whole GPU, "
                "including the desktop and other programs. The breakdown columns come from the memory pass (one arm "
                "loaded at a time): activations / workspace = peak allocated minus weights, LUT tables, KV cache, "
                "optimizer state, gradients and the evaluation buffers."]
    sheet = Sheet("PyTorch Memory", [Column(h, k, f) for h, k, f in MEMORY_COLUMNS], preamble=preamble)
    for r in memory_rows(results):
        sheet.lines.append([cell_value(r.get(h), k) if k != "text" else r.get(h) for h, k, _ in MEMORY_COLUMNS])
    return sheet


LUT_STORAGE_COLUMNS = [
    ("run_id", "text", "general"), ("variant", "text", "general"), ("arm", "text", "general"), ("kind", "text", "general"),
    ("stage", "text", "general"), ("layer", "text", "general"), ("op", "text", "general"), ("LUT architecture", "text", "general"),
    ("inputs per table (k)", "int", "int"), ("bits per input (b)", "int", "int"), ("out bits", "int", "int"),
    ("entries per table", "int", "int"), ("planned table bits", "int", "int"), ("tensor", "text", "general"),
    ("shape", "text", "general"), ("dtype", "text", "general"), ("bytes", "int", "int"), ("stored in", "text", "general"),
    ("device", "text", "general"), ("pinned", "text", "general"), ("memory-mapped file", "text", "general"),
    ("HtoD copies per decode step", "num", "0.00"), ("date (UTC)", "text", "general"), ("note", "text", "general")]


def lut_storage_rows(results: Results) -> list[dict[str, Any]]:
    """Every LUT table of every arm in each memory pass (bench/memory.py lut_inventory), or one row
    per arm saying it has none."""
    rows = []
    for rid in sorted(results.runs):
        run = results.runs[rid]
        meta, summary = run["meta"], run["summary"] or {}
        if not meta.get("name", "").startswith("memory_"):
            continue
        cfg, ts = meta.get("configs") or {}, meta.get("finished_utc") or ""
        for arm, out in (summary.get("arms") or {}).items():
            inv = out.get("lut_inventory") or {}
            copies = (out.get("htod_decode_window") or {}).get("per_step_copies")
            base = {"run_id": rid, "variant": variant_of(arm, cfg.get("lut_preset") or ""), "arm": arm,
                    "HtoD copies per decode step": copies, "date (UTC)": ts}
            if not inv.get("rows"):
                rows.append({**base, "stored in": inv.get("label") or "none", "note":
                             "no stage of this arm runs a LUT op, so there are no tables to store" if arm != "hf"
                             else "the library model has no LUT ops"})
                continue
            for r in inv["rows"]:
                rows.append({**base, "kind": r.get("kind") or "LUT op table", "stage": r.get("stage"),
                             "layer": "" if r.get("layer") is None else str(r["layer"]),
                             "op": r.get("op"), "LUT architecture": r.get("arch"), "inputs per table (k)": r.get("inputs_per_table"),
                             "bits per input (b)": r.get("bits_per_input"), "out bits": r.get("out_bits"),
                             "entries per table": r.get("entries_per_table"), "planned table bits": r.get("planned_table_bits"),
                             "tensor": r.get("name"), "shape": "x".join(str(d) for d in r["shape"]) if r.get("shape") else "",
                             "dtype": r.get("dtype"), "bytes": r.get("bytes"), "stored in": r.get("place"),
                             "device": r.get("device"), "pinned": "" if r.get("pinned") is None else ("yes" if r["pinned"] else "no"),
                             "memory-mapped file": r.get("mmap_file") or "",
                             "note": ("built once and sliced per step (a ROM in hardware); no stage of this arm runs a LUT op"
                                      if str(r.get("kind", "")).startswith("precomputed") and not inv.get("lut_stages") else "")})
    return rows


def lut_storage_sheet(results: Results) -> Sheet:
    preamble = ["Where each LUT table lives, per arm (bench/memory.py lut_inventory over the arm's ModelOps): GPU VRAM "
                "(cuda), host DRAM (cpu; pinned = page-locked, readable by DMA), or disk (memory-mapped). HtoD copies per "
                "decode step come from a torch.profiler pass: 0 means nothing (tables, weights, indices) is copied to the "
                "GPU during inference."]
    sheet = Sheet("LUT storage", [Column(h, k, f) for h, k, f in LUT_STORAGE_COLUMNS], preamble=preamble)
    for r in lut_storage_rows(results):
        sheet.lines.append([cell_value(r.get(h), k) if k != "text" else r.get(h) for h, k, _ in LUT_STORAGE_COLUMNS])
    return sheet


# ----------------------------------------------------------------------------------------------
# Averages over repeated runs (2026-11-24, overnight repeats for Alpha): mean, std, min, max, n and
# a 95 % confidence interval over the clean runs; contended runs are listed, never averaged
# ----------------------------------------------------------------------------------------------

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       25: 2.060, 30: 2.042}


def t95(df: int) -> float:
    """Student's t for a two-sided 95 % interval with df degrees of freedom (1.96 past 30)."""
    if df <= 0:
        return float("nan")
    if df in T95:
        return T95[df]
    lower = [k for k in T95 if k < df]
    return T95[max(lower)] if df <= 30 else 1.96


def describe(values: list[float]) -> dict[str, Any]:
    """n, mean, std (n - 1), min, max and the 95 % interval of the mean."""
    n = len(values)
    if not n:
        return {"n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    half = t95(n - 1) * std / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "std": std, "min": min(values), "max": max(values),
            "ci95_low": mean - half if half is not None else None, "ci95_high": mean + half if half is not None else None}


def run_conditions(results: Results) -> dict[str, str]:
    """run id -> the overnight runner's run_condition (clean, or busy: <HLS processes>)."""
    return {line["run_id"]: str(line["value"]) for line in results.extra if line.get("metric") == "run_condition"}


def contended_runs(results: Results, timing: list[dict[str, Any]]) -> dict[str, str]:
    """run id -> why it is contended: an HLS stage or build overlapped it (mark_contention), or the
    overnight runner saw HLS processes running when it started."""
    out = {r["run_id"]: r["contended"][len("contended: "):] for r in timing
           if r.get("component") in ("run total (meta start to finish)", "row total")
           and str(r.get("contended", "")).startswith("contended")}
    for rid, cond in run_conditions(results).items():
        if cond.startswith("busy"):
            out[rid] = (out.get(rid, "") + "; " if rid in out else "") + cond
    return out


# Changes to our code that move the timings, oldest first: (UTC from which runs use it, label).
# Repeated runs are only averaged within one code version, for every arm.
CODE_EPOCHS = [("2026-11-24T05:36:00+00:00", "after the rope_tables fix (2026-11-24)")]
BEFORE_FIRST_EPOCH = "before the rope_tables fix"


def code_version(results: Results, run_id: str, arm: str = "") -> str:
    """Which version of the benchmark code a run measured. It applies to every arm, the library model
    included, so an average only mixes runs of one measurement campaign (Alpha, 2026-11-24: "to make
    sure results are consistent")."""
    started = _parse_time((results.runs.get(run_id, {}).get("meta") or {}).get("started_utc"))
    label = BEFORE_FIRST_EPOCH
    for since, name in CODE_EPOCHS:
        if started is not None and started >= _parse_time(since):
            label = name
    return label


def _decode_step_ms(results: Results, run_id: str) -> float | None:
    for line in results.layers:
        if line["run_id"] == run_id and line["phase"] == "decode" and str(line["context"]) == "256":
            return num(line.get("step_ms_uninstrumented"))
    return None


def repeat_samples(results: Results) -> list[dict[str, Any]]:
    """One sample per (run, group, arm or setup, metric): every repeated benchmark's headline numbers."""
    out = []

    def add(rid: str, group: str, tokens: Any, who: str, metric: str, unit: str, value: Any) -> None:
        if isinstance(value, (int, float)):
            out.append({"run_id": rid, "group": group, "tokens": tokens, "who": who, "metric": metric, "unit": unit,
                        "value": float(value), "code": code_version(results, rid, who)})

    for rid in sorted(results.runs):
        run = results.runs[rid]
        meta, summary = run["meta"], run["summary"] or {}
        name, cfg = meta.get("name", ""), meta.get("configs") or {}
        if name.startswith("eval_decode") and summary.get("complete"):
            for arm in ("hf", "baseline", "lut"):
                info = summary.get(arm)
                if not isinstance(info, dict):
                    continue
                res, n = info.get("resources") or {}, info.get("scored_tokens")
                g = "inference, decode-step"
                add(rid, g, n, arm, "time for all decode steps", "s", info.get("decode_s"))
                add(rid, g, n, arm, "ms per decode step", "ms", info.get("ms_per_step"))
                add(rid, g, n, arm, "avg GPU power", "W", info.get("power_w_mean_steady"))
                add(rid, g, n, arm, "energy per decode step", "mJ", info.get("energy_per_step_mj"))
                add(rid, g, n, arm, "energy per scored token (secondary)", "mJ", info.get("energy_per_token_mj"))
                add(rid, g, n, arm, "top-1 accuracy", "fraction", info.get("top1_accuracy"))
                if cfg.get("loaded_only_arm"):          # earlier runs held both models: their peak is not one arm's
                    add(rid, g, n, arm, "peak GPU memory, PyTorch allocations", "MiB", res.get("torch_peak_allocated_mib"))
                add(rid, g, n, arm, "GPU temperature, mean", "C", res.get("temperature_gpu_mean"))
                add(rid, g, n, arm, "SM clock, mean", "MHz", res.get("clocks_sm_mean"))
        elif name.startswith("eval_power"):
            for arm in ("hf", "baseline", "lut"):
                info = summary.get(arm)
                if not isinstance(info, dict):
                    continue
                n, g = info.get("scored_tokens"), "inference, window-forward"
                add(rid, g, n, arm, "time per pass (median of passes)", "s", info.get("pass_s_median"))
                add(rid, g, n, arm, "avg GPU power", "W", info.get("power_w_mean_steady"))
                add(rid, g, n, arm, "energy per scored token", "mJ", info.get("energy_per_token_mj"))
                add(rid, g, n, arm, "top-1 accuracy", "fraction", info.get("top1_accuracy"))
        elif name.startswith("layer_timing"):
            add(rid, "per-stage timing, decode T=256", "", cfg.get("variant") or cfg.get("arm", ""), "decode step (uninstrumented)",
                "ms", _decode_step_ms(results, rid))
        elif name.startswith("train_throughput"):
            for setup in ("ours", "hf_fp32", "hf_bf16"):
                s = summary.get(setup) or {}
                if not s.get("tok_s"):
                    continue
                pw, g = s.get("power") or {}, f"training, batch {s.get('batch')} x {s.get('T')}"
                add(rid, g, "", setup, "ms per training step", "ms", s.get("ms_step"))
                add(rid, g, "", setup, "tokens per second", "tokens/s", s.get("tok_s"))
                add(rid, g, "", setup, "avg GPU power", "W", pw.get("power_w_mean_steady"))
                add(rid, g, "", setup, "peak GPU memory, PyTorch allocations", "MiB", pw.get("torch_peak_allocated_mib"))
    return out


AVERAGE_COLUMNS = [
    ("group", "text", "general"), ("tokens", "int", "int"), ("arm / setup", "text", "general"),
    ("code version", "text", "general"), ("metric", "text", "general"),
    ("unit", "text", "general"), ("n (clean)", "int", "int"), ("mean", "num", "general"), ("std", "num", "general"),
    ("min", "num", "general"), ("max", "num", "general"), ("95% CI low", "num", "general"), ("95% CI high", "num", "general"),
    ("contended runs (not averaged)", "int", "int"), ("clean run ids", "text", "general"),
    ("contended run ids", "text", "general"), ("note", "text", "general")]


def average_rows(results: Results, timing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (group, token set, arm or setup, metric): statistics over the clean runs."""
    contended = contended_runs(results, timing)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for sample in repeat_samples(results):
        groups.setdefault((sample["group"], sample["tokens"], sample["who"], sample["code"], sample["metric"], sample["unit"]),
                          []).append(sample)
    rows = []
    for (group, tokens, who, code, metric, unit), samples in groups.items():
        clean = [x for x in samples if x["run_id"] not in contended]
        busy = [x for x in samples if x["run_id"] in contended]
        stats = describe([x["value"] for x in clean])
        note = ""
        if metric == "top-1 accuracy" and len({x["value"] for x in samples}) > 1:
            note = "ACCURACY DIFFERS BETWEEN RUNS"
        elif metric == "top-1 accuracy":
            note = "identical in every run"
        if not clean and busy:
            note = (note + "; " if note else "") + "no clean run yet"
        rows.append({"group": group, "tokens": tokens or None, "arm / setup": who, "code version": code, "metric": metric,
                     "unit": unit,
                     "n (clean)": stats["n"], "mean": stats.get("mean"), "std": stats.get("std"), "min": stats.get("min"),
                     "max": stats.get("max"), "95% CI low": stats.get("ci95_low"), "95% CI high": stats.get("ci95_high"),
                     "contended runs (not averaged)": len(busy), "clean run ids": "; ".join(x["run_id"] for x in clean),
                     "contended run ids": "; ".join(f"{x['run_id']} ({contended[x['run_id']][:80]})" for x in busy),
                     "note": note})
    order = {"inference, decode-step": 0, "inference, window-forward": 1}
    rows.sort(key=lambda r: (order.get(r["group"], 2), r["group"], -(r["tokens"] or 0), r["arm / setup"], r["code version"],
                             r["metric"]))
    return rows


def averages_sheet(results: Results, timing: list[dict[str, Any]]) -> Sheet:
    preamble = ["Repeated runs: mean, sample std (n - 1), min, max and the 95 % confidence interval of the mean "
                "(Student's t) over the clean runs of one code version (a change that moves the timings, such as the "
                "rope_tables fix, starts a new version; bench/excel.py CODE_EPOCHS). A run is contended when an HLS stage or build overlapped it, or the "
                "overnight runner saw HLS processes when it started; contended runs are listed, not averaged. Every "
                "individual run stays in All data and the other tabs."]
    sheet = Sheet("Averages", [Column(h, k, f) for h, k, f in AVERAGE_COLUMNS], preamble=preamble)
    for r in average_rows(results, timing):
        sheet.lines.append([cell_value(r.get(h), k) if k != "text" else r.get(h) for h, k, _ in AVERAGE_COLUMNS])
    return sheet


def write_sheet_csv(sheet: Sheet, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([c.header for c in sheet.columns])
        for line in sheet.lines:
            writer.writerow(["" if v is None else (f"{v:.12g}" if isinstance(v, float) else v) for v in line])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: results-dir/benchmarks.xlsx")
    parser.add_argument("--csv-dir", type=Path, default=None, help="also write each sheet as a CSV here")
    parser.add_argument("--no-sync", action="store_true",
                        help="don't upload to the Google Sheet even if bench/gsheets.py is configured")
    args = parser.parse_args(argv)
    out = args.out or (WORKBOOK_PATH if args.results_dir == RESULTS_DIR else args.results_dir / WORKBOOK_PATH.name)
    sheets = build(args.results_dir)
    data = xlsx_bytes(sheets)
    write_atomically(out, data)
    print(f"{out}: {len(data):,} bytes; " + ", ".join(f"{s.name} {len(s.lines)}" for s in sheets))
    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        for sheet in sheets:
            write_sheet_csv(sheet, args.csv_dir / (re.sub(r"[^A-Za-z0-9]+", "_", sheet.name).strip("_").lower() + ".csv"))
        print(f"sheet CSVs in {args.csv_dir}")
    if not args.no_sync and args.results_dir == RESULTS_DIR:
        from bench import gsheets      # uploads changed tabs when a web_app_url is configured; never raises

        gsheets.sync_quietly(sheets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
