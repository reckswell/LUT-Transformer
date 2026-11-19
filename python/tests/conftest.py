"""Make python/ importable (baseline, LUT, common, bench) wherever `pytest` is started from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
