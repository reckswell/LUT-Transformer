"""bench/slot_parity.py runs and reports bit-identical logits on all three arms (a tiny model, a
couple of rounds; the timing itself is not checked here, only that it is measured and saved)."""

import json

import pytest

torch = pytest.importorskip("torch")

from baseline import host as base  # noqa: E402
from bench import slot_parity  # noqa: E402


def test_slot_parity_runs_and_logits_are_identical(tmp_path, capsys):
    code = slot_parity.main(["--model", str(base.PYTHON_DIR / "configs/model/core.yaml"), "--vocab", "37",
                             "--length", "12", "--decode-tokens", "4", "--rounds", "2", "--repeats", "1",
                             "--all-slots", "--results-dir", str(tmp_path)])
    assert code == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["identical_logits"] == {"baseline": True, "lut_default": True, "lut_all_slots": True}
    assert set(summary["workloads"]["prefill_decode"]) == set(slot_parity.ARMS)
    assert run_dir.name in capsys.readouterr().out
