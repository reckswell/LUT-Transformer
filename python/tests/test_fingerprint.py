"""bench/fingerprint.py: the PyTorch-side build key is stable and moves when an input moves."""

from bench import fingerprint


def test_the_fingerprint_is_stable_and_sensitive(tmp_path):
    snap = tmp_path / "17c2f1ea"
    snap.mkdir()
    (snap / "config.json").write_text('{"hidden_size": 288}\n')
    (snap / "model.safetensors").write_bytes(b"weights")
    a = fingerprint.combine(fingerprint.components("baseline", snap, ["part=xcu50"]))
    assert a == fingerprint.combine(fingerprint.components("baseline", snap, ["part=xcu50"]))
    assert len(a) == 64
    assert a != fingerprint.combine(fingerprint.components("lut", snap, ["part=xcu50"]))           # more code files
    assert a != fingerprint.combine(fingerprint.components("baseline", snap, ["part=xcu280"]))     # another build input
    (snap / "model.safetensors").write_bytes(b"other weights")
    assert a != fingerprint.combine(fingerprint.components("baseline", snap, ["part=xcu50"]))      # new weights
    (snap / "config.json").write_text('{"hidden_size": 288}\r\n')                                 # line endings only
    parts = fingerprint.components("baseline", snap, [])
    (snap / "config.json").write_text('{"hidden_size": 288}\n')
    assert parts["config"] == fingerprint.components("baseline", snap, [])["config"]
