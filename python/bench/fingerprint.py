"""A stable sha256 of the PyTorch-side inputs that HLS generation is built from, so an HLS build key
can include it and a build is reused when nothing it depends on changed (2026-11-24, Alpha: "if the
previous HLS code has not been changed (ie. the PyTorch code hasn't been changed), don't rebuild,
just rerun").

    cd python
    python -m bench.fingerprint                         # baseline variant, stories15m: one line, the sha256
    python -m bench.fingerprint --variant lut --json    # every component and its own hash

Components, each hashed on its own, then combined in a fixed order:
  code        the source files of the stages the kernel ports (inference only): baseline/inference.py
              and baseline/host.py; for --variant lut also LUT/host.py (the PARAMETERS block),
              LUT/inference_lut.py, LUT/stages.py, LUT/ops_registry.py and common/quant.py.
              Line endings are normalized; nothing else is (a comment change counts as a change,
              because the port follows the code by hand).
  config      the model's config.json in the installed snapshot (shapes, norm, activation, rope).
  checkpoint  the snapshot revision (its folder name) and the sha256 of every weight file in it
              (*.safetensors, *.bin, *.pt).
  extra       anything passed with --extra (for example the HLS part and clock), as given.
The same inputs always give the same hash; file timestamps and paths outside the repository don't
enter it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PYTHON_DIR = Path(__file__).resolve().parents[1]
CODE = {
    "baseline": ["baseline/inference.py", "baseline/host.py"],
    "lut": ["baseline/inference.py", "baseline/host.py", "LUT/host.py", "LUT/inference_lut.py", "LUT/stages.py",
            "LUT/ops_registry.py", "common/quant.py"],
}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path, text: bool = False) -> str:
    data = path.read_bytes()
    return _sha(data.replace(b"\r\n", b"\n") if text else data)


def components(variant: str, snapshot: Path | None, extra: list[str]) -> dict[str, Any]:
    """{"code": {relative path: sha256}, "config": sha256 | None, "checkpoint": {...}, "extra": [...]}."""
    code = {rel: file_sha(PYTHON_DIR / rel, text=True) for rel in CODE[variant]}
    out: dict[str, Any] = {"variant": variant, "code": code, "config": None, "checkpoint": None, "extra": sorted(extra)}
    if snapshot is not None and snapshot.is_dir():
        config = snapshot / "config.json"
        out["config"] = file_sha(config, text=True) if config.is_file() else None
        weights = {p.name: file_sha(p) for p in sorted(snapshot.iterdir())
                   if p.suffix in WEIGHT_SUFFIXES and p.is_file()}
        out["checkpoint"] = {"revision": snapshot.name, "weights": weights}
    return out


def combine(parts: dict[str, Any]) -> str:
    """One sha256 over the components, serialized with sorted keys (stable across runs and machines)."""
    return _sha(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(CODE), default="baseline")
    parser.add_argument("--model", default="stories15m", help="a model the sweep knows, or none")
    parser.add_argument("--snapshot", type=Path, default=None, help="the checkpoint folder (default: the installed one)")
    parser.add_argument("--extra", action="append", default=[], help="more inputs of the build key, e.g. part=xcu50")
    parser.add_argument("--json", action="store_true", help="print every component, not only the hash")
    args = parser.parse_args(argv)
    snapshot = args.snapshot
    if snapshot is None and args.model != "none":
        from bench import sweep
        from reference import local_cache

        source = sweep.resolve_model(args.model, local_cache.CHECKPOINT_DIR)
        snapshot = Path(source.load_name) if source.installed and Path(str(source.load_name)).is_dir() else None
    parts = components(args.variant, snapshot, args.extra)
    digest = combine(parts)
    print(json.dumps({"fingerprint": digest, **parts}, indent=2) if args.json else digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
