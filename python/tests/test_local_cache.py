"""Downloaded models and Hugging Face's cache live in ../LUT-Transformer-local, never in the
repository and never in ~/.cache (Alpha, 2026-11-19; reference/local_cache.py).

Checked: where CHECKPOINT_DIR and HF_HOME resolve (symlinks followed), that huggingface_hub
actually uses that HF_HOME for its hub and xet caches, that every from_pretrained call in python/
passes cache_dir=, and that a late import of huggingface_hub with another cache is refused.
"""

import ast
import os
import types
from pathlib import Path

import pytest

from reference import local_cache

PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PYTHON_DIR.parent
HOME_CACHE = Path.home() / ".cache"


def _inside(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def test_locations_are_outside_the_repository_and_home_cache():
    for path in (local_cache.CHECKPOINT_DIR, local_cache.HF_HOME, Path(os.environ["HF_HOME"])):
        assert not _inside(path, REPO_DIR), path
        assert not _inside(path, HOME_CACHE), path


def test_defaults_are_in_the_local_folder():
    default_local = REPO_DIR.parent / "LUT-Transformer-local"
    if "LUT_LOCAL_DIR" not in os.environ:
        assert local_cache.LOCAL_DIR == default_local
    if "LUT_CHECKPOINT_DIR" not in os.environ:
        assert local_cache.CHECKPOINT_DIR == local_cache.LOCAL_DIR / "checkpoints"
    assert Path(os.environ["HF_HOME"]).resolve() == local_cache.use_local_hf_cache()


def test_huggingface_hub_uses_that_cache():
    constants = pytest.importorskip("huggingface_hub.constants")
    home = Path(os.environ["HF_HOME"]).resolve()
    assert Path(constants.HF_HOME).resolve() == home
    for name in ("HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE"):
        if hasattr(constants, name) and f"{name}" not in os.environ:
            assert _inside(Path(getattr(constants, name)), home), (name, getattr(constants, name))


def test_every_from_pretrained_call_passes_cache_dir():
    missing = []
    for path in sorted(PYTHON_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_pretrained"
                    and "cache_dir" not in {kw.arg for kw in node.keywords}):
                missing.append(f"{path.relative_to(PYTHON_DIR)}:{node.lineno}")
    assert not missing, f"from_pretrained without cache_dir= (would download to ~/.cache): {missing}"


def test_a_hub_imported_with_another_cache_is_refused(monkeypatch):
    fake = types.SimpleNamespace(HF_HOME=str(HOME_CACHE / "huggingface"))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub.constants", fake)
    with pytest.raises(RuntimeError, match="imported before the reference package"):
        local_cache.use_local_hf_cache()
