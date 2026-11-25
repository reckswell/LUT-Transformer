"""The folder rules that keep the implementations separate (decision D9, agent.md 2.1).

    baseline/, LUT/   our model, from scratch: no torch.nn, torch.nn.functional, torch.autograd or
                      torch.func, and nothing from reference/, eval/ or transformers
    baseline/         also nothing from LUT/ (the baseline doesn't know the LUT model exists)
    reference/        library-built controls: nothing from LUT/ or eval/
    eval/             runs every arm, so it may import all of them

Checked by reading each file's syntax tree, so a forbidden import inside a function or a direct
call like torch.nn.functional.softmax(...) is caught too. Docstrings and comments are not code
and are not checked.
"""

import ast
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
LIBRARY_ML = ("torch.nn", "torch.autograd", "torch.func")

RULES = {   # folder -> module prefixes it must not import
    "baseline": ("LUT", "reference", "eval", "transformers", *LIBRARY_ML),
    "LUT": ("reference", "eval", "transformers", *LIBRARY_ML),
    "reference": ("LUT", "eval"),
}


def _imports_and_attribute_chains(path: Path) -> set[str]:
    """Every imported module name, and every dotted name used in code (torch.nn.functional...)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Attribute):
            parts, value = [node.attr], node.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                names.add(".".join([value.id, *reversed(parts)]))
    return names


def _violates(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_each_folder_imports_only_what_its_rule_allows():
    problems = []
    for folder, forbidden in RULES.items():
        files = sorted((PYTHON_DIR / folder).glob("*.py"))
        assert files, f"python/{folder}/ has no Python files"
        for path in files:
            for name in sorted(_imports_and_attribute_chains(path)):
                if any(_violates(name, prefix) for prefix in forbidden):
                    problems.append(f"{path.relative_to(PYTHON_DIR)}: {name}")
    assert not problems, "imports that break the separation:\n" + "\n".join(problems)


def test_the_checker_catches_a_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\nfrom reference import hf_gpt2\ny = torch.nn.functional.softmax(x, -1)\n")
    names = _imports_and_attribute_chains(bad)
    assert any(_violates(n, "reference") for n in names)
    assert any(_violates(n, "torch.nn") for n in names)
