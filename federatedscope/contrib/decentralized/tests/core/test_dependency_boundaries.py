"""Package layering: ``core -> lora -> methods -> simulation`` is one-way.

The import checks read the syntax tree, so a module named in a docstring is not
a violation and a real import cannot be hidden behind formatting. The checks for
method names in core/ and registry.py scan source lines instead.
"""
import ast
import pathlib

import pytest

PACKAGE = "federatedscope.contrib.decentralized"
ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../contrib/decentralized/


def _python_files(*relative):
    base = ROOT.joinpath(*relative)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in str(p))


def imported_modules(path: pathlib.Path):
    """Fully-qualified module names imported by ``path``, resolving relatives."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = [PACKAGE] + list(path.relative_to(ROOT).parts[:-1])

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                prefix = ".".join(base)
                found.add(f"{prefix}.{node.module}" if node.module else prefix)
            elif node.module:
                found.add(node.module)
    return found


def _violations(files, forbidden_prefixes):
    bad = []
    for path in files:
        for module in imported_modules(path):
            for prefix in forbidden_prefixes:
                if module == prefix or module.startswith(prefix + "."):
                    bad.append(f"{path.relative_to(ROOT)} imports {module}")
    return bad


def test_core_imports_neither_lora_nor_methods():
    bad = _violations(_python_files("core"), [f"{PACKAGE}.lora", f"{PACKAGE}.methods"])
    assert not bad, "core/ must stay method- and LoRA-agnostic:\n  " + "\n  ".join(bad)


def test_lora_imports_no_method():
    bad = _violations(_python_files("lora"), [f"{PACKAGE}.methods"])
    assert not bad, "lora/ must not depend on a concrete method:\n  " + "\n  ".join(bad)


def test_registry_imports_no_method():
    """The registry stays plugin-free so core/ can query it safely."""
    bad = _violations([ROOT / "registry.py"], [f"{PACKAGE}.methods", f"{PACKAGE}.lora"])
    assert not bad, "registry.py must import no plugin:\n  " + "\n  ".join(bad)


def test_no_method_imports_another_method():
    """No method plugin imports another."""
    methods_root = ROOT / "methods"
    if not methods_root.exists():
        pytest.skip("no methods/ directory")
    bad = []
    for entry in sorted(methods_root.iterdir()):
        if not entry.is_dir() or entry.name == "__pycache__":
            continue
        others = [
            f"{PACKAGE}.methods.{other.name}"
            for other in methods_root.iterdir()
            if other.is_dir() and other.name not in (entry.name, "__pycache__")
        ]
        if others:
            bad.extend(_violations(_python_files("methods", entry.name), others))
    assert not bad, "method plugins must be mutually independent:\n  " + "\n  ".join(bad)


def test_runtime_does_not_import_the_simulation_layer():
    """core/, lora/ and methods/ never import ``simulation``, which builds on all three."""
    files = _python_files("core") + _python_files("lora") + _python_files("methods")
    bad = _violations(files, [f"{PACKAGE}.simulation"])
    assert not bad, "the runtime must not depend on the simulation layer:\n  " + "\n  ".join(bad)


def test_runtime_names_no_concrete_method():
    """No line of core/ other than a comment line names a concrete method."""
    offenders = []
    for path in _python_files("core"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for name in ("dec_lora", "pulse_lora", "tad_lora"):
                if name in stripped:
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {stripped}")
    assert not offenders, "core/ must not name a concrete method:\n  " + "\n  ".join(offenders)


def test_registry_mentions_a_method_name_only_as_reserved_data():
    """registry.py may list names for cheap recognition, but must not branch on them."""
    import ast as _ast

    tree = _ast.parse((ROOT / "registry.py").read_text(encoding="utf-8"))
    reserved_literals = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.AnnAssign) and getattr(node.target, "id", "") == "RESERVED_METHOD_NAMES":
            reserved_literals = {
                elt.value for elt in getattr(node.value, "elts", []) if isinstance(elt, _ast.Constant)
            }
    assert {"dec_lora", "pulse_lora"} <= reserved_literals, \
        "expected dec_lora and pulse_lora declared in RESERVED_METHOD_NAMES"

    body = (ROOT / "registry.py").read_text(encoding="utf-8")
    body_without_reserved = body.replace('"dec_lora",', "").replace('"pulse_lora",', "")
    for lineno, line in enumerate(body_without_reserved.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        assert "dec_lora" not in stripped, f"registry.py:{lineno} branches on a method name: {stripped}"
