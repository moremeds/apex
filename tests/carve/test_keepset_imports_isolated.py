"""Prove the ENTIRE TA-signal keep-set imports with infrastructure stubbed out.

Importing a package's __init__ does not import its submodules, so we enumerate
every .py file under the keep-set dirs (plus the two single-file keep-set
modules) and import each one individually under the stub harness.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

# Directories whose every submodule must import; plus explicit single-file modules.
_KEEPSET_DIRS = [
    SRC / "domain" / "indicators",
    SRC / "domain" / "signals",
    SRC / "domain" / "strategy",
    SRC / "application" / "orchestrator" / "signal_pipeline",
]
_KEEPSET_FILES = [SRC / "application" / "services" / "ta_signal_service.py"]


def _module_name(py: Path) -> str:
    rel = py.relative_to(REPO).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _all_keepset_modules() -> list[str]:
    mods: list[str] = []
    for d in _KEEPSET_DIRS:
        mods += [_module_name(p) for p in sorted(d.rglob("*.py"))]
    mods += [_module_name(p) for p in _KEEPSET_FILES]
    return sorted(set(mods))


@pytest.mark.usefixtures("install_carve_stubs")
@pytest.mark.parametrize("modname", _all_keepset_modules())
def test_keepset_module_imports_with_stubs(modname: str) -> None:
    mod = importlib.import_module(modname)
    assert mod is not None
