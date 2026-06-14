"""Tests for the Phase 0 import-graph classifier."""

from __future__ import annotations

from pathlib import Path

from scripts.carve.import_graph import EdgeClass, classify_module, scan_keepset

REPO = Path(__file__).resolve().parents[2]


def test_clean_core_has_no_cut_edges() -> None:
    """domain/indicators imports nothing from infra/application (CUT).

    Intra-keepset domain imports are allowed (classified FOLLOW); only CUT
    edges (-> infrastructure/services/application) would break separability.
    """
    edges = classify_module(REPO / "src/domain/indicators")
    cuts = [e for e in edges if e.kind == EdgeClass.CUT]
    assert cuts == [], cuts


def test_signals_core_surfaces_known_cuts() -> None:
    """domain/signals must surface its infra/application coupling as CUT edges."""
    edges = classify_module(REPO / "src/domain/signals")
    targets = {e.target for e in edges if e.kind == EdgeClass.CUT}
    assert any("infrastructure.observability" in t for t in targets), targets
    assert any("services.historical_data_manager" in t for t in targets), targets


def test_scan_keepset_returns_all_modules() -> None:
    """The keepset scan classifies every module path it is given."""
    report = scan_keepset(REPO)
    assert set(report.keys()) >= {"domain/indicators", "domain/signals", "domain/strategy"}
    assert all("edges" in v for v in report.values())


def test_relative_imports_are_resolved(tmp_path) -> None:
    """A `from ...infrastructure.observability import X` edge resolves to CUT."""
    src = tmp_path / "src"
    pkg = src / "domain" / "signals"
    pkg.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "domain" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    # 3-dot relative: from src.domain.signals -> up 3 -> src, then infrastructure...
    (pkg / "mod.py").write_text(
        "from ...infrastructure.observability import get_logger\n"
    )
    edges = classify_module(pkg)
    rel = [e for e in edges if "infrastructure.observability" in e.target]
    assert rel, f"relative infra import not resolved; got {[e.target for e in edges]}"
    assert all(e.kind == EdgeClass.CUT for e in rel)
