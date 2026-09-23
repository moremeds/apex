"""``matrix.summarize``'s NOT_APPLICABLE gate rule: a record may be NOT_APPLICABLE
only for ``model.NOT_APPLICABLE_LEGACY_SERIES_REASON``; anything else must open the
gate. Synthetic case/result records (status records, not market data) written
directly to a tmp ``out`` directory in the exact shape ``matrix.py`` itself writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matrix  # type: ignore[import-not-found]  # sys.path set by conftest.py
from model import NOT_APPLICABLE_LEGACY_SERIES_REASON  # type: ignore[import-not-found]


def _case(case_id: str, operation: str = "bars") -> Dict[str, Any]:
    return {
        "id": case_id,
        "operation": operation,
        "dims": {"asset_class": "equity"},
        "request": {
            "transport": "http",
            "process": "raw",
            "path": "/v1/x",
            "params": {},
        },
        "expect": {"kind": "value", "check": "noop", "args": {}},
    }


def _result(case_id: str, status: str, detail: str = "") -> Dict[str, Any]:
    return {
        "id": case_id,
        "operation": "bars",
        "dims": {"asset_class": "equity"},
        "status": status,
        "detail": detail,
        "facts": {},
        "elapsed_ms": 1.0,
        "at": "2026-09-23T00:00:00Z",
    }


def _write_run(out: Path, cases: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(json.dumps(cases))
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")


def test_gate_passes_when_the_only_not_applicable_is_the_legacy_series_reason(
    tmp_path: Path,
) -> None:
    cases = [_case("c1"), _case("c2")]
    results = [
        _result("c1", "PASS"),
        _result("c2", "NOT_APPLICABLE", NOT_APPLICABLE_LEGACY_SERIES_REASON),
    ]
    _write_run(tmp_path, cases, results)
    matrix.summarize(tmp_path)
    summary = (tmp_path / "summary.md").read_text()
    assert "Gate (zero FAIL/NOT_RUN/BLOCKED_*" in summary
    assert ": PASS" in summary
    assert "0 record(s) violate it" in summary


def test_gate_fails_when_a_not_applicable_has_a_different_reason(
    tmp_path: Path,
) -> None:
    cases = [_case("c1"), _case("c2")]
    results = [
        _result("c1", "PASS"),
        _result("c2", "NOT_APPLICABLE", "some other, undocumented reason"),
    ]
    _write_run(tmp_path, cases, results)
    matrix.summarize(tmp_path)
    summary = (tmp_path / "summary.md").read_text()
    assert ": OPEN" in summary
    assert "1 record(s) violate it" in summary
    assert "c2" in summary


def test_gate_still_checks_fail_and_not_run_alongside_the_na_rule(
    tmp_path: Path,
) -> None:
    """The NOT_APPLICABLE rule is additive, not a replacement for the existing gate."""
    cases = [_case("c1"), _case("c2")]
    results = [
        _result("c1", "FAIL", "boom"),
        _result("c2", "NOT_APPLICABLE", NOT_APPLICABLE_LEGACY_SERIES_REASON),
    ]
    _write_run(tmp_path, cases, results)
    matrix.summarize(tmp_path)
    summary = (tmp_path / "summary.md").read_text()
    assert ": OPEN" in summary
    assert "0 record(s) violate it" in summary
