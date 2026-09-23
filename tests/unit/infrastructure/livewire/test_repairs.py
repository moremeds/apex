"""Repair reports: real report rows frozen from the mini lake on 2026-09-23
(repairs/tier_a_2026-09-21.json, tier_a_2026-09-04.json, decisions_2026-09-21.json)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.infrastructure.adapters.livewire.repairs import RepairsReader

_TIER_A_0921 = {
    "repairs": [
        {
            "symbol": "USDCLP",
            "asset_class": "fx",
            "timeframe": "1d",
            "gap": "G1",
            "sessions": ["2026-09-21"],
            "heal_by_days": None,
            "source": "yahoo",
        }
    ]
}
_TIER_A_0904 = {
    "repairs": [
        {
            "symbol": "DGS3",
            "asset_class": "rates",
            "timeframe": "1d",
            "gap": "G1",
            "sessions": ["2026-09-04"],
            "heal_by_days": None,
            "source": "fred",
        }
    ]
}
_DECISION = {
    "symbol": "CL_202609",
    "asset_class": "futures",
    "timeframe": "1d",
    "gap": "G1",
    "sessions": ["2026-09-17", "2026-09-18", "2026-09-21"],
    "heal_by_days": None,
    "source": "ib",
    "verdict": "inconclusive",
}


def _write(root: Path, name: str, payload: object) -> None:
    (root / name).write_text(json.dumps(payload))


def test_filters_by_series_and_session_window(tmp_path: Path) -> None:
    _write(tmp_path, "tier_a_2026-09-21.json", _TIER_A_0921)
    _write(tmp_path, "tier_a_2026-09-04.json", _TIER_A_0904)
    _write(tmp_path, "decisions_2026-09-21.json", [_DECISION])
    (tmp_path / "yahoo-split-repair-batch1").mkdir()  # batch dirs are never reports
    reader = RepairsReader(tmp_path)

    fx = reader.evidence("USDCLP", "fx", "1d", date(2026, 9, 1), date(2026, 9, 30))
    assert fx.state == "available" and fx.reports_read == 3
    assert [(e.report_kind, e.sessions, e.details["source"]) for e in fx.entries] == [
        ("tier_a", ("2026-09-21",), "yahoo")
    ]
    futures = reader.evidence("CL_202609", "futures", "1d", date(2026, 9, 18), date(2026, 9, 18))
    assert futures.entries[0].details["verdict"] == "inconclusive"
    assert reader.evidence("DGS3", "rates", "1d", date(2026, 9, 5), date(2026, 9, 30)).entries == ()


def test_exact_repeats_collapse_and_list_every_report(tmp_path: Path) -> None:
    _write(tmp_path, "tier_a_2026-09-16.json", _TIER_A_0921)
    _write(tmp_path, "tier_a_2026-09-21.json", _TIER_A_0921)
    (entry,) = (
        RepairsReader(tmp_path)
        .evidence("USDCLP", "fx", "1d", date(2026, 9, 21), date(2026, 9, 21))
        .entries
    )
    assert entry.reports == ("tier_a_2026-09-16.json", "tier_a_2026-09-21.json")
    assert entry.report_date == "2026-09-21"


def test_malformed_report_degrades_instead_of_reading_as_empty(tmp_path: Path) -> None:
    _write(tmp_path, "tier_a_2026-09-21.json", _TIER_A_0921)
    (tmp_path / "decisions_2026-09-22.json").write_text("{truncated")
    evidence = RepairsReader(tmp_path).evidence(
        "X", "fx", "1d", date(2026, 1, 1), date(2026, 12, 31)
    )
    assert evidence.state == "degraded" and evidence.entries == ()
    assert evidence.warnings and "decisions_2026-09-22.json" in evidence.warnings[0]


def test_unconfigured_and_absent_are_distinct(tmp_path: Path) -> None:
    assert RepairsReader(None).status()["state"] == "not_configured"
    assert RepairsReader(tmp_path / "missing").status()["state"] == "absent"
    # Production has no unresolved.json (checked 2026-09-23). Shape from the Livewire
    # contract (design §4); the reason text is a test value, not an observed report.
    _write(
        tmp_path,
        "unresolved.json",
        [
            {
                "symbol": "CL_202609",
                "asset_class": "futures",
                "timeframe": "1d",
                "session": "2026-09-21",
                "reason": "no provider row",
                "as_of": "2026-09-22",
            }
        ],
    )
    evidence = RepairsReader(tmp_path).evidence(
        "CL_202609", "futures", "1d", date(2026, 9, 21), date(2026, 9, 21)
    )
    assert evidence.entries[0].report_kind == "unresolved"
    assert evidence.entries[0].details["reason"] == "no provider row"


def test_readable_root_without_reports_is_absent(tmp_path: Path) -> None:
    (tmp_path / "yahoo-split-repair-batch1").mkdir()
    status = RepairsReader(tmp_path).status()
    assert status["state"] == "absent" and status["warnings"]
