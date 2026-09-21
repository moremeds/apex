"""GET /v1/equity/{symbol}/actions and /delisting, over a real-shaped lake.

These two were 501 until this change. Every value below was read off the production
macmini livewire lake on 2026-09-21 (Silver revision 76) and frozen here: TSLA's two
real splits (it has never paid a dividend), SPY's three real cash dividends, and
VSCO's one real security-master interval. The routes read the env vars, so the whole
path -- env gating, DuckDB read, envelope -- runs here.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


def _write_actions(bronze_root: Path, symbol: str, rows: list) -> None:
    directory = bronze_root / "asset_class=corporate_action" / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "action_id": [r[0] for r in rows],
            "provider": ["massive"] * len(rows),
            "event_revision": [1] * len(rows),
            "symbol": [symbol] * len(rows),
            "action_type": [r[1] for r in rows],
            "ex_date": [r[2] for r in rows],
            "split_from": [r[3] for r in rows],
            "split_to": [r[4] for r in rows],
            "cash_amount": [r[5] for r in rows],
            "currency": [r[6] for r in rows],
            "declaration_date": [r[7] for r in rows],
            "record_date": [r[8] for r in rows],
            "pay_date": [r[9] for r in rows],
            "status": ["active"] * len(rows),
        }
    ).to_parquet(directory / "events.parquet")


def _write_action_pair(bronze_root: Path, symbol: str, rows: list) -> None:
    """Like ``_write_actions``, but for rows that carry their own status and
    supersedes_action_id -- the shape the real AAA correction pair actually has.

    Each row: (action_id, event_revision, action_type, ex_date, split_from, split_to,
    cash_amount, currency, declaration_date, record_date, pay_date, status,
    supersedes_action_id).
    """
    directory = bronze_root / "asset_class=corporate_action" / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "action_id": [r[0] for r in rows],
            "provider": ["massive"] * len(rows),
            "event_revision": [r[1] for r in rows],
            "supersedes_action_id": [r[12] for r in rows],
            "symbol": [symbol] * len(rows),
            "action_type": [r[2] for r in rows],
            "ex_date": [r[3] for r in rows],
            "split_from": [r[4] for r in rows],
            "split_to": [r[5] for r in rows],
            "cash_amount": [r[6] for r in rows],
            "currency": [r[7] for r in rows],
            "declaration_date": [r[8] for r in rows],
            "record_date": [r[9] for r in rows],
            "pay_date": [r[10] for r in rows],
            "status": [r[11] for r in rows],
        }
    ).to_parquet(directory / "events.parquet")


# The real AAA correction pair (ADDENDUM 2026-09-21): only the active row is live.
_AAA = [
    (
        "260ef9a108a9bf5da5389ba5ee9e130e",
        1,
        "cash_dividend",
        dt.date(2026, 7, 31),
        None,
        None,
        0.0958,
        "USD",
        dt.date(2026, 6, 9),
        dt.date(2026, 7, 31),
        dt.date(2026, 8, 3),
        "corrected",
        None,
    ),
    (
        "4355ce0794445ebc7a16396b6762f1ab",
        2,
        "cash_dividend",
        dt.date(2026, 7, 31),
        None,
        None,
        0.09576,
        "USD",
        dt.date(2026, 6, 9),
        dt.date(2026, 7, 31),
        dt.date(2026, 8, 3),
        "active",
        "260ef9a108a9bf5da5389ba5ee9e130e",
    ),
]


# (action_id, action_type, ex_date, split_from, split_to, cash_amount, currency,
#  declaration_date, record_date, pay_date) -- TSLA has never paid a dividend, only
# the two real splits below.
_TSLA = [
    (
        "0180b889891c4fe985538b9bd78ec9c9",
        "split",
        dt.date(2020, 8, 31),
        1.0,
        5.0,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "994763a388abfb9be3c21baa7e160774",
        "split",
        dt.date(2022, 8, 25),
        1.0,
        3.0,
        None,
        None,
        None,
        None,
        None,
    ),
]
# SPY's three real cash dividends.
_SPY = [
    (
        "1288224e2bc8cde68ab33dac4734199b",
        "cash_dividend",
        dt.date(2021, 3, 19),
        None,
        None,
        1.277788,
        "USD",
        dt.date(2021, 1, 22),
        dt.date(2021, 3, 22),
        dt.date(2021, 4, 30),
    ),
    (
        "7d4a7c5839c36447e330d9fa0b2fd7ff",
        "cash_dividend",
        dt.date(2026, 6, 18),
        None,
        None,
        1.903516,
        "USD",
        dt.date(2026, 1, 2),
        dt.date(2026, 6, 18),
        dt.date(2026, 7, 31),
    ),
    (
        "f50d46ed96fc7f925eb0910d9a89a215",
        "cash_dividend",
        dt.date(2026, 9, 18),
        None,
        None,
        1.888834,
        "USD",
        dt.date(2026, 1, 2),
        dt.date(2026, 9, 18),
        dt.date(2026, 10, 30),
    ),
]


def _write_master(lake_root: Path) -> None:
    """The one real VSCO security-master row (event_id, security_id, and the
    +08:00-recorded interval, exactly as read from the production master)."""
    directory = lake_root / "security_master"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "event_id": ["fa106afa557b2596bf1ec07a11e0748b6aac85c0e8be760fe1e44fdcaa206f91"],
            "security_id": ["sec_a31d1beabb3e4c18bfdc59e5f6e40d4c"],
            "revision": [1],
            "symbol": ["VSCO"],
            "provider": ["massive"],
            "exchange_mic": ["XNYS"],
            "currency": ["USD"],
            "effective_from": pd.to_datetime(["2026-09-12T08:00:00+08:00"], utc=True),
            "effective_to": pd.to_datetime(["2026-09-14T08:00:00+08:00"], utc=True),
            "known_at": pd.to_datetime(["2026-09-14T08:00:00+08:00"], utc=True),
            "issuer_name": ["Victoria's Secret & Co."],
            "cik": [None],
            "composite_figi": [None],
            "share_class_figi": [None],
            "continuity_basis": ["regulator_filing"],
            "relationship_type": [None],
            "related_security_id": [None],
            "status": ["verified"],
            "supersedes": [None],
        }
    ).to_parquet(directory / "events.parquet")


@pytest.fixture
def lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_actions(tmp_path / "bronze", "TSLA", _TSLA)
    _write_actions(tmp_path / "bronze", "SPY", _SPY)
    _write_action_pair(tmp_path / "bronze", "AAA", _AAA)
    _write_master(tmp_path / "lake")
    monkeypatch.setenv("APEX_LIVEWIRE_ROOT", str(tmp_path / "bronze"))
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(tmp_path / "lake"))
    return tmp_path


async def _get(path: str) -> Any:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def test_actions_returns_the_live_log_ordered_by_ex_date(lake: Path) -> None:
    resp = await _get("/v1/equity/spy/actions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "SPY"
    assert body["identity"] == "ticker"
    assert body["source"] == "livewire_bronze_corporate_action"
    assert body["provider"] == "massive"
    assert [(a["action_type"], a["ex_date"]) for a in body["actions"]] == [
        ("cash_dividend", "2021-03-19"),
        ("cash_dividend", "2026-06-18"),
        ("cash_dividend", "2026-09-18"),
    ]
    assert body["actions"][0]["cash_amount"] == 1.277788
    assert body["actions"][2]["cash_amount"] == 1.888834


async def test_actions_returns_only_the_active_row_of_a_corrected_action(
    lake: Path,
) -> None:
    """The real AAA pair: only the active (revision-2) row is live."""
    resp = await _get("/v1/equity/AAA/actions")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["actions"]) == 1
    action = body["actions"][0]
    assert action["action_type"] == "cash_dividend"
    assert action["ex_date"] == "2026-07-31"
    assert action["cash_amount"] == 0.09576


async def test_actions_type_filter(lake: Path) -> None:
    body = (await _get("/v1/equity/TSLA/actions?type=split")).json()
    assert [a["ex_date"] for a in body["actions"]] == ["2020-08-31", "2022-08-25"]


async def test_actions_ex_date_window(lake: Path) -> None:
    body = (await _get("/v1/equity/SPY/actions?start=2021-01-01&end=2021-12-31")).json()
    assert [a["action_type"] for a in body["actions"]] == ["cash_dividend"]


async def test_actions_empty_window_is_a_200_not_a_404(lake: Path) -> None:
    """A log that exists and matches nothing is a quiet ticker, not an unknown one."""
    resp = await _get("/v1/equity/TSLA/actions?start=2030-01-01")
    assert resp.status_code == 200
    assert resp.json()["actions"] == []


async def test_actions_unknown_ticker_is_a_404(lake: Path) -> None:
    resp = await _get("/v1/equity/NVDA/actions")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_symbol"


async def test_actions_bad_type_is_a_400(lake: Path) -> None:
    resp = await _get("/v1/equity/TSLA/actions?type=spinoff")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_parameter"


async def test_actions_are_503_without_the_bronze_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEX_LIVEWIRE_ROOT", raising=False)
    resp = await _get("/v1/equity/TSLA/actions")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "provider_not_configured"


async def test_delisting_returns_identity_intervals(lake: Path) -> None:
    resp = await _get("/v1/equity/vsco/delisting")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "VSCO"
    assert body["identity"] == "ticker"
    assert body["source"] == "livewire_security_master"
    # Measured on the real master 2026-09-21: it records no reason for any delisting.
    assert body["delisting_reason_available"] is False
    interval = body["intervals"][0]
    assert interval["issuer_name"] == "Victoria's Secret & Co."
    assert interval["exchange_mic"] == "XNYS"
    assert interval["effective_from"].startswith("2026-09-12")
    assert interval["effective_to"].startswith("2026-09-14")
    assert interval["relationship_type"] is None


async def test_delisting_unknown_ticker_is_a_404(lake: Path) -> None:
    resp = await _get("/v1/equity/NVDA/delisting")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_symbol"


async def test_delisting_is_503_without_the_lake_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APEX_LIVEWIRE_LAKE_ROOT", raising=False)
    resp = await _get("/v1/equity/VSCO/delisting")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "provider_not_configured"


async def test_actions_reversed_bounds_are_a_400_not_an_empty_history(lake: Path) -> None:
    resp = await _get("/v1/equity/SPY/actions?start=2026-09-18&end=2021-03-19")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_parameter"
