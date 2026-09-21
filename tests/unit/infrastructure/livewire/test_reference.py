"""Corporate-action and security-master reads over real-shaped livewire parquet.

Every value below was read off the production macmini lake on 2026-09-21 (Silver
revision 76) and frozen here: TSLA's two real splits (it has never paid a dividend),
SPY's real March-2021 dividend, the real AAA correction pair (a revision-2 action
superseding a revision-1 one now marked ``status='corrected'``), and VSCO's one real
security-master interval plus its non-verified siblings. The fixtures are written at
runtime (*.parquet is gitignored) and mirror livewire's own column set, so a schema
drift upstream fails these tests rather than passing silently.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src.infrastructure.adapters.livewire.reference import LivewireReferenceReader

# livewire bronze asset_class=corporate_action, provider "massive", status "active".
_TSLA_ACTIONS = [
    # action_id, revision, type, ex_date, from, to, cash, currency, declared, record, pay
    (
        "0180b889891c4fe985538b9bd78ec9c9",
        1,
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
        1,
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
_SPY_ACTIONS = [
    (
        "1288224e2bc8cde68ab33dac4734199b",
        1,
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
]
# The real AAA correction pair (ADDENDUM 2026-09-21): a revision-1 action later
# re-marked "corrected" (no row supersedes it), and the revision-2 action whose
# supersedes_action_id names it. Only the active row is live.
_AAA_ACTIONS = [
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


def _write_actions(bronze_root: Path, symbol: str, rows: list) -> None:
    directory = bronze_root / "asset_class=corporate_action" / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "action_id": [r[0] for r in rows],
            "provider": ["massive"] * len(rows),
            "provider_event_id": [f"{r[0]}-src" for r in rows],
            "event_revision": [r[1] for r in rows],
            "supersedes_action_id": [None] * len(rows),
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
            "status": ["active"] * len(rows),
            "asset_class": ["corporate_action"] * len(rows),
        }
    ).to_parquet(directory / "events.parquet")


def _write_action_pair(bronze_root: Path, symbol: str, rows: list) -> None:
    """Like ``_write_actions``, but for rows that carry their own status and
    supersedes_action_id -- the shape the real AAA correction pair actually has."""
    directory = bronze_root / "asset_class=corporate_action" / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "action_id": [r[0] for r in rows],
            "provider": ["massive"] * len(rows),
            "provider_event_id": [f"{r[0]}-src" for r in rows],
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
            "asset_class": ["corporate_action"] * len(rows),
        }
    ).to_parquet(directory / "events.parquet")


# The five real WEN security-master rows (ADDENDUM 2026-09-21). Columns: event_id,
# security_id, revision, symbol, provider, exchange_mic, currency, effective_from,
# effective_to, known_at, issuer_name, cik, continuity_basis, status, supersedes.
# relationship_type/related_security_id are NULL on every row in the real master.
_WEN_ROWS = [
    (
        "b4bbb159cef982e3df4ae81a383aad6724ca91d6583e8a4274ff219e1e94da3e",
        "sec_69c7345102904d65848490832f1f191f",
        1,
        "WEN",
        "massive",
        "XNYS",
        "USD",
        "2008-09-30T08:00:00+08:00",
        None,
        "2026-09-16T16:23:40.342623+08:00",
        "WENDYS INTERNATIONAL",
        "0000105668",
        "provider_reference",
        "candidate",
        None,
    ),
    (
        "92409fa9d244477b891ec70bde78adde42f2c3745c6400b9ac39748141f017b3",
        "sec_f7f49b3d5a7145ca9a187b26c5daa388",
        1,
        "WEN",
        "wikipedia_sec_research",
        "XNYS",
        "USD",
        "2008-09-29T08:00:00+08:00",
        "2008-10-01T08:00:00+08:00",
        "2026-09-17T03:11:57.176216+08:00",
        "Wendy's International Inc.",
        "0000030697",
        "regulator_filing",
        "verified",
        None,
    ),
    (
        "612ddc00311967001ed74f25680105946bf6aa055a4cf176030cc39c3825d080",
        "sec_fed8e56a7978430e851923da5ba73c86",
        1,
        "WEN",
        "wikipedia_sec_research",
        "XNYS",
        "USD",
        "1996-01-01T08:00:00+08:00",
        "1996-01-03T08:00:00+08:00",
        "2026-09-17T15:36:55.444677+08:00",
        "Wendy's International, Inc.",
        "0000105668",
        "regulator_filing",
        "verified",
        None,
    ),
    (
        "ad29884f0cf9c3687f91874630eb504c2251cd046e58cc475452206222121cb8",
        "sec_f7f49b3d5a7145ca9a187b26c5daa388",
        2,
        "WEN",
        "wikipedia_sec_research",
        "XNYS",
        "USD",
        "2008-09-29T08:00:00+08:00",
        "2008-10-01T08:00:00+08:00",
        "2026-09-17T15:37:46.525792+08:00",
        "Wendy's International Inc.",
        "0000030697",
        "regulator_filing",
        "rejected",
        "92409fa9d244477b891ec70bde78adde42f2c3745c6400b9ac39748141f017b3",
    ),
    (
        "f223591e0a4fc5aa551b9a30f98c9562ac8a015ebaa169eb14abf86dc4cc3c8d",
        "sec_fed8e56a7978430e851923da5ba73c86",
        2,
        "WEN",
        "wikipedia_sec_research",
        "XNYS",
        "USD",
        "2008-09-29T08:00:00+08:00",
        "2008-10-01T08:00:00+08:00",
        "2026-09-17T15:37:46.525792+08:00",
        "Wendy's International, Inc.",
        "0000105668",
        "regulator_filing",
        "verified",
        None,
    ),
]
# ACN's two real rows: one candidate, one verified.
_ACN_ROWS = [
    (
        "ad3b3eb316d763439bddb1404dd69fa62aa62df482ff11134c5bbc89afc842b5",
        "sec_d52fe9e5d0d34dea9c8ddb0d1b039352",
        1,
        "ACN",
        "massive",
        "XNYS",
        "USD",
        "2011-07-06T08:00:00+08:00",
        None,
        "2026-09-16T13:57:21.987112+08:00",
        "Accenture PLC",
        "0001467373",
        "provider_reference",
        "candidate",
        None,
    ),
    (
        "6b8bc4b4a3faec6207a73bf5e71a2b0da222c18cf84833140167622f2a9d3684",
        "sec_f37ae752ddc743018424ef91604c179f",
        1,
        "ACN",
        "wikipedia_sec_research",
        "XNYS",
        "USD",
        "2011-07-05T08:00:00+08:00",
        "2011-07-07T08:00:00+08:00",
        "2026-09-17T03:11:57.176216+08:00",
        "Accenture plc",
        "0001467373",
        "regulator_filing",
        "verified",
        None,
    ),
]


def _write_security_master(lake_root: Path, rows: list) -> None:
    """Write real security-master rows in the shape ``fetch_identity`` reads."""
    directory = lake_root / "security_master"
    directory.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    effective_from = [r[7] for r in rows]
    effective_to = [r[8] for r in rows]
    known_at = [r[9] for r in rows]
    pd.DataFrame(
        {
            "event_id": [r[0] for r in rows],
            "security_id": [r[1] for r in rows],
            "revision": [r[2] for r in rows],
            "symbol": [r[3] for r in rows],
            "provider": [r[4] for r in rows],
            "exchange_mic": [r[5] for r in rows],
            "currency": [r[6] for r in rows],
            "effective_from": pd.to_datetime(effective_from, utc=True),
            "effective_to": pd.to_datetime(effective_to, utc=True),
            "known_at": pd.to_datetime(known_at, utc=True),
            "issuer_name": [r[10] for r in rows],
            "cik": [r[11] for r in rows],
            "composite_figi": [None] * n,
            "share_class_figi": [None] * n,
            "continuity_basis": [r[12] for r in rows],
            "relationship_type": [None] * n,
            "related_security_id": [None] * n,
            "status": [r[13] for r in rows],
            "supersedes": [r[14] for r in rows],
        }
    ).to_parquet(directory / "events.parquet")


def test_actions_are_ordered_by_ex_date(tmp_path: Path) -> None:
    _write_actions(tmp_path, "TSLA", _TSLA_ACTIONS)
    actions = LivewireReferenceReader(tmp_path).fetch_actions("TSLA")
    assert actions is not None
    assert [(a.action_type, a.ex_date, a.split_from, a.split_to) for a in actions] == [
        ("split", "2020-08-31", 1.0, 5.0),
        ("split", "2022-08-25", 1.0, 3.0),
    ]


def test_cash_dividend_carries_amount_and_the_three_dates(tmp_path: Path) -> None:
    _write_actions(tmp_path, "SPY", _SPY_ACTIONS)
    actions = LivewireReferenceReader(tmp_path).fetch_actions("SPY")
    assert actions is not None and len(actions) == 1
    action = actions[0]
    assert action.action_type == "cash_dividend"
    assert action.cash_amount == 1.277788
    assert action.currency == "USD"
    assert (action.declaration_date, action.record_date, action.pay_date) == (
        "2021-01-22",
        "2021-03-22",
        "2021-04-30",
    )
    assert action.split_from is None


def test_only_the_active_row_of_a_corrected_action_survives(tmp_path: Path) -> None:
    """The real AAA pair: the revision-1 action is re-marked "corrected" once the
    revision-2 action supersedes it, and only the active row is live."""
    _write_action_pair(tmp_path, "AAA", _AAA_ACTIONS)
    actions = LivewireReferenceReader(tmp_path).fetch_actions("AAA")
    assert actions is not None and len(actions) == 1
    action = actions[0]
    assert action.action_type == "cash_dividend"
    assert action.ex_date == "2026-07-31"
    assert action.cash_amount == 0.09576


def test_type_and_date_filters(tmp_path: Path) -> None:
    _write_actions(tmp_path, "TSLA", _TSLA_ACTIONS)
    reader = LivewireReferenceReader(tmp_path)
    splits = reader.fetch_actions("TSLA", action_type="cash_dividend")
    assert splits == []
    windowed = reader.fetch_actions("TSLA", start=dt.date(2021, 1, 1))
    assert windowed is not None
    assert [a.ex_date for a in windowed] == ["2022-08-25"]
    bounded = reader.fetch_actions("TSLA", end=dt.date(2021, 1, 1))
    assert bounded is not None
    assert [a.ex_date for a in bounded] == ["2020-08-31"]


def test_missing_log_is_none_not_empty(tmp_path: Path) -> None:
    """None means "no such ticker" (404); [] means "log exists, nothing matched"."""
    assert LivewireReferenceReader(tmp_path).fetch_actions("TSLA") is None
    assert LivewireReferenceReader(None).fetch_actions("TSLA") is None


def test_provider_is_reported_when_unambiguous(tmp_path: Path) -> None:
    _write_actions(tmp_path, "TSLA", _TSLA_ACTIONS)
    assert LivewireReferenceReader(tmp_path).fetch_provider("TSLA") == "massive"


def test_identity_is_verified_and_not_superseded(tmp_path: Path) -> None:
    """The real WEN rows (ADDENDUM): fetch_identity must match membership.py's live
    view -- status='verified' AND not superseded by any other row -- and return
    exactly the two live intervals. 92409fa... is dropped because a later row
    (ad2988..., itself rejected) supersedes it; b4bbb1... is dropped for being
    candidate, not verified."""
    _write_security_master(tmp_path, _WEN_ROWS)
    intervals = LivewireReferenceReader(None, tmp_path).fetch_identity("WEN")
    assert intervals is not None
    assert [(i.security_id, i.effective_from, i.effective_to) for i in intervals] == [
        (
            "sec_fed8e56a7978430e851923da5ba73c86",
            "1996-01-01 08:00:00+08:00",
            "1996-01-03 08:00:00+08:00",
        ),
        (
            "sec_fed8e56a7978430e851923da5ba73c86",
            "2008-09-29 08:00:00+08:00",
            "2008-10-01 08:00:00+08:00",
        ),
    ]
    assert all(i.status == "verified" for i in intervals)
    assert all(i.security_id == "sec_fed8e56a7978430e851923da5ba73c86" for i in intervals)
    # Measured on the real master 2026-09-21: no delisting reason exists anywhere.
    assert all(i.relationship_type is None for i in intervals)


def test_identity_serves_only_the_verified_acn_row(tmp_path: Path) -> None:
    """ACN's two real rows: the candidate is dropped, only the verified one is served."""
    _write_security_master(tmp_path, _ACN_ROWS)
    intervals = LivewireReferenceReader(None, tmp_path).fetch_identity("ACN")
    assert intervals is not None
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval.status == "verified"
    assert interval.security_id == "sec_f37ae752ddc743018424ef91604c179f"
    assert interval.issuer_name == "Accenture plc"


def test_identity_is_none_without_a_master(tmp_path: Path) -> None:
    assert LivewireReferenceReader(None, None).fetch_identity("VSCO") is None
    assert LivewireReferenceReader(None, tmp_path).fetch_identity("VSCO") is None
