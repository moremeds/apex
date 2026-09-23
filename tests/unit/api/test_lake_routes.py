"""The /v1/lake, /v1/security, /v1/futures and /gaps routes.

Real frozen inputs: the PIT/Silver builders in tests/support (FSLR/BIIB, Silver rev
76/77), the real membership fixture (MUNJ resolves on 2026-09-14), the conftest
coverage catalog and SPY daily rows from the mini lake. Mutations are labelled.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from src.infrastructure.adapters.livewire.repairs import RepairsReader
from tests.support.pit_manifest import (
    FSLR_ROWS,
    GENERATION_R76,
    pit_payload,
    publish_pit,
    write_daily,
)
from tests.support.silver_manifest import publish_manifest

MEMBERSHIP_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "livewire_membership"

# Real SPY daily rows, read from the mini lake on 2026-09-23 (price_basis unknown,
# source legacy). The lake has every XNYS session 2025-01-06..17 (2025-01-09 was a
# market closure). TEST MUTATION: 01-06, 01-14, 01-16 and 01-17 are dropped here to
# create a leading edge, one interior gap and a trailing edge.
_SPY_JAN_2025 = [
    (dt.date(2025, 1, 7), 597.42, 597.75, 586.78, 588.63, 60393052),
    (dt.date(2025, 1, 8), 588.7, 590.5799, 585.195, 589.49, 47304672),
    (dt.date(2025, 1, 10), 585.88, 585.95, 578.55, 580.49, 73105046),
    (dt.date(2025, 1, 13), 575.77, 581.75, 575.35, 581.39, 47910060),
    (dt.date(2025, 1, 15), 590.325, 593.94, 589.195, 592.78, 56900159),
]


# Real OJ futures rows (bronze asset_class=futures, read 2026-09-23): the last two
# sessions of each listed OJ contract. Columns follow the real file schema.
_OJ_COLUMNS = [
    "trade_date",
    "contract_id",
    "root_symbol",
    "expiry_date",
    "open",
    "high",
    "low",
    "close",
    "settlement",
    "volume",
    "open_interest",
]
_OJ = {
    "OJ_202611": [
        (dt.date(2026, 9, 21), 6850187916210034, "OJ", dt.date(2026, 11, 1), 143.95, 150.2, 141.35, 148.6, 148.6, 809, 0),
        (dt.date(2026, 9, 22), 6850187916210034, "OJ", dt.date(2026, 11, 1), 148.9, 152.7, 147.0, 152.5, 152.5, 754, 0),
    ],
    "OJ_202701": [
        (dt.date(2026, 9, 21), 4988956632683610, "OJ", dt.date(2027, 1, 1), 146.25, 153.0, 146.25, 151.8, 151.8, 67, 0),
        (dt.date(2026, 9, 22), 4988956632683610, "OJ", dt.date(2027, 1, 1), 152.35, 156.15, 150.85, 156.15, 156.15, 128, 0),
    ],
}  # fmt: skip


def _daily(root: Path, asset_class: str, symbol: str, frame: pd.DataFrame) -> None:
    directory = root / f"asset_class={asset_class}" / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "1d.parquet", index=False)


@pytest.fixture
def client(
    tmp_path: Path, catalog_db: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _daily(
        bronze,
        "equity",
        "SPY",
        pd.DataFrame(
            _SPY_JAN_2025,
            columns=["trade_date", "open", "high", "low", "close", "volume"],
        ),
    )
    for contract, rows in _OJ.items():
        _daily(bronze, "futures", contract, pd.DataFrame(rows, columns=_OJ_COLUMNS))
    r76 = silver / write_daily(silver, "FSLR", FSLR_ROWS[:4], GENERATION_R76, 75)
    publish_manifest(silver, 76, [r76])
    r77 = silver / write_daily(silver, "FSLR", FSLR_ROWS)
    publish_manifest(silver, 77, [r77])
    publish_pit(silver, pit_payload(silver, 1, index_id="sp500"))
    publish_pit(silver, pit_payload(silver, 2, index_id="ndx100", members=[]))
    monkeypatch.setenv("APEX_LIVEWIRE_LAKE_ROOT", str(MEMBERSHIP_FIXTURE))
    app = create_app()
    app.state.ohlc_provider = LivewireOhlcProvider(bronze, silver, "adjusted")
    from src.infrastructure.adapters.livewire.coverage import CoverageCatalog

    app.state.coverage_catalog = CoverageCatalog(catalog_db)
    app.state.pit_reader = PitRevisionReader(silver)
    app.state.repairs_reader = RepairsReader(None)
    # No `with`: the production lifespan (PG, xenon, subscriptions) is not this surface.
    yield TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/lake/status",
        "/v1/lake/coverage",
        "/v1/security/MUNJ",
        "/v1/lake/asset-classes",
    ],
)
def test_literal_routes_are_not_swallowed_by_the_instrument_catch_all(
    client: TestClient, path: str
) -> None:
    body = client.get(path, params={"as_of": "2026-09-14"}).json()
    assert "error" not in body or body["error"]["code"] != "unsupported_asset_class", body


def test_asset_classes_registry(client: TestClient) -> None:
    classes = {
        c["asset_class"]: c for c in client.get("/v1/lake/asset-classes").json()["asset_classes"]
    }
    assert set(classes) == {"equity", "volatility", "fx", "cmdty", "futures", "rates"}
    assert classes["rates"]["payload"] == "rates_series" and classes["equity"]["supports_adjusted"]


def test_status_summarizes_sources_without_paths(client: TestClient, tmp_path: Path) -> None:
    body = client.get("/v1/lake/status").json()
    sources = body["sources"]
    assert sources["silver"] == {
        "configured": True,
        "available": True,
        "current_revision": 77,
        "retained_revisions": 2,
    }
    assert sources["pit"]["latest_per_index"] == {
        "ndx100": {"revision": 2, "publisher_status": "PARTIAL"},
        "sp500": {"revision": 1, "publisher_status": "PARTIAL"},
    }
    assert sources["repairs"]["state"] == "not_configured"
    assert str(tmp_path) not in str(body)


def test_coverage_pages(client: TestClient) -> None:
    first = client.get("/v1/lake/coverage", params={"limit": 2}).json()
    assert first["returned"] == 2 and first["truncated"] and first["next_offset"] == 2
    assert first["catalog"]["size_bytes"] > 0
    assert client.get("/v1/lake/coverage", params={"limit": 0}).status_code == 400
    assert client.get("/v1/lake/coverage", params={"asset_class": "bonds"}).status_code == 400


def test_silver_revisions_list_and_detail(client: TestClient) -> None:
    listed = client.get("/v1/lake/silver-revisions").json()
    assert listed["current"] == 77
    assert [r["revision"] for r in listed["revisions"]] == [77, 76]
    detail = client.get("/v1/lake/silver-revisions/76").json()
    assert detail["is_current"] is False and detail["affected"][0]["symbol"] == "FSLR"
    missing = client.get("/v1/lake/silver-revisions/9").json()
    assert missing["error"]["code"] == "unknown_revision"


def test_pit_revisions_list_filter_and_detail(client: TestClient) -> None:
    listed = client.get("/v1/lake/pit-revisions", params={"index_id": "sp500"}).json()
    assert [r["revision"] for r in listed["revisions"]] == [1]
    assert listed["latest_per_index"] == {"sp500": 1}
    detail = client.get("/v1/lake/pit-revisions/1", params={"limit": 1}).json()
    assert detail["publisher_status"] == "PARTIAL" and detail["member_count"] == 3
    assert detail["returned"] == 1 and detail["truncated"] is True
    assert detail["inputs"]["security_master"]["path"] == "security_master/events.parquet"
    assert client.get("/v1/lake/pit-revisions/current").status_code == 422
    assert client.get("/v1/lake/pit-revisions/99").json()["error"]["code"] == "unknown_revision"


def test_bars_through_a_pit_revision(client: TestClient) -> None:
    body = client.get(
        "/v1/equity/FSLR/bars",
        params={"pit_revision": 1, "start": "2026-09-15T00:00:00Z"},
    ).json()
    assert [b["close"] for b in body["bars"]] == [
        202.34,
        191.07,
        201.16,
        195.96,
        199.84,
    ]
    assert body["provenance"]["pit"]["publisher_status"] == "PARTIAL"
    raw = client.get("/v1/equity/FSLR/bars", params={"pit_revision": 1, "price_mode": "raw"}).json()
    assert raw["error"]["code"] == "invalid_parameter"


def test_security_resolution(client: TestClient) -> None:
    body = client.get("/v1/security/MUNJ", params={"as_of": "2026-09-14"}).json()
    assert body["security_id"] == "sec_405d12b544ef24fee4a9ef06b721d90e"
    assert body["knowledge"] == "today"
    before = client.get("/v1/security/MUNJ", params={"as_of": "2026-08-25"})
    assert before.status_code == 404 and before.json()["error"]["code"] == "unknown_symbol"


def test_gaps_skip_the_2025_01_09_closure_and_report_edges(client: TestClient) -> None:
    body = client.get(
        "/v1/equity/SPY/gaps", params={"start": "2025-01-06", "end": "2025-01-17"}
    ).json()
    assert body["assessment"] == "session_presence"
    assert body["calendar"]["name"] == "XNYS" and body["calendar"]["certainty"] == "exchange"
    # 01-09 is a market closure (not expected); 01-14 is a missing session.
    assert body["gaps"] == [{"start": "2025-01-14", "end": "2025-01-14", "sessions": 1}]
    assert body["leading_unobserved"] == {
        "start": "2025-01-06",
        "end": "2025-01-06",
        "sessions": 1,
    }
    assert body["trailing_unobserved"] == {
        "start": "2025-01-16",
        "end": "2025-01-17",
        "sessions": 2,
    }
    assert body["status"] == "gaps" and body["present_sessions"] == 5
    empty = client.get("/v1/equity/SPY/gaps", params={"start": "2024-01-02", "end": "2024-01-31"})
    assert empty.json()["status"] == "no_data"
    assert client.get("/v1/equity/NOPE/gaps").json()["error"]["code"] == "unknown_symbol"
    assert client.get("/v1/equity/SPY/gaps", params={"timeframe": "4h"}).status_code == 400


def test_futures_contracts_come_from_the_partition_not_the_catalog(client: TestClient) -> None:
    body = client.get("/v1/futures/oj/contracts", params={"limit": 1}).json()
    assert body["root"] == "OJ" and body["truncated"] is True and body["next_offset"] == 1
    (contract,) = body["contracts"]
    assert contract == {
        "symbol": "OJ_202611",
        "contract_id": 6850187916210034,
        "root_symbol": "OJ",
        "expiry_date": "2026-11-01",
        "first_date": "2026-09-21",
        "last_date": "2026-09-22",
        "rows": 2,
        # The catalog fixture holds no futures rows, like 100 of 114 real contracts.
        "in_catalog": False,
    }
    second = client.get("/v1/futures/OJ/contracts", params={"offset": 1}).json()
    assert [c["symbol"] for c in second["contracts"]] == ["OJ_202701"]
    assert client.get("/v1/futures/ZZ/contracts").json()["error"]["code"] == "unknown_symbol"
    assert client.get("/v1/futures/O%2FJ/contracts").status_code in (400, 404)
