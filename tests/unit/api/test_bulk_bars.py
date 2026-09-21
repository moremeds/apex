"""GET /v1/equity/bars: the bulk read, and the route-matching it depends on.

Same frozen lake as test_bars_listing.py: real SPY and QQQ daily bars, and VSCO's real
bronze-delisted archive rows, all read off the production macmini lake on 2026-09-21
(Silver revision 76).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app
from src.infrastructure.adapters.livewire.ohlc_provider import (
    AdjustedDataUnavailable,
    LivewireOhlcProvider,
)
from tests.support.silver_manifest import publish_manifest

_SPY_RAW = [
    (dt.date(2026, 9, 16), 759.5, 761.67, 749.6, 754.05, 59_217_653),
    (dt.date(2026, 9, 17), 763.15, 763.57, 759.96, 762.6, 49_652_754),
]
# The same two SPY bars through Silver revision 76's price_adjustment_factor
# (0.9975231654864936 through 09-17) -- complete OHLCV, not close-only: the real
# Silver artifact carries adjusted open/high/low too, never a raw/adjusted mix.
_SPY_ADJUSTED = [
    (
        dt.date(2026, 9, 16),
        757.6188441869919,
        759.7834694560976,
        747.7433648486756,
        752.1823429350904,
        59_217_653,
    ),
    (
        dt.date(2026, 9, 17),
        761.2598037410175,
        761.6787634705219,
        758.0777048431156,
        760.711166,
        49_652_754,
    ),
]
# QQQ, Silver revision 76 -- adjusted equals raw for this window (factor 1.0; the last
# ex-dividend, 2026-06-22, predates it).
_QQQ_RAW = [
    (dt.date(2026, 9, 16), 708.0, 711.88, 700.0, 704.72, 35_687_958),
    (dt.date(2026, 9, 17), 715.95, 718.0401, 713.32, 716.92, 37_270_287),
]
# VSCO's real bronze-delisted rows: 2021-07-21..23 archive-only, 2026-05-29 / 06-01 the
# dates it also carries live. VSCO has no bars after 2026-06-01.
_VSCO_DELISTED = [
    (dt.date(2021, 7, 21), 55.0, 55.0, 39.99, 42.5, 80_637),
    (dt.date(2021, 7, 22), 42.75, 42.75, 39.79, 40.9, 352_595),
    (dt.date(2021, 7, 23), 41.98, 42.2, 40.99, 42.14, 75_030),
    (dt.date(2026, 5, 29), 58.5, 58.5, 55.0, 55.0, 3_956_703),
    (dt.date(2026, 6, 1), 52.26, 55.84, 51.0592, 54.3, 4_238_635),
]


def _write_daily(root: Path, symbol: str, rows: list) -> None:
    directory = root / "asset_class=equity" / f"symbol={symbol}"
    directory.mkdir(parents=True, exist_ok=True)
    close = [r[4] for r in rows]
    pd.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "symbol_id": [1] * len(rows),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": close,
            "adj_close": close,
            "volume": [r[5] for r in rows],
        }
    ).to_parquet(directory / "1d.parquet")


@pytest.fixture
def lake(tmp_path: Path) -> dict[str, Path]:
    bronze = tmp_path / "bronze"
    delisted = tmp_path / "bronze-delisted"
    _write_daily(bronze, "SPY", _SPY_RAW)
    _write_daily(bronze, "QQQ", _QQQ_RAW)
    _write_daily(delisted, "VSCO", _VSCO_DELISTED)
    return {"bronze": bronze, "delisted": delisted, "silver": tmp_path / "silver"}


def _provider(lake: dict[str, Path], price_mode: str = "raw") -> LivewireOhlcProvider:
    return LivewireOhlcProvider(
        bronze_root=lake["bronze"],
        silver_root=lake["silver"],
        price_mode=price_mode,  # type: ignore[arg-type]
        delisted_root=lake["delisted"],
    )


async def _get(provider: Any, path: str) -> Any:
    app = create_app()
    app.state.ohlc_provider = provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


_WINDOW = "start=2026-09-01T00:00:00Z&end=2026-09-30T00:00:00Z"


async def test_bars_is_not_swallowed_as_a_symbol(lake: dict[str, Path]) -> None:
    """/v1/equity/bars must reach the bulk route, not /v1/{asset_class}/{symbol} with
    symbol="bars". The bulk payload is the proof: it has a `symbols` map."""
    resp = await _get(_provider(lake), f"/v1/equity/bars?symbols=SPY&{_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["symbols"]) == {"SPY"}
    assert "listing_status" not in body  # the per-symbol payload's field, not this one


async def test_bulk_serves_several_symbols_on_one_basis(lake: dict[str, Path]) -> None:
    resp = await _get(_provider(lake), f"/v1/equity/bars?symbols=spy,qqq,SPY&{_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    # Upper-cased and de-duplicated, like /v1/equity/returns.
    assert sorted(body["symbols"]) == ["QQQ", "SPY"]
    assert (body["price_mode"], body["basis"]) == ("raw", "unadjusted")
    assert body["timeframe"] == "1d"
    assert body["adjustment_revision"] is None
    assert body["symbols"]["SPY"]["listing_status"] == "listed"
    assert [bar["close"] for bar in body["symbols"]["SPY"]["bars"]] == [754.05, 762.6]
    assert [bar["close"] for bar in body["symbols"]["QQQ"]["bars"]] == [704.72, 716.92]
    assert body["missing"] == {}


async def test_adjusted_bulk_pins_one_revision_for_every_symbol(
    lake: dict[str, Path],
) -> None:
    generation = lake["silver"] / "generations" / "76"
    _write_daily(generation, "SPY", _SPY_ADJUSTED)
    publish_manifest(lake["silver"], 76)
    resp = await _get(_provider(lake, "adjusted"), f"/v1/equity/bars?symbols=SPY&{_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    assert (body["price_mode"], body["basis"]) == ("adjusted", "split+dividend")
    assert body["adjustment_revision"] == 76
    last = body["symbols"]["SPY"]["bars"][-1]
    assert last["open"] == pytest.approx(761.2598037410175)
    assert last["high"] == pytest.approx(761.6787634705219)
    assert last["low"] == pytest.approx(758.0777048431156)
    assert last["close"] == pytest.approx(760.711166)


async def test_a_symbol_that_cannot_be_served_lands_in_missing(
    lake: dict[str, Path],
) -> None:
    """One bad ticker must not cost the other 199 their bars."""
    resp = await _get(_provider(lake), f"/v1/equity/bars?symbols=SPY,VSCO&{_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["symbols"]) == {"SPY"}
    assert "VSCO" in body["missing"]
    assert "no artifact" in body["missing"]["VSCO"]


async def test_adjusted_over_a_delisted_name_is_reported_per_symbol(
    lake: dict[str, Path],
) -> None:
    generation = lake["silver"] / "generations" / "76"
    _write_daily(generation, "SPY", _SPY_ADJUSTED)
    publish_manifest(lake["silver"], 76)
    resp = await _get(
        _provider(lake, "adjusted"),
        f"/v1/equity/bars?symbols=SPY,VSCO&listing=any&{_WINDOW}",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["symbols"]) == {"SPY"}
    assert body["missing"]["VSCO"] == "no Silver for delisted names; use price_mode=raw"


async def test_listing_any_labels_each_series(lake: dict[str, Path]) -> None:
    resp = await _get(_provider(lake), f"/v1/equity/bars?symbols=SPY,VSCO&listing=any&{_WINDOW}")
    assert resp.status_code == 200
    statuses = {sym: row["listing_status"] for sym, row in resp.json()["symbols"].items()}
    assert statuses == {"SPY": "listed", "VSCO": "delisted"}


async def test_delisted_tail_slices_from_the_epoch_when_start_is_omitted(
    lake: dict[str, Path],
) -> None:
    """Same `from_epoch` behavior as the single-symbol route: with no `start`, a
    delisted read tail-slices the whole archive instead of anchoring a lookback at
    now, so VSCO's real tail (2026-05-29, 2026-06-01) comes back."""
    resp = await _get(
        _provider(lake),
        "/v1/equity/bars?symbols=VSCO&listing=delisted&price_mode=raw&limit=2",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [bar["time"][:10] for bar in body["symbols"]["VSCO"]["bars"]] == [
        "2026-05-29",
        "2026-06-01",
    ]


async def test_symbols_is_required(lake: dict[str, Path]) -> None:
    resp = await _get(_provider(lake), "/v1/equity/bars?symbols=")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_parameter"


async def test_symbol_cap_is_enforced(lake: dict[str, Path]) -> None:
    symbols = ",".join(f"SYM{i}" for i in range(201))
    resp = await _get(_provider(lake), f"/v1/equity/bars?symbols={symbols}")
    assert resp.status_code == 400
    assert "at most 200" in resp.json()["error"]["message"]


async def test_unsupported_timeframe_is_a_400(lake: dict[str, Path]) -> None:
    resp = await _get(_provider(lake), "/v1/equity/bars?symbols=SPY&timeframe=3d")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_timeframe"


async def test_unknown_listing_is_a_400_not_an_empty_table(lake: dict[str, Path]) -> None:
    resp = await _get(_provider(lake), f"/v1/equity/bars?symbols=SPY&listing=maybe&{_WINDOW}")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_parameter"


async def test_pin_failure_is_a_typed_503_not_a_500(lake: dict[str, Path]) -> None:
    provider = _provider(lake, price_mode="adjusted")

    def _refuse() -> LivewireOhlcProvider:
        raise AdjustedDataUnavailable("Silver root is not configured")

    provider.pin_snapshot = _refuse  # type: ignore[method-assign]
    resp = await _get(provider, f"/v1/equity/bars?symbols=SPY&price_mode=adjusted&{_WINDOW}")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "adjusted_unavailable"
