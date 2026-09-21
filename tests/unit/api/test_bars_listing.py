"""GET /v1/equity/{symbol}/bars: adjustment basis and the listing filter.

Every fixture value below was read off the production macmini livewire lake on
2026-09-21 (Silver revision 76) and frozen here: SPY's raw and adjusted daily closes,
and VSCO's real archive/live bar rows. VSCO is a genuinely dual-resident ticker --
present in both bronze and bronze-delisted -- with no bars after 2026-06-01; never
write it a September 2026 bar. Bronze and bronze-delisted share one daily schema, so
both trees are written the same way here.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider
from tests.support.silver_manifest import publish_manifest

# (trade_date, open, high, low, close, volume) -- real SPY daily bars, 2026-09-21.
_SPY_RAW = [
    (dt.date(2026, 9, 16), 759.5, 761.67, 749.6, 754.05, 59_217_653),
    (dt.date(2026, 9, 17), 763.15, 763.57, 759.96, 762.6, 49_652_754),
]
# The same two bars through Silver revision 76's price_adjustment_factor
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

# VSCO's real bronze-delisted rows: 2021-07-21..23 are archive-only, and 2026-05-29 /
# 2026-06-01 are the two dates VSCO also carries in bronze/live (see _VSCO_LIVE).
# VSCO has no bars after 2026-06-01.
_VSCO_DELISTED = [
    (dt.date(2021, 7, 21), 55.0, 55.0, 39.99, 42.5, 80_637),
    (dt.date(2021, 7, 22), 42.75, 42.75, 39.79, 40.9, 352_595),
    (dt.date(2021, 7, 23), 41.98, 42.2, 40.99, 42.14, 75_030),
    (dt.date(2026, 5, 29), 58.5, 58.5, 55.0, 55.0, 3_956_703),
    (dt.date(2026, 6, 1), 52.26, 55.84, 51.0592, 54.3, 4_238_635),
]
# The same two shared dates as they appear in bronze/live -- identical prices, volume
# differing by exactly 1 from the archive. This is the real, observable "live wins"
# signal on a dual-resident read.
_VSCO_LIVE = [
    (dt.date(2026, 5, 29), 58.5, 58.5, 55.0, 55.0, 3_956_702),
    (dt.date(2026, 6, 1), 52.26, 55.84, 51.0592, 54.3, 4_238_634),
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
    silver = tmp_path / "silver"
    _write_daily(bronze, "SPY", _SPY_RAW)
    _write_daily(delisted, "VSCO", _VSCO_DELISTED)
    return {"bronze": bronze, "delisted": delisted, "silver": silver}


def _publish_silver(silver_root: Path) -> None:
    """One Silver generation holding SPY's adjusted daily artifact, published through
    the real manifest contract (SHA-256 verified on read)."""
    generation = silver_root / "generations" / "76"
    _write_daily(generation, "SPY", _SPY_ADJUSTED)
    publish_manifest(silver_root, 76)


async def _get(provider: Any, path: str) -> Any:
    app = create_app()
    app.state.ohlc_provider = provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


def _provider(lake: dict[str, Path], price_mode: str = "raw") -> LivewireOhlcProvider:
    return LivewireOhlcProvider(
        bronze_root=lake["bronze"],
        silver_root=lake["silver"],
        price_mode=price_mode,  # type: ignore[arg-type]
        delisted_root=lake["delisted"],
    )


_WINDOW = "start=2026-09-01T00:00:00Z&end=2026-09-30T00:00:00Z"
# VSCO's real bars run 2021-07-21 through 2026-06-01; the window used for any test
# that needs to see its actual rows.
_VSCO_WINDOW = "start=2021-01-01T00:00:00Z&end=2026-12-31T00:00:00Z"


async def test_raw_mode_reports_an_unadjusted_basis(lake: dict[str, Path]) -> None:
    resp = await _get(_provider(lake), f"/v1/equity/SPY/bars?{_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    assert (body["price_mode"], body["basis"]) == ("raw", "unadjusted")
    assert body["listing_status"] == "listed"
    assert body["bars"][-1]["close"] == 762.6


async def test_adjusted_mode_reports_a_split_dividend_basis(
    lake: dict[str, Path],
) -> None:
    """Silver's factor chain compounds dividends as well as splits, so the basis is
    named for both -- SPY's 2026-09-17 close moves 762.60 -> 760.711166 on a window
    with no split in it at all."""
    _publish_silver(lake["silver"])
    resp = await _get(_provider(lake, "adjusted"), f"/v1/equity/SPY/bars?{_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    assert (body["price_mode"], body["basis"]) == ("adjusted", "split+dividend")
    assert body["adjustment_revision"] == 76
    last = body["bars"][-1]
    assert last["open"] == pytest.approx(761.2598037410175)
    assert last["high"] == pytest.approx(761.6787634705219)
    assert last["low"] == pytest.approx(758.0777048431156)
    assert last["close"] == pytest.approx(760.711166)


async def test_delisted_serves_raw_bars_from_the_archived_tree(
    lake: dict[str, Path],
) -> None:
    resp = await _get(_provider(lake), f"/v1/equity/VSCO/bars?listing=delisted&{_VSCO_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["listing_status"] == "delisted"
    assert body["basis"] == "unadjusted"
    assert [bar["close"] for bar in body["bars"]] == [42.5, 40.9, 42.14, 55.0, 54.3]


async def test_delisted_tail_slices_from_the_epoch_when_start_is_omitted(
    lake: dict[str, Path],
) -> None:
    """With no `start`, a delisted read must not anchor its lookback at now -- VSCO's
    last real bar is 2026-06-01, years before any lookback window measured from today
    would reach. `_resolve_window`'s `from_epoch` path reads the whole archive and
    tail-slices to `limit` instead, so the tail of the real series comes back."""
    resp = await _get(_provider(lake), "/v1/equity/VSCO/bars?listing=delisted&limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert [bar["time"][:10] for bar in body["bars"]] == ["2026-05-29", "2026-06-01"]


async def test_adjusted_over_delisted_is_rejected_never_downgraded(
    lake: dict[str, Path],
) -> None:
    """Rule 12's spirit: there is no Silver over bronze-delisted, and answering an
    adjusted request with raw prices would be a silent basis swap."""
    _publish_silver(lake["silver"])
    resp = await _get(
        _provider(lake, "adjusted"),
        f"/v1/equity/VSCO/bars?listing=delisted&{_VSCO_WINDOW}",
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "adjusted_not_supported"
    assert error["message"] == "no Silver for delisted names; use price_mode=raw"


async def test_any_falls_back_to_the_archive_when_only_it_has_the_symbol(
    lake: dict[str, Path],
) -> None:
    resp = await _get(_provider(lake), f"/v1/equity/VSCO/bars?listing=any&{_VSCO_WINDOW}")
    assert resp.status_code == 200
    assert resp.json()["listing_status"] == "delisted"


async def test_any_stays_listed_when_only_the_live_tree_has_the_symbol(
    lake: dict[str, Path],
) -> None:
    resp = await _get(_provider(lake), f"/v1/equity/SPY/bars?listing=any&{_WINDOW}")
    assert resp.status_code == 200
    assert resp.json()["listing_status"] == "listed"


async def test_any_on_a_dual_resident_ticker_unions_with_the_live_tree_winning(
    lake: dict[str, Path],
) -> None:
    """VSCO is a REAL dual-resident ticker. bronze-delisted holds 2021-07-21..23
    (archive-only) plus 2026-05-29/06-01; bronze/live carries those same two 2026
    dates with identical prices and volume differing by exactly 1 -- the observable
    "live wins" signal. The union must keep the archive-only dates and report the
    LIVE volumes on the shared dates."""
    _write_daily(lake["bronze"], "VSCO", _VSCO_LIVE)
    resp = await _get(_provider(lake), f"/v1/equity/VSCO/bars?listing=any&{_VSCO_WINDOW}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["listing_status"] == "dual"
    bars = {bar["time"][:10]: bar for bar in body["bars"]}
    assert set(bars) == {
        "2021-07-21",
        "2021-07-22",
        "2021-07-23",
        "2026-05-29",
        "2026-06-01",
    }
    assert bars["2021-07-21"]["close"] == 42.5  # archive-only
    assert bars["2021-07-22"]["close"] == 40.9  # archive-only
    assert bars["2021-07-23"]["close"] == 42.14  # archive-only
    assert bars["2026-05-29"]["volume"] == 3_956_702  # shared -> live wins
    assert bars["2026-06-01"]["volume"] == 4_238_634  # shared -> live wins
    assert [bar["time"] for bar in body["bars"]] == sorted(bar["time"] for bar in body["bars"])


async def test_unknown_listing_filter_is_a_400(lake: dict[str, Path]) -> None:
    resp = await _get(_provider(lake), "/v1/equity/SPY/bars?listing=maybe")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_parameter"
