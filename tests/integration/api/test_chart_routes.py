"""Chart read-surface routes return schema-valid payloads (the argon chart contract).

ASGITransport does NOT run the lifespan, so each test pre-injects the fakes the
route reads off app.state (ohlc_provider / signal_repo).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

from httpx import ASGITransport, AsyncClient

from src.api.payload.validate import validate_payload
from src.api.server import create_app
from src.domain.events.domain_events import BarData

_DAY = timedelta(days=1)
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _series(n: int) -> List[BarData]:
    bars: List[BarData] = []
    price = 100.0
    for i in range(n):
        price += 1.0 if i % 3 else -0.7
        ts = _T0 + i * _DAY
        bars.append(
            BarData(
                symbol="AAPL",
                timeframe="1d",
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000 + i,
                vwap=price + 0.1,
                timestamp=ts,
                bar_start=ts,
            )
        )
    return bars


class _FakeProvider:
    def __init__(self, bars: List[BarData], dual_resident: set[str] | None = None) -> None:
        self._bars = bars
        # A real path that does not exist: _bars_response probes it to tell "no artifact"
        # (404) apart from "artifact exists, window empty" (200 with zero bars).
        self.bronze_root = Path("/nonexistent")
        self.delisted_root = None
        self._dual = dual_resident or set()

    def effective_price_mode(self, asset_class: str = "equity") -> str:
        return "raw"

    def is_dual_resident(self, symbol: str) -> bool:
        return symbol in self._dual

    async def fetch_rate_series(self, symbol: str, start: datetime, end: datetime) -> list:
        return []

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        asset_class: str = "equity",
        price_mode: str | None = None,
    ) -> List[BarData]:
        return [b for b in self._bars if start <= b.timestamp <= end]


def _series_ending(end_day: datetime, n: int) -> List[BarData]:
    """n daily bars ending at end_day (used to test the no-arg default window)."""
    bars: List[BarData] = []
    price = 100.0
    for i in range(n):
        price += 1.0 if i % 3 else -0.7
        ts = end_day - (n - 1 - i) * _DAY
        bars.append(
            BarData(
                symbol="AAPL",
                timeframe="1d",
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000 + i,
                vwap=price,
                timestamp=ts,
                bar_start=ts,
            )
        )
    return bars


class _FakeRepo:
    def __init__(self) -> None:
        self.last_limit: Any = None

    async def get_confluence_history(
        self,
        symbol: str,
        timeframe: str,
        start: Any = None,
        end: Any = None,
        limit: int = 100,
    ) -> List[dict]:
        self.last_limit = limit
        return [
            {
                "time": _T0,
                "alignment_score": 0.4,
                "bullish_count": 3,
                "bearish_count": 1,
                "neutral_count": 2,
                "total_indicators": 6,
                "dominant_direction": "bullish",
            }
        ]


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# --- /bars ---------------------------------------------------------------


async def test_get_bars_returns_valid_payload() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(50))
    async with _client(app) as c:
        resp = await c.get(
            "/bars/AAPL",
            params={
                "timeframe": "1d",
                "start": _T0.isoformat(),
                "end": (_T0 + 60 * _DAY).isoformat(),
            },
        )
    assert resp.status_code == 200
    payload = resp.json()
    validate_payload(payload, "bars_payload")
    assert payload["symbol"] == "AAPL"
    assert payload["count"] == 50
    assert payload["bars"][0]["close"] == 99.3  # i=0 -> i%3==0 -> price += -0.7


async def test_get_bars_503_when_provider_unconfigured() -> None:
    app = create_app()
    app.state.ohlc_provider = None
    async with _client(app) as c:
        resp = await c.get("/bars/AAPL", params={"timeframe": "1d"})
    assert resp.status_code == 503


async def test_get_bars_rejects_unsupported_timeframe() -> None:
    """Schema advertises 4h/15m/1w but livewire doesn't warehouse them -> 400, not 500."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(10))
    async with _client(app) as c:
        resp = await c.get("/bars/AAPL", params={"timeframe": "4h"})
    assert resp.status_code == 400


async def test_get_bars_limit_tail_slices() -> None:
    """No start/end -> tail-slice to `limit` most recent bars; the param is honored
    (the bug this fixes: /bars silently dropped limit and hard-capped at 500)."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(
        _series_ending(datetime(2026, 6, 1, tzinfo=timezone.utc), 600)
    )
    async with _client(app) as c:
        # default window keeps all 600 (default is 2000, above the series length)
        default_resp = await c.get("/bars/AAPL", params={"timeframe": "1d"})
        # explicit limit tail-slices
        limited_resp = await c.get("/bars/AAPL", params={"timeframe": "1d", "limit": "100"})
        # limit<=0 -> full history, no tail-slice
        full_resp = await c.get("/bars/AAPL", params={"timeframe": "1d", "limit": "0"})
    assert default_resp.json()["count"] == 600
    assert limited_resp.json()["count"] == 100
    assert full_resp.json()["count"] == 600


# --- /indicators ---------------------------------------------------------


async def test_get_indicators_returns_valid_payload() -> None:
    full = _series(200)
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(full)
    start, end = full[100].timestamp, full[150].timestamp
    async with _client(app) as c:
        resp = await c.get(
            "/indicators/AAPL",
            params={
                "timeframe": "1d",
                "indicator": "rsi",
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
    assert resp.status_code == 200
    payload = resp.json()
    validate_payload(payload, "indicator_series_payload")
    assert payload["indicator"] == "rsi"
    assert payload["count"] == 51  # inclusive 100..150
    assert "value" in payload["points"][0]["state"]


async def test_get_indicators_404_on_unknown_indicator() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(50))
    async with _client(app) as c:
        resp = await c.get("/indicators/AAPL", params={"timeframe": "1d", "indicator": "not_real"})
    assert resp.status_code == 404


async def test_get_indicators_503_when_provider_unconfigured() -> None:
    app = create_app()
    app.state.ohlc_provider = None
    async with _client(app) as c:
        resp = await c.get("/indicators/AAPL", params={"timeframe": "1d", "indicator": "rsi"})
    assert resp.status_code == 503


async def test_get_indicators_rejects_unsupported_timeframe() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(10))
    async with _client(app) as c:
        resp = await c.get("/indicators/AAPL", params={"timeframe": "1w", "indicator": "rsi"})
    assert resp.status_code == 400


# --- /confluence ---------------------------------------------------------


async def test_get_confluence_returns_valid_payload() -> None:
    app = create_app()
    app.state.signal_repo = _FakeRepo()
    async with _client(app) as c:
        resp = await c.get("/confluence/AAPL", params={"timeframe": "1d"})
    assert resp.status_code == 200
    payload = resp.json()
    validate_payload(payload, "confluence_payload")
    assert payload["points"][0]["dominant_direction"] == "bullish"


async def test_get_confluence_503_when_repo_unconfigured() -> None:
    app = create_app()
    app.state.signal_repo = None
    async with _client(app) as c:
        resp = await c.get("/confluence/AAPL", params={"timeframe": "1d"})
    assert resp.status_code == 503


async def test_get_confluence_passes_limit_to_repo() -> None:
    """Confluence must not silently cap at the repo default -> expose `limit`."""
    repo = _FakeRepo()
    app = create_app()
    app.state.signal_repo = repo
    async with _client(app) as c:
        resp = await c.get("/confluence/AAPL", params={"timeframe": "1d", "limit": "3"})
    assert resp.status_code == 200
    assert repo.last_limit == 3


async def test_get_confluence_rejects_out_of_range_limit() -> None:
    """limit is bounded (ge=1, le=5000) so a negative LIMIT can't reach Postgres."""
    app = create_app()
    app.state.signal_repo = _FakeRepo()
    async with _client(app) as c:
        resp = await c.get("/confluence/AAPL", params={"timeframe": "1d", "limit": "0"})
    assert resp.status_code == 422


# --- Task 8: /v1 routes and deprecated aliases -------------------------------------

import pytest  # noqa: E402

from src.infrastructure.adapters.livewire.ohlc_provider import (  # noqa: E402
    AdjustedDataUnavailable,
)


@pytest.mark.asyncio
async def test_v1_bars_route_serves_equity() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(5))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/equity/AAPL/bars", params={"timeframe": "1d"})
    assert r.status_code == 200
    body = r.json()
    assert body["asset_class"] == "equity"
    assert body["price_mode"] == "raw"
    assert body["listing_status"] == "listed"
    validate_payload(body, "bars_payload")


@pytest.mark.asyncio
async def test_flat_alias_matches_v1_and_is_marked_deprecated() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(5))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        legacy = await c.get("/bars/AAPL", params={"timeframe": "1d"})
        v1 = await c.get("/v1/equity/AAPL/bars", params={"timeframe": "1d"})
    assert legacy.json()["bars"] == v1.json()["bars"]
    assert legacy.headers["Deprecation"] == "true"
    assert "Sunset" in legacy.headers
    assert "successor-version" in legacy.headers["Link"]


@pytest.mark.asyncio
async def test_v1_indicators_route_works_off_equity() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(300))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(
            "/v1/equity/AAPL/indicators", params={"indicator": "rsi", "timeframe": "1d"}
        )
    assert r.status_code == 200
    validate_payload(r.json(), "indicator_series_payload")


@pytest.mark.asyncio
async def test_v1_confluence_route_is_registered() -> None:
    """PG-backed and equity-only: with no repo configured it must be a typed 503,
    not a 404 -- proving the route exists."""
    app = create_app()
    app.state.signal_repo = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/equity/AAPL/confluence")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "provider_not_configured"


@pytest.mark.asyncio
async def test_unknown_asset_class_is_400_with_a_code() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/crypto/BTC/bars")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_asset_class"


@pytest.mark.asyncio
async def test_intraday_on_a_daily_only_class_is_400_with_a_code() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/volatility/VIX/bars", params={"timeframe": "1m"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_timeframe"


# Explicit window covering the fixture series: the no-arg default window is anchored
# to now(), and a 1m default lookback (~14 days) would not reach a Jan-2026 fixture.
_WINDOW = {"start": _T0.isoformat(), "end": (_T0 + 30 * _DAY).isoformat()}


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe", ["5m", "30m", "1h", "1d"])
async def test_volatility_intraday_is_accepted(timeframe: str) -> None:
    """Measured on the production lake: volatility publishes 5m/30m/1h/1d (never 1m).
    Rejecting the intraday ladder would make real parquet files unreachable."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(5))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/volatility/VIX/bars", params={"timeframe": timeframe, **_WINDOW})
    assert r.status_code == 200, r.text
    assert r.json()["asset_class"] == "volatility"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe", ["1m", "5m", "30m", "1h", "1d"])
async def test_fx_intraday_is_accepted(timeframe: str) -> None:
    """All 21 production FX pairs publish the full 1m/5m/30m/1h/1d ladder."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(5))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/fx/DXY/bars", params={"timeframe": timeframe, **_WINDOW})
    assert r.status_code == 200, r.text
    assert r.json()["asset_class"] == "fx"


@pytest.mark.asyncio
async def test_rates_through_the_bars_route_is_redirected_not_500() -> None:
    """A yield has no OHLC; the bars route must name the right route rather than
    building a null-price payload that fails egress validation as a 500."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/rates/DGS10/bars")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_asset_class"
    assert "series" in r.json()["error"]["message"]


@pytest.mark.asyncio
async def test_rates_series_route_serves_the_yield_shape() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/rates/DGS10/series")
    assert r.status_code == 200
    body = r.json()
    assert body["asset_class"] == "rates"
    validate_payload(body, "rates_series_payload")


@pytest.mark.asyncio
async def test_adjusted_unavailable_is_503_not_500() -> None:
    """The 243 symbols with bronze but no Silver must be distinguishable from a crash."""

    class _Quarantined:
        bronze_root = Path("/nonexistent")
        delisted_root = None

        def effective_price_mode(self, asset_class: str = "equity") -> str:
            return "adjusted"

        async def fetch_bars(self, *a: object, **kw: object) -> list:
            raise AdjustedDataUnavailable("Silver daily artifact is missing for HON")

    app = create_app()
    app.state.ohlc_provider = _Quarantined()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/equity/HON/bars")
    assert r.status_code == 503
    body = r.json()["error"]
    assert body["code"] == "adjusted_unavailable"
    assert body["symbol"] == "HON"


@pytest.mark.asyncio
async def test_adjusted_requested_on_non_equity_is_400() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/fx/DXY/bars", params={"price_mode": "adjusted"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "adjusted_not_supported"


@pytest.mark.asyncio
async def test_listing_any_is_409_because_tickers_are_reused() -> None:
    """2,345 tickers exist in both the live and delisted trees (measured 2026-08-23).
    Only those are ambiguous; a live-only symbol under listing=any is served normally."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(_series(5), dual_resident={"BBBY"})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        ambiguous = await c.get("/v1/equity/BBBY/bars", params={"listing": "any"})
        unambiguous = await c.get("/v1/equity/AAPL/bars", params={"listing": "any"})
    assert ambiguous.status_code == 409
    assert ambiguous.json()["error"]["code"] == "ambiguous_symbol"
    assert unambiguous.status_code == 200
    assert unambiguous.json()["listing_status"] == "listed"


@pytest.mark.asyncio
async def test_delisted_is_a_typed_501_not_a_silent_empty() -> None:
    app = create_app()
    app.state.ohlc_provider = _FakeProvider([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/equity/BBBY/bars", params={"listing": "delisted"})
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "not_yet_available"


@pytest.mark.asyncio
async def test_missing_artifact_is_404_not_an_empty_200() -> None:
    """An absent parquet is a 404; an existing parquet with a quiet window is a 200."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider([])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/equity/NOSUCHTICKER/bars")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_symbol"
