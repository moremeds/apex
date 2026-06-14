"""Chart read-surface routes return schema-valid payloads (the argon chart contract).

ASGITransport does NOT run the lifespan, so each test pre-injects the fakes the
route reads off app.state (ohlc_provider / signal_repo).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    def __init__(self, bars: List[BarData]) -> None:
        self._bars = bars

    async def fetch_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
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
        self, symbol: str, timeframe: str, start: Any = None, end: Any = None, limit: int = 100
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


async def test_get_bars_default_window_tail_slices_to_500() -> None:
    """No start/end -> most recent 500 bars even when more exist in the lookback."""
    app = create_app()
    app.state.ohlc_provider = _FakeProvider(
        _series_ending(datetime(2026, 6, 1, tzinfo=timezone.utc), 600)
    )
    async with _client(app) as c:
        resp = await c.get("/bars/AAPL", params={"timeframe": "1d"})
    assert resp.status_code == 200
    assert resp.json()["count"] == 500


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
