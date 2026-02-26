"""Tests for /api/symbols and /api/history/{symbol} routes."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.events.domain_events import BarData, QuoteTick
from src.server.routes.symbols import create_symbols_router


def _make_app(**kwargs):
    app = FastAPI()
    app.include_router(create_symbols_router(**kwargs))
    return TestClient(app)


def _make_quote_adapter(quotes: dict):
    """Create a mock QuoteProvider with canned quotes."""
    adapter = MagicMock()
    ticks = {}
    for sym, price in quotes.items():
        ticks[sym] = QuoteTick(
            symbol=sym,
            last=price,
            bid=price - 0.01,
            ask=price + 0.01,
            volume=10000,
            source="test",
            timestamp=datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc),
        )
    adapter.get_all_quotes.return_value = ticks
    return adapter


def _make_historical_adapter(bars_map: dict = None):
    """Create a mock HistoricalSourcePort.

    supports_timeframe and get_supported_timeframes are sync on the real adapter,
    so we use MagicMock for them and only make fetch_bars async.
    """
    adapter = MagicMock()
    adapter.supports_timeframe.return_value = True
    adapter.get_supported_timeframes.return_value = ["1m", "5m", "1h", "4h", "1d"]

    if bars_map:

        async def fetch_bars(symbol, tf, start, end):
            return bars_map.get((symbol, tf), [])

        adapter.fetch_bars = fetch_bars
    else:

        async def fetch_bars_empty(symbol, tf, start, end):
            return []

        adapter.fetch_bars = fetch_bars_empty

    return adapter


class TestSymbolsEndpoint:
    def test_list_symbols_with_quotes(self):
        adapter = _make_quote_adapter({"AAPL": 185.5, "SPY": 600.0})
        client = _make_app(quote_adapter=adapter)
        resp = client.get("/api/symbols")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert "AAPL" in data["symbols"]
        assert data["symbols"]["AAPL"]["last"] == 185.5

    def test_list_symbols_empty(self):
        adapter = _make_quote_adapter({})
        client = _make_app(quote_adapter=adapter)
        resp = client.get("/api/symbols")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_symbols_no_adapter(self):
        client = _make_app()
        resp = client.get("/api/symbols")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestHistoryEndpoint:
    def test_history_returns_bars(self):
        bars = [
            BarData(
                symbol="AAPL",
                timeframe="1d",
                open=184.0,
                high=186.0,
                low=183.5,
                close=185.5,
                volume=50000,
                timestamp=datetime(2026, 2, 24, tzinfo=timezone.utc),
            ),
            BarData(
                symbol="AAPL",
                timeframe="1d",
                open=185.5,
                high=187.0,
                low=185.0,
                close=186.0,
                volume=45000,
                timestamp=datetime(2026, 2, 25, tzinfo=timezone.utc),
            ),
        ]
        adapter = _make_historical_adapter({("AAPL", "1d"): bars})
        client = _make_app(historical_adapter=adapter)
        resp = client.get("/api/history/AAPL?tf=1d&bars=500")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert data["timeframe"] == "1d"
        assert data["count"] == 2
        assert data["bars"][0]["o"] == 184.0
        assert data["bars"][1]["c"] == 186.0

    def test_history_no_adapter_503(self):
        client = _make_app()
        resp = client.get("/api/history/AAPL")
        assert resp.status_code == 503

    def test_history_unsupported_timeframe(self):
        adapter = MagicMock()
        adapter.supports_timeframe.return_value = False
        adapter.get_supported_timeframes.return_value = ["1d"]
        client = _make_app(historical_adapter=adapter)
        resp = client.get("/api/history/AAPL?tf=1s")
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_history_limits_bars(self):
        bars = [
            BarData(
                symbol="AAPL",
                timeframe="1d",
                open=180.0 + i,
                high=181.0 + i,
                low=179.0 + i,
                close=180.5 + i,
                volume=1000,
                timestamp=datetime(2026, 1, i + 1, tzinfo=timezone.utc),
            )
            for i in range(10)
        ]
        adapter = _make_historical_adapter({("AAPL", "1d"): bars})
        client = _make_app(historical_adapter=adapter)
        resp = client.get("/api/history/AAPL?tf=1d&bars=3")
        assert resp.status_code == 200
        assert resp.json()["count"] == 3  # Only last 3

    def test_history_empty_result(self):
        adapter = _make_historical_adapter()
        client = _make_app(historical_adapter=adapter)
        resp = client.get("/api/history/AAPL?tf=1d")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0
