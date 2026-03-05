"""Tests for Longbridge QuoteAdapter — tests adapter logic with mocked SDK."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.domain.events.domain_events import QuoteTick
from src.domain.interfaces.quote_provider import QuoteProvider


@pytest.fixture
def adapter():
    from src.infrastructure.adapters.longbridge.quote_adapter import LongbridgeQuoteAdapter

    return LongbridgeQuoteAdapter()


def test_implements_quote_provider(adapter):
    assert isinstance(adapter, QuoteProvider)


def test_not_connected_initially(adapter):
    assert adapter.is_connected() is False


def test_set_quote_callback(adapter):
    cb = MagicMock()
    adapter.set_quote_callback(cb)
    adapter._connected = True  # Simulate connected state

    # Simulate SDK pushing a quote — build a mock PushQuote
    mock_quote = MagicMock()
    mock_quote.symbol = "AAPL.US"
    mock_quote.last_done = 185.50
    mock_quote.volume = 1000
    mock_quote.turnover = 185500.0
    mock_quote.timestamp = datetime(2026, 2, 25, 15, 30, 0, tzinfo=timezone.utc)
    mock_quote.open = 184.0
    mock_quote.high = 186.0
    mock_quote.low = 183.5
    mock_quote.trade_status = MagicMock()

    adapter._on_sdk_quote("AAPL.US", mock_quote)
    cb.assert_called_once()
    tick = cb.call_args[0][0]
    assert isinstance(tick, QuoteTick)
    assert tick.symbol == "AAPL"  # stripped .US suffix
    assert tick.source == "longbridge"
    assert tick.last == 185.50
    assert tick.volume == 1000


def test_set_quote_callback_none(adapter):
    adapter.set_quote_callback(None)
    adapter._connected = True  # Simulate connected state
    # Should not crash when quote arrives with no callback
    mock_quote = MagicMock()
    mock_quote.symbol = "AAPL.US"
    mock_quote.last_done = 185.50
    mock_quote.volume = 1000
    mock_quote.turnover = 185500.0
    mock_quote.timestamp = datetime(2026, 2, 25, tzinfo=timezone.utc)
    mock_quote.open = 184.0
    mock_quote.high = 186.0
    mock_quote.low = 183.5
    mock_quote.trade_status = MagicMock()
    adapter._on_sdk_quote("AAPL.US", mock_quote)  # no crash


def test_symbol_mapping(adapter):
    """Internal AAPL ↔ Longbridge AAPL.US"""
    assert adapter._to_lb_symbol("AAPL") == "AAPL.US"
    assert adapter._to_internal_symbol("AAPL.US") == "AAPL"
    # Already has suffix
    assert adapter._to_lb_symbol("AAPL.US") == "AAPL.US"
    # HK symbols
    assert adapter._to_lb_symbol("0700.HK") == "0700.HK"
    assert adapter._to_internal_symbol("0700.HK") == "0700"


def test_subscribe_tracks_symbols(adapter):
    adapter._connected = True
    mock_ctx = MagicMock()
    adapter._ctx = mock_ctx

    asyncio.run(adapter.subscribe_quotes(["AAPL", "SPY"]))
    assert set(adapter.get_subscribed_symbols()) == {"AAPL", "SPY"}
    mock_ctx.subscribe.assert_called_once()


def test_unsubscribe_removes_symbols(adapter):
    adapter._connected = True
    mock_ctx = MagicMock()
    adapter._ctx = mock_ctx
    adapter._subscribed = {"AAPL", "SPY", "QQQ"}

    asyncio.run(adapter.unsubscribe_quotes(["AAPL"]))
    assert "AAPL" not in adapter.get_subscribed_symbols()
    assert "SPY" in adapter.get_subscribed_symbols()


def test_get_latest_quote(adapter):
    adapter._connected = True  # Simulate connected state for _on_sdk_quote
    mock_quote = MagicMock()
    mock_quote.symbol = "AAPL.US"
    mock_quote.last_done = 185.50
    mock_quote.volume = 1000
    mock_quote.turnover = 185500.0
    mock_quote.timestamp = datetime(2026, 2, 25, tzinfo=timezone.utc)
    mock_quote.open = 184.0
    mock_quote.high = 186.0
    mock_quote.low = 183.5
    mock_quote.trade_status = MagicMock()

    adapter._on_sdk_quote("AAPL.US", mock_quote)
    tick = adapter.get_latest_quote("AAPL")
    assert tick is not None
    assert tick.last == 185.50
    assert tick.symbol == "AAPL"


def test_get_latest_quote_missing(adapter):
    assert adapter.get_latest_quote("NVDA") is None


def test_get_all_quotes(adapter):
    assert adapter.get_all_quotes() == {}
