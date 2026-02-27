"""Tests for Longbridge HistoricalAdapter — mock-based tests for adapter logic."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.domain.events.domain_events import BarData
from src.domain.interfaces.historical_source import HistoricalSourcePort


@pytest.fixture
def adapter():
    from src.infrastructure.adapters.longbridge.historical_adapter import (
        LongbridgeHistoricalAdapter,
    )

    return LongbridgeHistoricalAdapter()


def test_implements_historical_source(adapter):
    assert isinstance(adapter, HistoricalSourcePort)


def test_source_name(adapter):
    assert adapter.source_name == "longbridge"


def test_supported_timeframes(adapter):
    assert adapter.supports_timeframe("1d")
    assert adapter.supports_timeframe("1h")
    assert adapter.supports_timeframe("1m")
    assert adapter.supports_timeframe("5m")
    assert adapter.supports_timeframe("4h")
    assert not adapter.supports_timeframe("1s")
    assert not adapter.supports_timeframe("tick")


def test_get_supported_timeframes(adapter):
    tfs = adapter.get_supported_timeframes()
    assert "1d" in tfs
    assert "1h" in tfs
    assert "1m" in tfs


@pytest.mark.asyncio
async def test_fetch_bars_returns_bar_data(adapter):
    # Mock the SDK context
    mock_candle = MagicMock()
    mock_candle.open = 184.0
    mock_candle.high = 186.0
    mock_candle.low = 183.5
    mock_candle.close = 185.5
    mock_candle.volume = 50000
    mock_candle.turnover = 9250000.0
    mock_candle.timestamp = datetime(2026, 2, 24, 5, 0, 0, tzinfo=timezone.utc)

    mock_ctx = MagicMock()
    mock_ctx.candlesticks.return_value = [mock_candle]
    adapter._ctx = mock_ctx
    adapter._connected = True

    start = datetime(2026, 2, 20, tzinfo=timezone.utc)
    end = datetime(2026, 2, 25, tzinfo=timezone.utc)
    bars = await adapter.fetch_bars("AAPL", "1d", start, end)

    assert len(bars) == 1
    bar = bars[0]
    assert isinstance(bar, BarData)
    assert bar.source == "longbridge"
    assert bar.symbol == "AAPL"
    assert bar.timeframe == "1d"
    assert bar.open == 184.0
    assert bar.close == 185.5
    assert bar.volume == 50000


@pytest.mark.asyncio
async def test_fetch_bars_empty_on_no_data(adapter):
    mock_ctx = MagicMock()
    mock_ctx.candlesticks.return_value = []
    adapter._ctx = mock_ctx
    adapter._connected = True

    start = datetime(2026, 2, 20, tzinfo=timezone.utc)
    end = datetime(2026, 2, 25, tzinfo=timezone.utc)
    bars = await adapter.fetch_bars("AAPL", "1d", start, end)
    assert bars == []


@pytest.mark.asyncio
async def test_fetch_bars_not_connected(adapter):
    with pytest.raises(ConnectionError):
        start = datetime(2026, 2, 20, tzinfo=timezone.utc)
        end = datetime(2026, 2, 25, tzinfo=timezone.utc)
        await adapter.fetch_bars("AAPL", "1d", start, end)
