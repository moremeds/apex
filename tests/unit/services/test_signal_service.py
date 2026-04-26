"""Tests for the signal service daemon event handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.events.domain_events import BarCloseEvent, TradingSignalEvent
from src.services.signal_service import _on_bar_close, _on_trading_signal


@pytest.mark.asyncio
async def test_event_handler_writes_bar_to_pg():
    repo = AsyncMock()
    event = BarCloseEvent(
        symbol="AAPL",
        timeframe="1d",
        open=150.0,
        high=155.0,
        low=149.0,
        close=153.0,
        volume=1000000,
        timestamp=MagicMock(),
    )
    await _on_bar_close(event, repo)
    repo.insert_bar.assert_awaited_once()


@pytest.mark.asyncio
async def test_event_handler_writes_signal_to_pg():
    repo = AsyncMock()
    event = TradingSignalEvent(
        symbol="AAPL",
        indicator="rsi",
        direction="bullish",
        strength=0.8,
        timeframe="1d",
        timestamp=MagicMock(),
    )
    await _on_trading_signal(event, repo)
    repo.insert_signal.assert_awaited_once()


@pytest.mark.asyncio
async def test_event_handler_saves_score_on_daily_bar():
    repo = AsyncMock()
    signal_engine = MagicMock()
    ie = MagicMock()
    ie.get_all_indicator_states.return_value = {
        ("AAPL", "1d", "regime_detector"): {
            "composite_score": 75.0,
            "trend_state": "up",
            "regime": "R0",
        }
    }
    signal_engine.indicator_engine = ie

    event = BarCloseEvent(
        symbol="AAPL",
        timeframe="1d",
        open=150.0,
        high=155.0,
        low=149.0,
        close=153.0,
        volume=1000000,
        timestamp=MagicMock(),
    )
    await _on_bar_close(event, repo, signal_engine=signal_engine)
    repo.save_score_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_daily_bar_skips_score_snapshot():
    repo = AsyncMock()
    signal_engine = MagicMock()

    event = BarCloseEvent(
        symbol="AAPL",
        timeframe="1h",
        open=150.0,
        high=155.0,
        low=149.0,
        close=153.0,
        volume=1000000,
        timestamp=MagicMock(),
    )
    await _on_bar_close(event, repo, signal_engine=signal_engine)
    repo.insert_bar.assert_awaited_once()
    repo.save_score_snapshot.assert_not_awaited()
