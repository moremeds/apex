"""Tests for signal service PG NOTIFY integration."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.signal_service import _on_bar_close, _on_trading_signal


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.insert_bar = AsyncMock()
    repo.save_score_snapshot = AsyncMock()
    repo.insert_signal = AsyncMock()
    return repo


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.notify = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_on_trading_signal_fires_notify(mock_repo, mock_db):
    """_on_trading_signal fires apex_signal NOTIFY after inserting signal."""
    event = MagicMock()
    event.symbol = "AAPL"
    event.direction = "bullish"
    event.strength = 0.85
    event.timeframe = "1h"
    event.indicator = "macd"
    event.trigger_rule = "macd_cross"
    event.timestamp = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)

    await _on_trading_signal(event, mock_repo, db=mock_db)

    mock_repo.insert_signal.assert_called_once()
    mock_db.notify.assert_called_once()
    call_args = mock_db.notify.call_args
    assert call_args[0][0] == "apex_signal"
    payload = call_args[0][1]
    assert payload["symbol"] == "AAPL"
    assert payload["direction"] == "bullish"
    assert payload["rule"] == "macd_cross"


@pytest.mark.asyncio
async def test_on_trading_signal_skips_notify_without_db(mock_repo):
    """_on_trading_signal still inserts but skips NOTIFY when db is None."""
    event = MagicMock()
    event.symbol = "AAPL"
    event.direction = "bullish"
    event.strength = 0.85
    event.timeframe = "1h"
    event.indicator = "macd"
    event.trigger_rule = "macd_cross"
    event.timestamp = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)

    await _on_trading_signal(event, mock_repo, db=None)

    mock_repo.insert_signal.assert_called_once()


@pytest.mark.asyncio
async def test_on_bar_close_fires_regime_notify_on_daily(mock_repo, mock_db):
    """_on_bar_close fires apex_regime notify on daily close with regime data."""
    event = MagicMock()
    event.symbol = "SPY"
    event.timeframe = "1d"
    event.open = 500.0
    event.high = 505.0
    event.low = 498.0
    event.close = 503.0
    event.volume = 100_000_000
    event.timestamp = datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc)

    mock_engine = MagicMock()
    ie = MagicMock()
    ie.get_all_indicator_states.return_value = {
        ("SPY", "1d", "regime_detector"): {
            "composite_score": 72.5,
            "trend_state": "uptrend",
            "regime": "R0",
            "regime_name": "Healthy Uptrend",
            "confidence": 88.0,
            "component_states": {"trend": "up", "vol": "low"},
        }
    }
    mock_engine.indicator_engine = ie

    await _on_bar_close(event, mock_repo, signal_engine=mock_engine, db=mock_db)

    mock_repo.insert_bar.assert_called_once()
    mock_repo.save_score_snapshot.assert_called_once()
    mock_db.notify.assert_called_once()
    call_args = mock_db.notify.call_args
    assert call_args[0][0] == "apex_regime"
    payload = call_args[0][1]
    assert payload["symbol"] == "SPY"
    assert payload["regime"] == "R0"
    assert payload["score"] == 72.5


@pytest.mark.asyncio
async def test_on_bar_close_no_regime_notify_on_intraday(mock_repo, mock_db):
    """_on_bar_close skips regime NOTIFY for non-daily timeframes."""
    event = MagicMock()
    event.symbol = "SPY"
    event.timeframe = "1h"
    event.open = 500.0
    event.high = 505.0
    event.low = 498.0
    event.close = 503.0
    event.volume = 1_000_000
    event.timestamp = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)

    mock_engine = MagicMock()
    await _on_bar_close(event, mock_repo, signal_engine=mock_engine, db=mock_db)

    mock_repo.insert_bar.assert_called_once()
    mock_db.notify.assert_not_called()
