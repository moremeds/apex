"""Tests for ServerPipeline — wires tick → bar → indicator → signal → WS hub."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.domain.events.domain_events import IndicatorUpdateEvent, QuoteTick, TradingSignalEvent
from src.server.pipeline import ServerPipeline, _map_regime_to_flex, STRATEGY_INDICATORS


class MockHub:
    """Tracks broadcast calls."""

    def __init__(self):
        self.bars_broadcast = 0
        self.indicators_broadcast = 0
        self.signals_broadcast = 0
        self.strategy_states_broadcast = 0
        self.last_strategy_state_args = None

    async def broadcast_bar(self, symbol, tf, bar):
        self.bars_broadcast += 1

    async def broadcast_indicator(self, symbol, tf, name, value):
        self.indicators_broadcast += 1

    async def broadcast_signal(self, symbol, signal):
        self.signals_broadcast += 1

    async def broadcast_strategy_state(self, symbol, tf, indicator, state):
        self.strategy_states_broadcast += 1
        self.last_strategy_state_args = (symbol, tf, indicator, state)


def make_tick(symbol: str, price: float, ts: datetime) -> QuoteTick:
    return QuoteTick(
        symbol=symbol,
        last=price,
        volume=100,
        source="test",
        timestamp=ts,
    )


def test_pipeline_creates():
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m"])
    assert pipeline is not None
    assert len(pipeline._aggregators) == 1


@pytest.mark.asyncio
async def test_pipeline_on_tick_feeds_aggregators():
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m", "5m"])
    await pipeline.start()

    tick = make_tick("AAPL", 185.5, datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc))
    pipeline.on_tick(tick)
    # Pipeline should have aggregators for each timeframe
    assert len(pipeline._aggregators) == 2

    await pipeline.stop()


@pytest.mark.asyncio
async def test_pipeline_start_stop():
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m"])
    await pipeline.start()
    assert pipeline._started
    await pipeline.stop()
    assert not pipeline._started


@pytest.mark.asyncio
async def test_pipeline_bar_close_triggers_broadcast():
    """When enough ticks close a bar, the hub should get a broadcast."""
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1m"])
    await pipeline.start()

    # Generate ticks spanning a full minute to close a bar
    base = datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc)
    for i in range(5):
        tick = make_tick("AAPL", 185.0 + i * 0.1, base.replace(second=i * 12))
        pipeline.on_tick(tick)

    # Now send a tick in the NEXT minute to close the first bar
    next_min = base.replace(minute=31, second=0)
    pipeline.on_tick(make_tick("AAPL", 186.0, next_min))

    # Give the async event bus time to process events and schedule broadcasts
    await asyncio.sleep(0.5)

    assert hub.bars_broadcast >= 1

    await pipeline.stop()


# ── Strategy state broadcasting tests ────────────────


@pytest.mark.asyncio
async def test_strategy_indicator_broadcasts_state():
    """Strategy indicators (dual_macd etc.) broadcast their full state dict."""
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1d"])
    await pipeline.start()

    event = IndicatorUpdateEvent(
        symbol="AAPL",
        timeframe="1d",
        indicator="dual_macd",
        value=0.5,
        state={"slow_histogram": 0.5, "trend_state": "BULLISH"},
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    pipeline._on_indicator_update(event)

    # Let the scheduled coroutine execute
    await asyncio.sleep(0.3)

    assert hub.indicators_broadcast == 1  # always broadcasts indicator value
    assert hub.strategy_states_broadcast == 1
    sym, tf, ind, state = hub.last_strategy_state_args
    assert sym == "AAPL"
    assert tf == "1d"
    assert ind == "dual_macd"
    assert state["slow_histogram"] == 0.5
    assert state["date"] == "2026-03-02T16:00:00+00:00"

    await pipeline.stop()


@pytest.mark.asyncio
async def test_non_strategy_indicator_skips_state_broadcast():
    """Non-strategy indicators (rsi, ema) do NOT broadcast strategy_state."""
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1d"])
    await pipeline.start()

    event = IndicatorUpdateEvent(
        symbol="AAPL",
        timeframe="1d",
        indicator="rsi",
        value=65.0,
        state={"value": 65.0},
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    pipeline._on_indicator_update(event)
    await asyncio.sleep(0.3)

    assert hub.indicators_broadcast == 1
    assert hub.strategy_states_broadcast == 0  # rsi is NOT a strategy indicator

    await pipeline.stop()


@pytest.mark.asyncio
async def test_regime_detector_maps_to_flex_format():
    """regime_detector state is mapped to RegimeFlexRow via _map_regime_to_flex."""
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1d"])
    await pipeline.start()

    event = IndicatorUpdateEvent(
        symbol="SPY",
        timeframe="1d",
        indicator="regime_detector",
        value=75.0,
        state={
            "regime": "R0",
            "regime_changed": True,
            "previous_regime": "R1",
            "confidence": 80,
            "composite_score": 75.0,
        },
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    pipeline._on_indicator_update(event)
    await asyncio.sleep(0.3)

    assert hub.strategy_states_broadcast == 1
    _, _, ind, state = hub.last_strategy_state_args
    assert ind == "regime_detector"
    # Mapped to flex format (not raw regime_detector state)
    assert state["regime"] == "R0"
    assert state["target_exposure"] == 1.0
    assert state["signal"] == "R1→R0"
    assert "date" in state

    await pipeline.stop()


@pytest.mark.asyncio
async def test_empty_state_skips_strategy_broadcast():
    """Empty state dict (falsy) should NOT trigger strategy_state broadcast."""
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1d"])
    await pipeline.start()

    event = IndicatorUpdateEvent(
        symbol="AAPL",
        timeframe="1d",
        indicator="dual_macd",
        value=0.0,
        state={},  # empty dict is falsy
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    pipeline._on_indicator_update(event)
    await asyncio.sleep(0.3)

    assert hub.indicators_broadcast == 1
    assert hub.strategy_states_broadcast == 0

    await pipeline.stop()


# ── Signal persistence tests ────────────────


@pytest.mark.asyncio
async def test_trading_signal_persists_to_duckdb():
    """Trading signals are persisted to DuckDB with timeframe and indicator."""
    from src.server.persistence import ServerPersistence

    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1d"])
    persistence = ServerPersistence(duckdb_path=":memory:")
    pipeline._persistence = persistence
    await pipeline.start()

    event = TradingSignalEvent(
        symbol="AAPL",
        timeframe="1d",
        direction="buy",
        strength=0.8,
        indicator="rsi",
        trigger_rule="rsi_oversold",
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    pipeline._on_trading_signal(event)
    await asyncio.sleep(0.3)

    signals = persistence.query_signals(symbol="AAPL")
    assert len(signals) == 1
    assert signals[0]["rule"] == "rsi_oversold"
    assert signals[0]["direction"] == "bullish"  # mapped from "buy"
    assert signals[0]["timeframe"] == "1d"
    assert signals[0]["indicator"] == "rsi"

    persistence.close()
    await pipeline.stop()


@pytest.mark.asyncio
async def test_trading_signal_persistence_failure_logs_not_crashes():
    """If persistence raises, signal processing continues without crashing."""
    hub = MockHub()
    pipeline = ServerPipeline(hub=hub, timeframes=["1d"])

    mock_persistence = MagicMock()
    mock_persistence.insert_signal.side_effect = RuntimeError("DB write failed")
    pipeline._persistence = mock_persistence
    await pipeline.start()

    event = TradingSignalEvent(
        symbol="AAPL",
        timeframe="1d",
        direction="sell",
        strength=0.6,
        indicator="macd",
        trigger_rule="macd_cross_down",
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    # Should not raise
    pipeline._on_trading_signal(event)
    await asyncio.sleep(0.3)

    # Signal was still broadcast to WS hub despite persistence failure
    assert hub.signals_broadcast == 1
    # Signal was still buffered for advisor
    assert len(pipeline._signal_buffer.get("AAPL", [])) == 1

    await pipeline.stop()
