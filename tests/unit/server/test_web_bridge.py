"""Tests for WebBridge — bridges domain events to WS hub, DuckDB, and Advisor."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.domain.events.domain_events import IndicatorUpdateEvent, QuoteTick, TradingSignalEvent
from src.domain.signals.signal_engine import SignalEngine
from src.server.web_bridge import WebBridge


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


class MockEventBus:
    """Simple mock event bus for WebBridge tests."""

    def __init__(self):
        self.subscriptions = {}

    def subscribe(self, event_type, callback):
        self.subscriptions.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type, callback):
        if event_type in self.subscriptions:
            self.subscriptions[event_type] = [
                cb for cb in self.subscriptions[event_type] if cb is not callback
            ]


def make_tick(symbol: str, price: float, ts: datetime) -> QuoteTick:
    return QuoteTick(
        symbol=symbol,
        last=price,
        volume=100,
        source="test",
        timestamp=ts,
    )


# ── SignalEngine creation tests ─────────────────────────────


def test_signal_engine_creates():
    bus = MockEventBus()
    engine = SignalEngine(event_bus=bus, timeframes=["1m"])
    engine.start()
    assert engine.is_started
    assert len(engine._aggregators) == 1
    engine.stop()


def test_signal_engine_on_tick_feeds_aggregators():
    bus = MockEventBus()
    engine = SignalEngine(event_bus=bus, timeframes=["1m", "5m"])
    engine.start()

    tick = make_tick("AAPL", 185.5, datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc))
    engine.on_tick(tick)
    assert len(engine._aggregators) == 2

    engine.stop()


def test_signal_engine_start_stop():
    bus = MockEventBus()
    engine = SignalEngine(event_bus=bus, timeframes=["1m"])
    engine.start()
    assert engine.is_started
    engine.stop()
    assert not engine.is_started


# ── WebBridge indicator broadcast tests ─────────────────────


def test_web_bridge_creates():
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)
    hub = MockHub()
    bridge = WebBridge(event_bus=bus, signal_engine=engine, hub=hub, persistence=None)
    assert bridge is not None


def test_web_bridge_start_stop():
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)
    hub = MockHub()
    bridge = WebBridge(event_bus=bus, signal_engine=engine, hub=hub, persistence=None)
    bridge.start()
    assert bridge._started
    bridge.stop()
    assert not bridge._started


@pytest.mark.asyncio
async def test_strategy_indicator_broadcasts_state():
    """Strategy indicators (dual_macd etc.) broadcast their full state dict."""
    hub = MockHub()
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)
    loop = asyncio.get_running_loop()
    bridge = WebBridge(event_bus=bus, signal_engine=engine, hub=hub, persistence=None, loop=loop)
    bridge.start()

    event = IndicatorUpdateEvent(
        symbol="AAPL",
        timeframe="1d",
        indicator="dual_macd",
        value=0.5,
        state={"slow_histogram": 0.5, "trend_state": "BULLISH"},
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    bridge._on_indicator_update(event)

    # Let the scheduled coroutines execute
    await asyncio.sleep(0.3)

    assert hub.indicators_broadcast == 1
    assert hub.strategy_states_broadcast == 1
    sym, tf, ind, state = hub.last_strategy_state_args
    assert sym == "AAPL"
    assert tf == "1d"
    assert ind == "dual_macd"
    assert state["slow_histogram"] == 0.5
    assert state["date"] == "2026-03-02T16:00:00+00:00"

    bridge.stop()


@pytest.mark.asyncio
async def test_non_strategy_indicator_skips_state_broadcast():
    """Non-strategy indicators (rsi, ema) do NOT broadcast strategy_state."""
    hub = MockHub()
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)
    loop = asyncio.get_running_loop()
    bridge = WebBridge(event_bus=bus, signal_engine=engine, hub=hub, persistence=None, loop=loop)
    bridge.start()

    event = IndicatorUpdateEvent(
        symbol="AAPL",
        timeframe="1d",
        indicator="rsi",
        value=65.0,
        state={"value": 65.0},
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    bridge._on_indicator_update(event)
    await asyncio.sleep(0.3)

    assert hub.indicators_broadcast == 1
    assert hub.strategy_states_broadcast == 0

    bridge.stop()


@pytest.mark.asyncio
async def test_regime_detector_maps_to_flex_format():
    """regime_detector state is mapped to RegimeFlexRow via map_regime_to_flex."""
    hub = MockHub()
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)
    loop = asyncio.get_running_loop()
    bridge = WebBridge(event_bus=bus, signal_engine=engine, hub=hub, persistence=None, loop=loop)
    bridge.start()

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
    bridge._on_indicator_update(event)
    await asyncio.sleep(0.3)

    assert hub.strategy_states_broadcast == 1
    _, _, ind, state = hub.last_strategy_state_args
    assert ind == "regime_detector"
    assert state["regime"] == "R0"
    assert state["target_exposure"] == 1.0
    assert state["signal"] == "R1→R0"
    assert "date" in state

    bridge.stop()


@pytest.mark.asyncio
async def test_empty_state_skips_strategy_broadcast():
    """Empty state dict (falsy) should NOT trigger strategy_state broadcast."""
    hub = MockHub()
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)
    loop = asyncio.get_running_loop()
    bridge = WebBridge(event_bus=bus, signal_engine=engine, hub=hub, persistence=None, loop=loop)
    bridge.start()

    event = IndicatorUpdateEvent(
        symbol="AAPL",
        timeframe="1d",
        indicator="dual_macd",
        value=0.0,
        state={},
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    bridge._on_indicator_update(event)
    await asyncio.sleep(0.3)

    assert hub.indicators_broadcast == 1
    assert hub.strategy_states_broadcast == 0

    bridge.stop()


# ── Signal persistence tests ────────────────


@pytest.mark.asyncio
async def test_trading_signal_persists_to_duckdb():
    """Trading signals are persisted to DuckDB with timeframe and indicator."""
    from src.server.persistence import ServerPersistence

    hub = MockHub()
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)
    persistence = ServerPersistence(duckdb_path=":memory:")
    loop = asyncio.get_running_loop()
    bridge = WebBridge(
        event_bus=bus, signal_engine=engine, hub=hub, persistence=persistence, loop=loop
    )
    bridge.start()

    event = TradingSignalEvent(
        symbol="AAPL",
        timeframe="1d",
        direction="buy",
        strength=0.8,
        indicator="rsi",
        trigger_rule="rsi_oversold",
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    bridge._on_trading_signal(event)
    await asyncio.sleep(0.3)

    signals = persistence.query_signals(symbol="AAPL")
    assert len(signals) == 1
    assert signals[0]["rule"] == "rsi_oversold"
    assert signals[0]["direction"] == "bullish"
    assert signals[0]["timeframe"] == "1d"
    assert signals[0]["indicator"] == "rsi"

    persistence.close()
    bridge.stop()


@pytest.mark.asyncio
async def test_trading_signal_persistence_failure_logs_not_crashes():
    """If persistence raises, signal processing continues without crashing."""
    hub = MockHub()
    bus = MockEventBus()
    engine = MagicMock(spec=SignalEngine)

    mock_persistence = MagicMock()
    mock_persistence.insert_signal.side_effect = RuntimeError("DB write failed")
    loop = asyncio.get_running_loop()
    bridge = WebBridge(
        event_bus=bus, signal_engine=engine, hub=hub, persistence=mock_persistence, loop=loop
    )
    bridge.start()

    event = TradingSignalEvent(
        symbol="AAPL",
        timeframe="1d",
        direction="sell",
        strength=0.6,
        indicator="macd",
        trigger_rule="macd_cross_down",
        timestamp=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
    )
    bridge._on_trading_signal(event)
    await asyncio.sleep(0.3)

    assert hub.signals_broadcast == 1
    assert len(bridge._signal_buffer.get("AAPL", [])) == 1

    bridge.stop()
