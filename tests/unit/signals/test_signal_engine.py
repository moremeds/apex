"""
Unit tests for SignalEngine — domain service for tick → bar → indicator → signal pipeline.

Tests the signal engine including:
- Start/stop lifecycle
- Options symbol detection (static method)
- Tick dispatch to aggregators
- Query API (inject_history, get_regime_states, etc.)
"""

from unittest.mock import MagicMock

from src.domain.signals.signal_engine import SignalEngine

from .conftest import MockEventBus

# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestSignalEngineLifecycle:
    """Test start/stop lifecycle behavior."""

    def test_init_defaults(self, mock_event_bus: MockEventBus) -> None:
        """Constructor should set default values."""
        engine = SignalEngine(mock_event_bus)

        assert engine.is_started is False
        assert engine.timeframes == ["1d"]

    def test_init_custom_timeframes(self, mock_event_bus: MockEventBus) -> None:
        """Constructor should accept custom timeframes."""
        engine = SignalEngine(mock_event_bus, timeframes=["1h", "4h", "1d"])

        assert engine.timeframes == ["1h", "4h", "1d"]

    def test_init_deduplicates_timeframes(self, mock_event_bus: MockEventBus) -> None:
        """Constructor should deduplicate timeframes."""
        engine = SignalEngine(mock_event_bus, timeframes=["1d", "1d", "1h"])

        assert engine.timeframes == ["1d", "1h"]

    def test_start_creates_components(self, mock_event_bus: MockEventBus) -> None:
        """start() should create bar aggregators and engines."""
        engine = SignalEngine(mock_event_bus, timeframes=["1d"])

        engine.start()

        assert engine.is_started is True
        assert len(engine._aggregators) == 1
        assert "1d" in engine._aggregators
        assert engine._indicator_engine is not None
        assert engine._rule_engine is not None

    def test_start_is_idempotent(self, mock_event_bus: MockEventBus) -> None:
        """Multiple start() calls should not create duplicates."""
        engine = SignalEngine(mock_event_bus)

        engine.start()
        engine.start()  # Second call should warn but not break

        assert engine.is_started is True

    def test_stop_clears_components(self, mock_event_bus: MockEventBus) -> None:
        """stop() should clean up all components."""
        engine = SignalEngine(mock_event_bus)
        engine.start()

        engine.stop()

        assert engine.is_started is False
        assert len(engine._aggregators) == 0
        assert engine._indicator_engine is None
        assert engine._rule_engine is None

    def test_stop_before_start(self, mock_event_bus: MockEventBus) -> None:
        """stop() before start() should be safe."""
        engine = SignalEngine(mock_event_bus)

        engine.stop()  # Should not raise

        assert engine.is_started is False

    def test_start_subscribes_to_tick_events(self, mock_event_bus: MockEventBus) -> None:
        """start() should subscribe to MARKET_DATA_TICK."""
        from src.domain.events.event_types import EventType

        engine = SignalEngine(mock_event_bus)
        engine.start()

        assert EventType.MARKET_DATA_TICK in mock_event_bus.subscriptions


# =============================================================================
# Options Symbol Detection Tests
# =============================================================================


class TestOptionsSymbolDetection:
    """Test options symbol pattern matching (static method)."""

    def test_stock_symbols_not_detected(self) -> None:
        """Short stock symbols should not be detected as options."""
        assert SignalEngine._is_options_symbol("AAPL") is False
        assert SignalEngine._is_options_symbol("TSLA") is False
        assert SignalEngine._is_options_symbol("SPY") is False
        assert SignalEngine._is_options_symbol("QQQ") is False
        assert SignalEngine._is_options_symbol("VIX") is False
        assert SignalEngine._is_options_symbol("GOOGL") is False

    def test_long_options_symbols_detected(self) -> None:
        """Long options symbols should be detected."""
        assert SignalEngine._is_options_symbol("AAPL  250117C00250000") is True
        assert SignalEngine._is_options_symbol("AAPL250117C00250000") is True

    def test_ib_style_options_detected(self) -> None:
        """IB-style options symbols should be detected."""
        assert SignalEngine._is_options_symbol("SPY240315C500") is True
        assert SignalEngine._is_options_symbol("AAPL240120P175") is True

    def test_colon_separated_options_detected(self) -> None:
        """Colon-separated options notation should be detected."""
        assert SignalEngine._is_options_symbol("AAPL:OPT:240117:175:C") is True
        assert SignalEngine._is_options_symbol("TSLA:opt:240315:200:P") is True

    def test_strike_pattern_detected(self) -> None:
        """C or P followed by digits should be detected."""
        assert SignalEngine._is_options_symbol("AAPLC00250") is True
        assert SignalEngine._is_options_symbol("TSLAP00175") is True

    def test_empty_symbol(self) -> None:
        """Empty symbol should not be detected as options."""
        assert SignalEngine._is_options_symbol("") is False
        assert SignalEngine._is_options_symbol(None) is False  # type: ignore


# =============================================================================
# Tick Dispatch Tests
# =============================================================================


class TestTickDispatch:
    """Test tick dispatch to aggregators."""

    def test_tick_ignored_when_not_started(self, mock_event_bus: MockEventBus) -> None:
        """Ticks should be ignored when engine is not started."""
        engine = SignalEngine(mock_event_bus)

        tick = MagicMock()
        tick.symbol = "AAPL"

        engine._on_tick(tick)  # Should not raise

    def test_tick_ignored_after_stop(self, mock_event_bus: MockEventBus) -> None:
        """Ticks should be ignored after engine stops."""
        engine = SignalEngine(mock_event_bus)
        engine.start()
        engine.stop()

        tick = MagicMock()
        tick.symbol = "AAPL"

        engine._on_tick(tick)  # Should not raise

    def test_options_tick_filtered(self, mock_event_bus: MockEventBus) -> None:
        """Options ticks should be filtered when exclude_options=True."""
        engine = SignalEngine(mock_event_bus, exclude_options=True)
        engine.start()

        tick = MagicMock()
        tick.symbol = "AAPL  250117C00250000"

        aggregator = engine._aggregators["1d"]
        aggregator.on_tick = MagicMock()

        engine._on_tick(tick)

        aggregator.on_tick.assert_not_called()

    def test_stock_tick_passed_to_aggregators(self, mock_event_bus: MockEventBus) -> None:
        """Stock ticks should be passed to all aggregators."""
        engine = SignalEngine(mock_event_bus, timeframes=["1d", "1h"], exclude_options=True)
        engine.start()

        for tf, agg in engine._aggregators.items():
            agg.on_tick = MagicMock()

        tick = MagicMock()
        tick.symbol = "AAPL"

        engine._on_tick(tick)

        for tf, agg in engine._aggregators.items():
            agg.on_tick.assert_called_once_with(tick)

    def test_direct_tick_injection(self, mock_event_bus: MockEventBus) -> None:
        """on_tick() should feed ticks directly to aggregators (Longbridge bypass)."""
        engine = SignalEngine(mock_event_bus, timeframes=["1m", "5m"])
        engine.start()

        for tf, agg in engine._aggregators.items():
            agg.on_tick = MagicMock()

        tick = MagicMock()
        tick.symbol = "AAPL"

        engine.on_tick(tick)

        for tf, agg in engine._aggregators.items():
            agg.on_tick.assert_called_once_with(tick)


# =============================================================================
# Stats Tests
# =============================================================================


class TestSignalEngineStats:
    """Test get_stats() functionality."""

    def test_stats_before_start(self, mock_event_bus: MockEventBus) -> None:
        """Stats should reflect stopped state before start."""
        engine = SignalEngine(mock_event_bus, timeframes=["1d"])

        stats = engine.get_stats()

        assert stats["started"] is False
        assert stats["timeframes"] == ["1d"]

    def test_stats_after_start(self, mock_event_bus: MockEventBus) -> None:
        """Stats should include component stats after start."""
        engine = SignalEngine(mock_event_bus, timeframes=["1d"])
        engine.start()

        stats = engine.get_stats()

        assert stats["started"] is True
        assert "bars_emitted" in stats
        assert "indicator_count" in stats


# =============================================================================
# Clear Cooldowns Tests
# =============================================================================


class TestClearCooldowns:
    """Test clear_cooldowns() delegation."""

    def test_clear_cooldowns_before_start(self, mock_event_bus: MockEventBus) -> None:
        """clear_cooldowns() before start should return 0."""
        engine = SignalEngine(mock_event_bus)

        cleared = engine.clear_cooldowns()
        assert cleared == 0

    def test_clear_cooldowns_delegates_to_rule_engine(self, mock_event_bus: MockEventBus) -> None:
        """clear_cooldowns() should delegate to rule engine."""
        engine = SignalEngine(mock_event_bus)
        engine.start()

        engine._rule_engine.clear_cooldowns = MagicMock(return_value=5)

        cleared = engine.clear_cooldowns()

        engine._rule_engine.clear_cooldowns.assert_called_once()
        assert cleared == 5
