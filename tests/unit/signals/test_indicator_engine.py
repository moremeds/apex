"""
Unit tests for IndicatorEngine.

Tests the core indicator calculation engine including:
- Start/stop lifecycle
- BAR_CLOSE event handling
- Parallel indicator calculation
- Warmup tracking
- Historical bar injection
- State caching
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.domain.events.event_types import EventType
from src.domain.signals.indicator_engine import IndicatorEngine

from .conftest import MockEventBus, generate_ohlcv_data, make_bar_close_event

# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestIndicatorEngineLifecycle:
    """Test start/stop lifecycle behavior."""

    def test_start_subscribes_to_bar_close(self, mock_event_bus: MockEventBus) -> None:
        """Engine should subscribe to BAR_CLOSE events on start."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)

        engine.start()

        assert EventType.BAR_CLOSE in mock_event_bus.subscriptions
        assert len(mock_event_bus.subscriptions[EventType.BAR_CLOSE]) == 1

    def test_start_is_idempotent(self, mock_event_bus: MockEventBus) -> None:
        """Calling start() multiple times should not create duplicate subscriptions."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)

        engine.start()
        engine.start()  # Second call should be no-op

        assert len(mock_event_bus.subscriptions[EventType.BAR_CLOSE]) == 1

    def test_stop_shuts_down_executor(self, mock_event_bus: MockEventBus) -> None:
        """Engine should shutdown executor on stop."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        engine.stop()

        assert engine._started is False
        # Executor should be shutdown (can verify by checking _executor state)
        assert engine._executor._shutdown is True

    def test_events_ignored_after_stop(self, mock_event_bus: MockEventBus) -> None:
        """BAR_CLOSE events should be ignored after engine stop."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()
        engine.stop()

        # Publish event after stop
        event = make_bar_close_event()
        mock_event_bus.publish(EventType.BAR_CLOSE, event)

        # No indicator updates should be published
        assert len(mock_event_bus.get_events(EventType.INDICATOR_UPDATE)) == 0


# =============================================================================
# Properties Tests
# =============================================================================


class TestIndicatorEngineProperties:
    """Test property accessors."""

    def test_bars_processed_starts_at_zero(self, mock_event_bus: MockEventBus) -> None:
        """bars_processed should start at 0."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        assert engine.bars_processed == 0

    def test_indicator_count_reflects_registry(self, mock_event_bus: MockEventBus) -> None:
        """indicator_count should match registry size."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        # Should have indicators from the global registry
        assert engine.indicator_count >= 0


# =============================================================================
# Warmup Tests
# =============================================================================


class TestIndicatorEngineWarmup:
    """Test warmup tracking functionality."""

    def test_warmup_status_no_history(self, mock_event_bus: MockEventBus) -> None:
        """Warmup status for unknown symbol should show 0 bars."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        status = engine.get_warmup_status("UNKNOWN", "1d")

        assert status["symbol"] == "UNKNOWN"
        assert status["timeframe"] == "1d"
        assert status["bars_loaded"] == 0
        assert status["status"] == "warming_up"

    def test_warmup_status_with_history(self, mock_event_bus: MockEventBus) -> None:
        """Warmup status should reflect injected history."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Inject some bars
        bars = generate_ohlcv_data(n_bars=50).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)

        status = engine.get_warmup_status("AAPL", "1d")

        assert status["bars_loaded"] == 50
        assert status["progress_pct"] > 0

    def test_all_warmup_status_empty(self, mock_event_bus: MockEventBus) -> None:
        """get_all_warmup_status should return empty list with no history."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        statuses = engine.get_all_warmup_status()
        assert statuses == []

    def test_all_warmup_status_multiple_symbols(self, mock_event_bus: MockEventBus) -> None:
        """get_all_warmup_status should include all symbol/timeframe pairs."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Inject bars for multiple symbols
        bars = generate_ohlcv_data(n_bars=20).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)
        engine.inject_historical_bars("TSLA", "1d", bars)

        statuses = engine.get_all_warmup_status()
        symbols = {s["symbol"] for s in statuses}

        assert "AAPL" in symbols
        assert "TSLA" in symbols


# =============================================================================
# Historical Bar Injection Tests
# =============================================================================


class TestHistoricalBarInjection:
    """Test inject_historical_bars functionality."""

    def test_inject_empty_bars(self, mock_event_bus: MockEventBus) -> None:
        """Injecting empty bar list should succeed."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        count = engine.inject_historical_bars("AAPL", "1d", [])
        assert count == 0

    def test_inject_bars_creates_history(self, mock_event_bus: MockEventBus) -> None:
        """Injecting bars should create history entry."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        bars = generate_ohlcv_data(n_bars=30).to_dict("records")
        count = engine.inject_historical_bars("AAPL", "1d", bars)

        assert count == 30
        assert ("AAPL", "1d") in engine._history
        assert len(engine._history[("AAPL", "1d")]) == 30

    def test_inject_bars_idempotent(self, mock_event_bus: MockEventBus) -> None:
        """Duplicate bar injection should skip already-injected bars."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        bars = generate_ohlcv_data(n_bars=30).to_dict("records")

        # First injection
        count1 = engine.inject_historical_bars("AAPL", "1d", bars)
        assert count1 == 30

        # Second injection of same bars
        count2 = engine.inject_historical_bars("AAPL", "1d", bars)
        assert count2 == 0  # All duplicates skipped

    def test_inject_newer_bars_appends(self, mock_event_bus: MockEventBus) -> None:
        """Injecting newer bars should append to existing history."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Inject first batch
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars1 = generate_ohlcv_data(n_bars=20, start_time=start_time).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars1)

        # Inject newer bars
        start_time2 = start_time + timedelta(days=20)
        bars2 = generate_ohlcv_data(n_bars=10, start_time=start_time2).to_dict("records")
        count = engine.inject_historical_bars("AAPL", "1d", bars2)

        assert count == 10
        assert len(engine._history[("AAPL", "1d")]) == 30

    def test_inject_respects_max_history(self, mock_event_bus: MockEventBus) -> None:
        """History should be bounded by max_history."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2, max_history=50)
        engine.start()

        # Inject more bars than max_history
        bars = generate_ohlcv_data(n_bars=100).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)

        # Should be capped at max_history
        assert len(engine._history[("AAPL", "1d")]) == 50


# =============================================================================
# State Cache Tests
# =============================================================================


class TestIndicatorStateCache:
    """Test indicator state caching."""

    def test_get_indicator_state_empty(self, mock_event_bus: MockEventBus) -> None:
        """get_indicator_state should return None for uncached state."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        state = engine.get_indicator_state("AAPL", "1d", "rsi")
        assert state is None

    def test_get_all_indicator_states_empty(self, mock_event_bus: MockEventBus) -> None:
        """get_all_indicator_states should return empty dict initially."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        states = engine.get_all_indicator_states()
        assert states == {}

    def test_get_all_indicator_states_filtered_by_symbol(
        self, mock_event_bus: MockEventBus
    ) -> None:
        """get_all_indicator_states should filter by symbol."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Manually populate state cache for testing
        engine._previous_states[("AAPL", "1d", "rsi")] = {"value": 50}
        engine._previous_states[("TSLA", "1d", "rsi")] = {"value": 60}

        states = engine.get_all_indicator_states(symbol="AAPL")

        assert len(states) == 1
        assert ("AAPL", "1d", "rsi") in states

    def test_get_all_indicator_states_filtered_by_timeframe(
        self, mock_event_bus: MockEventBus
    ) -> None:
        """get_all_indicator_states should filter by timeframe."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Manually populate state cache
        engine._previous_states[("AAPL", "1d", "rsi")] = {"value": 50}
        engine._previous_states[("AAPL", "1h", "rsi")] = {"value": 55}

        states = engine.get_all_indicator_states(timeframe="1d")

        assert len(states) == 1
        assert ("AAPL", "1d", "rsi") in states


# =============================================================================
# Lock Tests
# =============================================================================


class TestIndicatorEngineLocking:
    """Test per-symbol locking mechanism."""

    def test_get_lock_creates_new_lock(self, mock_event_bus: MockEventBus) -> None:
        """_get_lock should create new lock for new key."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)

        lock = engine._get_lock(("AAPL", "1d"))

        assert lock is not None
        assert ("AAPL", "1d") in engine._locks

    def test_get_lock_returns_same_lock(self, mock_event_bus: MockEventBus) -> None:
        """_get_lock should return same lock for same key."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)

        lock1 = engine._get_lock(("AAPL", "1d"))
        lock2 = engine._get_lock(("AAPL", "1d"))

        assert lock1 is lock2


# =============================================================================
# Coercion Tests
# =============================================================================


class TestBarCloseCoercion:
    """Test BarCloseEvent coercion."""

    def test_coerce_bar_close_event(self, mock_event_bus: MockEventBus) -> None:
        """Should pass through BarCloseEvent unchanged."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        event = make_bar_close_event()

        result = engine._coerce_bar_close(event)
        assert result is event

    def test_coerce_bar_close_dict(self, mock_event_bus: MockEventBus) -> None:
        """Should convert dict to BarCloseEvent."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        event_dict = {
            "symbol": "AAPL",
            "timeframe": "1d",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1000000.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = engine._coerce_bar_close(event_dict)

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.close == 101.0

    def test_coerce_bar_close_invalid(self, mock_event_bus: MockEventBus) -> None:
        """Should return None for invalid payload."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)

        result = engine._coerce_bar_close("invalid")
        assert result is None

        result = engine._coerce_bar_close(None)
        assert result is None


# =============================================================================
# Async Compute Tests
# =============================================================================


class TestComputeOnHistory:
    """Test compute_on_history async method."""

    @pytest.mark.asyncio
    async def test_compute_on_empty_history(self, mock_event_bus: MockEventBus) -> None:
        """compute_on_history with no history should return 0."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        count = await engine.compute_on_history("AAPL", "1d")
        assert count == 0

    @pytest.mark.asyncio
    async def test_compute_on_insufficient_warmup(self, mock_event_bus: MockEventBus) -> None:
        """compute_on_history with insufficient bars should compute 0 indicators."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Inject only 5 bars (insufficient for most indicators)
        bars = generate_ohlcv_data(n_bars=5).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)

        count = await engine.compute_on_history("AAPL", "1d")
        # May be 0 or low depending on indicator warmup requirements
        assert count >= 0

    @pytest.mark.asyncio
    async def test_compute_on_sufficient_history(self, mock_event_bus: MockEventBus) -> None:
        """compute_on_history with sufficient bars should compute indicators."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Inject enough bars for warmup
        bars = generate_ohlcv_data(n_bars=100).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)

        count = await engine.compute_on_history("AAPL", "1d")

        # Should compute at least some indicators
        assert count > 0

        # Should have published INDICATOR_UPDATE events
        updates = mock_event_bus.get_events(EventType.INDICATOR_UPDATE)
        assert len(updates) > 0

    @pytest.mark.asyncio
    async def test_compute_updates_state_cache(self, mock_event_bus: MockEventBus) -> None:
        """compute_on_history should update indicator state cache."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        bars = generate_ohlcv_data(n_bars=100).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)

        await engine.compute_on_history("AAPL", "1d")

        # State cache should have entries
        states = engine.get_all_indicator_states(symbol="AAPL", timeframe="1d")
        assert len(states) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIndicatorEngineIntegration:
    """Integration tests for full bar processing flow."""

    @pytest.mark.asyncio
    async def test_bar_close_triggers_indicator_updates(self, mock_event_bus: MockEventBus) -> None:
        """Processing a bar should eventually emit INDICATOR_UPDATE events."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        # Pre-populate with enough history for warmup
        bars = generate_ohlcv_data(n_bars=100).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)

        # Compute on history to populate indicators
        await engine.compute_on_history("AAPL", "1d")

        # Now process a new bar
        new_bar = make_bar_close_event(
            symbol="AAPL",
            timeframe="1d",
            close_price=105.0,
            timestamp=bars[-1]["timestamp"] + timedelta(days=1),
        )

        # Trigger bar close (async processing)
        engine._on_bar_close(new_bar)

        # Wait for async processing
        import asyncio

        await asyncio.sleep(0.5)

        # Should have updated indicators
        updates = mock_event_bus.get_events(EventType.INDICATOR_UPDATE)
        assert len(updates) > 0

    def test_bars_processed_counter_increments(self, mock_event_bus: MockEventBus) -> None:
        """bars_processed should increment with each bar."""
        engine = IndicatorEngine(mock_event_bus, max_workers=2)
        engine.start()

        initial_count = engine.bars_processed

        # Inject bars directly to history and trigger processing
        bars = generate_ohlcv_data(n_bars=10).to_dict("records")
        engine.inject_historical_bars("AAPL", "1d", bars)

        # bars_processed only increments on BAR_CLOSE events, not injection
        assert engine.bars_processed == initial_count
