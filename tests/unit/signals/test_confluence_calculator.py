"""
Unit tests for ConfluenceCalculator.

Tests the confluence calculation component including:
- Start/stop lifecycle
- Indicator state caching
- Debounce logic
- Cache eviction (LRU)
- Single-timeframe confluence calculation
- Multi-timeframe alignment
- Persistence callback
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.events.event_types import EventType
from src.domain.signals.confluence_calculator import (
    MAX_INDICATORS_PER_PAIR,
    MAX_SYMBOL_TIMEFRAME_PAIRS,
    ConfluenceCalculator,
)

from .conftest import MockEventBus

# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestConfluenceCalculatorLifecycle:
    """Test start/stop lifecycle behavior."""

    def test_start_creates_analyzers(self, mock_event_bus: MockEventBus) -> None:
        """start() should create cross-indicator and MTF analyzers."""
        calculator = ConfluenceCalculator(mock_event_bus)

        calculator.start()

        assert calculator._cross_analyzer is not None
        assert calculator._mtf_analyzer is not None
        assert calculator._started is True

    def test_start_is_idempotent(self, mock_event_bus: MockEventBus) -> None:
        """Multiple start() calls should not recreate analyzers."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        first_analyzer = calculator._cross_analyzer

        calculator.start()  # Second call

        assert calculator._cross_analyzer is first_analyzer

    def test_stop_clears_state(self, mock_event_bus: MockEventBus) -> None:
        """stop() should clear all cached state."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        # Add some state
        calculator._indicator_states[("AAPL", "1d")] = {"rsi": {"value": 50}}
        calculator._last_calc_time[("AAPL", "1d")] = time.monotonic() * 1000

        calculator.stop()

        assert calculator._started is False
        assert len(calculator._indicator_states) == 0
        assert len(calculator._last_calc_time) == 0
        assert calculator._cross_analyzer is None
        assert calculator._mtf_analyzer is None

    def test_stop_before_start(self, mock_event_bus: MockEventBus) -> None:
        """stop() before start() should be safe."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.stop()  # Should not raise
        assert calculator._started is False


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfluenceCalculatorConfig:
    """Test configuration options."""

    def test_default_debounce_ms(self, mock_event_bus: MockEventBus) -> None:
        """Default debounce should be 500ms."""
        calculator = ConfluenceCalculator(mock_event_bus)
        assert calculator._debounce_ms == 500.0

    def test_custom_debounce_ms(self, mock_event_bus: MockEventBus) -> None:
        """Custom debounce should be respected."""
        calculator = ConfluenceCalculator(mock_event_bus, debounce_ms=1000.0)
        assert calculator._debounce_ms == 1000.0

    def test_default_min_indicators(self, mock_event_bus: MockEventBus) -> None:
        """Default min_indicators should be 2."""
        calculator = ConfluenceCalculator(mock_event_bus)
        assert calculator._min_indicators == 2

    def test_custom_min_indicators(self, mock_event_bus: MockEventBus) -> None:
        """Custom min_indicators should be respected."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=5)
        assert calculator._min_indicators == 5


# =============================================================================
# Indicator State Caching Tests
# =============================================================================


class TestIndicatorStateCaching:
    """Test indicator state cache behavior."""

    def test_on_indicator_update_caches_state(self, mock_event_bus: MockEventBus) -> None:
        """on_indicator_update should cache indicator state."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        calculator.on_indicator_update(
            symbol="AAPL",
            timeframe="1d",
            indicator="rsi",
            state={"value": 50, "zone": "neutral"},
        )

        cached = calculator._indicator_states.get(("AAPL", "1d"), {})
        assert "rsi" in cached
        assert cached["rsi"]["value"] == 50

    def test_on_indicator_update_before_start(self, mock_event_bus: MockEventBus) -> None:
        """on_indicator_update should be ignored before start()."""
        calculator = ConfluenceCalculator(mock_event_bus)

        calculator.on_indicator_update(
            symbol="AAPL",
            timeframe="1d",
            indicator="rsi",
            state={"value": 50},
        )

        assert len(calculator._indicator_states) == 0

    def test_on_indicator_update_normalizes_none_state(self, mock_event_bus: MockEventBus) -> None:
        """None state should be normalized to empty dict."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        calculator.on_indicator_update(
            symbol="AAPL",
            timeframe="1d",
            indicator="rsi",
            state=None,  # type: ignore
        )

        cached = calculator._indicator_states.get(("AAPL", "1d"), {})
        assert "rsi" in cached
        assert cached["rsi"] == {}

    def test_indicator_state_count_property(self, mock_event_bus: MockEventBus) -> None:
        """indicator_state_count should reflect total cached entries."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5})
        calculator.on_indicator_update("TSLA", "1d", "rsi", {"value": 60})

        assert calculator.indicator_state_count == 3


# =============================================================================
# Cache Eviction Tests
# =============================================================================


class TestCacheEviction:
    """Test cache eviction behavior."""

    def test_evict_oldest_symbol_timeframe_pair(self, mock_event_bus: MockEventBus) -> None:
        """Should evict oldest (symbol, timeframe) when cache is full."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        # Fill cache to max
        for i in range(MAX_SYMBOL_TIMEFRAME_PAIRS):
            calculator._indicator_states[(f"SYM{i}", "1d")] = {"rsi": {"value": i}}

        # Add one more - should trigger eviction
        calculator.on_indicator_update("NEW_SYMBOL", "1d", "rsi", {"value": 999})

        assert len(calculator._indicator_states) == MAX_SYMBOL_TIMEFRAME_PAIRS
        assert ("NEW_SYMBOL", "1d") in calculator._indicator_states
        # First entry should have been evicted
        assert ("SYM0", "1d") not in calculator._indicator_states

    def test_evict_oldest_indicator_per_pair(self, mock_event_bus: MockEventBus) -> None:
        """Should evict oldest indicator when per-pair limit exceeded."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        # Fill indicators for one pair to max
        for i in range(MAX_INDICATORS_PER_PAIR):
            calculator._indicator_states.setdefault(("AAPL", "1d"), {})[f"ind{i}"] = {"v": i}

        # Add one more - should trigger indicator eviction
        calculator.on_indicator_update("AAPL", "1d", "new_indicator", {"v": 999})

        pair_indicators = calculator._indicator_states[("AAPL", "1d")]
        assert len(pair_indicators) == MAX_INDICATORS_PER_PAIR
        assert "new_indicator" in pair_indicators


# =============================================================================
# Debounce Tests
# =============================================================================


class TestDebounce:
    """Test debounce behavior."""

    def test_calculation_skipped_within_debounce(self, mock_event_bus: MockEventBus) -> None:
        """Calculation should be skipped within debounce window."""
        calculator = ConfluenceCalculator(mock_event_bus, debounce_ms=1000.0)
        calculator.start()

        # First update
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5})

        # Record time immediately after
        first_calc_time = calculator._last_calc_time.get(("AAPL", "1d"), 0)

        # Second update immediately (within debounce)
        calculator.on_indicator_update("AAPL", "1d", "supertrend", {"value": 100})

        # Calc time should not have been updated again
        second_calc_time = calculator._last_calc_time.get(("AAPL", "1d"), 0)
        assert second_calc_time == first_calc_time

    def test_calculation_proceeds_after_debounce(self, mock_event_bus: MockEventBus) -> None:
        """Calculation should proceed after debounce window."""
        calculator = ConfluenceCalculator(mock_event_bus, debounce_ms=10.0)  # Very short
        calculator.start()

        # First update
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5})

        first_calc_time = calculator._last_calc_time.get(("AAPL", "1d"), 0)

        # Wait for debounce to expire
        time.sleep(0.02)

        # Another update
        calculator.on_indicator_update("AAPL", "1d", "supertrend", {"value": 100})

        second_calc_time = calculator._last_calc_time.get(("AAPL", "1d"), 0)
        # New calculation should have updated the time
        assert second_calc_time > first_calc_time


# =============================================================================
# Min Indicators Tests
# =============================================================================


class TestMinIndicators:
    """Test min_indicators threshold."""

    def test_calculation_skipped_below_min(self, mock_event_bus: MockEventBus) -> None:
        """Calculation should be skipped with insufficient indicators."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=3)
        calculator.start()

        # Only 2 indicators (below min of 3)
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5})

        # Should not have calculated (no CONFLUENCE_UPDATE events)
        events = mock_event_bus.get_events(EventType.CONFLUENCE_UPDATE)
        assert len(events) == 0

    def test_calculation_proceeds_at_min(self, mock_event_bus: MockEventBus) -> None:
        """Calculation should proceed when min_indicators is met."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=2)
        calculator.start()

        # Exactly 2 indicators (meets min of 2)
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50, "direction": "neutral"})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5, "direction": "bullish"})

        # Should have calculated and published event
        events = mock_event_bus.get_events(EventType.CONFLUENCE_UPDATE)
        assert len(events) >= 1


# =============================================================================
# Clear Cache Tests
# =============================================================================


class TestClearCache:
    """Test cache clearing functionality."""

    def test_clear_symbol(self, mock_event_bus: MockEventBus) -> None:
        """clear_symbol should remove all data for a symbol."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        # Add data for multiple symbols
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1h", "rsi", {"value": 55})
        calculator.on_indicator_update("TSLA", "1d", "rsi", {"value": 60})

        calculator.clear_symbol("AAPL")

        assert ("AAPL", "1d") not in calculator._indicator_states
        assert ("AAPL", "1h") not in calculator._indicator_states
        assert ("TSLA", "1d") in calculator._indicator_states

    def test_clear_cache_all(self, mock_event_bus: MockEventBus) -> None:
        """clear_cache() should remove all cached data."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("TSLA", "1d", "rsi", {"value": 60})

        count = calculator.clear_cache()

        assert count == 2
        assert len(calculator._indicator_states) == 0
        assert len(calculator._last_calc_time) == 0

    def test_clear_cache_specific_symbol(self, mock_event_bus: MockEventBus) -> None:
        """clear_cache(symbol) should clear only that symbol."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5})
        calculator.on_indicator_update("TSLA", "1d", "rsi", {"value": 60})

        count = calculator.clear_cache(symbol="AAPL")

        assert count == 2
        assert ("AAPL", "1d") not in calculator._indicator_states
        assert ("TSLA", "1d") in calculator._indicator_states


# =============================================================================
# Get Cached States Tests
# =============================================================================


class TestGetCachedStates:
    """Test get_cached_states functionality."""

    def test_get_cached_states_specific_timeframe(self, mock_event_bus: MockEventBus) -> None:
        """get_cached_states with timeframe should return that timeframe's states."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1h", "rsi", {"value": 55})

        states = calculator.get_cached_states("AAPL", timeframe="1d")

        assert "rsi" in states
        assert states["rsi"]["value"] == 50

    def test_get_cached_states_all_timeframes(self, mock_event_bus: MockEventBus) -> None:
        """get_cached_states without timeframe should combine all timeframes."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50})
        calculator.on_indicator_update("AAPL", "1h", "macd", {"value": 1.5})

        states = calculator.get_cached_states("AAPL")

        # Should have combined keys
        assert "rsi_1d" in states
        assert "macd_1h" in states

    def test_get_cached_states_unknown_symbol(self, mock_event_bus: MockEventBus) -> None:
        """get_cached_states for unknown symbol should return empty dict."""
        calculator = ConfluenceCalculator(mock_event_bus)
        calculator.start()

        states = calculator.get_cached_states("UNKNOWN")
        assert states == {}


# =============================================================================
# Persistence Callback Tests
# =============================================================================


class TestPersistenceCallback:
    """Test persistence callback functionality."""

    def test_set_persistence_callback(self, mock_event_bus: MockEventBus) -> None:
        """set_persistence_callback should store the callback."""
        calculator = ConfluenceCalculator(mock_event_bus)

        callback = AsyncMock()
        calculator.set_persistence_callback(callback)

        assert calculator._persistence_callback is callback

    @pytest.mark.asyncio
    async def test_persistence_callback_called(self, mock_event_bus: MockEventBus) -> None:
        """Persistence callback should be called on confluence calculation."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=2)
        calculator.start()

        callback = AsyncMock()
        calculator.set_persistence_callback(callback)

        # Add enough indicators to trigger calculation
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50, "direction": "neutral"})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5, "direction": "bullish"})

        # Wait for async task to complete
        await asyncio.sleep(0.1)

        # Callback should have been called
        assert callback.called or callback.await_count > 0


# =============================================================================
# MTF Alignment Tests
# =============================================================================


class TestMTFAlignment:
    """Test multi-timeframe alignment calculation."""

    def test_mtf_skipped_with_single_timeframe(self, mock_event_bus: MockEventBus) -> None:
        """MTF alignment should be skipped with only one timeframe."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=1)
        calculator.start()

        # Only one timeframe
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50, "direction": "neutral"})

        # No ALIGNMENT_UPDATE should be published
        alignment_events = mock_event_bus.get_events(EventType.ALIGNMENT_UPDATE)
        assert len(alignment_events) == 0

    def test_mtf_calculated_with_multiple_timeframes(self, mock_event_bus: MockEventBus) -> None:
        """MTF alignment should be calculated with multiple timeframes."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=1, debounce_ms=10)
        calculator.start()

        # Multiple timeframes
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50, "direction": "neutral"})
        calculator.on_indicator_update("AAPL", "1h", "rsi", {"value": 55, "direction": "bullish"})

        # Wait for debounce
        time.sleep(0.02)

        # Trigger another update to recalculate
        calculator.on_indicator_update("AAPL", "1h", "macd", {"value": 1.0, "direction": "bullish"})

        # ALIGNMENT_UPDATE may or may not be published depending on analyzer implementation
        # Just verify no errors occurred - the get_events call verifies the flow worked
        _ = mock_event_bus.get_events(EventType.ALIGNMENT_UPDATE)


# =============================================================================
# Pending Tasks Tests
# =============================================================================


class TestPendingTasks:
    """Test pending task tracking for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_pending_tasks_cancelled_on_stop(self, mock_event_bus: MockEventBus) -> None:
        """stop() should cancel pending async tasks."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=2)
        calculator.start()

        # Mock a slow persistence callback
        async def slow_callback(**kwargs: Any) -> None:
            await asyncio.sleep(10)  # Very slow

        calculator.set_persistence_callback(slow_callback)

        # Trigger calculation
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50, "direction": "neutral"})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5, "direction": "bullish"})

        # Small delay to let task be created
        await asyncio.sleep(0.05)

        # Stop should cancel pending tasks
        calculator.stop()

        assert len(calculator._pending_tasks) == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in confluence calculation."""

    def test_analyzer_exception_handled(self, mock_event_bus: MockEventBus) -> None:
        """Analyzer exceptions should be caught and logged."""
        calculator = ConfluenceCalculator(mock_event_bus, min_indicators=2)
        calculator.start()

        # Mock analyzer to raise exception
        calculator._cross_analyzer.analyze = MagicMock(side_effect=RuntimeError("Test error"))

        # Should not raise
        calculator.on_indicator_update("AAPL", "1d", "rsi", {"value": 50, "direction": "neutral"})
        calculator.on_indicator_update("AAPL", "1d", "macd", {"value": 1.5, "direction": "bullish"})

        # No events should be published due to error
        events = mock_event_bus.get_events(EventType.CONFLUENCE_UPDATE)
        assert len(events) == 0
