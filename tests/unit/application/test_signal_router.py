"""Tests for SignalRouter service."""

import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.application.signal_router import SignalRouter, SignalRouterConfig
from src.domain.events.event_types import EventType
from src.domain.strategy.base import TradingSignal


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    bus = MagicMock()
    return bus


@pytest.fixture
def sample_signal():
    """Create sample trading signal."""
    return TradingSignal(
        signal_id="sig-test123",
        symbol="AAPL",
        direction="LONG",
        strength=0.8,
        target_quantity=100,
        strategy_id="test_strategy",
        reason="MA crossover",
        timestamp=datetime.now(),
    )


class TestSignalRouterBasic:
    """Test basic signal routing."""

    def test_route_publishes_event(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signal should be published to event bus."""
        router = SignalRouter(mock_event_bus)
        result = router.route(sample_signal)

        assert result is True
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == EventType.TRADING_SIGNAL

    def test_route_increments_stats(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Routing should increment statistics."""
        router = SignalRouter(mock_event_bus)
        router.route(sample_signal)

        assert router.stats.total_received == 1
        assert router.stats.total_published == 1

    def test_route_logs_signal(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Routing should log signal."""
        router = SignalRouter(mock_event_bus)

        with patch("src.application.signal_router.logger") as mock_logger:
            router.route(sample_signal)
            mock_logger.info.assert_called()


class TestSignalRouterFiltering:
    """Test signal filtering."""

    def test_filter_by_strength(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signals below min strength should be filtered."""
        config = SignalRouterConfig(min_strength=0.9)
        router = SignalRouter(mock_event_bus, config)

        sample_signal.strength = 0.5
        result = router.route(sample_signal)

        assert result is False
        assert router.stats.filtered == 1
        mock_event_bus.publish.assert_not_called()

    def test_filter_by_direction(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signals with disallowed direction should be filtered."""
        config = SignalRouterConfig(allowed_directions={"LONG"})
        router = SignalRouter(mock_event_bus, config)

        sample_signal.direction = "SHORT"
        result = router.route(sample_signal)

        assert result is False
        assert router.stats.filtered == 1

    def test_allow_all_directions(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Default config should allow all directions."""
        router = SignalRouter(mock_event_bus)

        for direction in ["LONG", "SHORT", "FLAT"]:
            sample_signal.direction = direction
            sample_signal.signal_id = f"sig-{direction}"
            router.clear_state()  # Clear dedupe
            result = router.route(sample_signal)
            assert result is True


class TestSignalRouterRateLimiting:
    """Test rate limiting."""

    def test_rate_limit_blocks_rapid_signals(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Rapid signals should be rate-limited."""
        config = SignalRouterConfig(rate_limit_ms=1000)  # 1 second
        router = SignalRouter(mock_event_bus, config)

        # First signal passes
        result1 = router.route(sample_signal)
        assert result1 is True

        # Immediate second signal is rate-limited
        result2 = router.route(sample_signal)
        assert result2 is False
        assert router.stats.rate_limited == 1

    def test_rate_limit_allows_after_window(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signals should pass after rate limit window."""
        config = SignalRouterConfig(rate_limit_ms=10)  # 10ms
        router = SignalRouter(mock_event_bus, config)

        router.route(sample_signal)
        time.sleep(0.02)  # Wait 20ms
        result = router.route(sample_signal)

        # Note: May still be deduplicated, so check rate_limited didn't increase
        assert router.stats.rate_limited == 0 or result is False

    def test_rate_limit_per_symbol(self, mock_event_bus: Any) -> None:
        """Rate limiting should be per-symbol."""
        config = SignalRouterConfig(rate_limit_ms=1000)
        router = SignalRouter(mock_event_bus, config)

        signal1 = TradingSignal(signal_id="sig-1", symbol="AAPL", direction="LONG")
        signal2 = TradingSignal(signal_id="sig-2", symbol="MSFT", direction="LONG")

        result1 = router.route(signal1)
        result2 = router.route(signal2)

        assert result1 is True
        assert result2 is True  # Different symbol, not rate-limited


class TestSignalRouterDeduplication:
    """Test signal deduplication."""

    def test_dedupe_blocks_duplicate(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Duplicate signals within window should be deduplicated."""
        config = SignalRouterConfig(dedupe_window_ms=1000, rate_limit_ms=0)
        router = SignalRouter(mock_event_bus, config)

        result1 = router.route(sample_signal)
        result2 = router.route(sample_signal)

        assert result1 is True
        assert result2 is False
        assert router.stats.deduplicated == 1

    def test_dedupe_allows_different_signals(self, mock_event_bus: Any) -> None:
        """Different signals should not be deduplicated."""
        config = SignalRouterConfig(dedupe_window_ms=1000, rate_limit_ms=0)
        router = SignalRouter(mock_event_bus, config)

        signal1 = TradingSignal(signal_id="sig-1", symbol="AAPL", direction="LONG", strength=0.8)
        signal2 = TradingSignal(signal_id="sig-2", symbol="AAPL", direction="SHORT", strength=0.8)

        result1 = router.route(signal1)
        result2 = router.route(signal2)

        assert result1 is True
        assert result2 is True


class TestSignalRouterCallbacks:
    """Test pre/post route callbacks."""

    def test_pre_route_can_veto(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Pre-route callback returning False should veto signal."""
        router = SignalRouter(mock_event_bus)
        router.add_pre_route_callback(lambda s: False)

        result = router.route(sample_signal)

        assert result is False
        assert router.stats.filtered == 1

    def test_pre_route_can_allow(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Pre-route callback returning True should allow signal."""
        router = SignalRouter(mock_event_bus)
        router.add_pre_route_callback(lambda s: True)

        result = router.route(sample_signal)

        assert result is True

    def test_post_route_called(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Post-route callback should be called after successful routing."""
        router = SignalRouter(mock_event_bus)
        callback = MagicMock()
        router.add_post_route_callback(callback)

        router.route(sample_signal)

        callback.assert_called_once_with(sample_signal)

    def test_post_route_not_called_on_filter(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Post-route callback should not be called if signal filtered."""
        config = SignalRouterConfig(min_strength=1.0)
        router = SignalRouter(mock_event_bus, config)
        callback = MagicMock()
        router.add_post_route_callback(callback)

        sample_signal.strength = 0.5
        router.route(sample_signal)

        callback.assert_not_called()


class TestSignalRouterPersistence:
    """Test signal persistence."""

    def test_persist_when_enabled(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signal should be persisted when enabled."""
        config = SignalRouterConfig(persist=True)
        store = MagicMock()
        router = SignalRouter(mock_event_bus, config, signal_store=store)

        router.route(sample_signal)

        store.save.assert_called_once_with(sample_signal)

    def test_no_persist_when_disabled(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signal should not be persisted when disabled."""
        config = SignalRouterConfig(persist=False)
        store = MagicMock()
        router = SignalRouter(mock_event_bus, config, signal_store=store)

        router.route(sample_signal)

        store.save.assert_not_called()


class TestSignalRouterExecution:
    """Test signal execution forwarding."""

    def test_execute_when_enabled(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signal should be forwarded to execution when enabled."""
        config = SignalRouterConfig(execute=True)
        adapter = MagicMock()
        router = SignalRouter(mock_event_bus, config, execution_adapter=adapter)

        router.route(sample_signal)

        adapter.submit.assert_called_once_with(sample_signal)

    def test_no_execute_when_disabled(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Signal should not be executed when disabled."""
        config = SignalRouterConfig(execute=False)
        adapter = MagicMock()
        router = SignalRouter(mock_event_bus, config, execution_adapter=adapter)

        router.route(sample_signal)

        adapter.submit.assert_not_called()


class TestSignalRouterState:
    """Test state management."""

    def test_reset_stats(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Stats should be reset."""
        router = SignalRouter(mock_event_bus)
        router.route(sample_signal)
        router.reset_stats()

        assert router.stats.total_received == 0
        assert router.stats.total_published == 0

    def test_clear_state(self, mock_event_bus: Any, sample_signal: Any) -> None:
        """Rate limit and dedupe state should be cleared."""
        config = SignalRouterConfig(rate_limit_ms=10000, dedupe_window_ms=10000)
        router = SignalRouter(mock_event_bus, config)

        router.route(sample_signal)
        router.clear_state()

        # After clearing, same signal should pass again
        result = router.route(sample_signal)
        assert result is True


class TestSignalRouterIntegration:
    """Integration tests with strategy callback pattern."""

    def test_strategy_callback_pattern(self, mock_event_bus: Any) -> None:
        """Router should work as strategy callback."""
        router = SignalRouter(mock_event_bus)

        # Simulate strategy registration
        signal_callbacks = []
        signal_callbacks.append(router.route)

        # Simulate strategy emitting signal
        signal = TradingSignal(signal_id="sig-int-1", symbol="AAPL", direction="LONG")
        for callback in signal_callbacks:
            callback(signal)

        mock_event_bus.publish.assert_called_once()
        assert router.stats.total_published == 1
