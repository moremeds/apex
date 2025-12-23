"""
Tests for MarketDataRouter (A2).

Verifies:
- Subscription refcount tracking
- Line limit enforcement
- Snapshot fallback when at limit
- Consumer callback fanout
- Unsubscribe behavior
- Metrics tracking
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.infrastructure.adapters.market_data_router import (
    MarketDataRouter,
    SubscriptionMode,
    SymbolSubscription,
)


class TestSubscriptionBasics:
    """Basic subscription operations."""

    @pytest.mark.asyncio
    async def test_subscribe_creates_entry(self):
        """First subscription creates registry entry."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            result = await router.subscribe("AAPL", consumer_id="risk_engine")

            assert result is True
            sub = router.get_subscription("AAPL")
            assert sub is not None
            assert sub.symbol == "AAPL"
            assert "risk_engine" in sub.consumers
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_subscribe_same_symbol_increments_refcount(self):
        """Same symbol subscription adds to refcount."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="risk_engine")
            await router.subscribe("AAPL", consumer_id="scanner")

            sub = router.get_subscription("AAPL")
            assert sub.refcount == 2
            assert "risk_engine" in sub.consumers
            assert "scanner" in sub.consumers
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_subscribe_different_symbols_separate_entries(self):
        """Different symbols create separate entries."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="consumer1")
            await router.subscribe("MSFT", consumer_id="consumer1")

            assert len(router.get_all_symbols()) == 2
            assert router.get_subscription("AAPL") is not None
            assert router.get_subscription("MSFT") is not None
        finally:
            await router.stop()


class TestLineLimitEnforcement:
    """Line limit behavior."""

    @pytest.mark.asyncio
    async def test_respects_max_lines_limit(self):
        """Streaming subscriptions respect max_lines when active."""
        router = MarketDataRouter(ib=None, max_lines=3)
        await router.start()

        try:
            # Subscribe to 3 symbols
            for i in range(3):
                result = await router.subscribe(
                    f"SYM{i}", consumer_id="test", mode=SubscriptionMode.STREAMING
                )
                assert result is True

            # Simulate active tickers by setting ticker on subscriptions
            # (normally set by IB connection)
            for symbol in router.get_all_symbols():
                sub = router.get_subscription(symbol)
                sub.ticker = MagicMock()  # Simulates active line

            # Now active_lines should be 3
            assert router.active_lines == 3

            # 4th streaming subscription should trigger line limit fallback
            result = await router.subscribe(
                "SYM4", consumer_id="test", mode=SubscriptionMode.STREAMING
            )
            assert result is True

            # Check that we got a line limit rejection (fell back to snapshot)
            assert router.get_metrics().line_limit_rejections >= 1
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_available_lines_tracks_correctly(self):
        """available_lines decreases as subscriptions are added."""
        router = MarketDataRouter(ib=None, max_lines=5)
        await router.start()

        try:
            initial_available = router.available_lines
            assert initial_available == 5

            await router.subscribe("AAPL", consumer_id="test")
            # Note: Without actual IB connection, ticker won't be set
            # so active_lines won't increase. This tests the logic structure.

            assert router.available_lines <= initial_available
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_snapshot_mode_doesnt_use_line(self):
        """Snapshot subscriptions don't count against line limit."""
        router = MarketDataRouter(ib=None, max_lines=2)
        await router.start()

        try:
            # Add snapshot subscription
            await router.subscribe(
                "SNAP1", consumer_id="test", mode=SubscriptionMode.SNAPSHOT
            )
            await router.subscribe(
                "SNAP2", consumer_id="test", mode=SubscriptionMode.SNAPSHOT
            )

            # Should not affect available lines (no ticker set without IB)
            assert router.get_metrics().line_limit_rejections == 0
        finally:
            await router.stop()


class TestUnsubscribe:
    """Unsubscribe behavior."""

    @pytest.mark.asyncio
    async def test_unsubscribe_decrements_refcount(self):
        """Unsubscribe removes consumer from refcount."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="consumer1")
            await router.subscribe("AAPL", consumer_id="consumer2")

            await router.unsubscribe("AAPL", consumer_id="consumer1")

            sub = router.get_subscription("AAPL")
            assert sub.refcount == 1
            assert "consumer1" not in sub.consumers
            assert "consumer2" in sub.consumers
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_entry_when_refcount_zero(self):
        """Entry removed when last consumer unsubscribes."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="consumer1")
            await router.unsubscribe("AAPL", consumer_id="consumer1")

            assert router.get_subscription("AAPL") is None
            assert "AAPL" not in router.get_all_symbols()
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_symbol_is_safe(self):
        """Unsubscribing from nonexistent symbol doesn't error."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            # Should not raise
            await router.unsubscribe("NONEXISTENT", consumer_id="test")
        finally:
            await router.stop()


class TestCallbackFanout:
    """Consumer callback distribution."""

    @pytest.mark.asyncio
    async def test_fanout_calls_registered_callbacks(self):
        """Fanout distributes data to all consumers."""
        router = MarketDataRouter(ib=None)

        callback1_data = []
        callback2_data = []

        router.register_callback("consumer1", lambda sym, data: callback1_data.append((sym, data)))
        router.register_callback("consumer2", lambda sym, data: callback2_data.append((sym, data)))

        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="consumer1")
            await router.subscribe("AAPL", consumer_id="consumer2")

            # Simulate market data update
            router._fanout("AAPL", {"price": 150.0})

            assert len(callback1_data) == 1
            assert len(callback2_data) == 1
            assert callback1_data[0] == ("AAPL", {"price": 150.0})
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_fanout_only_to_subscribed_consumers(self):
        """Fanout only calls callbacks for subscribed consumers."""
        router = MarketDataRouter(ib=None)

        callback1_data = []
        callback2_data = []

        router.register_callback("consumer1", lambda sym, data: callback1_data.append((sym, data)))
        router.register_callback("consumer2", lambda sym, data: callback2_data.append((sym, data)))

        await router.start()

        try:
            # Only consumer1 subscribes to AAPL
            await router.subscribe("AAPL", consumer_id="consumer1")

            router._fanout("AAPL", {"price": 150.0})

            assert len(callback1_data) == 1
            assert len(callback2_data) == 0  # Not subscribed
        finally:
            await router.stop()

    def test_unregister_callback_removes_it(self):
        """Unregistered callbacks are not called."""
        router = MarketDataRouter(ib=None)

        callback_data = []
        router.register_callback("consumer1", lambda sym, data: callback_data.append(data))
        router.unregister_callback("consumer1")

        # Callback should be gone
        assert "consumer1" not in router._callbacks


class TestMetrics:
    """Metrics tracking."""

    @pytest.mark.asyncio
    async def test_tracks_subscription_totals(self):
        """Metrics track total subscriptions."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="test")
            await router.subscribe("MSFT", consumer_id="test")

            metrics = router.get_metrics()
            assert metrics.subscriptions_total == 2
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_tracks_duplicate_requests(self):
        """Metrics track duplicate subscription requests."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="consumer1")
            await router.subscribe("AAPL", consumer_id="consumer2")  # Adds to existing

            metrics = router.get_metrics()
            assert metrics.duplicate_requests >= 1
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_get_stats_returns_dict(self):
        """get_stats() returns monitoring-friendly dict."""
        router = MarketDataRouter(ib=None)
        await router.start()

        try:
            await router.subscribe("AAPL", consumer_id="test")

            stats = router.get_stats()

            assert "running" in stats
            assert "active_lines" in stats
            assert "max_lines" in stats
            assert "available_lines" in stats
            assert stats["running"] is True
        finally:
            await router.stop()


class TestStartStop:
    """Lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self):
        """Start sets running to True."""
        router = MarketDataRouter(ib=None)

        await router.start()
        assert router._running is True

        await router.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_subscriptions(self):
        """Stop cancels all subscriptions."""
        router = MarketDataRouter(ib=None)
        await router.start()

        await router.subscribe("AAPL", consumer_id="test")
        await router.subscribe("MSFT", consumer_id="test")

        await router.stop()

        assert router._running is False
        assert len(router.get_all_symbols()) == 0

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        """Starting twice doesn't cause issues."""
        router = MarketDataRouter(ib=None)

        await router.start()
        await router.start()  # Should be no-op

        assert router._running is True
        await router.stop()
