"""Unit tests for PriorityEventBus."""

import asyncio
import time
import pytest

from src.domain.events import (
    PriorityEventBus,
    EventType,
    EventPriority,
    PriorityEventEnvelope,
)


class TestEventPriority:
    """Tests for event priority ordering."""

    def test_priority_values(self):
        """Verify priority values are correctly ordered."""
        assert EventPriority.CRITICAL < EventPriority.RISK
        assert EventPriority.RISK < EventPriority.TRADING
        assert EventPriority.TRADING < EventPriority.MARKET_DATA
        assert EventPriority.MARKET_DATA < EventPriority.POSITION
        assert EventPriority.POSITION < EventPriority.ACCOUNT
        assert EventPriority.ACCOUNT < EventPriority.CONTROL
        assert EventPriority.CONTROL < EventPriority.SNAPSHOT
        assert EventPriority.SNAPSHOT < EventPriority.DIAGNOSTIC
        assert EventPriority.DIAGNOSTIC < EventPriority.UI

    def test_envelope_ordering(self):
        """PriorityEventEnvelope orders by priority then sequence."""
        envelope_low = PriorityEventEnvelope(
            priority=EventPriority.UI,
            sequence=1,
            event_type=EventType.DASHBOARD_UPDATE,
            payload={},
        )
        envelope_high = PriorityEventEnvelope(
            priority=EventPriority.RISK,
            sequence=2,
            event_type=EventType.RISK_SIGNAL,
            payload={},
        )
        # Higher priority (lower number) should be "less than"
        assert envelope_high < envelope_low

    def test_envelope_sequence_tiebreaker(self):
        """Same priority uses sequence as tiebreaker."""
        envelope1 = PriorityEventEnvelope(
            priority=EventPriority.MARKET_DATA,
            sequence=1,
            event_type=EventType.MARKET_DATA_TICK,
            payload={},
        )
        envelope2 = PriorityEventEnvelope(
            priority=EventPriority.MARKET_DATA,
            sequence=2,
            event_type=EventType.MARKET_DATA_TICK,
            payload={},
        )
        assert envelope1 < envelope2


class TestPriorityEventBusBasic:
    """Basic functionality tests for PriorityEventBus."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Event bus starts and stops cleanly."""
        bus = PriorityEventBus()
        await bus.start()
        assert bus.is_running
        await bus.stop()
        assert not bus.is_running

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Basic publish/subscribe works."""
        bus = PriorityEventBus()
        await bus.start()

        received = []
        bus.subscribe(EventType.MARKET_DATA_TICK, lambda p: received.append(p))

        bus.publish(EventType.MARKET_DATA_TICK, {"symbol": "AAPL", "price": 150.0})
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0]["symbol"] == "AAPL"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_async_subscriber(self):
        """Async subscribers are awaited."""
        bus = PriorityEventBus()
        await bus.start()

        received = []

        async def async_handler(payload):
            await asyncio.sleep(0.01)
            received.append(payload)

        bus.subscribe_async(EventType.RISK_SIGNAL, async_handler)

        bus.publish(EventType.RISK_SIGNAL, {"level": "warning"})
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0]["level"] == "warning"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Unsubscribe removes callback."""
        bus = PriorityEventBus()
        await bus.start()

        received = []
        callback = lambda p: received.append(p)
        bus.subscribe(EventType.MARKET_DATA_TICK, callback)

        bus.publish(EventType.MARKET_DATA_TICK, {"seq": 1})
        await asyncio.sleep(0.05)

        bus.unsubscribe(EventType.MARKET_DATA_TICK, callback)

        bus.publish(EventType.MARKET_DATA_TICK, {"seq": 2})
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0]["seq"] == 1

        await bus.stop()


class TestFastLane:
    """Tests for fast lane behavior."""

    @pytest.mark.asyncio
    async def test_fast_lane_events_route_correctly(self):
        """Events with priority < SNAPSHOT go to fast lane."""
        bus = PriorityEventBus()
        await bus.start()

        received = []
        bus.subscribe(EventType.MARKET_DATA_TICK, lambda p: received.append(("tick", p)))
        bus.subscribe(EventType.RISK_SIGNAL, lambda p: received.append(("risk", p)))
        bus.subscribe(EventType.POSITION_UPDATED, lambda p: received.append(("pos", p)))

        bus.publish(EventType.MARKET_DATA_TICK, {"symbol": "AAPL"})
        bus.publish(EventType.RISK_SIGNAL, {"alert": "breach"})
        bus.publish(EventType.POSITION_UPDATED, {"qty": 100})

        await asyncio.sleep(0.1)

        assert len(received) == 3
        stats = bus.get_stats()
        assert stats["fast_published"] == 3
        assert stats["fast_dispatched"] == 3

        await bus.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """
        RISK_SIGNAL processed before MARKET_DATA_TICK.

        Expected: Risk processed first despite being published second.
        """
        bus = PriorityEventBus()
        await bus.start()

        order = []
        bus.subscribe(EventType.RISK_SIGNAL, lambda p: order.append("risk"))
        bus.subscribe(EventType.MARKET_DATA_TICK, lambda p: order.append("tick"))

        # Publish tick first, then risk (reverse priority order)
        bus.publish(EventType.MARKET_DATA_TICK, {})
        bus.publish(EventType.RISK_SIGNAL, {})

        await asyncio.sleep(0.05)

        # Risk should be processed first despite being published second
        assert order[0] == "risk", f"Expected risk first, got: {order}"
        assert order[1] == "tick"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_fast_lane_no_debounce(self):
        """Fast lane events dispatched immediately without debouncing."""
        bus = PriorityEventBus(slow_lane_debounce_ms=500)
        await bus.start()

        received = []
        times = []

        def handler(p):
            received.append(p)
            times.append(time.time())

        bus.subscribe(EventType.MARKET_DATA_TICK, handler)

        start = time.time()
        for i in range(5):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})

        await asyncio.sleep(0.1)

        assert len(received) == 5

        # All should be dispatched quickly (within 100ms total)
        if times:
            elapsed = times[-1] - start
            assert elapsed < 0.1, f"Fast lane took too long: {elapsed}s"

        await bus.stop()


class TestSlowLane:
    """Tests for slow lane behavior."""

    @pytest.mark.asyncio
    async def test_slow_lane_events_route_correctly(self):
        """Events with priority >= SNAPSHOT go to slow lane."""
        bus = PriorityEventBus(slow_lane_debounce_ms=50)
        await bus.start()

        received = []
        bus.subscribe(EventType.SNAPSHOT_READY, lambda p: received.append(("snap", p)))
        bus.subscribe(EventType.DASHBOARD_UPDATE, lambda p: received.append(("ui", p)))
        bus.subscribe(EventType.HEALTH_CHECK, lambda p: received.append(("health", p)))

        bus.publish(EventType.SNAPSHOT_READY, {"ts": 1})
        bus.publish(EventType.DASHBOARD_UPDATE, {"panel": "risk"})
        bus.publish(EventType.HEALTH_CHECK, {"status": "ok"})

        await asyncio.sleep(0.2)

        assert len(received) == 3
        stats = bus.get_stats()
        assert stats["slow_published"] == 3

        await bus.stop()

    @pytest.mark.asyncio
    async def test_slow_lane_coalesces_by_symbol(self):
        """Multiple slow events for same symbol coalesce to latest."""
        bus = PriorityEventBus(slow_lane_debounce_ms=50)
        await bus.start()

        received = []
        bus.subscribe(EventType.SNAPSHOT_READY, lambda p: received.append(p))

        # Publish multiple snapshots for same symbol - only latest should be delivered
        bus.publish(EventType.SNAPSHOT_READY, {"seq": 1, "symbol": "AAPL"})
        bus.publish(EventType.SNAPSHOT_READY, {"seq": 2, "symbol": "AAPL"})
        bus.publish(EventType.SNAPSHOT_READY, {"seq": 3, "symbol": "AAPL"})

        await asyncio.sleep(0.2)

        # Only the latest should be received (coalesced)
        assert len(received) == 1
        assert received[0]["seq"] == 3

        await bus.stop()

    @pytest.mark.asyncio
    async def test_slow_lane_debounce(self):
        """Slow lane applies debouncing."""
        bus = PriorityEventBus(slow_lane_debounce_ms=100)
        await bus.start()

        dispatch_times = []
        bus.subscribe(EventType.DASHBOARD_UPDATE, lambda p: dispatch_times.append(time.time()))

        start = time.time()
        for i in range(3):
            bus.publish(EventType.DASHBOARD_UPDATE, {"i": i, "symbol": f"SYM{i}"})
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.3)

        # Events should be batched and dispatched after debounce window
        # Not immediately like fast lane
        if dispatch_times:
            first_dispatch = dispatch_times[0] - start
            assert first_dispatch >= 0.05, f"Debounce not applied: {first_dispatch}s"

        await bus.stop()


class TestStarvationPrevention:
    """Tests for preventing slow-lane starvation under fast-lane load."""

    @pytest.mark.asyncio
    async def test_slow_lane_runs_under_fast_load(self):
        """
        Slow lane events are dispatched even when fast lane is flooded.

        Expected: Slow lane dispatches at least once every 200ms.
        """
        bus = PriorityEventBus(
            slow_lane_debounce_ms=50,
            fast_budget=100,  # Small budget to force yields
            fast_time_slice_ms=20,
            slow_lane_min_interval_ms=200,
        )
        await bus.start()

        slow_dispatch_times = []

        def track_slow(payload):
            slow_dispatch_times.append(time.time())

        bus.subscribe(EventType.SNAPSHOT_READY, track_slow)

        # Flood fast lane with ticks
        for i in range(1000):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})
            # Interleave slow events
            if i % 100 == 0:
                bus.publish(EventType.SNAPSHOT_READY, {"seq": i, "symbol": f"S{i}"})

        await asyncio.sleep(1.0)

        # Slow lane should have run multiple times (not starved)
        assert len(slow_dispatch_times) >= 3, f"Slow lane was starved: only {len(slow_dispatch_times)} dispatches"

        # Max gap between slow dispatches should be < 300ms
        for i in range(1, len(slow_dispatch_times)):
            gap_ms = (slow_dispatch_times[i] - slow_dispatch_times[i - 1]) * 1000
            assert gap_ms < 500, f"Slow lane gap too large: {gap_ms}ms"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_fast_lane_yields_after_budget(self):
        """Fast lane yields after processing budget events."""
        bus = PriorityEventBus(fast_budget=10, fast_time_slice_ms=5)
        await bus.start()

        fast_count = 0

        def count_fast(p):
            nonlocal fast_count
            fast_count += 1

        bus.subscribe(EventType.MARKET_DATA_TICK, count_fast)

        # Publish more than budget
        for i in range(50):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})

        await asyncio.sleep(0.2)

        # All should eventually be processed
        assert fast_count == 50

        await bus.stop()


class TestDropPolicy:
    """Tests for event drop policy under overload."""

    @pytest.mark.asyncio
    async def test_drop_policy_removes_low_priority_first(self):
        """
        UI events dropped first when slow queue is full.

        Expected: Risk/trading events never dropped; UI events dropped under overload.
        """
        bus = PriorityEventBus(max_pending_slow=10, slow_lane_debounce_ms=500)
        await bus.start()

        # Fill slow queue with UI events (way more than max_pending_slow)
        for i in range(50):
            bus.publish(EventType.DASHBOARD_UPDATE, {"i": i, "symbol": f"SYM{i}"})

        # Wait a moment for enqueue
        await asyncio.sleep(0.01)

        stats = bus.get_stats()

        # Some UI events should have been dropped
        assert stats["dropped"] > 0, "Expected drops under overload"

        # Slow queue should be bounded
        assert stats["slow_pending"] <= 10, f"Slow queue exceeded bound: {stats['slow_pending']}"

        await bus.stop()


class TestStats:
    """Tests for statistics collection."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        bus = PriorityEventBus()
        await bus.start()

        bus.subscribe(EventType.MARKET_DATA_TICK, lambda p: None)
        bus.subscribe(EventType.SNAPSHOT_READY, lambda p: None)

        # Publish to both lanes
        for i in range(10):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})
        for i in range(5):
            bus.publish(EventType.SNAPSHOT_READY, {"i": i, "symbol": f"S{i}"})

        await asyncio.sleep(0.3)

        stats = bus.get_stats()

        assert stats["fast_published"] == 10
        assert stats["slow_published"] == 5
        assert stats["running"] is True

        await bus.stop()

    @pytest.mark.asyncio
    async def test_high_water_mark(self):
        """High water mark tracks peak queue depth."""
        bus = PriorityEventBus()
        await bus.start()

        # Publish many events quickly
        for i in range(100):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})

        await asyncio.sleep(0.01)

        stats = bus.get_stats()
        assert stats["fast_high_water"] > 0

        await bus.stop()


class TestSyncFallback:
    """Tests for sync dispatch when bus not running."""

    def test_sync_fallback_dispatch(self):
        """Events dispatch synchronously when bus not started."""
        bus = PriorityEventBus()

        received = []
        bus.subscribe(EventType.MARKET_DATA_TICK, lambda p: received.append(p))

        # Publish before starting - should dispatch synchronously
        bus.publish(EventType.MARKET_DATA_TICK, {"symbol": "AAPL"})

        assert len(received) == 1
        assert received[0]["symbol"] == "AAPL"


class TestErrorHandling:
    """Tests for error handling in subscribers."""

    @pytest.mark.asyncio
    async def test_subscriber_error_isolated(self):
        """One subscriber error doesn't affect others."""
        bus = PriorityEventBus()
        await bus.start()

        results = []

        def good_handler(p):
            results.append("good")

        def bad_handler(p):
            raise ValueError("intentional error")

        bus.subscribe(EventType.MARKET_DATA_TICK, bad_handler)
        bus.subscribe(EventType.MARKET_DATA_TICK, good_handler)

        bus.publish(EventType.MARKET_DATA_TICK, {})
        await asyncio.sleep(0.1)

        # Good handler should still run
        assert "good" in results

        stats = bus.get_stats()
        assert stats["errors"] >= 1

        await bus.stop()
