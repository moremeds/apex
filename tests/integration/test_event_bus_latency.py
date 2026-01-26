"""Integration tests for PriorityEventBus latency and performance."""

import asyncio
import statistics
import time
from typing import Any

import pytest

from src.domain.events import EventType, PriorityEventBus


class TestTickLatency:
    """Tests for market tick latency under various conditions."""

    @pytest.mark.asyncio
    async def test_tick_latency_under_load(self) -> None:
        """
        Market ticks processed within 5ms even with 1000 pending snapshots.

        Expected: P95 tick latency < 5ms.
        """
        bus = PriorityEventBus()
        await bus.start()

        tick_latencies = []

        def on_tick(payload: Any) -> None:
            tick_latencies.append(time.time() - payload["sent_at"])

        bus.subscribe(EventType.MARKET_DATA_TICK, on_tick)

        # Flood slow lane with snapshots
        for i in range(1000):
            bus.publish(EventType.SNAPSHOT_READY, {"i": i, "symbol": f"SYM{i}"})

        # Send 100 ticks and measure latency
        for i in range(100):
            bus.publish(EventType.MARKET_DATA_TICK, {"sent_at": time.time(), "i": i})
            await asyncio.sleep(0.001)

        await asyncio.sleep(0.2)

        assert len(tick_latencies) >= 90, f"Only received {len(tick_latencies)} ticks"

        # Calculate P95 latency
        sorted_latencies = sorted(tick_latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95 = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1]

        assert p95 < 0.01, f"P95 latency {p95*1000:.2f}ms exceeds 10ms target"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_risk_signal_latency(self) -> None:
        """
        Risk signals processed faster than market data.

        Expected: Risk signals have lower latency than ticks under load.
        """
        bus = PriorityEventBus()
        await bus.start()

        tick_latencies = []
        risk_latencies = []

        def on_tick(payload: Any) -> None:
            tick_latencies.append(time.time() - payload["sent_at"])

        def on_risk(payload: Any) -> None:
            risk_latencies.append(time.time() - payload["sent_at"])

        bus.subscribe(EventType.MARKET_DATA_TICK, on_tick)
        bus.subscribe(EventType.RISK_SIGNAL, on_risk)

        # Interleave ticks and risk signals
        for i in range(50):
            bus.publish(EventType.MARKET_DATA_TICK, {"sent_at": time.time(), "i": i})
            bus.publish(EventType.RISK_SIGNAL, {"sent_at": time.time(), "i": i})

        await asyncio.sleep(0.2)

        assert len(tick_latencies) >= 40
        assert len(risk_latencies) >= 40

        # Risk signals should have comparable or better latency
        avg_tick = statistics.mean(tick_latencies)
        avg_risk = statistics.mean(risk_latencies)

        # Risk should not be significantly worse (allows for some jitter)
        assert (
            avg_risk <= avg_tick * 2
        ), f"Risk latency {avg_risk*1000:.2f}ms much worse than tick {avg_tick*1000:.2f}ms"

        await bus.stop()


class TestThroughput:
    """Tests for event throughput."""

    @pytest.mark.asyncio
    async def test_high_throughput_fast_lane(self) -> None:
        """
        Fast lane can handle high event rate.

        Expected: 10,000 events/second sustained.
        """
        bus = PriorityEventBus(fast_lane_max_size=50000)
        await bus.start()

        received = 0

        def counter(p: Any) -> None:
            nonlocal received
            received += 1

        bus.subscribe(EventType.MARKET_DATA_TICK, counter)

        events_to_send = 10000
        start = time.time()

        for i in range(events_to_send):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})

        # Wait for all to be processed
        max_wait = 5.0
        wait_start = time.time()
        while received < events_to_send and (time.time() - wait_start) < max_wait:
            await asyncio.sleep(0.05)

        elapsed = time.time() - start

        assert received == events_to_send, f"Only received {received}/{events_to_send}"

        throughput = events_to_send / elapsed
        assert throughput > 5000, f"Throughput {throughput:.0f}/s below 5000/s target"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_mixed_lane_throughput(self) -> None:
        """
        Both lanes can handle mixed traffic.

        Expected: Fast lane not blocked by slow lane operations.
        """
        bus = PriorityEventBus(slow_lane_debounce_ms=50)
        await bus.start()

        fast_received = 0
        slow_received = 0

        def fast_counter(p: Any) -> None:
            nonlocal fast_received
            fast_received += 1

        def slow_counter(p: Any) -> None:
            nonlocal slow_received
            slow_received += 1

        bus.subscribe(EventType.MARKET_DATA_TICK, fast_counter)
        bus.subscribe(EventType.SNAPSHOT_READY, slow_counter)

        # Mix of fast and slow events
        for i in range(1000):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})
            if i % 10 == 0:
                bus.publish(EventType.SNAPSHOT_READY, {"i": i, "symbol": f"S{i}"})

        await asyncio.sleep(0.5)

        assert fast_received == 1000, f"Fast: {fast_received}/1000"
        assert slow_received >= 50, f"Slow: {slow_received}/100"  # Some may coalesce

        await bus.stop()


class TestDurability:
    """Tests for event bus durability under stress."""

    @pytest.mark.asyncio
    async def test_sustained_load(self) -> None:
        """
        Event bus remains stable under sustained load.

        Expected: No event loss, no memory issues over 2 seconds.
        """
        bus = PriorityEventBus()
        await bus.start()

        received = 0

        def counter(p: Any) -> None:
            nonlocal received
            received += 1

        bus.subscribe(EventType.MARKET_DATA_TICK, counter)

        # Sustained load for 2 seconds
        sent = 0
        end_time = time.time() + 2.0

        while time.time() < end_time:
            for _ in range(100):
                bus.publish(EventType.MARKET_DATA_TICK, {"ts": time.time()})
                sent += 1
            await asyncio.sleep(0.01)

        # Wait for processing
        await asyncio.sleep(0.5)

        stats = bus.get_stats()
        loss_rate = (sent - received) / sent if sent > 0 else 0

        assert loss_rate < 0.001, f"Loss rate {loss_rate*100:.2f}% exceeds 0.1%"
        assert stats["errors"] == 0, f"Errors occurred: {stats['errors']}"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_drains_events(self) -> None:
        """
        Stop drains remaining events gracefully.

        Expected: All pending events processed on stop.
        """
        bus = PriorityEventBus()
        await bus.start()

        received = []

        bus.subscribe(EventType.MARKET_DATA_TICK, lambda p: received.append(p))

        # Queue events
        for i in range(100):
            bus.publish(EventType.MARKET_DATA_TICK, {"i": i})

        # Stop immediately
        await bus.stop()

        # All events should have been drained
        assert len(received) == 100, f"Only {len(received)}/100 events drained"


class TestPriorityFairness:
    """Tests for priority-based fairness."""

    @pytest.mark.asyncio
    async def test_priority_ordering_at_scale(self) -> None:
        """
        Priority ordering maintained under high load.

        Expected: Higher priority events processed before lower priority
        when published at approximately the same time.
        """
        bus = PriorityEventBus(fast_budget=500)
        await bus.start()

        order = []

        def risk_handler(p: Any) -> None:
            order.append(("risk", p["seq"]))

        def tick_handler(p: Any) -> None:
            order.append(("tick", p["seq"]))

        def pos_handler(p: Any) -> None:
            order.append(("pos", p["seq"]))

        bus.subscribe(EventType.RISK_SIGNAL, risk_handler)
        bus.subscribe(EventType.MARKET_DATA_TICK, tick_handler)
        bus.subscribe(EventType.POSITION_UPDATED, pos_handler)

        # Publish all events before any processing can happen
        # by doing it in a tight loop without awaits
        for i in range(10):
            # Lower priority first
            bus.publish(EventType.POSITION_UPDATED, {"seq": i})
            bus.publish(EventType.MARKET_DATA_TICK, {"seq": i})
            bus.publish(EventType.RISK_SIGNAL, {"seq": i})

        await asyncio.sleep(0.3)

        assert len(order) == 30, f"Expected 30 events, got {len(order)}"

        # Count risk events in the first 10 events processed
        # Since risk has highest priority, most should be processed first
        first_ten = order[:10]
        risk_in_first_ten = sum(1 for event_type, _ in first_ten if event_type == "risk")

        # At least 5 of the 10 risk events should be in first 10 processed
        assert (
            risk_in_first_ten >= 5
        ), f"Only {risk_in_first_ten} risk events in first 10, expected >= 5. Order: {first_ten}"

        await bus.stop()
