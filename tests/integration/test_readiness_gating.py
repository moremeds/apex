"""Integration tests for ReadinessManager with event bus."""

import asyncio
from datetime import datetime
from typing import Any, List, Tuple

import pytest

from src.application.readiness_manager import (
    ReadinessManager,
    ReadinessState,
)
from src.domain.events import EventType, PriorityEventBus


class TestReadinessWithEventBus:
    """Tests for ReadinessManager integration with PriorityEventBus."""

    @pytest.mark.asyncio
    async def test_readiness_events_flow_through_bus(self) -> None:
        """Readiness events are published through the event bus."""
        bus = PriorityEventBus()
        await bus.start()

        received_events: List[Tuple[str, Any]] = []

        def capture_broker_connected(payload: Any) -> None:
            received_events.append(("broker_connected", payload))

        def capture_positions_ready(payload: Any) -> None:
            received_events.append(("positions_ready", payload))

        def capture_market_data_ready(payload: Any) -> None:
            received_events.append(("market_data_ready", payload))

        def capture_system_ready(payload: Any) -> None:
            received_events.append(("system_ready", payload))

        bus.subscribe(EventType.BROKER_CONNECTED, capture_broker_connected)
        bus.subscribe(EventType.POSITIONS_READY, capture_positions_ready)
        bus.subscribe(EventType.MARKET_DATA_READY, capture_market_data_ready)
        bus.subscribe(EventType.SYSTEM_READY, capture_system_ready)

        manager = ReadinessManager(bus, required_brokers=["ib"])

        # Simulate startup sequence
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)  # 90%

        # Wait for event processing
        await asyncio.sleep(0.1)

        await bus.stop()

        # Verify events were received
        event_types = [e[0] for e in received_events]
        assert "broker_connected" in event_types
        assert "positions_ready" in event_types
        assert "market_data_ready" in event_types
        assert "system_ready" in event_types

    @pytest.mark.asyncio
    async def test_degraded_event_on_disconnect(self) -> None:
        """SYSTEM_DEGRADED event flows through bus on disconnect."""
        bus = PriorityEventBus()
        await bus.start()

        degraded_events = []

        def capture_degraded(payload: Any) -> None:
            degraded_events.append(payload)

        bus.subscribe(EventType.SYSTEM_DEGRADED, capture_degraded)

        manager = ReadinessManager(bus, required_brokers=["ib"])

        # Get to ready state
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)

        await asyncio.sleep(0.05)

        # Disconnect broker
        manager.on_broker_disconnected("ib", "Test disconnect")

        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(degraded_events) == 1
        assert degraded_events[0]["reason"] == "No brokers connected"

    @pytest.mark.asyncio
    async def test_fast_startup_with_quick_data(self) -> None:
        """
        System becomes ready quickly when data arrives fast.

        This replaces the arbitrary 30-second timeout with event-driven readiness.
        """
        bus = PriorityEventBus()
        await bus.start()

        ready_time = None

        def capture_ready(payload: Any) -> None:
            nonlocal ready_time
            ready_time = datetime.now()

        bus.subscribe(EventType.SYSTEM_READY, capture_ready)

        manager = ReadinessManager(bus, required_brokers=["ib"])
        start_time = datetime.now()

        # Rapid data arrival
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 100)
        manager.on_market_data_update(100, 90)

        await asyncio.sleep(0.1)
        await bus.stop()

        assert ready_time is not None
        elapsed = (ready_time - start_time).total_seconds()

        # Should be ready in well under 1 second
        assert elapsed < 1.0, f"Took {elapsed}s to become ready"

    @pytest.mark.asyncio
    async def test_gating_pattern(self) -> None:
        """
        Demonstrate gating pattern for downstream operations.

        Operations should wait for SYSTEM_READY before proceeding.
        """
        bus = PriorityEventBus()
        await bus.start()

        operation_performed = False
        operation_performed_at_state = None

        manager = ReadinessManager(bus, required_brokers=["ib"])

        def gated_operation(payload: Any) -> None:
            nonlocal operation_performed, operation_performed_at_state
            operation_performed = True
            operation_performed_at_state = manager.state

        # Subscribe to SYSTEM_READY to trigger gated operation
        bus.subscribe(EventType.SYSTEM_READY, gated_operation)

        # Partial readiness - operation should not run
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        await asyncio.sleep(0.05)

        assert not operation_performed
        assert not manager.is_ready()

        # Complete readiness - operation should run
        manager.on_market_data_update(50, 45)
        await asyncio.sleep(0.1)

        await bus.stop()

        assert operation_performed
        assert operation_performed_at_state == ReadinessState.SYSTEM_READY

    @pytest.mark.asyncio
    async def test_multiple_subscribers_notified(self) -> None:
        """Multiple components can subscribe to readiness events."""
        bus = PriorityEventBus()
        await bus.start()

        notifications = []

        def component_a(payload: Any) -> None:
            notifications.append("A")

        def component_b(payload: Any) -> None:
            notifications.append("B")

        def component_c(payload: Any) -> None:
            notifications.append("C")

        bus.subscribe(EventType.SYSTEM_READY, component_a)
        bus.subscribe(EventType.SYSTEM_READY, component_b)
        bus.subscribe(EventType.SYSTEM_READY, component_c)

        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)

        await asyncio.sleep(0.1)
        await bus.stop()

        # All components should be notified
        assert "A" in notifications
        assert "B" in notifications
        assert "C" in notifications


class TestReadinessRecovery:
    """Tests for system recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recovery_after_degradation(self) -> None:
        """System can recover and emit SYSTEM_READY again."""
        bus = PriorityEventBus()
        await bus.start()

        ready_count = 0

        def count_ready(payload: Any) -> None:
            nonlocal ready_count
            ready_count += 1

        bus.subscribe(EventType.SYSTEM_READY, count_ready)

        manager = ReadinessManager(bus, required_brokers=["ib"])

        # First ready
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)
        await asyncio.sleep(0.05)

        assert ready_count == 1

        # Degrade
        manager.on_broker_disconnected("ib")
        await asyncio.sleep(0.05)

        assert manager.is_degraded()

        # Recover
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)
        await asyncio.sleep(0.1)

        await bus.stop()

        # Should emit SYSTEM_READY again after recovery
        assert ready_count == 2

    @pytest.mark.asyncio
    async def test_partial_broker_failure(self) -> None:
        """System handles partial broker failures."""
        bus = PriorityEventBus()
        await bus.start()

        manager = ReadinessManager(bus, required_brokers=["ib", "futu"])

        # Both brokers connect
        manager.on_broker_connected("ib")
        manager.on_broker_connected("futu")
        manager.on_positions_loaded("ib", 30)
        manager.on_positions_loaded("futu", 20)
        manager.on_market_data_update(50, 45)

        await asyncio.sleep(0.05)
        assert manager.is_ready()

        # One broker disconnects
        manager.on_broker_disconnected("futu")
        await asyncio.sleep(0.05)

        # Still not degraded - one broker remains
        # (State depends on implementation - could still have IB connected)
        snapshot = manager.get_snapshot()
        assert snapshot.brokers["ib"].connected
        assert not snapshot.brokers["futu"].connected

        await bus.stop()


class TestReadinessPerformance:
    """Tests for readiness system performance."""

    @pytest.mark.asyncio
    async def test_rapid_updates_handled(self) -> None:
        """System handles rapid status updates without issues."""
        bus = PriorityEventBus()
        await bus.start()

        manager = ReadinessManager(bus, required_brokers=["ib"])
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 100)

        # Rapid market data updates
        for i in range(100):
            coverage = 50 + i // 2  # Gradually increase coverage
            manager.on_market_data_update(100, coverage)

        await asyncio.sleep(0.2)
        await bus.stop()

        # Should end up ready with high coverage
        assert manager.is_ready()
        assert manager.coverage_ratio == 0.99

    @pytest.mark.asyncio
    async def test_concurrent_broker_updates(self) -> None:
        """Handles concurrent broker status updates."""
        bus = PriorityEventBus()
        await bus.start()

        manager = ReadinessManager(bus, required_brokers=["ib", "futu", "manual"])

        # Concurrent connections
        manager.on_broker_connected("ib")
        manager.on_broker_connected("futu")
        manager.on_broker_connected("manual")

        # Concurrent position loads
        manager.on_positions_loaded("ib", 30)
        manager.on_positions_loaded("futu", 20)
        manager.on_positions_loaded("manual", 10)

        manager.on_market_data_update(60, 54)  # 90%

        await asyncio.sleep(0.1)
        await bus.stop()

        assert manager.is_ready()
        snapshot = manager.get_snapshot()
        assert snapshot.brokers["ib"].position_count == 30
        assert snapshot.brokers["futu"].position_count == 20
        assert snapshot.brokers["manual"].position_count == 10
