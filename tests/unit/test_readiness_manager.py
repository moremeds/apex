"""Unit tests for ReadinessManager."""

import pytest
from datetime import datetime, timedelta
from typing import Any, List, Tuple

from src.application.readiness_manager import (
    ReadinessManager,
    ReadinessState,
    BrokerStatus,
    MarketDataStatus,
    DataFreshness,
)
from src.domain.events.event_types import EventType


class MockEventBus:
    """Mock event bus for testing."""

    def __init__(self):
        self.published_events: List[Tuple[EventType, Any]] = []

    def publish(self, event_type: EventType, payload: Any) -> None:
        self.published_events.append((event_type, payload))

    def get_events_of_type(self, event_type: EventType) -> List[Any]:
        return [p for et, p in self.published_events if et == event_type]

    def clear(self):
        self.published_events.clear()


class TestReadinessManagerInit:
    """Tests for ReadinessManager initialization."""

    def test_initial_state_is_starting(self):
        """Manager starts in STARTING state."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        assert manager.state == ReadinessState.STARTING
        assert not manager.is_ready()
        assert not manager.is_positions_ready()
        assert not manager.is_market_data_ready()

    def test_broker_status_initialized(self):
        """Required brokers are tracked."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib", "futu"])

        snapshot = manager.get_snapshot()
        assert "ib" in snapshot.brokers
        assert "futu" in snapshot.brokers
        assert not snapshot.brokers["ib"].connected
        assert not snapshot.brokers["futu"].connected


class TestBrokerReadiness:
    """Tests for broker connection tracking."""

    def test_broker_connected_event_emitted(self):
        """BROKER_CONNECTED event emitted on connection."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")

        events = bus.get_events_of_type(EventType.BROKER_CONNECTED)
        assert len(events) == 1
        assert events[0]["broker"] == "ib"

    def test_broker_disconnected_event_emitted(self):
        """BROKER_DISCONNECTED event emitted on disconnection."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_broker_disconnected("ib", "Connection lost")

        events = bus.get_events_of_type(EventType.BROKER_DISCONNECTED)
        assert len(events) == 1
        assert events[0]["broker"] == "ib"
        assert events[0]["error"] == "Connection lost"

    def test_state_transitions_to_brokers_connecting(self):
        """State transitions to BROKERS_CONNECTING when broker connects."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        assert manager.state == ReadinessState.STARTING

        manager.on_broker_connected("ib")

        assert manager.state == ReadinessState.BROKERS_CONNECTING

    def test_unknown_broker_added_dynamically(self):
        """Unknown brokers are tracked when they connect."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("futu")

        snapshot = manager.get_snapshot()
        assert "futu" in snapshot.brokers
        assert snapshot.brokers["futu"].connected


class TestPositionsReadiness:
    """Tests for positions loading tracking."""

    def test_positions_ready_when_all_brokers_loaded(self):
        """POSITIONS_READY emitted when all connected brokers have loaded."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib", "futu"])

        manager.on_broker_connected("ib")
        manager.on_broker_connected("futu")
        manager.on_positions_loaded("ib", 50)

        # Not ready yet - futu hasn't loaded
        assert not manager.is_positions_ready()

        manager.on_positions_loaded("futu", 20)

        # Now ready
        assert manager.is_positions_ready()
        events = bus.get_events_of_type(EventType.POSITIONS_READY)
        assert len(events) == 1
        assert events[0]["total_positions"] == 70
        assert events[0]["brokers"]["ib"] == 50
        assert events[0]["brokers"]["futu"] == 20

    def test_positions_ready_with_single_broker(self):
        """POSITIONS_READY works with single broker."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 100)

        assert manager.is_positions_ready()
        events = bus.get_events_of_type(EventType.POSITIONS_READY)
        assert len(events) == 1
        assert events[0]["total_positions"] == 100

    def test_positions_ready_only_emitted_once(self):
        """POSITIONS_READY event only emitted once."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_positions_loaded("ib", 60)  # Second load

        events = bus.get_events_of_type(EventType.POSITIONS_READY)
        assert len(events) == 1

    def test_state_transitions_to_data_loading_after_positions(self):
        """State transitions to DATA_LOADING after positions loaded (when positions > 0)."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)

        # With positions, state immediately goes to DATA_LOADING (waiting for market data)
        assert manager.state == ReadinessState.DATA_LOADING


class TestMarketDataReadiness:
    """Tests for market data coverage tracking."""

    def test_market_data_ready_at_threshold(self):
        """MARKET_DATA_READY emitted at coverage threshold."""
        bus = MockEventBus()
        manager = ReadinessManager(
            bus,
            required_brokers=["ib"],
            market_data_coverage_threshold=0.9
        )

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 100)

        # 80% - not ready
        manager.on_market_data_update(100, 80)
        assert not manager.is_market_data_ready()

        # 90% - ready
        manager.on_market_data_update(100, 90)
        assert manager.is_market_data_ready()

        events = bus.get_events_of_type(EventType.MARKET_DATA_READY)
        assert len(events) == 1
        assert events[0]["coverage_ratio"] == 0.9

    def test_market_data_ready_only_emitted_once(self):
        """MARKET_DATA_READY event only emitted once."""
        bus = MockEventBus()
        manager = ReadinessManager(
            bus,
            required_brokers=["ib"],
            market_data_coverage_threshold=0.9
        )

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 100)
        manager.on_market_data_update(100, 90)
        manager.on_market_data_update(100, 95)  # Higher coverage

        events = bus.get_events_of_type(EventType.MARKET_DATA_READY)
        assert len(events) == 1

    def test_coverage_ratio_property(self):
        """Coverage ratio property returns current value."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        assert manager.coverage_ratio == 0.0

        manager.on_market_data_update(100, 75)
        assert manager.coverage_ratio == 0.75


class TestSystemReadiness:
    """Tests for full system readiness."""

    def test_system_ready_when_both_conditions_met(self):
        """SYSTEM_READY emitted when positions and market data ready."""
        bus = MockEventBus()
        manager = ReadinessManager(
            bus,
            required_brokers=["ib"],
            market_data_coverage_threshold=0.9
        )

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)  # 90%

        assert manager.is_ready()
        assert manager.state == ReadinessState.SYSTEM_READY

        events = bus.get_events_of_type(EventType.SYSTEM_READY)
        assert len(events) == 1

    def test_system_ready_requires_positions_first(self):
        """Must have positions before system can be ready."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        # Market data without positions
        manager.on_market_data_update(50, 45)

        assert not manager.is_ready()

    def test_system_ready_with_empty_portfolio(self):
        """System can be ready with zero positions."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 0)  # Empty portfolio

        # With no positions, no market data needed
        assert manager.state == ReadinessState.SYSTEM_READY


class TestDegradedState:
    """Tests for system degradation handling."""

    def test_system_degraded_when_broker_disconnects(self):
        """SYSTEM_DEGRADED emitted when all brokers disconnect."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        # Get to SYSTEM_READY
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)  # 90%

        assert manager.is_ready()

        # Broker disconnects
        manager.on_broker_disconnected("ib", "Connection lost")

        assert not manager.is_ready()
        assert manager.state == ReadinessState.DEGRADED
        assert manager.is_degraded()

        events = bus.get_events_of_type(EventType.SYSTEM_DEGRADED)
        assert len(events) == 1
        assert "No brokers connected" in events[0]["reason"]

    def test_system_degraded_when_coverage_drops(self):
        """SYSTEM_DEGRADED emitted when coverage drops below threshold."""
        bus = MockEventBus()
        manager = ReadinessManager(
            bus,
            required_brokers=["ib"],
            market_data_coverage_threshold=0.9
        )

        # Get to SYSTEM_READY at 90%
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 100)
        manager.on_market_data_update(100, 90)

        assert manager.is_ready()

        # Coverage drops below 72% (90% * 0.8)
        manager.on_market_data_update(100, 70)

        assert manager.state == ReadinessState.DEGRADED

    def test_recovery_from_degraded(self):
        """System can recover from DEGRADED state."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        # Get to SYSTEM_READY
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)

        # Disconnect -> DEGRADED
        manager.on_broker_disconnected("ib")
        assert manager.is_degraded()

        # Reconnect and reload
        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)

        # Should recover (but need to re-emit POSITIONS_READY)
        # Note: In current impl, positions_ready_emitted stays True
        # So recovery just needs broker connected + coverage threshold
        assert manager.state == ReadinessState.SYSTEM_READY


class TestDataFreshness:
    """Tests for data freshness tracking."""

    def test_freshness_initial_state(self):
        """Initial freshness state shows all data as stale."""
        freshness = DataFreshness()

        assert freshness.is_tick_stale()
        assert freshness.is_position_stale()
        assert freshness.is_exec_stale()
        assert freshness.any_stale()

    def test_freshness_updated_on_tick(self):
        """Tick freshness updated when tick received."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_tick_received()

        snapshot = manager.get_snapshot()
        assert snapshot.freshness.last_tick_time is not None
        assert not snapshot.freshness.is_tick_stale()

    def test_freshness_updated_on_positions(self):
        """Position freshness updated when positions loaded."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)

        snapshot = manager.get_snapshot()
        assert snapshot.freshness.last_position_time is not None

    def test_freshness_updated_on_exec_heartbeat(self):
        """Exec freshness updated on heartbeat."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_exec_heartbeat()

        snapshot = manager.get_snapshot()
        assert snapshot.freshness.last_exec_heartbeat is not None
        assert not snapshot.freshness.is_exec_stale()

    def test_stale_reasons(self):
        """Stale reasons reported correctly."""
        freshness = DataFreshness()

        reasons = freshness.stale_reasons()
        assert "tick_stale" in reasons
        assert "position_stale" in reasons
        assert "exec_heartbeat_stale" in reasons


class TestSnapshot:
    """Tests for readiness snapshot."""

    def test_snapshot_contains_all_info(self):
        """Snapshot contains complete readiness info."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib", "futu"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)

        snapshot = manager.get_snapshot()

        # Only IB is connected and has positions, so state goes to SYSTEM_READY
        # (even though futu is required but not connected, IB alone satisfies readiness)
        # Note: With only one connected broker that has loaded, system can proceed
        assert snapshot.state in (ReadinessState.DATA_LOADING, ReadinessState.SYSTEM_READY)
        assert len(snapshot.brokers) == 2
        assert snapshot.brokers["ib"].connected
        assert snapshot.brokers["ib"].position_count == 50
        assert snapshot.market_data.coverage_ratio == 0.9
        assert isinstance(snapshot.state_since, datetime)


class TestShutdown:
    """Tests for shutdown handling."""

    def test_shutdown_transitions_state(self):
        """Shutdown method transitions to SHUTDOWN state."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        manager.on_broker_connected("ib")
        manager.on_positions_loaded("ib", 50)
        manager.on_market_data_update(50, 45)

        assert manager.is_ready()

        manager.shutdown()

        assert manager.state == ReadinessState.SHUTDOWN
        assert not manager.is_ready()


class TestStateTransitions:
    """Tests for state machine transitions."""

    def test_full_startup_sequence(self):
        """Complete startup sequence transitions correctly."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib"])

        assert manager.state == ReadinessState.STARTING

        manager.on_broker_connected("ib")
        assert manager.state == ReadinessState.BROKERS_CONNECTING

        manager.on_positions_loaded("ib", 50)
        # With positions > 0, immediately transitions to DATA_LOADING
        assert manager.state == ReadinessState.DATA_LOADING

        manager.on_market_data_update(50, 30)  # 60%, below threshold
        assert manager.state == ReadinessState.DATA_LOADING

        manager.on_market_data_update(50, 45)  # 90%, at threshold
        # State machine cascades: DATA_LOADING -> DATA_READY -> SYSTEM_READY
        assert manager.state == ReadinessState.SYSTEM_READY

    def test_multi_broker_startup(self):
        """Multi-broker startup waits for all."""
        bus = MockEventBus()
        manager = ReadinessManager(bus, required_brokers=["ib", "futu"])

        manager.on_broker_connected("ib")
        manager.on_broker_connected("futu")
        manager.on_positions_loaded("ib", 30)

        # Still BROKERS_CONNECTING - futu hasn't loaded
        assert manager.state == ReadinessState.BROKERS_CONNECTING

        manager.on_positions_loaded("futu", 20)

        # Now DATA_LOADING (cascades from BROKERS_READY since positions > 0)
        assert manager.state == ReadinessState.DATA_LOADING
