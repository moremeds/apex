"""System readiness state machine and event emitter."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Protocol, Any

from ..domain.events.event_types import EventType
from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


class EventPublisher(Protocol):
    """Protocol for event publishing."""
    def publish(self, event_type: EventType, payload: Any) -> None: ...


class ReadinessState(Enum):
    """System readiness states."""
    STARTING = "starting"
    BROKERS_CONNECTING = "brokers_connecting"
    BROKERS_READY = "brokers_ready"
    DATA_LOADING = "data_loading"
    DATA_READY = "data_ready"
    SYSTEM_READY = "system_ready"
    DEGRADED = "degraded"
    SHUTDOWN = "shutdown"


@dataclass
class BrokerStatus:
    """Status of a single broker."""
    name: str
    connected: bool = False
    positions_loaded: bool = False
    position_count: int = 0
    last_error: Optional[str] = None
    connected_at: Optional[datetime] = None


@dataclass
class MarketDataStatus:
    """Status of market data coverage."""
    total_symbols: int = 0
    symbols_with_data: int = 0
    coverage_ratio: float = 0.0
    last_update: Optional[datetime] = None


@dataclass
class DataFreshness:
    """
    Data freshness tracking for degradation detection.

    Even if READY, stale data should trigger DEGRADED/monitor-only mode.
    """
    last_tick_time: Optional[datetime] = None
    last_position_time: Optional[datetime] = None
    last_exec_heartbeat: Optional[datetime] = None

    # Thresholds (configurable)
    tick_stale_threshold_sec: float = 30.0      # No tick for 30s = stale
    position_stale_threshold_sec: float = 60.0  # No position update for 60s = stale
    exec_heartbeat_threshold_sec: float = 15.0  # Exec adapter heartbeat timeout

    def is_tick_stale(self) -> bool:
        if self.last_tick_time is None:
            return True
        age = (datetime.now() - self.last_tick_time).total_seconds()
        return age > self.tick_stale_threshold_sec

    def is_position_stale(self) -> bool:
        if self.last_position_time is None:
            return True
        age = (datetime.now() - self.last_position_time).total_seconds()
        return age > self.position_stale_threshold_sec

    def is_exec_stale(self) -> bool:
        if self.last_exec_heartbeat is None:
            return True
        age = (datetime.now() - self.last_exec_heartbeat).total_seconds()
        return age > self.exec_heartbeat_threshold_sec

    def any_stale(self) -> bool:
        return self.is_tick_stale() or self.is_position_stale() or self.is_exec_stale()

    def stale_reasons(self) -> list[str]:
        reasons = []
        if self.is_tick_stale():
            reasons.append("tick_stale")
        if self.is_position_stale():
            reasons.append("position_stale")
        if self.is_exec_stale():
            reasons.append("exec_heartbeat_stale")
        return reasons


@dataclass
class ReadinessSnapshot:
    """Current readiness state snapshot."""
    state: ReadinessState
    brokers: Dict[str, BrokerStatus]
    market_data: MarketDataStatus
    freshness: DataFreshness
    state_since: datetime
    errors: list[str] = field(default_factory=list)


class ReadinessManager:
    """
    Manages system readiness state machine.

    Emits events when state transitions occur:
    - BROKER_CONNECTED / BROKER_DISCONNECTED
    - POSITIONS_READY
    - MARKET_DATA_READY
    - SYSTEM_READY / SYSTEM_DEGRADED

    State machine:
        STARTING -> BROKERS_CONNECTING -> BROKERS_READY -> DATA_LOADING -> DATA_READY -> SYSTEM_READY
                                      ↓                                                      ↓
                                   DEGRADED  <-----------------------------------------------+
    """

    def __init__(
        self,
        event_bus: EventPublisher,
        required_brokers: list[str],
        market_data_coverage_threshold: float = 0.9,
        startup_timeout_sec: float = 60.0,
    ):
        """
        Initialize ReadinessManager.

        Args:
            event_bus: Event bus for publishing readiness events
            required_brokers: List of broker names that must connect
            market_data_coverage_threshold: Coverage ratio required (0.0-1.0)
            startup_timeout_sec: Maximum time to wait for readiness
        """
        self._event_bus = event_bus
        self._required_brokers = set(required_brokers)
        self._coverage_threshold = market_data_coverage_threshold
        self._startup_timeout = startup_timeout_sec

        # State
        self._state = ReadinessState.STARTING
        self._state_since = datetime.now()
        self._broker_status: Dict[str, BrokerStatus] = {
            name: BrokerStatus(name=name) for name in required_brokers
        }
        self._market_data_status = MarketDataStatus()
        self._freshness = DataFreshness()

        # Tracking flags (for one-time event emission)
        self._positions_ready_emitted = False
        self._market_data_ready_emitted = False
        self._system_ready_emitted = False

    # -------------------------------------------------------------------------
    # Broker Readiness
    # -------------------------------------------------------------------------

    def on_broker_connected(self, broker_name: str) -> None:
        """Handle broker connection."""
        if broker_name not in self._broker_status:
            # Dynamically add unknown brokers
            self._broker_status[broker_name] = BrokerStatus(name=broker_name)

        status = self._broker_status[broker_name]
        status.connected = True
        status.connected_at = datetime.now()
        status.last_error = None

        logger.info(f"Broker connected: {broker_name}")

        self._event_bus.publish(EventType.BROKER_CONNECTED, {
            "broker": broker_name,
            "timestamp": datetime.now().isoformat(),
        })

        self._evaluate_state()

    def on_broker_disconnected(self, broker_name: str, error: Optional[str] = None) -> None:
        """Handle broker disconnection."""
        if broker_name in self._broker_status:
            status = self._broker_status[broker_name]
            status.connected = False
            status.positions_loaded = False
            status.last_error = error

            logger.warning(f"Broker disconnected: {broker_name} ({error})")

            self._event_bus.publish(EventType.BROKER_DISCONNECTED, {
                "broker": broker_name,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            })

            # May transition to DEGRADED
            self._evaluate_state()

    def on_positions_loaded(self, broker_name: str, count: int) -> None:
        """Handle positions loaded from a broker."""
        if broker_name not in self._broker_status:
            self._broker_status[broker_name] = BrokerStatus(name=broker_name)

        status = self._broker_status[broker_name]
        status.positions_loaded = True
        status.position_count = count

        # Update freshness
        self._freshness.last_position_time = datetime.now()

        logger.info(f"Positions loaded from {broker_name}: {count}")

        self._check_positions_ready()
        self._evaluate_state()

    def _check_positions_ready(self) -> None:
        """Check if all connected brokers have reported positions."""
        if self._positions_ready_emitted:
            return

        connected_brokers = [
            status for status in self._broker_status.values()
            if status.connected
        ]

        if not connected_brokers:
            return

        all_loaded = all(status.positions_loaded for status in connected_brokers)

        if all_loaded:
            total_positions = sum(s.position_count for s in connected_brokers)

            logger.info(f"POSITIONS_READY: {total_positions} positions from {len(connected_brokers)} brokers")

            self._event_bus.publish(EventType.POSITIONS_READY, {
                "total_positions": total_positions,
                "brokers": {
                    status.name: status.position_count
                    for status in connected_brokers
                },
                "timestamp": datetime.now().isoformat(),
            })

            self._positions_ready_emitted = True

    # -------------------------------------------------------------------------
    # Market Data Readiness
    # -------------------------------------------------------------------------

    def on_market_data_update(self, total_symbols: int, symbols_with_data: int) -> None:
        """Handle market data coverage update."""
        self._market_data_status.total_symbols = total_symbols
        self._market_data_status.symbols_with_data = symbols_with_data
        self._market_data_status.coverage_ratio = (
            symbols_with_data / total_symbols if total_symbols > 0 else 0.0
        )
        self._market_data_status.last_update = datetime.now()

        # Update freshness
        self._freshness.last_tick_time = datetime.now()

        self._check_market_data_ready()
        self._evaluate_state()

    def on_tick_received(self) -> None:
        """Update tick freshness timestamp."""
        self._freshness.last_tick_time = datetime.now()

    def on_exec_heartbeat(self) -> None:
        """Update execution adapter heartbeat timestamp."""
        self._freshness.last_exec_heartbeat = datetime.now()

    def _check_market_data_ready(self) -> None:
        """Check if market data coverage threshold is met."""
        if self._market_data_ready_emitted:
            return

        if self._market_data_status.coverage_ratio >= self._coverage_threshold:
            logger.info(
                f"MARKET_DATA_READY: {self._market_data_status.symbols_with_data}/"
                f"{self._market_data_status.total_symbols} symbols "
                f"({self._market_data_status.coverage_ratio:.1%})"
            )

            self._event_bus.publish(EventType.MARKET_DATA_READY, {
                "total_symbols": self._market_data_status.total_symbols,
                "symbols_with_data": self._market_data_status.symbols_with_data,
                "coverage_ratio": self._market_data_status.coverage_ratio,
                "timestamp": datetime.now().isoformat(),
            })

            self._market_data_ready_emitted = True

    # -------------------------------------------------------------------------
    # State Machine
    # -------------------------------------------------------------------------

    def _evaluate_state(self) -> None:
        """Evaluate and transition state based on current conditions."""
        # Loop until no more transitions (allows cascading state changes)
        max_iterations = 10  # Safety limit
        for _ in range(max_iterations):
            old_state = self._state

            # Check for degraded conditions
            connected_brokers = sum(1 for s in self._broker_status.values() if s.connected)

            if connected_brokers == 0 and self._state not in (ReadinessState.STARTING, ReadinessState.SHUTDOWN):
                self._transition_to(ReadinessState.DEGRADED, "No brokers connected")
                return

            # State transitions
            if self._state == ReadinessState.STARTING:
                if connected_brokers > 0:
                    self._transition_to(ReadinessState.BROKERS_CONNECTING)

            elif self._state == ReadinessState.BROKERS_CONNECTING:
                if self._positions_ready_emitted:
                    self._transition_to(ReadinessState.BROKERS_READY)

            elif self._state == ReadinessState.BROKERS_READY:
                # Check if we have any positions to get market data for
                total_positions = sum(s.position_count for s in self._broker_status.values() if s.connected)

                if total_positions > 0:
                    # Need to wait for market data
                    self._transition_to(ReadinessState.DATA_LOADING)
                else:
                    # Empty portfolio - no market data needed
                    self._transition_to(ReadinessState.SYSTEM_READY)

            elif self._state == ReadinessState.DATA_LOADING:
                if self._market_data_ready_emitted:
                    self._transition_to(ReadinessState.DATA_READY)

            elif self._state == ReadinessState.DATA_READY:
                # System ready when both positions and market data are ready
                if self._positions_ready_emitted and self._market_data_ready_emitted:
                    self._transition_to(ReadinessState.SYSTEM_READY)

            elif self._state == ReadinessState.SYSTEM_READY:
                # Check for degradation (coverage dropped significantly)
                # Only check coverage if we have positions that need market data
                total_positions = sum(s.position_count for s in self._broker_status.values() if s.connected)
                if total_positions > 0:
                    if self._market_data_status.coverage_ratio < self._coverage_threshold * 0.8:
                        self._transition_to(ReadinessState.DEGRADED, "Market data coverage dropped")

            elif self._state == ReadinessState.DEGRADED:
                # Check for recovery
                if (connected_brokers > 0 and
                    self._positions_ready_emitted and
                    self._market_data_status.coverage_ratio >= self._coverage_threshold):
                    self._transition_to(ReadinessState.SYSTEM_READY, "Recovered")

            # If no transition occurred, we're done
            if self._state == old_state:
                break

    def _transition_to(self, new_state: ReadinessState, reason: str = "") -> None:
        """Transition to new state and emit events."""
        old_state = self._state
        self._state = new_state
        self._state_since = datetime.now()

        logger.info(f"Readiness: {old_state.value} -> {new_state.value}" + (f" ({reason})" if reason else ""))

        if new_state == ReadinessState.SYSTEM_READY and not self._system_ready_emitted:
            self._event_bus.publish(EventType.SYSTEM_READY, {
                "state": new_state.value,
                "brokers": {
                    name: {"connected": s.connected, "positions": s.position_count}
                    for name, s in self._broker_status.items()
                },
                "market_data_coverage": self._market_data_status.coverage_ratio,
                "timestamp": datetime.now().isoformat(),
            })
            self._system_ready_emitted = True

        elif new_state == ReadinessState.DEGRADED:
            self._system_ready_emitted = False  # Allow re-emit when recovered
            self._event_bus.publish(EventType.SYSTEM_DEGRADED, {
                "state": new_state.value,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_snapshot(self) -> ReadinessSnapshot:
        """Get current readiness snapshot."""
        return ReadinessSnapshot(
            state=self._state,
            brokers=dict(self._broker_status),
            market_data=self._market_data_status,
            freshness=self._freshness,
            state_since=self._state_since,
        )

    def is_ready(self) -> bool:
        """Check if system is ready for operations."""
        return self._state == ReadinessState.SYSTEM_READY

    def is_positions_ready(self) -> bool:
        """Check if positions are loaded."""
        return self._positions_ready_emitted

    def is_market_data_ready(self) -> bool:
        """Check if market data coverage is sufficient."""
        return self._market_data_ready_emitted

    def is_degraded(self) -> bool:
        """Check if system is in degraded state."""
        return self._state == ReadinessState.DEGRADED

    @property
    def state(self) -> ReadinessState:
        """Get current state."""
        return self._state

    @property
    def coverage_ratio(self) -> float:
        """Get current market data coverage ratio."""
        return self._market_data_status.coverage_ratio

    def shutdown(self) -> None:
        """Mark system as shutting down."""
        self._transition_to(ReadinessState.SHUTDOWN, "Shutdown requested")
