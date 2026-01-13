"""Event types and priorities for the dual-lane event bus."""

from __future__ import annotations
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, Type, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from .domain_events import DomainEvent


class EventPriority(IntEnum):
    """Event priorities (lower number = higher priority)."""
    CRITICAL = 0     # SHUTDOWN, CONNECTION_LOST
    RISK = 10        # RISK_SIGNAL, RISK_BREACH
    TRADING = 20     # ORDER_SUBMITTED, FILL_RECEIVED, TRADING_SIGNAL
    MARKET_DATA = 30 # MARKET_DATA_TICK, MARKET_DATA_BATCH
    POSITION_DELTA = 35  # POSITION_DELTA - Fast path for TUI streaming
    POSITION = 40    # POSITION_UPDATED, POSITIONS_BATCH
    ACCOUNT = 50     # ACCOUNT_UPDATED
    CONTROL = 60     # TIMER_TICK, RECONCILIATION_ISSUE, INDICATOR_UPDATE, FULL_RESYNC
    SIGNAL_DATA = 65 # BAR_CLOSE (fast lane, after control but before snapshot)
    SNAPSHOT = 70    # SNAPSHOT_READY
    DIAGNOSTIC = 80  # HEALTH_CHECK, STATS
    UI = 90          # DASHBOARD_UPDATE


class EventType(Enum):
    """System event types with priority categorization."""
    # Critical (Priority 0)
    SHUTDOWN = "shutdown"
    CONNECTION_LOST = "connection_lost"

    # Risk (Priority 10)
    RISK_SIGNAL = "risk_signal"
    RISK_BREACH = "risk_breach"

    # Trading (Priority 20)
    TRADING_SIGNAL = "trading_signal"
    CONFLUENCE_UPDATE = "confluence_update"
    ALIGNMENT_UPDATE = "alignment_update"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"

    # Market Data (Priority 30)
    MARKET_DATA_TICK = "market_data_tick"
    MARKET_DATA_BATCH = "market_data_batch"
    MARKET_DATA_STALE = "market_data_stale"

    # Position Delta (Priority 35) - Fast path for TUI streaming
    POSITION_DELTA = "position_delta"

    # Position (Priority 40)
    POSITION_UPDATED = "position_updated"
    POSITIONS_BATCH = "positions_batch"
    POSITIONS_READY = "positions_ready"

    # Account (Priority 50)
    ACCOUNT_UPDATED = "account_updated"

    # Control (Priority 60)
    TIMER_TICK = "timer_tick"
    RECONCILIATION_ISSUE = "reconciliation_issue"
    MARKET_DATA_READY = "market_data_ready"
    CONNECTION_RESTORED = "connection_restored"
    SYSTEM_READY = "system_ready"
    SYSTEM_DEGRADED = "system_degraded"
    BROKER_CONNECTED = "broker_connected"
    BROKER_DISCONNECTED = "broker_disconnected"
    INDICATOR_UPDATE = "indicator_update"
    FULL_RESYNC = "full_resync"  # Triggers complete portfolio state rebuild on reconnect

    # Signal Data (Priority 65)
    BAR_CLOSE = "bar_close"

    # Snapshot (Priority 70)
    SNAPSHOT_READY = "snapshot_ready"

    # Diagnostic (Priority 80)
    HEALTH_CHECK = "health_check"
    MDQC_VALIDATION_TRIGGER = "mdqc_validation_trigger"

    # UI (Priority 90)
    DASHBOARD_UPDATE = "dashboard_update"


# Map EventType to priority
EVENT_PRIORITY_MAP: dict[EventType, EventPriority] = {
    EventType.SHUTDOWN: EventPriority.CRITICAL,
    EventType.CONNECTION_LOST: EventPriority.CRITICAL,
    EventType.RISK_SIGNAL: EventPriority.RISK,
    EventType.RISK_BREACH: EventPriority.RISK,
    EventType.TRADING_SIGNAL: EventPriority.TRADING,
    EventType.CONFLUENCE_UPDATE: EventPriority.TRADING,
    EventType.ALIGNMENT_UPDATE: EventPriority.TRADING,
    EventType.ORDER_SUBMITTED: EventPriority.TRADING,
    EventType.ORDER_FILLED: EventPriority.TRADING,
    EventType.ORDER_CANCELLED: EventPriority.TRADING,
    EventType.MARKET_DATA_TICK: EventPriority.MARKET_DATA,
    EventType.MARKET_DATA_BATCH: EventPriority.MARKET_DATA,
    EventType.MARKET_DATA_STALE: EventPriority.MARKET_DATA,
    EventType.POSITION_DELTA: EventPriority.POSITION_DELTA,
    EventType.POSITION_UPDATED: EventPriority.POSITION,
    EventType.POSITIONS_BATCH: EventPriority.POSITION,
    EventType.POSITIONS_READY: EventPriority.POSITION,
    EventType.ACCOUNT_UPDATED: EventPriority.ACCOUNT,
    EventType.TIMER_TICK: EventPriority.CONTROL,
    EventType.RECONCILIATION_ISSUE: EventPriority.CONTROL,
    EventType.MARKET_DATA_READY: EventPriority.CONTROL,
    EventType.CONNECTION_RESTORED: EventPriority.CONTROL,
    EventType.SYSTEM_READY: EventPriority.CONTROL,
    EventType.SYSTEM_DEGRADED: EventPriority.CONTROL,
    EventType.BROKER_CONNECTED: EventPriority.CONTROL,
    EventType.BROKER_DISCONNECTED: EventPriority.CONTROL,
    EventType.INDICATOR_UPDATE: EventPriority.CONTROL,
    EventType.FULL_RESYNC: EventPriority.CONTROL,
    EventType.BAR_CLOSE: EventPriority.SIGNAL_DATA,
    EventType.SNAPSHOT_READY: EventPriority.SNAPSHOT,
    EventType.HEALTH_CHECK: EventPriority.DIAGNOSTIC,
    EventType.MDQC_VALIDATION_TRIGGER: EventPriority.DIAGNOSTIC,
    EventType.DASHBOARD_UPDATE: EventPriority.UI,
}


# Events that go to fast lane (priority < SNAPSHOT)
FAST_LANE_THRESHOLD = EventPriority.SNAPSHOT


# ============================================================================
# DROP POLICY (Explicit - from implementation feedback)
# ============================================================================
# When queues are full, events are dropped in this order (lowest priority first):
#
# DROP ORDER (first to drop -> last to drop):
# 1. DASHBOARD_UPDATE (UI) - Always safe to drop
# 2. HEALTH_CHECK, DIAGNOSTIC - Can skip cycles
# 3. SNAPSHOT_READY - Can wait for next snapshot
# 4. MARKET_DATA_TICK - Drop oldest, keep newest per symbol
# 5. POSITION/ACCOUNT - Drop older updates per symbol
# 6. NEVER DROP: RISK_*, TRADING_*, ORDER_*, SYSTEM_*, BROKER_*

DROPPABLE_EVENTS: dict[EventType, int] = {
    EventType.DASHBOARD_UPDATE: 1,    # First to drop
    EventType.HEALTH_CHECK: 2,
    EventType.SNAPSHOT_READY: 3,
    # Market data coalesced, not dropped outright
}

NEVER_DROP: set[EventType] = {
    EventType.SHUTDOWN,
    EventType.CONNECTION_LOST,
    EventType.RISK_SIGNAL,
    EventType.RISK_BREACH,
    EventType.TRADING_SIGNAL,
    EventType.ORDER_SUBMITTED,
    EventType.ORDER_FILLED,
    EventType.ORDER_CANCELLED,
    EventType.POSITIONS_READY,
    EventType.MARKET_DATA_READY,
    EventType.SYSTEM_READY,
    EventType.SYSTEM_DEGRADED,
    EventType.BROKER_CONNECTED,
    EventType.BROKER_DISCONNECTED,
    EventType.POSITION_DELTA,  # Critical for TUI streaming
    EventType.FULL_RESYNC,  # Critical for state recovery
}
# ============================================================================


@dataclass(order=True)
class PriorityEventEnvelope:
    """Event wrapper with priority ordering for PriorityQueue."""
    priority: int
    sequence: int = field(compare=True)  # Tie-breaker for same priority
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    source: str = field(default="", compare=False)


# =============================================================================
# EventType to DomainEvent Mapping
# =============================================================================
# Maps each EventType to its expected payload type (DomainEvent subclass).
# This enables:
# - Runtime validation of event payloads
# - Type-safe subscription callbacks
# - Documentation of expected payload types
#
# Note: Imported lazily to avoid circular imports

def get_event_type_mapping() -> Dict[EventType, Type["DomainEvent"]]:
    """
    Get mapping from EventType to expected DomainEvent class.

    Returns:
        Dict mapping EventType to its expected payload class.
    """
    from .domain_events import (
        QuoteTick, BarData, TradeFill, OrderUpdate,
        PositionSnapshot, AccountSnapshot, ConnectionEvent, RiskBreachEvent,
        MarketDataTickEvent, BarCloseEvent, IndicatorUpdateEvent,
        PositionDeltaEvent, FullResyncEvent,
    )

    return {
        # Market Data Events - C3: Use MarketDataTickEvent (what's actually published)
        EventType.MARKET_DATA_TICK: MarketDataTickEvent,
        EventType.MARKET_DATA_BATCH: BarData,

        # Position Delta Events - Fast path for TUI streaming
        EventType.POSITION_DELTA: PositionDeltaEvent,

        # Signal Engine Events
        EventType.BAR_CLOSE: BarCloseEvent,
        EventType.INDICATOR_UPDATE: IndicatorUpdateEvent,

        # Trading Events
        EventType.ORDER_SUBMITTED: OrderUpdate,
        EventType.ORDER_FILLED: TradeFill,
        EventType.ORDER_CANCELLED: OrderUpdate,

        # Position Events
        EventType.POSITION_UPDATED: PositionSnapshot,

        # Account Events
        EventType.ACCOUNT_UPDATED: AccountSnapshot,

        # Connection Events
        EventType.BROKER_CONNECTED: ConnectionEvent,
        EventType.BROKER_DISCONNECTED: ConnectionEvent,
        EventType.CONNECTION_LOST: ConnectionEvent,
        EventType.CONNECTION_RESTORED: ConnectionEvent,

        # Risk Events
        EventType.RISK_BREACH: RiskBreachEvent,

        # Control Events
        EventType.FULL_RESYNC: FullResyncEvent,
    }


def validate_event_payload(event_type: EventType, payload: Any) -> bool:
    """
    Validate that payload type matches expected type for event.

    Args:
        event_type: The event type being published.
        payload: The payload to validate.

    Returns:
        True if payload is valid type (or no mapping exists), False otherwise.
    """
    mapping = get_event_type_mapping()
    expected_type = mapping.get(event_type)

    if expected_type is None:
        # No mapping defined - allow any payload (backward compatibility)
        return True

    return isinstance(payload, expected_type)
