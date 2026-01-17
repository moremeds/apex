"""Domain events module for priority-based event bus."""

from .domain_events import (  # Base; Market Data Events; Trading Events; Position/Account Events; System Events; Registry
    EVENT_REGISTRY,
    AccountSnapshot,
    BarData,
    ConnectionEvent,
    DomainEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdate,
    PositionSnapshot,
    QuoteTick,
    RiskBreachEvent,
    Timeframe,
    TradeFill,
    deserialize_event,
    deserialize_events,
)
from .event_types import (
    DROPPABLE_EVENTS,
    EVENT_PRIORITY_MAP,
    FAST_LANE_THRESHOLD,
    NEVER_DROP,
    EventPriority,
    EventType,
    PriorityEventEnvelope,
    get_event_type_mapping,
    validate_event_payload,
)
from .priority_event_bus import PriorityEventBus

__all__ = [
    # Event types and priorities
    "EventType",
    "EventPriority",
    "EVENT_PRIORITY_MAP",
    "FAST_LANE_THRESHOLD",
    "DROPPABLE_EVENTS",
    "NEVER_DROP",
    "PriorityEventEnvelope",
    "PriorityEventBus",
    # Type mapping and validation
    "get_event_type_mapping",
    "validate_event_payload",
    # Domain events
    "DomainEvent",
    "Timeframe",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "QuoteTick",
    "BarData",
    "TradeFill",
    "OrderUpdate",
    "PositionSnapshot",
    "AccountSnapshot",
    "ConnectionEvent",
    "RiskBreachEvent",
    "EVENT_REGISTRY",
    "deserialize_event",
    "deserialize_events",
]
