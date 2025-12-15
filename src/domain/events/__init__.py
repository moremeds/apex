"""Domain events module for priority-based event bus."""

from .event_types import (
    EventType,
    EventPriority,
    EVENT_PRIORITY_MAP,
    FAST_LANE_THRESHOLD,
    DROPPABLE_EVENTS,
    NEVER_DROP,
    PriorityEventEnvelope,
    get_event_type_mapping,
    validate_event_payload,
)
from .priority_event_bus import PriorityEventBus
from .domain_events import (
    # Base
    DomainEvent,
    Timeframe,
    OrderSide,
    OrderStatus,
    OrderType,
    # Market Data Events
    QuoteTick,
    BarData,
    # Trading Events
    TradeFill,
    OrderUpdate,
    # Position/Account Events
    PositionSnapshot,
    AccountSnapshot,
    # System Events
    ConnectionEvent,
    RiskBreachEvent,
    # Registry
    EVENT_REGISTRY,
    deserialize_event,
    deserialize_events,
)

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
