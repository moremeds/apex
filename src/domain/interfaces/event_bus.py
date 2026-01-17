"""Event bus interface for publish-subscribe pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

# Re-export EventType from the enhanced event_types module
# This provides backward compatibility for existing imports
from ..events.event_types import EventType

if TYPE_CHECKING:
    from ..events.domain_events import DomainEvent

# Type variable for typed event callbacks
T = TypeVar("T", bound="DomainEvent")


class EventBus(ABC):
    """
    Event bus for decoupled component communication.

    Supports both typed domain events (preferred) and legacy dict payloads.

    Usage with typed events:
        from src.domain.events.domain_events import QuoteTick

        # Publishing
        tick = QuoteTick(symbol="AAPL", last=150.0)
        event_bus.publish(EventType.MARKET_DATA_TICK, tick)

        # Subscribing
        def on_tick(tick: QuoteTick) -> None:
            print(f"Got tick: {tick.symbol} @ {tick.last}")

        event_bus.subscribe(EventType.MARKET_DATA_TICK, on_tick)
    """

    @abstractmethod
    def publish(self, event_type: EventType, payload: Any, priority: Optional[int] = None) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event being published.
            payload: Event data - preferably a DomainEvent subclass,
                    but dict is still supported for backward compatibility.
            priority: Optional priority for event processing (used by PriorityEventBus).
        """

    @abstractmethod
    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to listen for.
            callback: Function to call when event is published.
                     For type safety, use typed callbacks matching
                     the expected domain event for this event type.

        Example:
            def on_tick(tick: QuoteTick) -> None:
                print(tick.symbol)

            event_bus.subscribe(EventType.MARKET_DATA_TICK, on_tick)
        """

    @abstractmethod
    def unsubscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """Unsubscribe a callback from an event type."""
