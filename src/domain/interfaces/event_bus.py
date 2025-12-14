"""Event bus interface for publish-subscribe pattern."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any

# Re-export EventType from the enhanced event_types module
# This provides backward compatibility for existing imports
from ..events.event_types import EventType, EventPriority


class EventBus(ABC):
    """Event bus for decoupled component communication."""

    @abstractmethod
    def publish(self, event_type: EventType, payload: Any) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event being published.
            payload: Event data (dict, dataclass, etc).
        """
        pass

    @abstractmethod
    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to listen for.
            callback: Function to call when event is published.
        """
        pass

    @abstractmethod
    def unsubscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """Unsubscribe a callback from an event type."""
        pass
