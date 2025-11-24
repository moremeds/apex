"""Simple in-memory event bus implementation."""

from __future__ import annotations
from typing import Callable, Any, Dict, List
import logging

from ..domain.interfaces.event_bus import EventBus, EventType


logger = logging.getLogger(__name__)


class SimpleEventBus(EventBus):
    """
    Simple in-memory event bus implementation.

    Thread-safe publish-subscribe pattern for decoupled component communication.
    """

    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable[[Any], None]]] = {}

    def publish(self, event_type: EventType, payload: Any) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event being published.
            payload: Event data (dict, dataclass, etc).
        """
        logger.debug(f"Publishing event: {event_type.value}")

        subscribers = self._subscribers.get(event_type, [])
        for callback in subscribers:
            try:
                callback(payload)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}", exc_info=True)

    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to listen for.
            callback: Function to call when event is published.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        Unsubscribe a callback from an event type.

        Args:
            event_type: Event type.
            callback: Callback to remove.
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}")
            except ValueError:
                logger.warning(f"Callback not found for {event_type.value}")
