"""Event bus interface for publish-subscribe pattern."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any
from enum import Enum


class EventType(Enum):
    """System event types."""
    # Legacy events (keep for compatibility)
    POSITION_CHANGED = "position_changed"
    LIMIT_BREACHED = "limit_breached"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    MARKET_DATA_STALE = "market_data_stale"
    RECONCILIATION_ISSUE = "reconciliation_issue"
    HEALTH_CHECK_FAILED = "health_check_failed"

    # Granular position events
    POSITION_UPDATED = "position_updated"      # Single position update (from push)
    POSITIONS_BATCH = "positions_batch"        # Batch from adapter poll

    # Market data events
    MARKET_DATA_TICK = "market_data_tick"      # Single symbol tick (streaming)
    MARKET_DATA_BATCH = "market_data_batch"    # Batch from poll

    # Account events
    ACCOUNT_UPDATED = "account_updated"        # Account snapshot updated

    # Computed events
    SNAPSHOT_READY = "snapshot_ready"          # Risk snapshot computed
    RISK_SIGNAL = "risk_signal"                # Risk signal detected

    # System events
    TIMER_TICK = "timer_tick"                  # Periodic tick for reconciliation
    SHUTDOWN = "shutdown"                      # Graceful shutdown signal


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
