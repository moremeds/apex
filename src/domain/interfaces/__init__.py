"""Domain interfaces for dependency injection."""

from .bar_provider import BarProvider
from .event_bus import EventBus, EventType
from .execution_provider import OrderRequest, OrderResult
from .historical_source import DateRange, HistoricalSourcePort
from .live_feed import LiveFeedPort

# New provider protocols (Phase 2)
from .signal_persistence import SignalPersistencePort

__all__ = [
    "EventBus",
    "EventType",
    "BarProvider",
    "OrderRequest",
    "OrderResult",
    "HistoricalSourcePort",
    "DateRange",
    "LiveFeedPort",
    "SignalPersistencePort",
]
