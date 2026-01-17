"""Domain interfaces for dependency injection."""

from .account_provider import AccountProvider
from .bar_provider import BarProvider
from .broker_adapter import BrokerAdapter
from .event_bus import EventBus, EventType
from .execution_provider import ExecutionProvider, OrderRequest, OrderResult
from .historical_source import DateRange, HistoricalSourcePort
from .market_data_provider import MarketDataProvider
from .position_provider import PositionProvider

# New provider protocols (Phase 2)
from .quote_provider import QuoteProvider
from .signal_introspection import SignalIntrospectionPort
from .signal_persistence import SignalPersistencePort

__all__ = [
    # Legacy interfaces (kept for compatibility)
    "BrokerAdapter",
    "MarketDataProvider",
    "EventBus",
    "EventType",
    # New provider protocols
    "QuoteProvider",
    "BarProvider",
    "ExecutionProvider",
    "OrderRequest",
    "OrderResult",
    "PositionProvider",
    "AccountProvider",
    # Historical data management
    "HistoricalSourcePort",
    "DateRange",
    # Signal persistence
    "SignalPersistencePort",
    # Signal introspection (real-time read-only)
    "SignalIntrospectionPort",
]
