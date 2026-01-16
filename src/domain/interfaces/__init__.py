"""Domain interfaces for dependency injection."""

from .broker_adapter import BrokerAdapter
from .market_data_provider import MarketDataProvider
from .event_bus import EventBus, EventType

# New provider protocols (Phase 2)
from .quote_provider import QuoteProvider
from .bar_provider import BarProvider
from .execution_provider import ExecutionProvider, OrderRequest, OrderResult
from .position_provider import PositionProvider
from .account_provider import AccountProvider
from .historical_source import HistoricalSourcePort, DateRange
from .signal_persistence import SignalPersistencePort
from .signal_introspection import SignalIntrospectionPort

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
