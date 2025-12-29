"""
Interactive Brokers adapter package.

Provides multiple specialized IB adapters:
- IbCompositeAdapter: Main adapter - wraps split adapters via connection pool
- IbLiveAdapter: Real-time streaming (quotes, positions, account)
- IbHistoricalAdapter: Historical bar data
- IbExecutionAdapter: Order submission and management
- IbConnectionPool: Multiple IB connections on same event loop

Each adapter uses a reserved client ID to allow simultaneous connections.
"""

from .base import IbBaseAdapter
from .live_adapter import IbLiveAdapter
from .historical_adapter import IbHistoricalAdapter
from .execution_adapter import IbExecutionAdapter
from .composite_adapter import IbCompositeAdapter
from .connection_pool import IbConnectionPool, ConnectionPoolConfig

__all__ = [
    # Main adapter
    "IbCompositeAdapter",
    # Split adapters (used internally by composite)
    "IbBaseAdapter",
    "IbLiveAdapter",
    "IbHistoricalAdapter",
    "IbExecutionAdapter",
    # Connection pool (multiple IB on same loop)
    "IbConnectionPool",
    "ConnectionPoolConfig",
]
