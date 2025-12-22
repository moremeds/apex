"""
Interactive Brokers adapter package.

Provides multiple specialized IB adapters:
- IbAdapter: Legacy monolithic adapter (for backward compatibility)
- IbLiveAdapter: Real-time streaming (quotes, positions, account)
- IbHistoricalAdapter: Historical bar data
- IbExecutionAdapter: Order submission and management
- IbConnectionPool: Multiple IB connections on same event loop

Each adapter uses a reserved client ID to allow simultaneous connections.
"""

from .adapter import IbAdapter  # Legacy - kept for backward compatibility
from .base import IbBaseAdapter
from .live_adapter import IbLiveAdapter
from .historical_adapter import IbHistoricalAdapter
from .execution_adapter import IbExecutionAdapter
from .connection_pool import IbConnectionPool, ConnectionPoolConfig

__all__ = [
    # Legacy adapter
    "IbAdapter",
    # New split adapters (Phase 2)
    "IbBaseAdapter",
    "IbLiveAdapter",
    "IbHistoricalAdapter",
    "IbExecutionAdapter",
    # Connection pool (multiple IB on same loop)
    "IbConnectionPool",
    "ConnectionPoolConfig",
]
