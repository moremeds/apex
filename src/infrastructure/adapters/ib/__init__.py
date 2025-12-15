"""
Interactive Brokers adapter package.

Provides multiple specialized IB adapters:
- IbAdapter: Legacy monolithic adapter (for backward compatibility)
- IbLiveAdapter: Real-time streaming (quotes, positions, account)
- IbHistoricalAdapter: Historical bar data
- IbExecutionAdapter: Order submission and management

Each adapter uses a different client ID to allow simultaneous connections.
"""

from .adapter import IbAdapter  # Legacy - kept for backward compatibility
from .base import IbBaseAdapter
from .live_adapter import IbLiveAdapter
from .historical_adapter import IbHistoricalAdapter
from .execution_adapter import IbExecutionAdapter
from .client_manager import ClientIdManager, get_client_id_manager

__all__ = [
    # Legacy adapter
    "IbAdapter",
    # New split adapters (Phase 2)
    "IbBaseAdapter",
    "IbLiveAdapter",
    "IbHistoricalAdapter",
    "IbExecutionAdapter",
    # Client ID management
    "ClientIdManager",
    "get_client_id_manager",
]
