"""
Client ID manager for IB adapters.

Each IB adapter type uses a different client ID to allow simultaneous connections:
- Live adapter: base_id + 0
- Historical adapter: base_id + 1
- Execution adapter: base_id + 2

This prevents "Duplicate client ID" errors from TWS/Gateway.
"""

from __future__ import annotations
from typing import Dict, Optional
from threading import Lock

from ....utils.logging_setup import get_logger


logger = get_logger(__name__)


class ClientIdManager:
    """
    Manages client IDs for IB adapters.

    Ensures each adapter type gets a unique client ID and tracks
    which IDs are currently in use.
    """

    # Client ID offsets for each adapter type
    OFFSETS = {
        "live": 0,
        "historical": 1,
        "execution": 2,
    }

    def __init__(self, base_client_id: int = 1):
        """
        Initialize client ID manager.

        Args:
            base_client_id: Base client ID (from config).
                           Adapter IDs will be base + offset.
        """
        self._base_id = base_client_id
        self._in_use: Dict[str, int] = {}
        self._lock = Lock()

    def get_client_id(self, adapter_type: str) -> int:
        """
        Get client ID for an adapter type.

        Args:
            adapter_type: One of "live", "historical", "execution".

        Returns:
            Unique client ID for this adapter type.

        Raises:
            ValueError: If adapter_type is unknown.
        """
        if adapter_type not in self.OFFSETS:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. "
                f"Must be one of: {list(self.OFFSETS.keys())}"
            )

        with self._lock:
            offset = self.OFFSETS[adapter_type]
            client_id = self._base_id + offset
            self._in_use[adapter_type] = client_id
            logger.debug(f"Assigned client ID {client_id} to {adapter_type} adapter")
            return client_id

    def release_client_id(self, adapter_type: str) -> None:
        """
        Release a client ID when adapter disconnects.

        Args:
            adapter_type: The adapter type releasing its ID.
        """
        with self._lock:
            if adapter_type in self._in_use:
                client_id = self._in_use.pop(adapter_type)
                logger.debug(f"Released client ID {client_id} from {adapter_type} adapter")

    def get_in_use(self) -> Dict[str, int]:
        """Get dict of currently in-use client IDs."""
        with self._lock:
            return self._in_use.copy()

    def is_in_use(self, adapter_type: str) -> bool:
        """Check if an adapter type's client ID is in use."""
        with self._lock:
            return adapter_type in self._in_use


# Global singleton instance
_manager: Optional[ClientIdManager] = None


def get_client_id_manager(base_client_id: int = 1) -> ClientIdManager:
    """
    Get the global ClientIdManager instance.

    Args:
        base_client_id: Base client ID (only used on first call).

    Returns:
        The singleton ClientIdManager instance.
    """
    global _manager
    if _manager is None:
        _manager = ClientIdManager(base_client_id)
    return _manager


def reset_client_id_manager() -> None:
    """Reset the global ClientIdManager (for testing)."""
    global _manager
    _manager = None
