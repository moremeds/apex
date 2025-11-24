"""Position provider interface for dependency injection."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from ...models.position import Position


class PositionProvider(ABC):
    """Interface for position data sources (IBKR, manual YAML, etc)."""

    @abstractmethod
    async def fetch_positions(self) -> List[Position]:
        """
        Fetch current positions from the source.

        Returns:
            List of Position objects with source field populated.

        Raises:
            ConnectionError: If unable to connect to data source.
            DataError: If data is malformed or invalid.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the provider is connected and ready."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
