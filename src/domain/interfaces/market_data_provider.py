"""Market data provider interface for dependency injection."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from ...models import Position
from ...models.market_data import MarketData


class MarketDataProvider(ABC):
    """Interface for market data sources (IBKR, vendors, etc)."""

    @abstractmethod
    async def fetch_market_data(self, positions: List[Position]) -> List[MarketData]:
        """
        Fetch market data (prices + Greeks) for given positions.

        Args:
            positions: List of positions to fetch market data for.

        Returns:
            List of MarketData objects. Missing positions may be omitted.

        Raises:
            ConnectionError: If unable to connect to data source.
        """
        pass

    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data updates."""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data updates."""
        pass

    @abstractmethod
    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data for a symbol."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        pass
