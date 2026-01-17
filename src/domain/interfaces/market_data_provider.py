"""Market data provider interface for dependency injection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from ...models import Position
from ...models.market_data import MarketData


class MarketDataProvider(ABC):
    """
    Interface for market data sources (IBKR, Yahoo Finance, CCXT, etc).

    All market data providers must implement this interface.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the market data source.

        Raises:
            ConnectionError: If unable to connect.
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the market data source."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the provider is connected."""

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

    @abstractmethod
    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch quotes for a list of symbols (without position context).

        Useful for market indicators (VIX, SPY) that aren't positions.

        Args:
            symbols: List of symbols to fetch quotes for.

        Returns:
            Dict mapping symbol to MarketData.
        """

    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data updates."""

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data updates."""

    @abstractmethod
    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest cached market data for a symbol."""

    def set_streaming_callback(self, callback: Optional[Callable[[str, MarketData], None]]) -> None:
        """
        Set callback for streaming market data updates.

        Args:
            callback: Function to call with (symbol, market_data) on updates.
                     Set to None to disable streaming callbacks.
        """
        pass  # Default no-op, override if streaming is supported

    def enable_streaming(self) -> None:
        """Enable streaming mode (if supported by provider)."""
        pass  # Default no-op

    def disable_streaming(self) -> None:
        """Disable streaming mode (if supported by provider)."""
        pass  # Default no-op

    def supports_streaming(self) -> bool:
        """Check if this provider supports real-time streaming."""
        return False  # Default to False, override if supported

    def supports_greeks(self) -> bool:
        """Check if this provider supports Greeks (options data)."""
        return False  # Default to False, override if supported
