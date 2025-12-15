"""Bar provider protocol for historical market data."""

from __future__ import annotations
from typing import Protocol, List, Optional, Callable, runtime_checkable
from datetime import datetime

from ..events.domain_events import BarData, Timeframe


@runtime_checkable
class BarProvider(Protocol):
    """
    Protocol for historical bar/candle data providers.

    Implementations:
    - IbHistoricalAdapter
    - ParquetDataFeed (future)

    Usage:
        provider: BarProvider = IbHistoricalAdapter(...)
        bars = await provider.fetch_bars("AAPL", "1d", start, end)
    """

    async def connect(self) -> None:
        """
        Connect to the historical data source.

        Raises:
            ConnectionError: If unable to connect.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the historical data source."""
        ...

    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        ...

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[BarData]:
        """
        Fetch historical bars for a symbol.

        Args:
            symbol: Symbol to fetch bars for.
            timeframe: Bar timeframe (e.g., "1m", "5m", "1h", "1d").
            start: Start datetime (inclusive). If None, fetches from earliest.
            end: End datetime (inclusive). If None, fetches to latest.
            limit: Maximum number of bars to return. If None, no limit.

        Returns:
            List of BarData sorted by timestamp ascending.

        Raises:
            ConnectionError: If not connected.
            ValueError: If invalid timeframe or date range.
        """
        ...

    async def fetch_latest_bar(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[BarData]:
        """
        Fetch the most recent completed bar.

        Args:
            symbol: Symbol to fetch.
            timeframe: Bar timeframe.

        Returns:
            Latest completed BarData or None.
        """
        ...

    async def subscribe_bars(
        self,
        symbol: str,
        timeframe: str,
    ) -> None:
        """
        Subscribe to real-time bar updates.

        New bars are delivered via the callback set with set_bar_callback.

        Args:
            symbol: Symbol to subscribe.
            timeframe: Bar timeframe to subscribe.
        """
        ...

    async def unsubscribe_bars(
        self,
        symbol: str,
        timeframe: str,
    ) -> None:
        """
        Unsubscribe from bar updates.

        Args:
            symbol: Symbol to unsubscribe.
            timeframe: Bar timeframe to unsubscribe.
        """
        ...

    def set_bar_callback(
        self,
        callback: Optional[Callable[[BarData], None]]
    ) -> None:
        """
        Set callback for incoming bar updates.

        Args:
            callback: Function to call with each new bar.
                     Set to None to disable callbacks.
        """
        ...

    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes.

        Returns:
            List of timeframe strings (e.g., ["1m", "5m", "1h", "1d"]).
        """
        ...

    async def fetch_bars_batch(
        self,
        requests: List[dict],
    ) -> dict:
        """
        Fetch bars for multiple symbols efficiently.

        Args:
            requests: List of request dicts with keys:
                - symbol: str
                - timeframe: str
                - start: Optional[datetime]
                - end: Optional[datetime]
                - limit: Optional[int]

        Returns:
            Dict mapping symbol to List[BarData].
        """
        ...
