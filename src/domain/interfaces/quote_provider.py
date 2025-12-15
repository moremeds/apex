"""Quote provider protocol for real-time market data streaming."""

from __future__ import annotations
from typing import Protocol, List, Optional, Callable, Dict, runtime_checkable

from ..events.domain_events import QuoteTick


@runtime_checkable
class QuoteProvider(Protocol):
    """
    Protocol for real-time quote streaming providers.

    Implementations:
    - IbLiveAdapter
    - FutuQuoteAdapter (future)

    Usage:
        provider: QuoteProvider = IbLiveAdapter(...)
        await provider.subscribe_quotes(["AAPL", "MSFT"])
        provider.set_quote_callback(handle_quote)
    """

    async def connect(self) -> None:
        """
        Connect to the quote source.

        Raises:
            ConnectionError: If unable to connect.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the quote source."""
        ...

    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        ...

    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """
        Subscribe to real-time quotes for symbols.

        Args:
            symbols: List of symbols to subscribe to.

        Raises:
            ConnectionError: If not connected.
        """
        ...

    async def unsubscribe_quotes(self, symbols: List[str]) -> None:
        """
        Unsubscribe from quotes for symbols.

        Args:
            symbols: List of symbols to unsubscribe from.
        """
        ...

    def set_quote_callback(
        self,
        callback: Optional[Callable[[QuoteTick], None]]
    ) -> None:
        """
        Set callback for incoming quotes.

        The callback receives QuoteTick events as they arrive.

        Args:
            callback: Function to call with each quote tick.
                     Set to None to disable callbacks.
        """
        ...

    def get_latest_quote(self, symbol: str) -> Optional[QuoteTick]:
        """
        Get the most recent cached quote for a symbol.

        Args:
            symbol: Symbol to look up.

        Returns:
            Latest QuoteTick or None if not available.
        """
        ...

    def get_all_quotes(self) -> Dict[str, QuoteTick]:
        """
        Get all cached quotes.

        Returns:
            Dict mapping symbol to latest QuoteTick.
        """
        ...

    def get_subscribed_symbols(self) -> List[str]:
        """
        Get list of currently subscribed symbols.

        Returns:
            List of subscribed symbols.
        """
        ...

    async def fetch_snapshot(self, symbols: List[str]) -> Dict[str, QuoteTick]:
        """
        Fetch one-time quote snapshot for symbols.

        Unlike subscribe, this fetches current data without ongoing updates.

        Args:
            symbols: List of symbols to fetch.

        Returns:
            Dict mapping symbol to QuoteTick snapshot.
        """
        ...
