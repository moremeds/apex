"""Position provider protocol for position data."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from ..events.domain_events import PositionSnapshot


@runtime_checkable
class PositionProvider(Protocol):
    """
    Protocol for position data providers.

    Implementations:
    - IbLiveAdapter
    - FutuAdapter
    - BrokerManager (aggregates multiple providers)

    Usage:
        provider: PositionProvider = IbLiveAdapter(...)
        positions = await provider.fetch_positions()
    """

    async def connect(self) -> None:
        """
        Connect to the position source.

        Raises:
            ConnectionError: If unable to connect.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the position source."""
        ...

    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        ...

    async def fetch_positions(self) -> List[PositionSnapshot]:
        """
        Fetch all current positions.

        Returns:
            List of PositionSnapshot for all open positions.

        Raises:
            ConnectionError: If not connected.
        """
        ...

    async def fetch_position(self, symbol: str) -> Optional[PositionSnapshot]:
        """
        Fetch a specific position by symbol.

        Args:
            symbol: Symbol to look up.

        Returns:
            PositionSnapshot or None if no position.
        """
        ...

    async def fetch_positions_by_underlying(self, underlying: str) -> List[PositionSnapshot]:
        """
        Fetch all positions for an underlying.

        Useful for getting all options on a stock.

        Args:
            underlying: Underlying symbol.

        Returns:
            List of PositionSnapshot for the underlying.
        """
        ...

    async def subscribe_positions(self) -> None:
        """
        Subscribe to position updates.

        Position changes are delivered via set_position_callback.
        """
        ...

    def unsubscribe_positions(self) -> None:
        """Unsubscribe from position updates."""
        ...

    def set_position_callback(
        self, callback: Optional[Callable[[List[PositionSnapshot]], None]]
    ) -> None:
        """
        Set callback for position updates.

        The callback receives the full list of positions on any change.

        Args:
            callback: Function to call with updated positions.
                     Set to None to disable.
        """
        ...

    def get_cached_positions(self) -> List[PositionSnapshot]:
        """
        Get cached positions without fetching.

        Returns:
            List of cached PositionSnapshot (may be stale).
        """
        ...

    def get_position_count(self) -> int:
        """
        Get number of open positions.

        Returns:
            Count of open positions.
        """
        ...

    def get_positions_by_asset_type(self, asset_type: str) -> List[PositionSnapshot]:
        """
        Get positions filtered by asset type.

        Args:
            asset_type: "STOCK", "OPTION", "FUTURE".

        Returns:
            List of matching PositionSnapshot.
        """
        ...
