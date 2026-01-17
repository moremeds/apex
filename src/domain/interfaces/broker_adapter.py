"""Broker adapter interface for order and trade operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ...models.account import AccountInfo
from ...models.order import Order, Trade
from ...models.position import Position


class BrokerAdapter(ABC):
    """
    Interface for broker adapters (IB, Futu, etc).

    Provides a unified interface for:
    - Connection management
    - Position fetching
    - Account info fetching
    - Order history fetching
    - Trade (execution) history fetching

    Terminology:
    - Order: A request to buy/sell a security (has lifecycle: pending -> filled/cancelled)
    - Trade: An execution/fill of an order (immutable record of what happened)
    """

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the broker.

        Raises:
            ConnectionError: If unable to connect to broker.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the adapter is connected and ready."""
        pass

    # -------------------------------------------------------------------------
    # Position Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def fetch_positions(self) -> List[Position]:
        """
        Fetch current positions from the broker.

        Returns:
            List of Position objects.

        Raises:
            ConnectionError: If unable to connect to broker.
        """
        pass

    # -------------------------------------------------------------------------
    # Account Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def fetch_account_info(self) -> AccountInfo:
        """
        Fetch account information from the broker.

        Returns:
            AccountInfo object with balances, margin, etc.

        Raises:
            ConnectionError: If unable to connect to broker.
        """
        pass

    # -------------------------------------------------------------------------
    # Order Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """
        Fetch order history from the broker.

        Args:
            include_open: Include open/pending orders.
            include_completed: Include filled/cancelled/expired orders.
            days_back: Number of days to look back for completed orders.

        Returns:
            List of Order objects.

        Raises:
            ConnectionError: If unable to connect to broker.
        """
        pass

    # -------------------------------------------------------------------------
    # Trade Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def fetch_trades(self, days_back: int = 30) -> List[Trade]:
        """
        Fetch trade (execution) history from the broker.

        Args:
            days_back: Number of days to look back.

        Returns:
            List of Trade objects (executions/fills).

        Raises:
            ConnectionError: If unable to connect to broker.
        """
        pass
