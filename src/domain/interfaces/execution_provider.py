"""Execution provider protocol for order management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, runtime_checkable

from ..events.domain_events import OrderUpdate, TradeFill


@dataclass
class OrderRequest:
    """
    Order request for submission.

    Represents the parameters needed to submit an order.
    """

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: str = "LIMIT"  # "MARKET", "LIMIT", "STOP", "STOP_LIMIT"

    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Time in force
    tif: str = "DAY"  # "DAY", "GTC", "IOC", "FOK"

    # Asset details (for options)
    underlying: Optional[str] = None
    asset_type: str = "STOCK"  # "STOCK", "OPTION", "FUTURE"
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None  # "C" or "P"
    multiplier: int = 1

    # Optional identifiers
    client_order_id: Optional[str] = None
    account_id: Optional[str] = None

    # Bracket/OCO orders
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None


@dataclass
class OrderResult:
    """
    Result of an order operation.

    Returned by submit_order, cancel_order, modify_order.
    """

    success: bool
    order_id: Optional[str] = None
    message: str = ""
    error_code: Optional[str] = None


@runtime_checkable
class ExecutionProvider(Protocol):
    """
    Protocol for order execution providers.

    Implementations:
    - IbExecutionAdapter
    - FutuExecutionAdapter (future)

    Usage:
        provider: ExecutionProvider = IbExecutionAdapter(...)
        result = await provider.submit_order(OrderRequest(...))
    """

    async def connect(self) -> None:
        """
        Connect to the execution venue.

        Raises:
            ConnectionError: If unable to connect.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the execution venue."""
        ...

    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        ...

    # -------------------------------------------------------------------------
    # Order Submission
    # -------------------------------------------------------------------------

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """
        Submit a new order.

        Args:
            request: Order request with all parameters.

        Returns:
            OrderResult with success status and order_id if successful.
        """
        ...

    async def submit_bracket_order(
        self,
        entry: OrderRequest,
        take_profit: OrderRequest,
        stop_loss: OrderRequest,
    ) -> List[OrderResult]:
        """
        Submit a bracket order (entry + take profit + stop loss).

        Args:
            entry: Entry order request.
            take_profit: Take profit order request.
            stop_loss: Stop loss order request.

        Returns:
            List of OrderResults for each leg.
        """
        ...

    # -------------------------------------------------------------------------
    # Order Management
    # -------------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel an open order.

        Args:
            order_id: ID of order to cancel.

        Returns:
            OrderResult indicating success/failure.
        """
        ...

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Cancel all open orders.

        Args:
            symbol: If provided, only cancel orders for this symbol.

        Returns:
            List of OrderResults for each cancelled order.
        """
        ...

    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_limit_price: Optional[float] = None,
        new_stop_price: Optional[float] = None,
    ) -> OrderResult:
        """
        Modify an existing order.

        Args:
            order_id: ID of order to modify.
            new_quantity: New quantity (if changing).
            new_limit_price: New limit price (if changing).
            new_stop_price: New stop price (if changing).

        Returns:
            OrderResult indicating success/failure.
        """
        ...

    # -------------------------------------------------------------------------
    # Order Queries
    # -------------------------------------------------------------------------

    async def get_order(self, order_id: str) -> Optional[OrderUpdate]:
        """
        Get current state of an order.

        Args:
            order_id: ID of order to query.

        Returns:
            OrderUpdate with current state or None if not found.
        """
        ...

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderUpdate]:
        """
        Get all open orders.

        Args:
            symbol: If provided, filter to this symbol.

        Returns:
            List of OrderUpdate for open orders.
        """
        ...

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        days_back: int = 30,
    ) -> List[OrderUpdate]:
        """
        Get historical orders.

        Args:
            symbol: If provided, filter to this symbol.
            days_back: Number of days to look back.

        Returns:
            List of OrderUpdate for historical orders.
        """
        ...

    # -------------------------------------------------------------------------
    # Execution Queries
    # -------------------------------------------------------------------------

    async def get_fills(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        days_back: int = 7,
    ) -> List[TradeFill]:
        """
        Get trade fills/executions.

        Args:
            order_id: If provided, filter to this order.
            symbol: If provided, filter to this symbol.
            days_back: Number of days to look back.

        Returns:
            List of TradeFill for executions.
        """
        ...

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_order_callback(self, callback: Optional[Callable[[OrderUpdate], None]]) -> None:
        """
        Set callback for order status updates.

        Args:
            callback: Function to call on order updates.
                     Set to None to disable.
        """
        ...

    def set_fill_callback(self, callback: Optional[Callable[[TradeFill], None]]) -> None:
        """
        Set callback for trade fills.

        Args:
            callback: Function to call on fills.
                     Set to None to disable.
        """
        ...

    # -------------------------------------------------------------------------
    # Risk Controls
    # -------------------------------------------------------------------------

    def set_max_order_size(self, max_quantity: float) -> None:
        """
        Set maximum order size limit.

        Orders exceeding this will be rejected.

        Args:
            max_quantity: Maximum quantity per order.
        """
        ...

    def set_max_position_size(self, symbol: str, max_quantity: float) -> None:
        """
        Set maximum position size for a symbol.

        Orders that would exceed this will be rejected.

        Args:
            symbol: Symbol to set limit for.
            max_quantity: Maximum position size.
        """
        ...

    def enable_trading(self) -> None:
        """Enable order submission (default state)."""
        ...

    def disable_trading(self, reason: str = "") -> None:
        """
        Disable order submission (kill switch).

        Args:
            reason: Reason for disabling trading.
        """
        ...

    def is_trading_enabled(self) -> bool:
        """Check if trading is currently enabled."""
        ...
