"""Order and execution history models for IB and Futu.

Terminology:
- Order: A request to buy/sell a security. Has a lifecycle (pending → filled/cancelled).
  One order can have multiple executions (partial fills).
- Execution (Trade/Fill/Deal): An actual fill that occurred. Immutable record of what happened.
  IB calls this "Fill" or "Execution", Futu calls this "Deal".
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from enum import Enum


class OrderSource(Enum):
    """Order data source."""
    IB = "IB"
    FUTU = "FUTU"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    """
    Unified order model for IB and Futu.

    This model captures order data from both brokers with a common schema.
    Used for order history tracking and reconciliation.
    """

    # Unique identifiers
    order_id: str  # Broker-assigned order ID
    source: OrderSource  # IB or FUTU
    account_id: str  # Account identifier

    # Security identification
    symbol: str  # Local symbol (e.g., "AAPL", "AAPL240119C190000")
    underlying: str  # Underlying symbol
    asset_type: str  # "STOCK", "OPTION", "FUTURE"

    # Order details
    side: OrderSide  # BUY or SELL
    order_type: OrderType  # MARKET, LIMIT, etc.
    quantity: float  # Order quantity
    limit_price: Optional[float] = None  # Limit price (if applicable)
    stop_price: Optional[float] = None  # Stop price (if applicable)

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: float = 0.0

    # Timestamps
    created_time: Optional[datetime] = None  # When order was created
    submitted_time: Optional[datetime] = None  # When order was submitted to exchange
    filled_time: Optional[datetime] = None  # When order was fully filled
    updated_time: Optional[datetime] = None  # Last update time

    # Option-specific fields
    expiry: Optional[str] = None  # YYYYMMDD format
    strike: Optional[float] = None
    right: Optional[Literal["C", "P"]] = None

    # Additional metadata
    broker_order_id: Optional[str] = None  # Alternative ID from broker
    exchange: Optional[str] = None  # Exchange where order was routed
    time_in_force: Optional[str] = None  # DAY, GTC, IOC, etc.
    notes: Optional[str] = None  # Any additional notes

    def key(self) -> tuple:
        """
        Composite key for order deduplication.

        Returns:
            Tuple of (source, order_id, account_id)
        """
        return (self.source.value, self.order_id, self.account_id)

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_open(self) -> bool:
        """Check if order is still open (pending/submitted/partially filled)."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def fill_ratio(self) -> float:
        """Get fill ratio (0.0 to 1.0)."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


@dataclass
class Trade:
    """
    Unified trade model for IB and Futu.

    Represents a single fill/execution of an order. One order can have
    multiple trades (partial fills).

    Broker terminology:
    - IB: "Fill" or "Execution"
    - Futu: "Deal"
    """

    # Unique identifiers
    trade_id: str  # Broker-assigned trade/execution/deal ID
    order_id: str  # Parent order ID
    source: OrderSource  # IB or FUTU
    account_id: str  # Account identifier

    # Security identification
    symbol: str  # Local symbol
    underlying: str  # Underlying symbol
    asset_type: str  # "STOCK", "OPTION", "FUTURE"

    # Trade details
    side: OrderSide  # BUY or SELL
    quantity: float  # Executed quantity
    price: float  # Execution price
    commission: float = 0.0  # Commission for this trade

    # Timestamp
    trade_time: datetime = field(default_factory=datetime.now)

    # Option-specific fields
    expiry: Optional[str] = None  # YYYYMMDD format
    strike: Optional[float] = None
    right: Optional[Literal["C", "P"]] = None

    # Additional metadata
    exchange: Optional[str] = None  # Exchange where trade executed
    liquidity: Optional[str] = None  # "ADD" or "REMOVE" (maker/taker)

    def key(self) -> tuple:
        """
        Composite key for trade deduplication.

        Returns:
            Tuple of (source, trade_id, account_id)
        """
        return (self.source.value, self.trade_id, self.account_id)

    @property
    def notional_value(self) -> float:
        """Calculate notional value of trade."""
        # For options, multiply by contract multiplier (typically 100)
        multiplier = 100 if self.asset_type == "OPTION" else 1
        return self.quantity * self.price * multiplier

    @property
    def total_cost(self) -> float:
        """Calculate total cost including commission."""
        return self.notional_value + self.commission
