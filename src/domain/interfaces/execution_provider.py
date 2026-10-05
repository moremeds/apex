"""Order request/result value types used by the frozen backtest and strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
