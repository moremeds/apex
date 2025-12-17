"""
Fill models for order execution simulation.

Fill models determine how and when orders get filled:
- Immediate: Fill at current price (for testing)
- Next bar: Fill at next bar's open
- Probabilistic: Partial fills based on queue position
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum
import random


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class FillResult:
    """Result of fill simulation."""
    filled: bool  # Whether any quantity was filled
    filled_quantity: float  # Quantity filled
    remaining_quantity: float  # Quantity unfilled
    fill_price: float  # Execution price
    partial: bool = False  # Whether this is a partial fill
    reject_reason: Optional[str] = None  # Reason if rejected


class FillModel(ABC):
    """
    Abstract base class for fill models.

    Fill models determine execution behavior for orders.
    """

    @abstractmethod
    def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_price: float = 0.0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> FillResult:
        """
        Simulate order fill.

        Args:
            symbol: Symbol being traded.
            side: "BUY" or "SELL".
            quantity: Order quantity.
            order_type: Type of order.
            limit_price: Limit price (for LIMIT orders).
            stop_price: Stop price (for STOP orders).
            current_price: Current market price.
            bid: Current bid.
            ask: Current ask.
            volume: Recent volume.

        Returns:
            FillResult with execution details.
        """
        ...


class ImmediateFillModel(FillModel):
    """
    Immediate fill at current price.

    All market orders fill immediately.
    Limit orders fill if price is favorable.
    """

    def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_price: float = 0.0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> FillResult:
        # Determine fill price
        if side == "BUY":
            fill_price = ask if ask else current_price
        else:
            fill_price = bid if bid else current_price

        if fill_price <= 0:
            return FillResult(
                filled=False,
                filled_quantity=0,
                remaining_quantity=quantity,
                fill_price=0,
                reject_reason="No valid price",
            )

        # Market orders always fill
        if order_type == OrderType.MARKET:
            return FillResult(
                filled=True,
                filled_quantity=quantity,
                remaining_quantity=0,
                fill_price=fill_price,
            )

        # Limit orders check price
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                return FillResult(
                    filled=False,
                    filled_quantity=0,
                    remaining_quantity=quantity,
                    fill_price=0,
                    reject_reason="No limit price",
                )

            # Check if limit is favorable
            if side == "BUY" and fill_price <= limit_price:
                return FillResult(
                    filled=True,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    fill_price=fill_price,
                )
            elif side == "SELL" and fill_price >= limit_price:
                return FillResult(
                    filled=True,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    fill_price=fill_price,
                )
            else:
                return FillResult(
                    filled=False,
                    filled_quantity=0,
                    remaining_quantity=quantity,
                    fill_price=0,
                    reject_reason="Limit not reached",
                )

        # Stop orders
        if order_type == OrderType.STOP:
            if stop_price is None:
                return FillResult(
                    filled=False,
                    filled_quantity=0,
                    remaining_quantity=quantity,
                    fill_price=0,
                    reject_reason="No stop price",
                )

            # Check if stop is triggered
            if side == "BUY" and current_price >= stop_price:
                return FillResult(
                    filled=True,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    fill_price=fill_price,
                )
            elif side == "SELL" and current_price <= stop_price:
                return FillResult(
                    filled=True,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    fill_price=fill_price,
                )
            else:
                return FillResult(
                    filled=False,
                    filled_quantity=0,
                    remaining_quantity=quantity,
                    fill_price=0,
                    reject_reason="Stop not triggered",
                )

        return FillResult(
            filled=False,
            filled_quantity=0,
            remaining_quantity=quantity,
            fill_price=0,
            reject_reason=f"Unknown order type: {order_type}",
        )


class NextBarFillModel(FillModel):
    """
    Fill at next bar's open.

    Orders are queued and filled at the open of the next bar.
    More realistic for daily backtests.
    """

    def __init__(self):
        self._pending_orders: List[dict] = []
        self._next_bar_open: Optional[float] = None

    def set_next_bar_open(self, price: float) -> None:
        """Set the next bar's open price."""
        self._next_bar_open = price

    def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_price: float = 0.0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> FillResult:
        # If we have a next bar open, fill market orders at that price
        if order_type == OrderType.MARKET:
            fill_price = self._next_bar_open if self._next_bar_open else current_price
            if fill_price > 0:
                return FillResult(
                    filled=True,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    fill_price=fill_price,
                )

        # For limit orders, check if next bar open would fill
        if order_type == OrderType.LIMIT and limit_price:
            fill_price = self._next_bar_open if self._next_bar_open else current_price
            if side == "BUY" and fill_price <= limit_price:
                return FillResult(
                    filled=True,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    fill_price=fill_price,
                )
            elif side == "SELL" and fill_price >= limit_price:
                return FillResult(
                    filled=True,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    fill_price=fill_price,
                )

        # Order not filled yet
        return FillResult(
            filled=False,
            filled_quantity=0,
            remaining_quantity=quantity,
            fill_price=0,
            reject_reason="Waiting for next bar",
        )


class ProbabilisticFillModel(FillModel):
    """
    Probabilistic fill model with partial fills.

    Models realistic fill behavior:
    - Market orders: High probability, possible partial fill
    - Limit orders: Lower probability based on distance from price
    - Considers volume and queue position
    """

    def __init__(
        self,
        market_fill_prob: float = 0.99,
        partial_fill_prob: float = 0.1,
        volume_ratio_threshold: float = 0.1,  # Max volume fraction for full fill
        seed: Optional[int] = None,
    ):
        """
        Initialize probabilistic fill model.

        Args:
            market_fill_prob: Probability of market order fill.
            partial_fill_prob: Probability of partial fill.
            volume_ratio_threshold: Volume ratio above which partial fills likely.
            seed: Random seed for reproducibility.
        """
        self.market_fill_prob = market_fill_prob
        self.partial_fill_prob = partial_fill_prob
        self.volume_ratio_threshold = volume_ratio_threshold
        self._random = random.Random(seed)

    def simulate_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_price: float = 0.0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> FillResult:
        fill_price = (ask if side == "BUY" else bid) or current_price

        if fill_price <= 0:
            return FillResult(
                filled=False,
                filled_quantity=0,
                remaining_quantity=quantity,
                fill_price=0,
                reject_reason="No valid price",
            )

        # Market orders
        if order_type == OrderType.MARKET:
            if self._random.random() < self.market_fill_prob:
                # Check for partial fill based on volume
                filled_qty = self._calculate_filled_quantity(quantity, volume)
                return FillResult(
                    filled=True,
                    filled_quantity=filled_qty,
                    remaining_quantity=quantity - filled_qty,
                    fill_price=fill_price,
                    partial=filled_qty < quantity,
                )
            else:
                return FillResult(
                    filled=False,
                    filled_quantity=0,
                    remaining_quantity=quantity,
                    fill_price=0,
                    reject_reason="Market order not filled",
                )

        # Limit orders
        if order_type == OrderType.LIMIT and limit_price:
            # Calculate fill probability based on limit vs current
            fill_prob = self._calculate_limit_fill_probability(
                side, limit_price, current_price, bid, ask
            )

            if self._random.random() < fill_prob:
                filled_qty = self._calculate_filled_quantity(quantity, volume)
                actual_fill_price = min(limit_price, fill_price) if side == "BUY" else max(limit_price, fill_price)
                return FillResult(
                    filled=True,
                    filled_quantity=filled_qty,
                    remaining_quantity=quantity - filled_qty,
                    fill_price=actual_fill_price,
                    partial=filled_qty < quantity,
                )
            else:
                return FillResult(
                    filled=False,
                    filled_quantity=0,
                    remaining_quantity=quantity,
                    fill_price=0,
                    reject_reason="Limit order not filled",
                )

        return FillResult(
            filled=False,
            filled_quantity=0,
            remaining_quantity=quantity,
            fill_price=0,
            reject_reason=f"Unsupported order type: {order_type}",
        )

    def _calculate_filled_quantity(
        self,
        quantity: float,
        volume: Optional[float],
    ) -> float:
        """Calculate filled quantity considering volume."""
        # Check if partial fill based on volume
        if volume and volume > 0:
            volume_ratio = quantity / volume
            if volume_ratio > self.volume_ratio_threshold:
                # Likely partial fill
                if self._random.random() < self.partial_fill_prob:
                    fill_fraction = self._random.uniform(0.5, 0.95)
                    return quantity * fill_fraction

        return quantity

    def _calculate_limit_fill_probability(
        self,
        side: str,
        limit_price: float,
        current_price: float,
        bid: Optional[float],
        ask: Optional[float],
    ) -> float:
        """Calculate probability of limit order fill."""
        if side == "BUY":
            market_price = ask if ask else current_price
            if limit_price >= market_price:
                return 0.95  # Very likely to fill
            else:
                # Decreasing probability as limit is further below market
                distance = (market_price - limit_price) / market_price
                return max(0.1, 0.95 - distance * 10)
        else:
            market_price = bid if bid else current_price
            if limit_price <= market_price:
                return 0.95  # Very likely to fill
            else:
                distance = (limit_price - market_price) / market_price
                return max(0.1, 0.95 - distance * 10)
