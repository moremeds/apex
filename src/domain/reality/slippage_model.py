"""
Slippage models for realistic execution simulation.

Slippage represents the difference between expected and actual execution price.
Sources include:
- Bid-ask spread crossing
- Market impact from order size
- Adverse selection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class SlippageResult:
    """Result of slippage calculation."""
    slippage_amount: float  # Absolute slippage in price units
    slippage_bps: float  # Slippage in basis points
    adjusted_price: float  # Price after slippage


class SlippageModel(ABC):
    """
    Abstract base class for slippage models.

    Slippage models adjust execution price based on market conditions.
    """

    @abstractmethod
    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SlippageResult:
        """
        Calculate slippage for a trade.

        Args:
            symbol: Symbol being traded.
            side: "BUY" or "SELL".
            quantity: Order quantity.
            price: Reference price (mid or last).
            bid: Current bid price (optional).
            ask: Current ask price (optional).
            volume: Recent volume (optional).

        Returns:
            SlippageResult with adjusted price.
        """
        ...


class ZeroSlippageModel(SlippageModel):
    """Zero slippage - fills at exact price."""

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SlippageResult:
        return SlippageResult(
            slippage_amount=0.0,
            slippage_bps=0.0,
            adjusted_price=price,
        )


class ConstantSlippageModel(SlippageModel):
    """
    Constant slippage in basis points.

    Simple model that applies fixed slippage regardless of order size.
    """

    def __init__(self, slippage_bps: float = 5.0):
        """
        Initialize constant slippage model.

        Args:
            slippage_bps: Slippage in basis points (5.0 = 0.05%).
        """
        self.slippage_bps = slippage_bps

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SlippageResult:
        slippage_pct = self.slippage_bps / 10000
        slippage_amount = price * slippage_pct

        # Buys pay more, sells receive less
        if side == "BUY":
            adjusted_price = price + slippage_amount
        else:
            adjusted_price = price - slippage_amount

        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_bps=self.slippage_bps,
            adjusted_price=adjusted_price,
        )


class SpreadSlippageModel(SlippageModel):
    """
    Spread-based slippage model.

    Fills at bid/ask price plus optional additional slippage.
    """

    def __init__(
        self,
        additional_bps: float = 0.0,
        spread_fraction: float = 0.5,  # Fraction of spread to cross
    ):
        """
        Initialize spread slippage model.

        Args:
            additional_bps: Additional slippage beyond spread.
            spread_fraction: Fraction of spread to cross (0.5 = half spread).
        """
        self.additional_bps = additional_bps
        self.spread_fraction = spread_fraction

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SlippageResult:
        # If no bid/ask, use price-based estimation
        if bid is None or ask is None:
            # Estimate 0.1% spread
            estimated_spread = price * 0.001
            bid = price - estimated_spread / 2
            ask = price + estimated_spread / 2

        spread = ask - bid
        mid = (bid + ask) / 2

        # Calculate spread crossing cost
        spread_cost = spread * self.spread_fraction

        # Add additional slippage
        additional = mid * (self.additional_bps / 10000)

        if side == "BUY":
            adjusted_price = mid + spread_cost / 2 + additional
            slippage_amount = adjusted_price - mid
        else:
            adjusted_price = mid - spread_cost / 2 - additional
            slippage_amount = mid - adjusted_price

        slippage_bps = (slippage_amount / mid) * 10000 if mid > 0 else 0

        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_bps=slippage_bps,
            adjusted_price=adjusted_price,
        )


class VolumeSlippageModel(SlippageModel):
    """
    Volume-based market impact model.

    Slippage increases with order size relative to market volume.
    Uses square-root market impact model: impact = eta * sqrt(quantity / volume)
    """

    def __init__(
        self,
        base_bps: float = 2.0,  # Base slippage
        impact_factor: float = 0.1,  # Market impact coefficient
        volume_window: float = 0.05,  # Fraction of volume before impact
    ):
        """
        Initialize volume-based slippage model.

        Args:
            base_bps: Base slippage in basis points.
            impact_factor: Market impact coefficient.
            volume_window: Volume fraction before impact starts.
        """
        self.base_bps = base_bps
        self.impact_factor = impact_factor
        self.volume_window = volume_window

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SlippageResult:
        # Base slippage
        slippage_pct = self.base_bps / 10000

        # Add volume-based impact
        if volume and volume > 0:
            volume_fraction = quantity / volume
            if volume_fraction > self.volume_window:
                # Square-root impact model
                excess = volume_fraction - self.volume_window
                impact = self.impact_factor * (excess ** 0.5)
                slippage_pct += impact

        slippage_amount = price * slippage_pct

        if side == "BUY":
            adjusted_price = price + slippage_amount
        else:
            adjusted_price = price - slippage_amount

        slippage_bps = slippage_pct * 10000

        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_bps=slippage_bps,
            adjusted_price=adjusted_price,
        )


class RandomSlippageModel(SlippageModel):
    """
    Random slippage within bounds.

    Adds noise to simulate real-world variance.
    """

    def __init__(
        self,
        min_bps: float = 0.0,
        max_bps: float = 10.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize random slippage model.

        Args:
            min_bps: Minimum slippage in basis points.
            max_bps: Maximum slippage in basis points.
            seed: Random seed for reproducibility.
        """
        self.min_bps = min_bps
        self.max_bps = max_bps
        self._random = random.Random(seed)

    def calculate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SlippageResult:
        slippage_bps = self._random.uniform(self.min_bps, self.max_bps)
        slippage_pct = slippage_bps / 10000
        slippage_amount = price * slippage_pct

        if side == "BUY":
            adjusted_price = price + slippage_amount
        else:
            adjusted_price = price - slippage_amount

        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_bps=slippage_bps,
            adjusted_price=adjusted_price,
        )
