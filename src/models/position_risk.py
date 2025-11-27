"""Position Risk Model - Calculated risk metrics for a single position."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .position import Position


@dataclass
class PositionRisk:
    """
    Calculated risk metrics for a single position.

    This model contains all pre-calculated metrics for a position,
    computed by the RiskEngine. The presentation layer (dashboard)
    should use these values directly without recalculation.
    """

    # Original position reference
    position: Position

    # Market data
    mark_price: Optional[float] = None
    iv: Optional[float] = None  # Implied volatility for options
    is_using_close: bool = False  # True if mark_price is from yesterday's close (no live data)

    # P&L calculations
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0

    # Greeks (position-level, already multiplied by quantity * multiplier)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    # Derived metrics
    delta_dollars: float = 0.0  # delta * mark * quantity * multiplier
    notional: float = 0.0  # mark * quantity * multiplier

    # Data quality flags
    has_market_data: bool = False
    has_greeks: bool = False
    is_stale: bool = False
    is_using_close: bool = False  # True if mark_price is from yesterday's close (no live data)

    # Timestamp
    calculated_at: datetime = field(default_factory=datetime.now)

    @property
    def symbol(self) -> str:
        """Get position symbol."""
        return self.position.symbol

    @property
    def underlying(self) -> str:
        """Get underlying symbol."""
        return self.position.underlying

    @property
    def quantity(self) -> float:
        """Get position quantity."""
        return self.position.quantity

    @property
    def asset_type(self):
        """Get asset type."""
        return self.position.asset_type

    @property
    def expiry(self) -> Optional[str]:
        """Get expiry date."""
        return self.position.expiry

    @property
    def strike(self) -> Optional[float]:
        """Get strike price."""
        return self.position.strike

    @property
    def right(self) -> Optional[str]:
        """Get option right (C/P)."""
        return self.position.right

    def get_display_name(self) -> str:
        """Get formatted display name for the position."""
        return self.position.get_display_name()

    def days_to_expiry(self) -> Optional[int]:
        """Get days to expiry."""
        return self.position.days_to_expiry()

    def expiry_bucket(self) -> str:
        """Get expiry bucket classification."""
        return self.position.expiry_bucket()
