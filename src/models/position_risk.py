"""Position risk model for per-position risk calculations."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .position import Position


@dataclass
class PositionRisk:
    """
    Per-position risk metrics.

    This is the SINGLE SOURCE OF TRUTH for all position-level calculations.
    Created by RiskEngine._create_position_risk() and used by the dashboard
    without recalculation.
    """

    # Reference to the underlying position
    position: "Position"

    # Market data
    mark_price: Optional[float] = None
    iv: Optional[float] = None

    # Values
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0

    # Greeks (position-level contributions)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    # Dollar exposure
    delta_dollars: float = 0.0
    notional: float = 0.0

    # Data quality flags
    has_market_data: bool = False
    has_greeks: bool = False
    is_stale: bool = False

    # Metadata
    calculated_at: datetime = field(default_factory=datetime.now)

    # Convenience properties to access position attributes
    @property
    def symbol(self) -> str:
        """Get symbol from underlying position."""
        return self.position.symbol

    @property
    def underlying(self) -> str:
        """Get underlying from underlying position."""
        return self.position.underlying

    @property
    def quantity(self) -> float:
        """Get quantity from underlying position."""
        return self.position.quantity

    @property
    def expiry(self) -> Optional[str]:
        """Get expiry from underlying position."""
        return self.position.expiry

    @property
    def strike(self) -> Optional[float]:
        """Get strike from underlying position."""
        return self.position.strike

    @property
    def right(self) -> Optional[str]:
        """Get right (C/P) from underlying position."""
        return self.position.right

    def get_display_name(self) -> str:
        """Get display name from underlying position."""
        return self.position.get_display_name()
