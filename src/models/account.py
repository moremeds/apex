"""Account information model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class AccountInfo:
    """Account balance and margin data."""

    net_liquidation: float
    total_cash: float
    buying_power: float
    margin_used: float
    margin_available: float

    # Margin metrics
    maintenance_margin: float
    init_margin_req: float
    excess_liquidity: float

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Metadata
    timestamp: Optional[datetime] = None
    account_id: Optional[str] = None

    def margin_utilization(self) -> float:
        """
        Calculate margin utilization ratio.

        Returns:
            Margin utilization as a ratio (0.0 to ~2.0+).
            Returns 0.0 if buying_power is zero/negative or result is invalid.
        """
        import math

        if self.buying_power <= 0:
            return 0.0

        ratio = self.margin_used / self.buying_power

        # Sanity check: ratio should be non-negative and not NaN/infinity
        if math.isnan(ratio) or math.isinf(ratio) or ratio < 0:
            return 0.0

        return ratio
