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
        """Calculate margin utilization ratio."""
        if self.buying_power == 0:
            return 0.0
        return self.margin_used / self.buying_power
