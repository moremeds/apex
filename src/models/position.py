"""Position model with source tracking and reconciliation support."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Literal
from enum import Enum


class AssetType(Enum):
    """Asset type enumeration."""
    STOCK = "STOCK"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    CASH = "CASH"


class PositionSource(Enum):
    """Position data source for reconciliation."""
    IB = "IB"  # Interactive Brokers
    MANUAL = "MANUAL"  # Manual YAML file
    CACHED = "CACHED"  # Previous snapshot


@dataclass
class Position:
    """Unified position model for stocks, options, and futures with reconciliation support."""

    # Core identification
    symbol: str
    underlying: str
    asset_type: AssetType
    quantity: float
    avg_price: float
    multiplier: int = 1

    # Option/Future specific
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[Literal["C", "P"]] = None

    # Reconciliation & metadata
    source: PositionSource = PositionSource.IB
    strategy_tag: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    account_id: Optional[str] = None

    def key(self) -> tuple:
        """
        Composite key used for de-duplication and reconciliation across sources.

        Returns:
            Tuple of (symbol, underlying, asset_type, expiry, strike, right).
        """
        return (
            self.symbol,
            self.underlying,
            self.asset_type,
            self.expiry,
            self.strike,
            self.right,
        )

    def get_display_name(self) -> str:
        if self.asset_type == AssetType.STOCK:
            return f"{self.symbol}"
        if self.asset_type == AssetType.OPTION:
            return f"{self.underlying} {self.expiry} {self.strike}{self.right}"
        return "Err Name"

    def days_to_expiry(self, ref_date: Optional[date] = None) -> Optional[int]:
        """Calculate days to expiry from reference date (default: today)."""
        if self.expiry is None or self.expiry=="":
            return None
        expiry_date = datetime.strptime(self.expiry, "%Y%m%d").date()
        ref = ref_date or date.today()
        return (expiry_date - ref).days

    def expiry_bucket(self, ref_date: Optional[date] = None) -> str:
        """
        Classify position into expiry bucket.

        Returns:
            One of: "0DTE", "1_7D", "8_30D", "31_90D", "90D_PLUS", "NO_EXPIRY"
        """
        dte = self.days_to_expiry(ref_date)
        if dte is None:
            return "NO_EXPIRY"
        if dte == 0:
            return "0DTE"
        if dte <= 7:
            return "1_7D"
        if dte <= 30:
            return "8_30D"
        if dte <= 90:
            return "31_90D"
        return "90D_PLUS"
