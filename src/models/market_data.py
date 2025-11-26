"""Market data model with Greeks source tracking and quality flags."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class GreeksSource(Enum):
    """Source of Greeks calculation."""
    IBKR = "IBKR"  # From Interactive Brokers
    VENDOR = "VENDOR"  # From data vendor
    CALCULATED = "CALCULATED"  # Locally calculated (future use)
    MOCK = "MOCK"  # Mock data for testing
    MISSING = "MISSING"  # No Greeks available


class DataQuality(Enum):
    """Market data quality flag."""
    GOOD = "GOOD"  # Fresh and valid
    STALE = "STALE"  # Exceeded staleness threshold
    SUSPICIOUS = "SUSPICIOUS"  # Failed bid/ask sanity check
    MISSING = "MISSING"  # No data available
    ZERO_QUOTE = "ZERO_QUOTE"  # Zero bid or ask


@dataclass
class MarketData:
    """Market data with prices, Greeks, and quality metadata."""

    symbol: str

    # Price data
    last: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    volume: Optional[int] = None

    # Greeks (from IBKR in MVP)
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None

    # Underlying price (for options delta dollars calculation)
    underlying_price: Optional[float] = None

    # Metadata
    yesterday_close: Optional[float] = None
    timestamp: Optional[datetime] = None
    greeks_source: GreeksSource = GreeksSource.MISSING
    quality: DataQuality = DataQuality.MISSING

    def effective_mid(self) -> Optional[float]:
        """
        Return the best available mark price (mid -> last -> theoretical).

        Returns:
            Best available price, or None if no price data.
        """
        if self.mid is not None:
            return self.mid
        return self.last

    def is_stale(self, stale_seconds: int = 10) -> bool:
        """Check if market data exceeds staleness threshold."""
        if self.timestamp is None:
            return True
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > stale_seconds

    def has_greeks(self) -> bool:
        """Check if Greeks are available."""
        return self.greeks_source != GreeksSource.MISSING and self.delta is not None

    def bid_ask_valid(self) -> bool:
        """Validate bid <= ask sanity check."""
        if self.bid is None or self.ask is None:
            return True  # Can't validate if missing
        return self.bid <= self.ask
