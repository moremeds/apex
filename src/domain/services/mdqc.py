"""
Market Data Quality Control (MDQC).

Validates market data for:
- Staleness (timestamp exceeds threshold)
- Bid/ask sanity (bid <= ask)
- Zero quotes (bid or ask = 0)
- NaN/infinity values (corrupted data)
- Missing Greeks vs missing prices
"""

from __future__ import annotations

import math
from typing import Dict, List

from ...models.market_data import DataQuality, MarketData


class MDQC:
    """Market Data Quality Control service."""

    def __init__(
        self,
        stale_seconds: int = 10,
        ignore_zero_quotes: bool = True,
        enforce_bid_ask_sanity: bool = True,
    ):
        """
        Initialize MDQC.

        Args:
            stale_seconds: Staleness threshold in seconds.
            ignore_zero_quotes: Flag zero bid/ask quotes as suspicious.
            enforce_bid_ask_sanity: Validate bid <= ask.
        """
        self.stale_seconds = stale_seconds
        self.ignore_zero_quotes = ignore_zero_quotes
        self.enforce_bid_ask_sanity = enforce_bid_ask_sanity

    def validate_all(self, market_data: Dict[str, MarketData]) -> Dict[str, MarketData]:
        """
        Validate all market data and update quality flags.

        Args:
            market_data: Dict of symbol -> MarketData.

        Returns:
            Same dict with updated quality flags.
        """
        for symbol, md in market_data.items():
            md.quality = self.validate_single(md)
        return market_data

    def validate_single(self, md: MarketData) -> DataQuality:
        """
        Validate a single MarketData object.

        Returns:
            DataQuality enum value.
        """
        # Check if data is present
        if md.last is None and md.bid is None and md.ask is None:
            return DataQuality.MISSING

        # Check for NaN/infinity in price fields (corrupted IBKR data)
        price_fields = [md.bid, md.ask, md.last, md.mid]
        for price in price_fields:
            if price is not None and (math.isnan(price) or math.isinf(price)):
                return DataQuality.SUSPICIOUS

        # Check for NaN/infinity in Greeks fields
        greeks_fields = [md.delta, md.gamma, md.vega, md.theta, md.iv]
        for greek in greeks_fields:
            if greek is not None and (math.isnan(greek) or math.isinf(greek)):
                return DataQuality.SUSPICIOUS

        # Check staleness
        if md.is_stale(self.stale_seconds):
            return DataQuality.STALE

        # Check zero quotes
        if self.ignore_zero_quotes:
            if (md.bid is not None and md.bid == 0.0) or (md.ask is not None and md.ask == 0.0):
                return DataQuality.ZERO_QUOTE

        # Check bid/ask sanity
        if self.enforce_bid_ask_sanity:
            if not md.bid_ask_valid():
                return DataQuality.SUSPICIOUS

        return DataQuality.GOOD

    def get_stale_symbols(self, market_data: Dict[str, MarketData]) -> List[str]:
        """Get list of symbols with stale market data."""
        return [symbol for symbol, md in market_data.items() if md.is_stale(self.stale_seconds)]

    def get_missing_greeks(self, market_data: Dict[str, MarketData]) -> List[str]:
        """Get list of symbols missing Greeks."""
        return [symbol for symbol, md in market_data.items() if not md.has_greeks()]

    def get_suspicious_symbols(self, market_data: Dict[str, MarketData]) -> List[str]:
        """Get list of symbols with suspicious data."""
        return [
            symbol for symbol, md in market_data.items() if md.quality == DataQuality.SUSPICIOUS
        ]
