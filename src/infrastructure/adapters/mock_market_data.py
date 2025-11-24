"""
Mock market data provider for testing when IB is not connected.

Generates sample market data based on positions to allow testing
the risk engine and dashboard without a live IB connection.
"""

from __future__ import annotations
from typing import List, Dict
from datetime import datetime
import logging
import random

from ...models.position import Position
from ...models.market_data import MarketData, GreeksSource, DataQuality


logger = logging.getLogger(__name__)


class MockMarketDataProvider:
    """
    Mock market data provider for testing.

    Generates reasonable market data for positions when IB is not available.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize mock provider.

        Args:
            seed: Random seed for reproducible data.
        """
        self.seed = seed
        random.seed(seed)

        # Base prices for common stocks
        self.base_prices = {
            "AAPL": 180.00,
            "MSFT": 380.00,
            "GOOGL": 140.00,
            "AMZN": 150.00,
            "TSLA": 240.00,
            "NVDA": 490.00,
            "SPY": 450.00,
            "QQQ": 380.00,
            "IWM": 200.00,
            "DIA": 360.00,
            "GLD": 185.00,
            "SLV": 24.00,
            "TLT": 95.00,
            "VIX": 15.00,
            "VOO": 420.00,
        }

    def generate_market_data(self, positions: List[Position]) -> List[MarketData]:
        """
        Generate mock market data for given positions.

        Args:
            positions: List of positions to generate data for.

        Returns:
            List of MarketData objects.
        """
        market_data_list = []

        for pos in positions:
            md = self._generate_for_symbol(pos.symbol, pos.underlying, pos.asset_type.value)
            if md:
                market_data_list.append(md)

        logger.info(f"Generated mock market data for {len(market_data_list)} symbols")
        return market_data_list

    def _generate_for_symbol(
        self, symbol: str, underlying: str, asset_type: str
    ) -> MarketData | None:
        """Generate market data for a single symbol."""

        # Get base price
        base_price = self.base_prices.get(underlying, 100.0)

        # Add some random movement (+/- 2%)
        price_movement = random.uniform(-0.02, 0.02)
        current_price = base_price * (1 + price_movement)

        # Create bid/ask spread (0.01% - 0.1%)
        spread_pct = random.uniform(0.0001, 0.001)
        spread = current_price * spread_pct

        bid = current_price - spread / 2
        ask = current_price + spread / 2
        mid = (bid + ask) / 2

        if asset_type == "STOCK":
            # Stock market data
            return MarketData(
                symbol=symbol,
                last=current_price,
                bid=bid,
                ask=ask,
                mid=mid,
                volume=random.randint(1000000, 10000000),
                delta=1.0,  # Stocks always have delta=1
                timestamp=datetime.now(),
                greeks_source=GreeksSource.MOCK,
                quality=DataQuality.GOOD,
            )

        elif asset_type == "OPTION":
            # Option market data
            # Generate reasonable option prices (simplified)
            option_price = base_price * random.uniform(0.01, 0.10)

            option_bid = option_price * 0.98
            option_ask = option_price * 1.02
            option_mid = (option_bid + option_ask) / 2

            # Generate Greeks (simplified, not accurate)
            delta = random.uniform(-0.8, 0.8)
            gamma = random.uniform(0.001, 0.05)
            vega = random.uniform(0.05, 0.30)
            theta = random.uniform(-0.10, -0.01)

            return MarketData(
                symbol=symbol,
                last=option_price,
                bid=option_bid,
                ask=option_ask,
                mid=option_mid,
                volume=random.randint(100, 10000),
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                timestamp=datetime.now(),
                greeks_source=GreeksSource.MOCK,
                quality=DataQuality.GOOD,
            )

        return None
