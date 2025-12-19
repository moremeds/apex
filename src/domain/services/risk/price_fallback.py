"""
Price fallback chain for risk calculations.

When live market data is unavailable, this module provides a structured
fallback mechanism to ensure risk visibility is preserved rather than
zeroing out exposure (which would hide risk).

Fallback order:
1. Live mid/last price from primary provider
2. Yesterday's close from market data
3. Yesterday's close from Yahoo Finance (stocks only)
4. Average cost basis (last resort - preserves notional visibility)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

from ....models.market_data import MarketData
from ....models.position import Position
from ....utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ....infrastructure.adapters.yahoo_adapter import YahooFinanceAdapter

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PriceResult:
    """Result of price resolution with fallback information."""

    price: Optional[float]
    source: str  # "live", "md_yesterday_close", "yahoo_yesterday_close", "avg_cost_basis", "none"
    is_fallback: bool  # True if not using live price

    @property
    def has_price(self) -> bool:
        """Check if a valid price was resolved."""
        return self.price is not None and self.price > 0


class PriceFallbackChain:
    """
    Resolves prices using a fallback chain when live data is unavailable.

    This is critical for risk management - we want to preserve visibility
    into positions even when live pricing is temporarily unavailable,
    rather than showing zero exposure which hides risk.

    Usage:
        chain = PriceFallbackChain(yahoo_adapter=yahoo_adapter)
        result = chain.resolve(position, market_data)
        if result.has_price:
            mark = result.price
            if result.is_fallback:
                logger.warning(f"Using fallback: {result.source}")
    """

    def __init__(self, yahoo_adapter: Optional["YahooFinanceAdapter"] = None):
        """
        Initialize the fallback chain.

        Args:
            yahoo_adapter: Optional Yahoo Finance adapter for historical prices.
        """
        self._yahoo_adapter = yahoo_adapter

    def resolve(
        self,
        position: Position,
        market_data: Dict[str, MarketData],
    ) -> PriceResult:
        """
        Resolve price for a position using the fallback chain.

        Args:
            position: The position needing a price.
            market_data: Current market data dictionary.

        Returns:
            PriceResult with resolved price and source information.
        """
        md = market_data.get(position.symbol)

        # 1. Try live price from market data
        if md is not None and md.has_live_price():
            return PriceResult(
                price=md.effective_mid(),
                source="live",
                is_fallback=False,
            )

        # 2. Try yesterday's close from market data
        if md is not None and md.yesterday_close is not None and md.yesterday_close > 0:
            return PriceResult(
                price=md.yesterday_close,
                source="md_yesterday_close",
                is_fallback=True,
            )

        # 3. Try Yahoo Finance for yesterday's close (stocks only)
        if self._yahoo_adapter is not None and position.asset_type.value == "STOCK":
            yahoo_md = self._yahoo_adapter.get_latest(position.symbol)
            if yahoo_md and yahoo_md.yesterday_close and yahoo_md.yesterday_close > 0:
                return PriceResult(
                    price=yahoo_md.yesterday_close,
                    source="yahoo_yesterday_close",
                    is_fallback=True,
                )

        # 4. Last resort: average cost basis
        if position.avg_price and position.avg_price > 0:
            return PriceResult(
                price=position.avg_price,
                source="avg_cost_basis",
                is_fallback=True,
            )

        # No price available
        return PriceResult(
            price=None,
            source="none",
            is_fallback=True,
        )

    def resolve_with_logging(
        self,
        position: Position,
        market_data: Dict[str, MarketData],
    ) -> PriceResult:
        """
        Resolve price with debug logging for fallback usage.

        Same as resolve() but logs when fallback is used.
        """
        result = self.resolve(position, market_data)

        if result.is_fallback and result.has_price:
            logger.debug(
                f"Using fallback price for {position.symbol}: "
                f"{result.source}=${result.price:.2f}"
            )
        elif not result.has_price:
            logger.warning(f"No price available for {position.symbol}")

        return result
