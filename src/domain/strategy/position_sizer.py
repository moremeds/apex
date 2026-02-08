"""
PositionSizer - ATR-based volatility position sizing.

Calculates position sizes based on portfolio risk budget, ATR-derived
stop distance, and optional confidence/regime scaling factors.

Formula:
    shares = (portfolio_value * risk_per_trade) / (atr * stop_mult)
    shares *= confidence * regime_size_factor
    shares = min(shares, max_position_pct * portfolio_value / price)

Integrates with CostEstimator for break-even validation.

Usage:
    sizer = PositionSizer(
        portfolio_value=100_000,
        risk_per_trade_pct=0.02,
        max_position_pct=0.10,
    )

    result = sizer.calculate(
        symbol="AAPL",
        price=185.0,
        atr=2.5,
        stop_distance_atr_mult=3.0,
    )
    print(f"Size: {result.shares} shares, risk: ${result.dollar_risk:.0f}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing calculation."""

    shares: int
    dollar_risk: float
    notional: float
    risk_pct_of_portfolio: float
    stop_distance: float
    reason: str = ""

    @property
    def is_valid(self) -> bool:
        return self.shares > 0


class PositionSizer:
    """
    ATR-based volatility position sizer.

    Sizes positions so that each trade risks a fixed percentage of
    portfolio value, with the stop distance derived from ATR.
    """

    def __init__(
        self,
        portfolio_value: float = 100_000.0,
        risk_per_trade_pct: float = 0.02,
        max_position_pct: float = 0.10,
        min_shares: int = 1,
        max_shares: int = 10_000,
    ) -> None:
        """
        Args:
            portfolio_value: Total portfolio value in dollars.
            risk_per_trade_pct: Max risk per trade as fraction (0.02 = 2%).
            max_position_pct: Max single position as fraction of portfolio.
            min_shares: Minimum shares per trade.
            max_shares: Maximum shares per trade.
        """
        self._portfolio_value = portfolio_value
        self._risk_per_trade_pct = risk_per_trade_pct
        self._max_position_pct = max_position_pct
        self._min_shares = min_shares
        self._max_shares = max_shares

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @portfolio_value.setter
    def portfolio_value(self, value: float) -> None:
        self._portfolio_value = value

    def calculate(
        self,
        symbol: str,
        price: float,
        atr: float,
        stop_distance_atr_mult: float = 3.0,
        confidence: float = 1.0,
        regime_size_factor: float = 1.0,
    ) -> SizingResult:
        """
        Calculate position size.

        Args:
            symbol: Ticker symbol (for logging).
            price: Current price.
            atr: Current ATR value.
            stop_distance_atr_mult: Stop distance as multiple of ATR.
            confidence: Signal confidence multiplier (0.0 to 1.0).
            regime_size_factor: Regime-based size scaling (0.0 to 1.0).

        Returns:
            SizingResult with shares, risk, and sizing details.
        """
        if price <= 0 or atr <= 0:
            return SizingResult(
                shares=0,
                dollar_risk=0.0,
                notional=0.0,
                risk_pct_of_portfolio=0.0,
                stop_distance=0.0,
                reason=f"Invalid inputs: price={price}, atr={atr}",
            )

        # Calculate stop distance in dollars
        stop_distance = atr * stop_distance_atr_mult

        # Base risk budget
        dollar_risk_budget = self._portfolio_value * self._risk_per_trade_pct

        # Raw shares from risk budget
        raw_shares = dollar_risk_budget / stop_distance
        raw_shares *= max(0.0, min(1.0, confidence))
        raw_shares *= max(0.0, min(1.0, regime_size_factor))

        # Apply max position constraint
        max_notional = self._portfolio_value * self._max_position_pct
        max_shares_by_notional = max_notional / price
        raw_shares = min(raw_shares, max_shares_by_notional)

        # Round down and clamp; if raw_shares < 1, signal/regime says near-zero
        floored = int(math.floor(raw_shares))
        if floored < 1:
            return SizingResult(
                shares=0,
                dollar_risk=0.0,
                notional=0.0,
                risk_pct_of_portfolio=0.0,
                stop_distance=stop_distance,
                reason=f"{symbol}: sized to 0 (raw={raw_shares:.2f})",
            )
        shares = max(self._min_shares, min(self._max_shares, floored))

        # Recalculate actual risk
        actual_dollar_risk = shares * stop_distance
        notional = shares * price
        risk_pct = actual_dollar_risk / self._portfolio_value if self._portfolio_value > 0 else 0.0

        return SizingResult(
            shares=shares,
            dollar_risk=actual_dollar_risk,
            notional=notional,
            risk_pct_of_portfolio=risk_pct,
            stop_distance=stop_distance,
            reason=f"{symbol}: {shares} shares @ ${price:.2f}, risk ${actual_dollar_risk:.0f}",
        )

    def calculate_from_stop_price(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        confidence: float = 1.0,
        regime_size_factor: float = 1.0,
    ) -> SizingResult:
        """
        Calculate position size from explicit stop price.

        Args:
            symbol: Ticker symbol.
            entry_price: Planned entry price.
            stop_price: Explicit stop-loss price.
            confidence: Signal confidence multiplier.
            regime_size_factor: Regime-based scaling.

        Returns:
            SizingResult with shares and risk details.
        """
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return SizingResult(
                shares=0,
                dollar_risk=0.0,
                notional=0.0,
                risk_pct_of_portfolio=0.0,
                stop_distance=0.0,
                reason="Stop distance is zero",
            )

        # Use stop_distance directly (atr_mult=1 since distance is already absolute)
        dollar_risk_budget = self._portfolio_value * self._risk_per_trade_pct
        raw_shares = dollar_risk_budget / stop_distance
        raw_shares *= max(0.0, min(1.0, confidence))
        raw_shares *= max(0.0, min(1.0, regime_size_factor))

        max_notional = self._portfolio_value * self._max_position_pct
        max_shares_by_notional = max_notional / entry_price
        raw_shares = min(raw_shares, max_shares_by_notional)

        floored = int(math.floor(raw_shares))
        if floored < 1:
            return SizingResult(
                shares=0,
                dollar_risk=0.0,
                notional=0.0,
                risk_pct_of_portfolio=0.0,
                stop_distance=stop_distance,
                reason=f"{symbol}: sized to 0 (raw={raw_shares:.2f})",
            )
        shares = max(self._min_shares, min(self._max_shares, floored))
        actual_dollar_risk = shares * stop_distance
        notional = shares * entry_price
        risk_pct = actual_dollar_risk / self._portfolio_value if self._portfolio_value > 0 else 0.0

        return SizingResult(
            shares=shares,
            dollar_risk=actual_dollar_risk,
            notional=notional,
            risk_pct_of_portfolio=risk_pct,
            stop_distance=stop_distance,
            reason=f"{symbol}: {shares} shares @ ${entry_price:.2f}, stop ${stop_price:.2f}",
        )
