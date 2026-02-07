"""
Strategy data providers - protocols for regime and confluence access.

These protocols allow strategies to access regime detection and confluence
scoring without creating hard dependencies on infrastructure layer.

Providers are injected into StrategyContext by the runner (backtest or live).
Uses TYPE_CHECKING imports to avoid circular dependencies.

Usage:
    class MyStrategy(Strategy):
        def on_bar(self, bar: BarData) -> None:
            regime = self.context.get_regime(bar.symbol)
            if regime and regime.final_regime == MarketRegime.R0_HEALTHY_UPTREND:
                # Full trading allowed
                ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..signals.indicators.regime.models import MarketRegime, RegimeOutput
    from ..signals.models import ConfluenceScore


@runtime_checkable
class RegimeProvider(Protocol):
    """
    Protocol for providing regime detection data to strategies.

    Implementations may wrap the live RegimeDetector or backtest-computed
    regime series. Strategies access regime data through StrategyContext
    convenience methods rather than using this protocol directly.
    """

    def get_regime(self, symbol: str) -> Optional["RegimeOutput"]:
        """
        Get current regime output for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            RegimeOutput with final_regime, confidence, component_states,
            or None if regime data is unavailable (e.g., warmup period).
        """
        ...

    def get_market_regime(self) -> Optional["MarketRegime"]:
        """
        Get the current market-level regime (from SPY/QQQ).

        Returns:
            MarketRegime enum or None if unavailable.
        """
        ...


@runtime_checkable
class ConfluenceProvider(Protocol):
    """
    Protocol for providing confluence scoring data to strategies.

    Implementations may wrap the live ConfluenceCalculator or provide
    pre-computed confluence scores for backtesting.
    """

    def get_confluence(self, symbol: str, timeframe: str = "1d") -> Optional["ConfluenceScore"]:
        """
        Get current confluence score for a symbol/timeframe.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe (default "1d").

        Returns:
            ConfluenceScore with alignment_score, bullish/bearish counts,
            or None if insufficient indicator data.
        """
        ...
