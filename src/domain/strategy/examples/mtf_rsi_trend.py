"""
Multi-Timeframe RSI Trend Strategy.

A trend-following strategy that uses multiple timeframes:
- Higher timeframe (e.g., daily) for trend direction via RSI
- Lower timeframe (e.g., hourly) for entry timing via RSI

Entry Logic:
- LONG: Daily RSI > 50 (bullish trend) AND Hourly RSI < 30 (oversold dip)
- SHORT: Daily RSI < 50 (bearish trend) AND Hourly RSI > 70 (overbought rally)

This strategy demonstrates:
- Multi-timeframe data consumption via on_bars()
- Trend/momentum alignment across timeframes
- Using higher timeframe for context, lower for precision

"""
from collections import deque
from typing import Deque, Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from ..base import Strategy, StrategyContext
from ..registry import register_strategy
from ..signals.indicators import rsi as talib_rsi
from ...events.domain_events import BarData, QuoteTick, TradeFill
from ...interfaces.execution_provider import OrderRequest

logger = logging.getLogger(__name__)


@register_strategy(
    "mtf_rsi_trend",
    description="Multi-Timeframe RSI Trend Strategy (daily trend + hourly entry)",
    author="Apex",
    version="1.0",
)
class MTFRsiTrendStrategy(Strategy):
    """
    Multi-Timeframe RSI Trend Strategy.

    Uses daily RSI for trend direction and hourly RSI for entry timing.
    Aligns trend momentum with counter-trend entries for better risk/reward.

    Parameters:
        primary_timeframe: Higher timeframe for trend (default: "1d")
        secondary_timeframe: Lower timeframe for entry (default: "1h")
        trend_rsi_period: RSI period for trend timeframe (default: 14)
        entry_rsi_period: RSI period for entry timeframe (default: 14)
        trend_threshold: RSI level dividing bull/bear (default: 50)
        entry_oversold: Entry RSI oversold level (default: 30)
        entry_overbought: Entry RSI overbought level (default: 70)
        position_size: Shares per trade (default: 100)

    Multi-Timeframe Flow:
        1. on_bars() receives Dict[timeframe, BarData] aligned to primary bar close
        2. Calculate RSI on each timeframe's price history
        3. Use primary (daily) RSI for trend bias
        4. Use secondary (hourly) RSI for entry signal
        5. Enter only when both align (trend + counter-trend dip)
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        primary_timeframe: str = "1d",
        secondary_timeframe: str = "1h",
        trend_rsi_period: int = 14,
        entry_rsi_period: int = 14,
        trend_threshold: float = 50.0,
        entry_oversold: float = 30.0,
        entry_overbought: float = 70.0,
        position_size: float = 100,
    ):
        super().__init__(strategy_id, symbols, context)

        self.primary_tf = primary_timeframe
        self.secondary_tf = secondary_timeframe
        self.trend_rsi_period = trend_rsi_period
        self.entry_rsi_period = entry_rsi_period
        self.trend_threshold = trend_threshold
        self.entry_oversold = entry_oversold
        self.entry_overbought = entry_overbought
        self.position_size = position_size

        # Price history per symbol per timeframe
        # Structure: {symbol: {timeframe: deque of closes}}
        self._prices: Dict[str, Dict[str, Deque[float]]] = {}
        for symbol in symbols:
            self._prices[symbol] = {
                primary_timeframe: deque(maxlen=trend_rsi_period + 1),
                secondary_timeframe: deque(maxlen=entry_rsi_period + 1),
            }

        # Cached RSI values
        self._trend_rsi: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._entry_rsi: Dict[str, Optional[float]] = {s: None for s in symbols}

    def on_start(self) -> None:
        """Initialize strategy."""
        logger.info(
            f"MTF RSI Trend started: {self.symbols} "
            f"(primary={self.primary_tf}, secondary={self.secondary_tf}, "
            f"trend_period={self.trend_rsi_period}, entry_period={self.entry_rsi_period})"
        )

    def on_bars(self, bars: Dict[str, BarData]) -> None:
        """
        Process aligned multi-timeframe bars.

        This is called once per primary timeframe bar close, with the most
        recent bar from each configured timeframe aligned to that moment.

        Args:
            bars: Dict mapping timeframe (e.g., "1d", "1h") to BarData
        """
        # Get primary bar (must exist)
        primary_bar = bars.get(self.primary_tf)
        if not primary_bar:
            return

        symbol = primary_bar.symbol
        price_primary = primary_bar.close

        if price_primary is None or price_primary <= 0:
            return

        # Update primary timeframe price history
        self._prices[symbol][self.primary_tf].append(price_primary)

        # Update secondary timeframe if available
        secondary_bar = bars.get(self.secondary_tf)
        if secondary_bar and secondary_bar.close:
            self._prices[symbol][self.secondary_tf].append(secondary_bar.close)

        # Calculate RSI on each timeframe
        self._trend_rsi[symbol] = self._calculate_rsi(
            self._prices[symbol][self.primary_tf],
            self.trend_rsi_period,
        )
        self._entry_rsi[symbol] = self._calculate_rsi(
            self._prices[symbol][self.secondary_tf],
            self.entry_rsi_period,
        )

        # Check for signals
        trend_rsi = self._trend_rsi[symbol]
        entry_rsi = self._entry_rsi[symbol]

        if trend_rsi is None:
            return  # Not enough data for trend

        current_position = self.context.get_position_quantity(symbol)

        # Bullish trend + oversold entry = BUY
        if trend_rsi > self.trend_threshold:
            # In uptrend - look for oversold dips to buy
            if entry_rsi is not None and entry_rsi < self.entry_oversold:
                if current_position <= 0:
                    self._enter_long(symbol, price_primary, trend_rsi, entry_rsi)

            # Exit short if we're in one
            elif current_position < 0:
                self._exit_position(symbol, price_primary, current_position, "trend_reversal")

        # Bearish trend + overbought entry = SELL
        elif trend_rsi < self.trend_threshold:
            # In downtrend - look for overbought rallies to sell
            if entry_rsi is not None and entry_rsi > self.entry_overbought:
                if current_position >= 0:
                    self._enter_short(symbol, price_primary, trend_rsi, entry_rsi)

            # Exit long if we're in one
            elif current_position > 0:
                self._exit_position(symbol, price_primary, current_position, "trend_reversal")

    def on_bar(self, bar: BarData) -> None:
        """
        Fallback for single-timeframe mode.

        When no secondary timeframe is configured, we still work with just
        the primary timeframe data.
        """
        # Convert single bar to MTF format and delegate
        bars = {self.primary_tf: bar}
        self.on_bars(bars)

    def _calculate_rsi(self, prices: Deque[float], period: int) -> Optional[float]:
        """
        Calculate RSI using TA-Lib via indicators module.

        Args:
            prices: Deque of closing prices
            period: RSI period

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(prices) < period + 1:
            return None

        # Convert deque to pandas Series for TA-Lib wrapper
        price_series = pd.Series(list(prices))
        rsi_series = talib_rsi(price_series, period=period)

        # Return the last (most recent) RSI value
        last_rsi = rsi_series.iloc[-1]
        return None if np.isnan(last_rsi) else float(last_rsi)

    def _enter_long(
        self, symbol: str, price: float, trend_rsi: float, entry_rsi: float
    ) -> None:
        """Enter long position."""
        logger.info(
            f"[{self.strategy_id}] BUY SIGNAL: {symbol} @ {price:.2f} | "
            f"Trend RSI={trend_rsi:.1f} (>{self.trend_threshold}), "
            f"Entry RSI={entry_rsi:.1f} (<{self.entry_oversold})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="BUY",
            quantity=self.position_size,
            order_type="MARKET",
        )
        self.request_order(order)

    def _enter_short(
        self, symbol: str, price: float, trend_rsi: float, entry_rsi: float
    ) -> None:
        """Enter short position."""
        logger.info(
            f"[{self.strategy_id}] SELL SIGNAL: {symbol} @ {price:.2f} | "
            f"Trend RSI={trend_rsi:.1f} (<{self.trend_threshold}), "
            f"Entry RSI={entry_rsi:.1f} (>{self.entry_overbought})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="SELL",
            quantity=self.position_size,
            order_type="MARKET",
        )
        self.request_order(order)

    def _exit_position(
        self, symbol: str, price: float, quantity: float, reason: str
    ) -> None:
        """Exit current position."""
        side = "SELL" if quantity > 0 else "BUY"

        logger.info(
            f"[{self.strategy_id}] EXIT {side}: {symbol} @ {price:.2f} | "
            f"reason={reason}, qty={abs(quantity)}"
        )

        order = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            order_type="MARKET",
        )
        self.request_order(order)

    def on_fill(self, fill: TradeFill) -> None:
        """Handle trade fill."""
        logger.info(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """Handle tick data - not used for this bar-based strategy."""
        pass  # MTF strategy uses on_bar/on_bars for daily/hourly data

    def get_state(self) -> dict:
        """Get current strategy state for monitoring."""
        return {
            symbol: {
                "trend_rsi": self._trend_rsi.get(symbol),
                "entry_rsi": self._entry_rsi.get(symbol),
                "primary_bars": len(self._prices[symbol][self.primary_tf]),
                "secondary_bars": len(self._prices[symbol][self.secondary_tf]),
            }
            for symbol in self.symbols
        }
