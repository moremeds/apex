"""Multi-Timeframe RSI Trend SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .indicators import rsi


class MTFRsiTrendSignalGenerator:
    """
    Multi-Timeframe RSI strategy signal generation with directional support.

    Uses two timeframes:
    - Primary (e.g., daily): Trend direction via RSI > 50 (bullish) or < 50 (bearish)
    - Secondary (e.g., hourly): Entry timing via oversold/overbought levels

    Entry Logic:
    - LONG: Primary RSI > trend_threshold (uptrend) AND Secondary RSI < oversold
    - SHORT: Primary RSI < trend_threshold (downtrend) AND Secondary RSI > overbought

    Exit Logic:
    - Exit LONG: Secondary RSI > overbought (take profit in uptrend)
    - Exit SHORT: Secondary RSI < oversold (take profit in downtrend)

    This generator implements DirectionalSignalGenerator protocol for proper
    long/short signal separation in VectorBT.

    Parameters:
        trend_rsi_period: RSI period for primary timeframe trend (default 14)
        entry_rsi_period: RSI period for secondary timeframe entry (default 14)
        trend_threshold: RSI level dividing bullish/bearish (default 50)
        entry_oversold: Oversold threshold for entry (default 30)
        entry_overbought: Overbought threshold for entry (default 70)
    """

    @property
    def warmup_bars(self) -> int:
        """Max RSI warmup period."""
        return 21  # Max of trend_rsi_period and entry_rsi_period + buffer

    def _get_secondary_rsi(
        self,
        index: pd.DatetimeIndex,
        close: pd.Series,
        entry_period: int,
        secondary_data: Optional[Dict[str, pd.DataFrame]],
    ) -> pd.Series:
        """Extract and align secondary timeframe RSI to primary index."""
        secondary_df = None
        if secondary_data:
            # Try common hourly keys in order of preference
            for key in ["1h", "1H", "4h", "4H"]:
                if key in secondary_data:
                    secondary_df = secondary_data[key]
                    break
            # Fallback to first available timeframe
            if secondary_df is None:
                secondary_df = next(iter(secondary_data.values()), None)

        if secondary_df is not None and not secondary_df.empty:
            secondary_close = secondary_df["close"]
            secondary_rsi_values = rsi(secondary_close, entry_period)
            # Align secondary RSI to primary index via forward-fill
            return secondary_rsi_values.reindex(index, method="ffill")
        else:
            # Fallback: use primary timeframe RSI for entry
            return rsi(close, entry_period)

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate combined entry/exit signals (long-only compatible).

        For proper long/short handling, use generate_directional() instead.

        Returns:
            (entries, exits): Combined signals (long entries only for safety).
        """
        long_entries, long_exits, _, _ = self.generate_directional(data, params, secondary_data)
        # Return long-only signals for backward compatibility
        return long_entries, long_exits

    def generate_directional(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate separate long and short entry/exit signals.

        Args:
            data: Primary timeframe OHLCV DataFrame (e.g., daily).
            params: Strategy parameters.
            secondary_data: Dict with secondary timeframe data.

        Returns:
            (long_entries, long_exits, short_entries, short_exits):
            Four boolean series aligned to primary timeframe index.
        """
        close = data["close"]
        index = close.index

        # Extract parameters
        trend_period = int(params.get("trend_rsi_period", 14))
        entry_period = int(params.get("entry_rsi_period", 14))
        trend_threshold = float(params.get("trend_threshold", 50))
        oversold = float(params.get("entry_oversold", 30))
        overbought = float(params.get("entry_overbought", 70))

        # Calculate primary timeframe RSI for trend
        primary_rsi = rsi(close, trend_period)

        # Trend direction
        bullish_trend = primary_rsi > trend_threshold
        bearish_trend = primary_rsi < trend_threshold

        # Get aligned secondary RSI
        secondary_rsi = self._get_secondary_rsi(index, close, entry_period, secondary_data)

        # Long signals: bullish trend + oversold entry
        long_entries = bullish_trend & (secondary_rsi < oversold)
        # Long exits: trend reversal (matches Apex strategy logic)
        long_exits = bearish_trend

        # Short signals: bearish trend + overbought entry
        short_entries = bearish_trend & (secondary_rsi > overbought)
        # Short exits: trend reversal (matches Apex strategy logic)
        short_exits = bullish_trend

        return (
            long_entries.fillna(False),
            long_exits.fillna(False),
            short_entries.fillna(False),
            short_exits.fillna(False),
        )
