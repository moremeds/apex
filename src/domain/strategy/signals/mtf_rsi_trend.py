"""Multi-Timeframe RSI Trend SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .indicators import rsi


class MTFRsiTrendSignalGenerator:
    """
    Multi-Timeframe RSI strategy signal generation.

    Uses two timeframes:
    - Primary (e.g., daily): Trend direction via RSI > 50 (bullish) or < 50 (bearish)
    - Secondary (e.g., hourly): Entry timing via oversold/overbought levels

    Entry Logic:
    - LONG: Primary RSI > trend_threshold (uptrend) AND Secondary RSI < oversold
    - SHORT: Primary RSI < trend_threshold (downtrend) AND Secondary RSI > overbought

    Exit Logic:
    - Exit LONG: Secondary RSI > overbought (take profit in uptrend)
    - Exit SHORT: Secondary RSI < oversold (take profit in downtrend)

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

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate MTF RSI trend signals.

        Args:
            data: Primary timeframe OHLCV DataFrame (e.g., daily).
            params: Strategy parameters.
            secondary_data: Dict with secondary timeframe data.
                Expected key: "1h" (or other configured secondary timeframe).

        Returns:
            (entries, exits): Boolean series aligned to primary timeframe index.
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

        # Get secondary timeframe data
        # Default to primary if no secondary data available
        secondary_df = None
        if secondary_data:
            # Try common hourly keys in order of preference
            for key in ["1h", "1H", "4h", "4H"]:
                if key in secondary_data:
                    secondary_df = secondary_data[key]
                    break
            # Fallback to first available timeframe
            if secondary_df is None and secondary_data:
                secondary_df = next(iter(secondary_data.values()))

        if secondary_df is not None and not secondary_df.empty:
            # Resample secondary RSI to primary timeframe
            # Use the last secondary RSI value within each primary bar period
            secondary_close = secondary_df["close"]
            secondary_rsi_values = rsi(secondary_close, entry_period)

            # Align secondary RSI to primary index via forward-fill
            # Each primary bar gets the most recent secondary RSI value
            secondary_rsi = secondary_rsi_values.reindex(index, method="ffill")
        else:
            # Fallback: use primary timeframe RSI for entry if no secondary data
            secondary_rsi = rsi(close, entry_period)

        # Entry signals
        long_entry = bullish_trend & (secondary_rsi < oversold)
        short_entry = bearish_trend & (secondary_rsi > overbought)
        entries = long_entry | short_entry

        # Exit signals (take profit at opposite extreme)
        exits = (
            (bullish_trend & (secondary_rsi > overbought)) |
            (bearish_trend & (secondary_rsi < oversold))
        )

        return entries.fillna(False), exits.fillna(False)
