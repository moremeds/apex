"""TA Metrics SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import macd, rsi, sma


class TAMetricsSignalGenerator:
    """
    Vectorized TA metrics matrix signal generation using TA-Lib.

    Computes a composite score from multiple technical indicators:
    - MA Score: Based on fast/slow SMA crossover
    - RSI Score: Based on overbought/oversold levels
    - MACD Score: Based on histogram direction
    - Trend Score: Based on price relative to MAs

    Entry when total score >= min_score, exit when <= -min_score.

    Parameters:
        sma_fast: Fast SMA period (default 20)
        sma_slow: Slow SMA period (default 50)
        rsi_period: RSI period (default 14)
        rsi_oversold: Oversold threshold (default 30)
        rsi_overbought: Overbought threshold (default 70)
        macd_fast: MACD fast period (default 12)
        macd_slow: MACD slow period (default 26)
        macd_signal: MACD signal period (default 9)
        min_score: Minimum score for entry/exit (default 3)
    """

    @property
    def warmup_bars(self) -> int:
        """Max of all indicator warmup periods."""
        return 50  # max(sma_slow, macd_slow + macd_signal)

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate TA metrics matrix signals.

        Args:
            data: OHLCV DataFrame with 'close' column.
            params: Strategy parameters for each indicator.

        Returns:
            (entries, exits): Boolean series based on composite score.
        """
        close = data["close"]
        index = close.index

        # Extract parameters
        sma_fast_period = int(params.get("sma_fast", 20))
        sma_slow_period = int(params.get("sma_slow", 50))
        rsi_period = int(params.get("rsi_period", 14))
        rsi_oversold = float(params.get("rsi_oversold", 30))
        rsi_overbought = float(params.get("rsi_overbought", 70))
        macd_fast = int(params.get("macd_fast", 12))
        macd_slow = int(params.get("macd_slow", 26))
        macd_signal_period = int(params.get("macd_signal", 9))
        min_score = float(params.get("min_score", 3))

        # Calculate indicators using TA-Lib wrappers
        sma_fast_series = sma(close, sma_fast_period)
        sma_slow_series = sma(close, sma_slow_period)
        rsi_values = rsi(close, rsi_period)
        _, _, macd_hist_series = macd(close, macd_fast, macd_slow, macd_signal_period)

        # MA Score: +2 for bullish cross, +1 for bullish, -2 for bearish cross, -1 for bearish
        bullish = sma_fast_series > sma_slow_series
        bearish = sma_fast_series < sma_slow_series
        bullish_cross = bullish & (sma_fast_series.shift(1) <= sma_slow_series.shift(1))
        bearish_cross = bearish & (sma_fast_series.shift(1) >= sma_slow_series.shift(1))

        ma_score = pd.Series(
            np.select(
                [bullish_cross, bullish, bearish_cross, bearish],
                [2, 1, -2, -1],
                default=0,
            ),
            index=index,
        )

        # RSI Score: +2 oversold, +1 low, -2 overbought, -1 high
        rsi_score = pd.Series(
            np.select(
                [
                    rsi_values < rsi_oversold,
                    rsi_values < 40,
                    rsi_values > rsi_overbought,
                    rsi_values > 60,
                ],
                [2, 1, -2, -1],
                default=0,
            ),
            index=index,
        )

        # MACD Score: +2 for bullish cross, +1 for positive, -2 for bearish cross, -1 for negative
        prev_hist = macd_hist_series.shift(1)
        macd_score = pd.Series(
            np.select(
                [
                    (macd_hist_series > 0) & (prev_hist <= 0),
                    macd_hist_series > 0,
                    (macd_hist_series < 0) & (prev_hist >= 0),
                    macd_hist_series < 0,
                ],
                [2, 1, -2, -1],
                default=0,
            ),
            index=index,
        )

        # Trend Score: +1 for bullish alignment, -1 for bearish alignment
        trend_score = pd.Series(
            np.select(
                [
                    (close > sma_fast_series) & (sma_fast_series > sma_slow_series),
                    (close < sma_fast_series) & (sma_fast_series < sma_slow_series),
                ],
                [1, -1],
                default=0,
            ),
            index=index,
        )

        # Total score
        total_score = ma_score + rsi_score + macd_score + trend_score

        # Entry when score >= min_score, exit when <= -min_score
        entries = (total_score >= min_score).fillna(False)
        exits = (total_score <= -min_score).fillna(False)

        return entries, exits
