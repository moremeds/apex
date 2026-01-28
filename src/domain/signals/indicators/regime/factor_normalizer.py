"""
Factor Normalizer - Consistent percentile rank normalization for regime factors.

Phase 5: Ensures all factors are normalized to [0, 1] range using rolling percentile rank.
This makes factors comparable across assets and time periods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .components.helpers import rolling_percentile_rank


@dataclass
class NormalizedFactors:
    """Container for normalized factor scores (all in [0, 1] range)."""

    trend: pd.Series  # Higher = stronger uptrend (EMA20/50 - longer term)
    trend_short: pd.Series  # Higher = recent momentum (EMA10/20 - shorter term)
    momentum: pd.Series  # Higher = more overbought
    volatility: pd.Series  # Higher = more stress (inverted for composite)
    # Dual MACD factors
    macd_trend: pd.Series  # Long MACD histogram (55/89) - trend direction
    macd_momentum: pd.Series  # Short MACD histogram (13/21) - momentum vs trend
    breadth: Optional[pd.Series] = None  # Higher = outperforming benchmark

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for model training."""
        data = {
            "trend": self.trend,
            "trend_short": self.trend_short,
            "momentum": self.momentum,
            "volatility": self.volatility,
            "macd_trend": self.macd_trend,
            "macd_momentum": self.macd_momentum,
        }
        if self.breadth is not None:
            data["breadth"] = self.breadth
        return pd.DataFrame(data)


class FactorNormalizer:
    """
    Normalize raw indicator values to [0, 1] using rolling percentile rank.

    Uses existing rolling_percentile_rank() from helpers.py for consistency.
    """

    def __init__(
        self,
        lookback_short: int = 63,  # ~3 months
        lookback_long: int = 252,  # ~1 year
    ) -> None:
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def normalize(self, raw: np.ndarray, lookback: Optional[int] = None) -> np.ndarray:
        """
        Normalize raw values to [0, 1] using rolling percentile rank.

        Args:
            raw: Raw indicator values
            lookback: Window size (defaults to lookback_long)

        Returns:
            Normalized values in [0, 1] range
        """
        window = lookback or self.lookback_long

        # Adjust window if data is too short (graceful degradation)
        n_valid = np.sum(~np.isnan(raw))
        if n_valid < window:
            # Use at least 60% of available data, minimum 20 bars
            window = max(20, int(n_valid * 0.6))

        ranks = rolling_percentile_rank(raw, window)
        return ranks / 100.0  # Scale from 0-100 to 0-1

    def normalize_series(self, raw: pd.Series, lookback: Optional[int] = None) -> pd.Series:
        """Normalize pd.Series, preserving index."""
        normalized = self.normalize(raw.values, lookback)
        return pd.Series(normalized, index=raw.index, name=f"{raw.name}_norm")

    def compute_trend_factor(
        self,
        close: np.ndarray,
        ma_short: np.ndarray,
        ma_long: np.ndarray,
    ) -> np.ndarray:
        """
        Compute trend factor from MA relationships.

        Higher score = stronger uptrend.
        Uses EMA difference normalized by percentile rank.
        """
        # Raw: normalized EMA difference
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_diff = (ma_short - ma_long) / np.where(ma_long != 0, ma_long, np.nan)

        return self.normalize(raw_diff, self.lookback_long)

    def compute_momentum_factor(self, rsi: np.ndarray) -> np.ndarray:
        """
        Compute momentum factor from RSI.

        Higher score = more overbought (relative to history).
        NOT raw RSI/100 - uses percentile rank of RSI values.
        """
        return self.normalize(rsi, self.lookback_short)

    def compute_volatility_factor(self, atr_pct: np.ndarray) -> np.ndarray:
        """
        Compute volatility factor from ATR%.

        Higher score = more volatility stress.
        Note: In composite score, this is inverted (1 - vol) so high vol = lower score.
        """
        return self.normalize(atr_pct, self.lookback_short)

    def compute_breadth_factor(
        self,
        asset_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Compute breadth factor from relative returns.

        Higher score = outperforming benchmark.
        """
        relative = asset_returns - benchmark_returns
        return self.normalize(relative, self.lookback_short)

    def compute_macd_factor(self, macd_hist: np.ndarray) -> np.ndarray:
        """
        Compute MACD histogram factor.

        Normalizes MACD histogram to [0, 1] using percentile rank.
        Higher score = more bullish momentum.
        """
        return self.normalize(macd_hist, self.lookback_short)


def compute_normalized_factors(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    lookback_short: int = 63,
    lookback_long: int = 252,
) -> NormalizedFactors:
    """
    Compute all normalized factors from OHLCV DataFrame.

    Args:
        df: DataFrame with OHLCV columns
        benchmark_df: Optional benchmark DataFrame for breadth calculation
        lookback_short: Short lookback window (default 63 bars)
        lookback_long: Long lookback window (default 252 bars)

    Returns:
        NormalizedFactors with all scores in [0, 1] range
    """
    import talib

    normalizer = FactorNormalizer(lookback_short, lookback_long)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Trend (long-term): EMA 20/50 difference
    ema_20 = talib.EMA(close, timeperiod=20)
    ema_50 = talib.EMA(close, timeperiod=50)
    trend = normalizer.compute_trend_factor(close, ema_20, ema_50)

    # Trend (short-term): EMA 10/20 difference - more sensitive to recent changes
    ema_10 = talib.EMA(close, timeperiod=10)
    trend_short = normalizer.compute_trend_factor(close, ema_10, ema_20)

    # Momentum: RSI 14
    rsi = talib.RSI(close, timeperiod=14)
    momentum = normalizer.compute_momentum_factor(rsi)

    # Volatility: ATR 14 as % of price
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_pct = np.where(close != 0, atr / close, 0)
    volatility = normalizer.compute_volatility_factor(atr_pct)

    # Dual MACD: Long (55/89) for trend, Short (13/21) for momentum vs trend
    # Long MACD - trend direction
    ema_55 = talib.EMA(close, timeperiod=55)
    ema_89 = talib.EMA(close, timeperiod=89)
    macd_long = ema_55 - ema_89
    macd_long_signal = talib.EMA(macd_long, timeperiod=9)
    macd_long_hist = macd_long - macd_long_signal
    macd_trend = normalizer.compute_macd_factor(macd_long_hist)

    # Short MACD - momentum relative to trend
    ema_13 = talib.EMA(close, timeperiod=13)
    ema_21 = talib.EMA(close, timeperiod=21)
    macd_short = ema_13 - ema_21
    macd_short_signal = talib.EMA(macd_short, timeperiod=9)
    macd_short_hist = macd_short - macd_short_signal
    macd_momentum = normalizer.compute_macd_factor(macd_short_hist)

    # Breadth: relative return vs benchmark (if provided)
    breadth = None
    if benchmark_df is not None and "close" in benchmark_df.columns:
        asset_ret = pd.Series(close).pct_change(20).values
        # Normalize benchmark index timezone to match df.index
        bench_close = benchmark_df["close"].copy()
        if bench_close.index.tz is not None and df.index.tz is None:
            bench_close.index = bench_close.index.tz_localize(None)
        elif bench_close.index.tz is None and df.index.tz is not None:
            bench_close.index = bench_close.index.tz_localize(df.index.tz)
        bench_ret = bench_close.pct_change(20).reindex(df.index, method="ffill").values
        breadth = normalizer.compute_breadth_factor(asset_ret, bench_ret)
        breadth = pd.Series(breadth, index=df.index, name="breadth")

    return NormalizedFactors(
        trend=pd.Series(trend, index=df.index, name="trend"),
        trend_short=pd.Series(trend_short, index=df.index, name="trend_short"),
        momentum=pd.Series(momentum, index=df.index, name="momentum"),
        volatility=pd.Series(volatility, index=df.index, name="volatility"),
        macd_trend=pd.Series(macd_trend, index=df.index, name="macd_trend"),
        macd_momentum=pd.Series(macd_momentum, index=df.index, name="macd_momentum"),
        breadth=breadth,
    )
