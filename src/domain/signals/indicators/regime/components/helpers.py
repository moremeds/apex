"""
Shared Helper Functions for Regime Component Calculators.

Contains common utility functions used across multiple component calculators
to avoid code duplication and ensure consistent behavior.
"""

from __future__ import annotations

import numpy as np

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


def rolling_percentile_rank(data: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorized rolling percentile rank calculation.

    Computes the percentile rank of the current value within a rolling window.
    This is faster than using pandas apply(percentileofscore) for large datasets.

    Args:
        data: Input array of values
        window: Rolling window size for percentile calculation

    Returns:
        Array of percentile ranks (0-100), with NaN for positions
        where window is not yet filled or data is insufficient.

    Formula:
        percentile_rank = (count of values < current) / (window_valid_count - 1) * 100

    Note:
        Requires at least 2 valid values in the window to avoid division by zero.
        The formula uses (window_valid_count - 1) to normalize to 0-100 range
        where the minimum value gets 0 and maximum gets 100.
    """
    n = len(data)
    result = np.full(n, np.nan)

    if n < window:
        return result

    for i in range(window - 1, n):
        window_data = data[i - window + 1 : i + 1]
        valid_mask = ~np.isnan(window_data)
        valid_count = np.sum(valid_mask)
        # Need at least 2 valid values to avoid division by zero (len-1 >= 1)
        if valid_count >= 2 and not np.isnan(data[i]):
            valid_data = window_data[valid_mask]
            count_less = np.sum(valid_data < data[i])
            # Percentile rank: (values less than current) / (total - 1) * 100
            result[i] = (count_less / (valid_count - 1)) * 100

    return result


def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Calculate True Range for each bar.

    True Range is the maximum of:
    - Current high - current low
    - Abs(current high - previous close)
    - Abs(current low - previous close)

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices

    Returns:
        Array of True Range values
    """
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return tr


def calculate_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Calculate Average True Range.

    Uses TA-Lib if available, otherwise falls back to manual calculation
    using Wilder's smoothing method.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period

    Returns:
        Array of ATR values (NaN for warmup period)
    """
    if HAS_TALIB:
        return talib.ATR(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            timeperiod=period,
        )

    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return atr

    # True Range calculation
    tr = calculate_true_range(high, low, close)

    # Initial ATR = SMA of TR
    atr[period - 1] = np.mean(tr[:period])

    # Subsequent ATR = smoothed (Wilder's smoothing)
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Uses TA-Lib if available for performance, otherwise uses
    cumsum-based vectorized calculation.

    Args:
        data: Input array of values
        period: SMA period

    Returns:
        Array of SMA values (NaN for warmup period)
    """
    if HAS_TALIB:
        return talib.SMA(data.astype(np.float64), timeperiod=period)

    n = len(data)
    result = np.full(n, np.nan)
    if n < period:
        return result

    cumsum = np.cumsum(data)
    result[period - 1 :] = (
        cumsum[period - 1 :] - np.concatenate([[0], cumsum[:-period]])
    ) / period
    return result
