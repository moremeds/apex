"""
TA-Lib indicator wrappers for signal generation.

All indicators return pandas Series aligned to input index.
These wrappers should be used by both:
- SignalGenerator implementations (vectorized)
- Event-driven Strategy implementations (for parity)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

try:
    import talib
except ImportError as exc:
    raise ImportError(
        "TA-Lib is required for indicators. Install with: pip install TA-Lib"
    ) from exc


def _to_array(series: "pd.Series[float]") -> np.ndarray:  # type: ignore[type-arg]
    """Convert pandas Series to float64 numpy array for TA-Lib."""
    arr: np.ndarray = series.to_numpy(dtype="float64", copy=False)
    return arr


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average via TA-Lib.

    Args:
        series: Price series (typically close prices).
        period: Lookback period.

    Returns:
        SMA series aligned to input index. First (period-1) values are NaN.
    """
    values = talib.SMA(_to_array(series), timeperiod=period)
    return pd.Series(values, index=series.index)


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average via TA-Lib.

    Args:
        series: Price series.
        period: Lookback period.

    Returns:
        EMA series aligned to input index.
    """
    values = talib.EMA(_to_array(series), timeperiod=period)
    return pd.Series(values, index=series.index)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index via TA-Lib.

    Args:
        series: Price series.
        period: Lookback period (default 14).

    Returns:
        RSI series (0-100 range) aligned to input index.
    """
    values = talib.RSI(_to_array(series), timeperiod=period)
    return pd.Series(values, index=series.index)


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence via TA-Lib.

    Args:
        series: Price series.
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) Series.
    """
    macd_line, signal_line, hist = talib.MACD(
        _to_array(series),
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period,
    )
    index = series.index
    return (
        pd.Series(macd_line, index=index),
        pd.Series(signal_line, index=index),
        pd.Series(hist, index=index),
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range via TA-Lib.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback period (default 14).

    Returns:
        ATR series aligned to close.index.
    """
    values = talib.ATR(
        _to_array(high),
        _to_array(low),
        _to_array(close),
        timeperiod=period,
    )
    return pd.Series(values, index=close.index)


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Momentum indicator via TA-Lib.

    Args:
        series: Price series.
        period: Lookback period (default 10).

    Returns:
        Momentum series (current - period ago) aligned to input index.
    """
    values = talib.MOM(_to_array(series), timeperiod=period)
    return pd.Series(values, index=series.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index via TA-Lib.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback period (default 14).

    Returns:
        ADX series (0-100 range) aligned to close.index.
    """
    values = talib.ADX(
        _to_array(high),
        _to_array(low),
        _to_array(close),
        timeperiod=period,
    )
    return pd.Series(values, index=close.index)


def bbands(
    series: pd.Series,
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands via TA-Lib.

    Args:
        series: Price series.
        period: Lookback period (default 20).
        nbdevup: Upper band std dev multiplier (default 2.0).
        nbdevdn: Lower band std dev multiplier (default 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band) Series.
    """
    upper, middle, lower = talib.BBANDS(
        _to_array(series),
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
    )
    index = series.index
    return (
        pd.Series(upper, index=index),
        pd.Series(middle, index=index),
        pd.Series(lower, index=index),
    )
