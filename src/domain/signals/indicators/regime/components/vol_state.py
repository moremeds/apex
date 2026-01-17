"""
Volatility State Component Calculator.

Classifies realized volatility using dual-window ATR percentiles:
- HighVol: ATR_pct_63 > 80 OR ATR_pct_252 > 85
- LowVol: ATR_pct_63 < 20 AND ATR_pct_252 < 25
- NormalVol: otherwise

Uses dual-window approach to capture both short-term spikes and
longer-term regime shifts in volatility.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..models import VolState
from .helpers import calculate_atr, rolling_percentile_rank


def calculate_vol_state(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate VolState for each bar using dual-window ATR percentiles.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        params: Parameters dict with:
            - atr_period: Period for ATR calculation (default 20)
            - atr_pct_short_window: Short percentile window (default 63, ~3 months)
            - atr_pct_long_window: Long percentile window (default 252, ~1 year)
            - vol_high_short_pct: High vol threshold for short window (default 80)
            - vol_high_long_pct: High vol threshold for long window (default 85)
            - vol_low_pct: Low vol threshold (default 20)

    Returns:
        Tuple of:
        - vol_state: Array of VolState enum values
        - details: Dict with intermediate values (atr, atr_pct, atr_pct_63, atr_pct_252)
    """
    n = len(close)
    atr_period = params.get("atr_period", 20)
    short_window = params.get("atr_pct_short_window", 63)
    long_window = params.get("atr_pct_long_window", 252)
    vol_high_short = params.get("vol_high_short_pct", 80)
    vol_high_long = params.get("vol_high_long_pct", 85)
    vol_low = params.get("vol_low_pct", 20)

    # Initialize outputs
    vol_state = np.array([VolState.NORMAL] * n, dtype=object)
    atr = np.full(n, np.nan)
    atr_pct = np.full(n, np.nan)  # ATR as % of close
    atr_pct_63 = np.full(n, 50.0)  # Percentile (default to 50)
    atr_pct_252 = np.full(n, 50.0)

    if n < atr_period:
        return vol_state, {
            "atr": atr,
            "atr_pct": atr_pct,
            "atr_pct_63": atr_pct_63,
            "atr_pct_252": atr_pct_252,
        }

    # Calculate ATR (uses TA-Lib if available, fallback otherwise)
    atr = calculate_atr(high, low, close, atr_period)

    # Calculate ATR as percentage of close
    for i in range(n):
        if not np.isnan(atr[i]) and close[i] > 0:
            atr_pct[i] = atr[i] / close[i]

    # Calculate rolling percentiles
    atr_pct_63 = rolling_percentile_rank(atr, short_window)
    atr_pct_252 = rolling_percentile_rank(atr, long_window)

    # Classify volatility state
    for i in range(n):
        if np.isnan(atr_pct_63[i]) or np.isnan(atr_pct_252[i]):
            vol_state[i] = VolState.NORMAL
            continue

        # HighVol: ATR_pct_63 > 80 OR ATR_pct_252 > 85
        if atr_pct_63[i] > vol_high_short or atr_pct_252[i] > vol_high_long:
            vol_state[i] = VolState.HIGH
        # LowVol: ATR_pct_63 < 20 AND ATR_pct_252 < 25
        elif atr_pct_63[i] < vol_low and atr_pct_252[i] < vol_low + 5:
            vol_state[i] = VolState.LOW
        else:
            vol_state[i] = VolState.NORMAL

    return vol_state, {
        "atr": atr,
        "atr_pct": atr_pct,
        "atr_pct_63": atr_pct_63,
        "atr_pct_252": atr_pct_252,
    }
