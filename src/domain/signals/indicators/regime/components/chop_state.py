"""
Choppiness State Component Calculator.

Classifies market choppiness using CHOP index percentile and MA20 crosses:
- CHOPPY: CHOP_pct_252 > 70 OR MA20_crosses >= 4 (in last 10 bars)
- TRENDING: CHOP_pct_252 < 30 AND MA20_crosses <= 1
- NEUTRAL: otherwise

The Choppiness Index (CHOP) measures the degree to which a market is
trending vs. consolidating. Values near 100 indicate extreme consolidation,
while values near 0 indicate strong trending.

Note: 61.8/38.2 Fibonacci levels are traditional thresholds but we use
percentile-based classification for better regime detection.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..models import ChopState
from .helpers import calculate_sma, calculate_true_range, rolling_percentile_rank


def calculate_chop_state(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate ChopState for each bar.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        params: Parameters dict with:
            - chop_period: CHOP calculation period (default 14)
            - chop_pct_window: Percentile window (default 252)
            - ma20_period: MA period for cross counting (default 20)
            - ma20_cross_lookback: Bars to look back for crosses (default 10)
            - chop_high_pct: High chop threshold (default 70)
            - chop_low_pct: Low chop threshold (default 30)
            - chop_cross_high: Cross count for choppy (default 4)
            - chop_cross_low: Cross count for trending (default 1)

    Returns:
        Tuple of:
        - chop_state: Array of ChopState enum values
        - details: Dict with intermediate values (chop, chop_pct_252, ma20, ma20_crosses)
    """
    n = len(close)
    chop_period = params.get("chop_period", 14)
    chop_pct_window = params.get("chop_pct_window", 252)
    ma20_period = params.get("ma20_period", 20)
    cross_lookback = params.get("ma20_cross_lookback", 10)
    chop_high = params.get("chop_high_pct", 70)
    chop_low = params.get("chop_low_pct", 30)
    cross_high = params.get("chop_cross_high", 4)
    cross_low = params.get("chop_cross_low", 1)

    # Initialize outputs
    chop_state = np.array([ChopState.NEUTRAL] * n, dtype=object)
    chop = np.full(n, np.nan)
    chop_pct_252 = np.full(n, 50.0)
    ma20 = np.full(n, np.nan)
    ma20_crosses = np.zeros(n, dtype=int)

    if n < max(chop_period, ma20_period):
        return chop_state, {
            "chop": chop,
            "chop_pct_252": chop_pct_252,
            "ma20": ma20,
            "ma20_crosses": ma20_crosses,
        }

    # Calculate Choppiness Index
    chop = _calculate_chop(high, low, close, chop_period)

    # Calculate CHOP percentile
    chop_pct_252 = rolling_percentile_rank(chop, chop_pct_window)

    # Calculate MA20 (uses TA-Lib if available, fallback otherwise)
    ma20 = calculate_sma(close, ma20_period)

    # Count MA20 crosses in lookback window
    ma20_crosses = _count_ma_crosses(close, ma20, cross_lookback)

    # Classify choppiness state
    for i in range(n):
        if np.isnan(chop_pct_252[i]):
            chop_state[i] = ChopState.NEUTRAL
            continue

        # CHOPPY: CHOP_pct_252 > 70 OR MA20_crosses >= 4
        if chop_pct_252[i] > chop_high or ma20_crosses[i] >= cross_high:
            chop_state[i] = ChopState.CHOPPY
        # TRENDING: CHOP_pct_252 < 30 AND MA20_crosses <= 1
        elif chop_pct_252[i] < chop_low and ma20_crosses[i] <= cross_low:
            chop_state[i] = ChopState.TRENDING
        else:
            chop_state[i] = ChopState.NEUTRAL

    return chop_state, {
        "chop": chop,
        "chop_pct_252": chop_pct_252,
        "ma20": ma20,
        "ma20_crosses": ma20_crosses,
    }


def _calculate_chop(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """
    Calculate Choppiness Index.

    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)

    Values:
    - Near 100: Market is very choppy (consolidating)
    - Near 0: Market is strongly trending
    """
    n = len(close)
    chop = np.full(n, np.nan)

    if n < period:
        return chop

    # Calculate True Range using shared helper
    tr = calculate_true_range(high, low, close)

    # Calculate CHOP
    log_period = np.log10(period)
    for i in range(period - 1, n):
        # Sum of ATR over period
        atr_sum = np.sum(tr[i - period + 1 : i + 1])

        # Highest high and lowest low over period
        hh = np.max(high[i - period + 1 : i + 1])
        ll = np.min(low[i - period + 1 : i + 1])

        # Avoid division by zero
        range_hl = hh - ll
        if range_hl > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_hl) / log_period

    return chop


def _count_ma_crosses(
    close: np.ndarray,
    ma: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Count the number of times close crosses MA in the lookback window.

    A cross occurs when close goes from above MA to below (or vice versa).
    """
    n = len(close)
    crosses = np.zeros(n, dtype=int)

    if n < 2:
        return crosses

    # Determine if close is above or below MA
    above_ma = close > ma

    # Count crosses
    for i in range(lookback, n):
        count = 0
        for j in range(i - lookback + 1, i + 1):
            if j > 0 and not np.isnan(ma[j]) and not np.isnan(ma[j - 1]):
                # Check for cross
                if above_ma[j] != above_ma[j - 1]:
                    count += 1
        crosses[i] = count

    return crosses
