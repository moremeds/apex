"""
Trend State Component Calculator.

Classifies trend based on MA relationships and slope:
- TrendUp: close > MA200 AND MA50_slope > 0 AND close > MA50
- TrendDown: close < MA200 AND MA50_slope < 0
- Neutral: otherwise
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..models import TrendState
from .helpers import calculate_sma


def calculate_trend_state(
    close: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate TrendState for each bar.

    Args:
        close: Array of close prices
        params: Parameters dict with:
            - ma50_period: Period for MA50 (default 50)
            - ma200_period: Period for MA200 (default 200)
            - slope_lookback: Bars for slope calculation (default 20)

    Returns:
        Tuple of:
        - trend_state: Array of TrendState enum values
        - details: Dict with intermediate values (ma50, ma200, ma50_slope)
    """
    n = len(close)
    ma50_period = params.get("ma50_period", 50)
    ma200_period = params.get("ma200_period", 200)
    slope_lookback = params.get("slope_lookback", 20)

    # Initialize outputs
    trend_state = np.array([TrendState.NEUTRAL] * n, dtype=object)
    ma50 = np.full(n, np.nan)
    ma200 = np.full(n, np.nan)
    ma50_slope = np.full(n, np.nan)

    if n < ma200_period:
        return trend_state, {"ma50": ma50, "ma200": ma200, "ma50_slope": ma50_slope}

    # Calculate MAs (uses TA-Lib if available, fallback otherwise)
    ma50 = calculate_sma(close, ma50_period)
    ma200 = calculate_sma(close, ma200_period)

    # Calculate MA50 slope (rate of change over slope_lookback bars)
    # Slope = (MA50[i] - MA50[i - slope_lookback]) / MA50[i - slope_lookback]
    for i in range(slope_lookback, n):
        if not np.isnan(ma50[i]) and not np.isnan(ma50[i - slope_lookback]):
            if ma50[i - slope_lookback] != 0:
                ma50_slope[i] = (ma50[i] - ma50[i - slope_lookback]) / ma50[i - slope_lookback]

    # Classify trend state
    for i in range(n):
        if np.isnan(ma50[i]) or np.isnan(ma200[i]) or np.isnan(ma50_slope[i]):
            trend_state[i] = TrendState.NEUTRAL
            continue

        # TrendUp: close > MA200 AND MA50_slope > 0 AND close > MA50
        if close[i] > ma200[i] and ma50_slope[i] > 0 and close[i] > ma50[i]:
            trend_state[i] = TrendState.UP
        # TrendDown: close < MA200 AND MA50_slope < 0
        elif close[i] < ma200[i] and ma50_slope[i] < 0:
            trend_state[i] = TrendState.DOWN
        else:
            trend_state[i] = TrendState.NEUTRAL

    return trend_state, {"ma50": ma50, "ma200": ma200, "ma50_slope": ma50_slope}
