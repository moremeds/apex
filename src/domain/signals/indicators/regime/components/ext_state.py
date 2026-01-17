"""
Extension State Component Calculator.

Classifies how extended price is from its mean:
- ext = (close - MA20) / ATR20

Classification:
- OVERBOUGHT: ext > 2.0
- OVERSOLD: ext < -2.0
- SLIGHTLY_HIGH: ext > 1.5
- SLIGHTLY_LOW: ext < -1.5
- NEUTRAL: otherwise

This measures "how far" price has moved from its mean in ATR terms,
helping identify overextended conditions that may revert.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..models import ExtState
from .helpers import calculate_atr, calculate_sma


def calculate_ext_state(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculate ExtState for each bar.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        params: Parameters dict with:
            - ma20_period: MA period (default 20)
            - atr_period: ATR period (default 20)
            - ext_overbought: Overbought threshold (default 2.0)
            - ext_oversold: Oversold threshold (default -2.0)
            - ext_slightly_high: Slightly high threshold (default 1.5)
            - ext_slightly_low: Slightly low threshold (default -1.5)

    Returns:
        Tuple of:
        - ext_state: Array of ExtState enum values
        - details: Dict with intermediate values (ext, ma20, atr20)
    """
    n = len(close)
    ma20_period = params.get("ma20_period", 20)
    atr_period = params.get("atr_period", 20)
    overbought = params.get("ext_overbought", 2.0)
    oversold = params.get("ext_oversold", -2.0)
    slightly_high = params.get("ext_slightly_high", 1.5)
    slightly_low = params.get("ext_slightly_low", -1.5)

    # Initialize outputs
    ext_state = np.array([ExtState.NEUTRAL] * n, dtype=object)
    ext = np.full(n, np.nan)
    ma20 = np.full(n, np.nan)
    atr20 = np.full(n, np.nan)

    if n < max(ma20_period, atr_period):
        return ext_state, {"ext": ext, "ma20": ma20, "atr20": atr20}

    # Calculate MA20 (uses TA-Lib if available, fallback otherwise)
    ma20 = calculate_sma(close, ma20_period)

    # Calculate ATR20 (uses TA-Lib if available, fallback otherwise)
    atr20 = calculate_atr(high, low, close, atr_period)

    # Calculate extension: (close - MA20) / ATR20
    for i in range(n):
        if not np.isnan(ma20[i]) and not np.isnan(atr20[i]) and atr20[i] > 0:
            ext[i] = (close[i] - ma20[i]) / atr20[i]

    # Classify extension state
    for i in range(n):
        if np.isnan(ext[i]):
            ext_state[i] = ExtState.NEUTRAL
            continue

        if ext[i] > overbought:
            ext_state[i] = ExtState.OVERBOUGHT
        elif ext[i] < oversold:
            ext_state[i] = ExtState.OVERSOLD
        elif ext[i] > slightly_high:
            ext_state[i] = ExtState.SLIGHTLY_HIGH
        elif ext[i] < slightly_low:
            ext_state[i] = ExtState.SLIGHTLY_LOW
        else:
            ext_state[i] = ExtState.NEUTRAL

    return ext_state, {"ext": ext, "ma20": ma20, "atr20": atr20}
