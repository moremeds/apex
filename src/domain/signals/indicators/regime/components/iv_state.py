"""
Implied Volatility State Component Calculator.

Classifies implied volatility using VIX/VXN percentile.
ONLY applicable at MARKET level (QQQ/SPY).

For short put strategies, IV often rises BEFORE realized vol spikes,
making this an important leading indicator for risk management.

Classification:
- IV_HIGH: VIX_pct_63 > 75
- IV_ELEVATED: VIX_pct_63 in [50, 75]
- IV_LOW: VIX_pct_63 < 25
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..models import IVState
from .helpers import rolling_percentile_rank

# IV symbol mapping - use correct IV index per market benchmark
IV_SYMBOL_MAP = {
    "SPY": "VIX",  # VIX is based on S&P 500 options
    "QQQ": "VXN",  # VXN is based on Nasdaq-100/NDX options
    "IWM": "RVX",  # Russell 2000 volatility
    "DIA": "VXD",  # DJIA volatility
}


def get_iv_symbol(market_symbol: str) -> str:
    """
    Get the correct IV index for a market benchmark.

    Args:
        market_symbol: Market ETF symbol (SPY, QQQ, etc.)

    Returns:
        IV index symbol (VIX, VXN, etc.) or VIX as default
    """
    return IV_SYMBOL_MAP.get(market_symbol.upper(), "VIX")


def calculate_iv_state(
    iv_data: Optional[np.ndarray],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool]:
    """
    Calculate IVState using VIX/VXN percentile.

    FALLBACK LOGIC:
        1. Try VIX/VXN data from data feed
        2. If unavailable: Return NA state (don't use fake data)

    Args:
        iv_data: Array of VIX/VXN values (may be None or empty)
        params: Parameters dict with:
            - iv_pct_window: Percentile window (default 63)
            - iv_high_pct: High IV threshold (default 75)
            - iv_elevated_pct: Elevated IV threshold (default 50)
            - iv_low_pct: Low IV threshold (default 25)

    Returns:
        Tuple of:
        - iv_state: Array of IVState enum values
        - details: Dict with intermediate values (iv_pct_63)
        - is_available: Whether IV data is available
    """
    pct_window = params.get("iv_pct_window", 63)
    iv_high = params.get("iv_high_pct", 75)
    iv_elevated = params.get("iv_elevated_pct", 50)
    iv_low = params.get("iv_low_pct", 25)

    # Check if IV data is available
    if iv_data is None or len(iv_data) == 0:
        return np.array([IVState.NA]), {"iv_pct_63": np.array([np.nan])}, False

    n = len(iv_data)
    if n < pct_window:
        # Not enough data for percentile calculation
        iv_state = np.array([IVState.NA] * n, dtype=object)
        iv_pct_63 = np.full(n, np.nan)
        return iv_state, {"iv_pct_63": iv_pct_63}, False

    # Initialize outputs
    iv_state = np.array([IVState.NA] * n, dtype=object)

    # Calculate rolling percentile rank
    iv_pct_63 = rolling_percentile_rank(iv_data, pct_window)

    # Classify IV state
    for i in range(n):
        if np.isnan(iv_pct_63[i]):
            iv_state[i] = IVState.NA
            continue

        if iv_pct_63[i] > iv_high:
            iv_state[i] = IVState.HIGH
        elif iv_pct_63[i] > iv_elevated:
            iv_state[i] = IVState.ELEVATED
        elif iv_pct_63[i] < iv_low:
            iv_state[i] = IVState.LOW
        else:
            # Between LOW and ELEVATED thresholds (25-50)
            iv_state[i] = IVState.NORMAL

    return iv_state, {"iv_pct_63": iv_pct_63}, True
