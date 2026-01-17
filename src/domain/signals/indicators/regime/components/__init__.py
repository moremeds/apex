"""
Regime Detection Component Calculators.

Each component calculates a specific aspect of market state:
- TrendState: Price vs MA relationships and slope
- VolState: Dual-window ATR percentile volatility
- IVState: Implied volatility (VIX/VXN) for market level
- ChopState: Choppiness index and MA20 crosses
- ExtState: Extension from mean (distance from MA20)

Shared helpers available:
- rolling_percentile_rank: Vectorized percentile rank calculation
- calculate_true_range: True Range calculation
- calculate_atr: Average True Range (TA-Lib or fallback)
- calculate_sma: Simple Moving Average (TA-Lib or fallback)
"""

from .chop_state import calculate_chop_state
from .ext_state import calculate_ext_state
from .helpers import (
    calculate_atr,
    calculate_sma,
    calculate_true_range,
    rolling_percentile_rank,
)
from .iv_state import calculate_iv_state, get_iv_symbol
from .trend_state import calculate_trend_state
from .vol_state import calculate_vol_state

__all__ = [
    "calculate_trend_state",
    "calculate_vol_state",
    "calculate_iv_state",
    "get_iv_symbol",
    "calculate_chop_state",
    "calculate_ext_state",
    # Shared helpers
    "rolling_percentile_rank",
    "calculate_true_range",
    "calculate_atr",
    "calculate_sma",
]
