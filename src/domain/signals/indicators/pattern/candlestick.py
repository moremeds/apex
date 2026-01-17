"""
Candlestick Patterns Indicator.

Detects Japanese candlestick patterns using TA-Lib.
Supports 61 patterns including:
- Reversal: Hammer, Doji, Engulfing, Morning Star, Evening Star
- Continuation: Three White Soldiers, Rising Three Methods
- Indecision: Spinning Top, High Wave

Each pattern returns:
- 100: Bullish pattern
- -100: Bearish pattern
- 0: No pattern detected
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set

import numpy as np
import pandas as pd

try:
    import talib

    HAS_TALIB = True

    # All 61 TA-Lib candlestick pattern functions
    PATTERN_FUNCTIONS: Dict[str, Callable[..., np.ndarray]] = {
        # Single candle patterns
        "doji": talib.CDLDOJI,
        "doji_star": talib.CDLDOJISTAR,
        "dragonfly_doji": talib.CDLDRAGONFLYDOJI,
        "gravestone_doji": talib.CDLGRAVESTONEDOJI,
        "long_legged_doji": talib.CDLLONGLEGGEDDOJI,
        "hammer": talib.CDLHAMMER,
        "inverted_hammer": talib.CDLINVERTEDHAMMER,
        "hanging_man": talib.CDLHANGINGMAN,
        "shooting_star": talib.CDLSHOOTINGSTAR,
        "marubozu": talib.CDLMARUBOZU,
        "closing_marubozu": talib.CDLCLOSINGMARUBOZU,
        "spinning_top": talib.CDLSPINNINGTOP,
        "high_wave": talib.CDLHIGHWAVE,
        "long_line": talib.CDLLONGLINE,
        "short_line": talib.CDLSHORTLINE,
        "rickshaw_man": talib.CDLRICKSHAWMAN,
        "takuri": talib.CDLTAKURI,
        # Two candle patterns
        "engulfing": talib.CDLENGULFING,
        "harami": talib.CDLHARAMI,
        "harami_cross": talib.CDLHARAMICROSS,
        "piercing": talib.CDLPIERCING,
        "dark_cloud_cover": talib.CDLDARKCLOUDCOVER,
        "kicking": talib.CDLKICKING,
        "kicking_by_length": talib.CDLKICKINGBYLENGTH,
        "belt_hold": talib.CDLBELTHOLD,
        "counterattack": talib.CDLCOUNTERATTACK,
        "matching_low": talib.CDLMATCHINGLOW,
        "homing_pigeon": talib.CDLHOMINGPIGEON,
        "on_neck": talib.CDLONNECK,
        "in_neck": talib.CDLINNECK,
        "thrusting": talib.CDLTHRUSTING,
        "separating_lines": talib.CDLSEPARATINGLINES,
        "two_crows": talib.CDL2CROWS,
        "hikkake": talib.CDLHIKKAKE,
        "hikkake_mod": talib.CDLHIKKAKEMOD,
        # Three candle patterns
        "morning_star": talib.CDLMORNINGSTAR,
        "evening_star": talib.CDLEVENINGSTAR,
        "morning_doji_star": talib.CDLMORNINGDOJISTAR,
        "evening_doji_star": talib.CDLEVENINGDOJISTAR,
        "three_white_soldiers": talib.CDL3WHITESOLDIERS,
        "three_black_crows": talib.CDL3BLACKCROWS,
        "three_inside": talib.CDL3INSIDE,
        "three_outside": talib.CDL3OUTSIDE,
        "three_line_strike": talib.CDL3LINESTRIKE,
        "three_stars_in_south": talib.CDL3STARSINSOUTH,
        "identical_three_crows": talib.CDLIDENTICAL3CROWS,
        "abandoned_baby": talib.CDLABANDONEDBABY,
        "advance_block": talib.CDLADVANCEBLOCK,
        "stalled_pattern": talib.CDLSTALLEDPATTERN,
        "stick_sandwich": talib.CDLSTICKSANDWICH,
        "tristar": talib.CDLTRISTAR,
        "unique_three_river": talib.CDLUNIQUE3RIVER,
        "upside_gap_two_crows": talib.CDLUPSIDEGAP2CROWS,
        "tasuki_gap": talib.CDLTASUKIGAP,
        "gap_side_side_white": talib.CDLGAPSIDESIDEWHITE,
        # Multi-candle patterns
        "breakaway": talib.CDLBREAKAWAY,
        "concealing_baby_swallow": talib.CDLCONCEALBABYSWALL,
        "ladder_bottom": talib.CDLLADDERBOTTOM,
        "mat_hold": talib.CDLMATHOLD,
        "rise_fall_three_methods": talib.CDLRISEFALL3METHODS,
        "xside_gap_three_methods": talib.CDLXSIDEGAP3METHODS,
    }

    # Patterns that require the 'penetration' parameter
    # Only star patterns and abandoned_baby actually accept penetration in TA-Lib
    PENETRATION_PATTERNS = {
        "morning_star",
        "evening_star",
        "morning_doji_star",
        "evening_doji_star",
        "abandoned_baby",
    }
except ImportError:
    HAS_TALIB = False
    PATTERN_FUNCTIONS = {}  # type: ignore[misc]
    PENETRATION_PATTERNS = set()  # type: ignore[misc]

from ...models import SignalCategory
from ..base import IndicatorBase


class CandlestickPatternsIndicator(IndicatorBase):
    """
    Candlestick Pattern detector using TA-Lib.

    Supports all 61 TA-Lib candlestick patterns including:
    - Single candle: Doji, Hammer, Marubozu, Spinning Top, etc.
    - Two candle: Engulfing, Harami, Piercing, Dark Cloud Cover, etc.
    - Three candle: Morning/Evening Star, Three White Soldiers, etc.
    - Multi-candle: Breakaway, Rising/Falling Three Methods, etc.

    Default Parameters:
        patterns: List of patterns to detect (default: all 61 patterns)
        penetration: Penetration factor for star patterns (default: 0.3)

    State Output:
        bullish_patterns: List of detected bullish patterns
        bearish_patterns: List of detected bearish patterns
        strongest_signal: "bullish", "bearish", or "neutral"
        pattern_count: Number of patterns detected
    """

    name = "candlestick"
    category = SignalCategory.PATTERN
    required_fields = ["open", "high", "low", "close"]
    warmup_periods = 10

    _default_params = {
        "patterns": list(PATTERN_FUNCTIONS.keys()) if HAS_TALIB else [],
        "penetration": 0.3,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate candlestick pattern signals."""
        patterns_to_check = params.get("patterns", list(PATTERN_FUNCTIONS.keys()))
        penetration = params.get("penetration", 0.3)

        if len(data) == 0 or not HAS_TALIB:
            return pd.DataFrame(
                {f"cdl_{p}": pd.Series(dtype=float) for p in patterns_to_check},
                index=data.index,
            )

        open_p = data["open"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        results = {}
        for pattern_name in patterns_to_check:
            if pattern_name in PATTERN_FUNCTIONS:
                func = PATTERN_FUNCTIONS[pattern_name]
                if pattern_name in PENETRATION_PATTERNS:
                    results[f"cdl_{pattern_name}"] = func(
                        open_p, high, low, close, penetration=penetration
                    )
                else:
                    results[f"cdl_{pattern_name}"] = func(open_p, high, low, close)

        return pd.DataFrame(results, index=data.index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract candlestick pattern state for rule evaluation."""
        bullish_patterns = []
        bearish_patterns = []

        for col, value in current.items():
            if col.startswith("cdl_") and not pd.isna(value):
                pattern_name = col[4:]  # Remove "cdl_" prefix
                if value > 0:
                    bullish_patterns.append(pattern_name)
                elif value < 0:
                    bearish_patterns.append(pattern_name)

        pattern_count = len(bullish_patterns) + len(bearish_patterns)

        if len(bullish_patterns) > len(bearish_patterns):
            strongest_signal = "bullish"
        elif len(bearish_patterns) > len(bullish_patterns):
            strongest_signal = "bearish"
        else:
            strongest_signal = "neutral"

        return {
            "bullish_patterns": bullish_patterns,
            "bearish_patterns": bearish_patterns,
            "strongest_signal": strongest_signal,
            "pattern_count": pattern_count,
        }
