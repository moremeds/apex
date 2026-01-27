"""
DualMACD Indicator - Overlapping Long and Short MACD Histograms.

Philosophy:
- Long MACD (55/89 EMA): Shows trend direction, strength, and divergence rhythm
- Short MACD (13/21 EMA): Shows overbought/oversold within trend, micro entry signals
- Overlaying both reveals relative strength and avoids misjudging trend strength

Signals:
- Trend Confirmation: Both histograms positive/negative
- Momentum Divergence: Short histogram diverges from long (early reversal warning)
- Overbought in Trend: Long positive, short stretched relative to long
- Entry Timing: Short histogram crossing zero while long maintains direction
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from ...models import SignalCategory
from ..base import IndicatorBase


class DualMACDIndicator(IndicatorBase):
    """
    Dual MACD indicator with overlapping long and short timeframe histograms.

    Long MACD (55/89/9): Trend direction indicator
    Short MACD (13/21/9): Momentum timing indicator

    Default Parameters:
        long_fast: 55 (Fibonacci)
        long_slow: 89 (Fibonacci)
        short_fast: 13 (Fibonacci)
        short_slow: 21 (Fibonacci)
        signal_period: 9

    State Output:
        long_macd: Long MACD line (EMA55 - EMA89)
        long_signal: Long signal line
        long_histogram: Long MACD histogram (trend direction)
        short_macd: Short MACD line (EMA13 - EMA21)
        short_signal: Short signal line
        short_histogram: Short MACD histogram (momentum timing)
        relative_strength: short_histogram / max(abs(long_histogram), 0.001)
        trend_direction: "bullish", "bearish", or "neutral"
        momentum_state: "aligned", "diverging", or "neutral"
        signal_type: Specific signal type if conditions met
    """

    name = "dual_macd"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 98  # long_slow + signal_period

    _default_params = {
        "long_fast": 55,
        "long_slow": 89,
        "short_fast": 13,
        "short_slow": 21,
        "signal_period": 9,
        "histogram_multiplier": 2,  # For visualization scaling
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate dual MACD lines, signals, and histograms."""
        long_fast = params["long_fast"]
        long_slow = params["long_slow"]
        short_fast = params["short_fast"]
        short_slow = params["short_slow"]
        signal_period = params["signal_period"]
        multiplier = params["histogram_multiplier"]

        n = len(data)
        if n == 0:
            return pd.DataFrame(
                {
                    "long_macd": pd.Series(dtype=float),
                    "long_signal": pd.Series(dtype=float),
                    "long_histogram": pd.Series(dtype=float),
                    "short_macd": pd.Series(dtype=float),
                    "short_signal": pd.Series(dtype=float),
                    "short_histogram": pd.Series(dtype=float),
                    "relative_strength": pd.Series(dtype=float),
                },
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)

        if HAS_TALIB:
            # Long MACD (55/89)
            ema_long_fast = talib.EMA(close, timeperiod=long_fast)
            ema_long_slow = talib.EMA(close, timeperiod=long_slow)
            long_macd = ema_long_fast - ema_long_slow
            long_signal = talib.EMA(long_macd, timeperiod=signal_period)
            long_histogram = multiplier * (long_macd - long_signal)

            # Short MACD (13/21)
            ema_short_fast = talib.EMA(close, timeperiod=short_fast)
            ema_short_slow = talib.EMA(close, timeperiod=short_slow)
            short_macd = ema_short_fast - ema_short_slow
            short_signal = talib.EMA(short_macd, timeperiod=signal_period)
            short_histogram = multiplier * (short_macd - short_signal)
        else:
            long_macd, long_signal, long_histogram = self._calculate_macd_manual(
                close, long_fast, long_slow, signal_period, multiplier
            )
            short_macd, short_signal, short_histogram = self._calculate_macd_manual(
                close, short_fast, short_slow, signal_period, multiplier
            )

        # Relative strength: shows how extended short histogram is vs long
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_strength = np.where(
                np.abs(long_histogram) > 0.001,
                short_histogram / np.abs(long_histogram),
                np.where(short_histogram > 0, 100, np.where(short_histogram < 0, -100, 0)),
            )

        return pd.DataFrame(
            {
                "long_macd": long_macd,
                "long_signal": long_signal,
                "long_histogram": long_histogram,
                "short_macd": short_macd,
                "short_signal": short_signal,
                "short_histogram": short_histogram,
                "relative_strength": relative_strength,
            },
            index=data.index,
        )

    def _calculate_macd_manual(
        self,
        close: np.ndarray,
        fast: int,
        slow: int,
        signal: int,
        multiplier: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD without TA-Lib."""
        n = len(close)

        def ema(arr: np.ndarray, period: int) -> np.ndarray:
            result = np.full(n, np.nan, dtype=np.float64)
            if n < period:
                return result
            alpha = 2.0 / (period + 1)
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, n):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
            return result

        fast_ema = ema(close, fast)
        slow_ema = ema(close, slow)
        macd_line = fast_ema - slow_ema

        # Signal line is EMA of MACD
        signal_line = np.full(n, np.nan, dtype=np.float64)
        valid_start = slow - 1 + signal - 1
        if n > valid_start:
            alpha = 2.0 / (signal + 1)
            signal_line[valid_start] = np.nanmean(macd_line[slow - 1 : valid_start + 1])
            for i in range(valid_start + 1, n):
                if not np.isnan(macd_line[i]):
                    signal_line[i] = alpha * macd_line[i] + (1 - alpha) * signal_line[i - 1]

        histogram = multiplier * (macd_line - signal_line)
        return macd_line, signal_line, histogram

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract DualMACD state for rule evaluation."""
        long_hist = current.get("long_histogram", 0)
        short_hist = current.get("short_histogram", 0)
        relative_strength = current.get("relative_strength", 0)

        if pd.isna(long_hist) or pd.isna(short_hist):
            return {
                "long_macd": 0,
                "long_signal": 0,
                "long_histogram": 0,
                "short_macd": 0,
                "short_signal": 0,
                "short_histogram": 0,
                "relative_strength": 0,
                "trend_direction": "neutral",
                "momentum_state": "neutral",
                "signal_type": None,
            }

        # Determine trend direction from long MACD
        if long_hist > 0:
            trend_direction = "bullish"
        elif long_hist < 0:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        # Determine momentum state (aligned or diverging)
        if (long_hist > 0 and short_hist > 0) or (long_hist < 0 and short_hist < 0):
            momentum_state = "aligned"
        elif (long_hist > 0 and short_hist < 0) or (long_hist < 0 and short_hist > 0):
            momentum_state = "diverging"
        else:
            momentum_state = "neutral"

        # Detect specific signal types
        signal_type = self._detect_signal_type(
            current, previous, trend_direction, momentum_state, relative_strength
        )

        return {
            "long_macd": float(current.get("long_macd", 0)),
            "long_signal": float(current.get("long_signal", 0)),
            "long_histogram": float(long_hist),
            "short_macd": float(current.get("short_macd", 0)),
            "short_signal": float(current.get("short_signal", 0)),
            "short_histogram": float(short_hist),
            "relative_strength": float(relative_strength),
            "trend_direction": trend_direction,
            "momentum_state": momentum_state,
            "signal_type": signal_type,
        }

    def _detect_signal_type(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        trend_direction: str,
        momentum_state: str,
        relative_strength: float,
    ) -> Optional[str]:
        """
        Detect specific dual MACD signal types.

        Signal Types:
        - trend_confirmation: Both histograms confirm same direction
        - momentum_divergence: Short diverges from long (early warning)
        - overbought_in_uptrend: Long bullish, short extremely stretched
        - oversold_in_downtrend: Long bearish, short extremely stretched
        - bullish_momentum_entry: Short crosses up while long positive
        - bearish_momentum_entry: Short crosses down while long negative
        """
        if previous is None:
            return None

        long_hist = float(current.get("long_histogram", 0))
        short_hist = float(current.get("short_histogram", 0))
        prev_short_hist = float(previous.get("short_histogram", 0)) if previous is not None else 0

        # Trend confirmation with both aligned
        if momentum_state == "aligned":
            if long_hist > 0 and short_hist > 0:
                return "trend_confirmation_bullish"
            elif long_hist < 0 and short_hist < 0:
                return "trend_confirmation_bearish"

        # Early warning: momentum diverging from trend
        if momentum_state == "diverging":
            if long_hist > 0 and short_hist < 0:
                return "momentum_divergence_bearish_warning"
            elif long_hist < 0 and short_hist > 0:
                return "momentum_divergence_bullish_warning"

        # Overbought/oversold within trend (relative strength extreme)
        if abs(relative_strength) > 2.0:
            if long_hist > 0 and relative_strength > 2.0:
                return "overbought_in_uptrend"
            elif long_hist < 0 and relative_strength < -2.0:
                return "oversold_in_downtrend"

        # Entry signals: short histogram zero cross while long maintains direction
        if not pd.isna(prev_short_hist):
            # Short crosses up from negative
            if prev_short_hist < 0 and short_hist >= 0 and long_hist > 0:
                return "bullish_momentum_entry"
            # Short crosses down from positive
            if prev_short_hist > 0 and short_hist <= 0 and long_hist < 0:
                return "bearish_momentum_entry"

        return None
