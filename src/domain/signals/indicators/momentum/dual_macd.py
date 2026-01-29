"""
DualMACD Indicator v1.0 - Tactical Dip/Rally Detection via Dual-Timeframe MACD.

Philosophy:
- Slow MACD (55/89, signal 34): Structural trend direction and slope
- Fast MACD (13/21, signal 9): Tactical timing within trend
- Overlaying both reveals dip-buy / rally-sell opportunities with confidence

State Output (every bar, no NaN in classified fields):
- slow_histogram, fast_histogram: Raw histogram values (×multiplier)
- slow_hist_delta, fast_hist_delta: Slope of histograms (ΔH over slope_lookback)
- trend_state: BULLISH | BEARISH | IMPROVING | DETERIORATING
- tactical_signal: DIP_BUY | RALLY_SELL | NONE
- momentum_balance: FAST_DOMINANT | SLOW_DOMINANT | BALANCED
- confidence: 0.0-1.0 (curvature-based, for tactical signals only)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


class TrendState(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    IMPROVING = "IMPROVING"
    DETERIORATING = "DETERIORATING"


class TacticalSignal(str, Enum):
    DIP_BUY = "DIP_BUY"
    RALLY_SELL = "RALLY_SELL"
    NONE = "NONE"


class MomentumBalance(str, Enum):
    FAST_DOMINANT = "FAST_DOMINANT"
    SLOW_DOMINANT = "SLOW_DOMINANT"
    BALANCED = "BALANCED"


@dataclass(frozen=True)
class DualMACDConfig:
    slow_fast: int = 55
    slow_slow: int = 89
    slow_signal: int = 34
    fast_fast: int = 13
    fast_slow: int = 21
    fast_signal: int = 9
    slope_lookback: int = 3
    hist_norm_window: int = 252
    histogram_multiplier: float = 2.0
    eps: float = 1e-3


class DualMACDIndicator(IndicatorBase):
    """
    Dual MACD indicator v1.0 with tactical dip/rally detection.

    Slow MACD (55/89/34): Structural trend direction
    Fast MACD (13/21/9): Tactical timing within trend

    Default Parameters:
        slow_fast: 55, slow_slow: 89, slow_signal: 34
        fast_fast: 13, fast_slow: 21, fast_signal: 9
        slope_lookback: 3
        hist_norm_window: 252
        histogram_multiplier: 2.0
    """

    name = "dual_macd"
    category = SignalCategory.MOMENTUM
    required_fields = ["close"]
    warmup_periods = 255  # max(slow_slow=89, hist_norm_window=252) + slope_lookback=3

    _default_params = {
        "slow_fast": 55,
        "slow_slow": 89,
        "slow_signal": 34,
        "fast_fast": 13,
        "fast_slow": 21,
        "fast_signal": 9,
        "slope_lookback": 3,
        "hist_norm_window": 252,
        "histogram_multiplier": 2.0,
        "eps": 1e-3,
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate dual MACD histograms and deltas."""
        n = len(data)
        if n == 0:
            return pd.DataFrame(
                {
                    "slow_histogram": pd.Series(dtype=float),
                    "fast_histogram": pd.Series(dtype=float),
                    "slow_hist_delta": pd.Series(dtype=float),
                    "fast_hist_delta": pd.Series(dtype=float),
                    "fast_hist_delta2": pd.Series(dtype=float),
                    "slow_hist_norm": pd.Series(dtype=float),
                    "fast_hist_norm": pd.Series(dtype=float),
                },
                index=data.index,
            )

        close = data["close"].values.astype(np.float64)
        multiplier = params["histogram_multiplier"]
        slope_lb = params["slope_lookback"]
        norm_window = params["hist_norm_window"]

        if HAS_TALIB:
            slow_hist = self._calc_macd_talib(
                close, params["slow_fast"], params["slow_slow"], params["slow_signal"], multiplier
            )
            fast_hist = self._calc_macd_talib(
                close, params["fast_fast"], params["fast_slow"], params["fast_signal"], multiplier
            )
        else:
            slow_hist = self._calc_macd_manual(
                close, params["slow_fast"], params["slow_slow"], params["slow_signal"], multiplier
            )
            fast_hist = self._calc_macd_manual(
                close, params["fast_fast"], params["fast_slow"], params["fast_signal"], multiplier
            )

        # Deltas (slope)
        slow_hist_delta = np.full(n, np.nan, dtype=np.float64)
        fast_hist_delta = np.full(n, np.nan, dtype=np.float64)
        for i in range(slope_lb, n):
            if not np.isnan(slow_hist[i]) and not np.isnan(slow_hist[i - slope_lb]):
                slow_hist_delta[i] = slow_hist[i] - slow_hist[i - slope_lb]
            if not np.isnan(fast_hist[i]) and not np.isnan(fast_hist[i - slope_lb]):
                fast_hist_delta[i] = fast_hist[i] - fast_hist[i - slope_lb]

        # Second derivative of fast histogram
        fast_hist_delta2 = np.full(n, np.nan, dtype=np.float64)
        for i in range(slope_lb + 1, n):
            if not np.isnan(fast_hist_delta[i]) and not np.isnan(fast_hist_delta[i - 1]):
                fast_hist_delta2[i] = fast_hist_delta[i] - fast_hist_delta[i - 1]

        # Rolling percentile rank of |histogram| (causal, 0-1)
        slow_hist_norm = self._rolling_pctile_rank(np.abs(slow_hist), norm_window)
        fast_hist_norm = self._rolling_pctile_rank(np.abs(fast_hist), norm_window)

        return pd.DataFrame(
            {
                "slow_histogram": slow_hist,
                "fast_histogram": fast_hist,
                "slow_hist_delta": slow_hist_delta,
                "fast_hist_delta": fast_hist_delta,
                "fast_hist_delta2": fast_hist_delta2,
                "slow_hist_norm": slow_hist_norm,
                "fast_hist_norm": fast_hist_norm,
            },
            index=data.index,
        )

    def _calc_macd_talib(
        self,
        close: np.ndarray,
        fast: int,
        slow: int,
        signal: int,
        multiplier: float,
    ) -> np.ndarray:
        """Calculate MACD histogram using TA-Lib."""
        ema_fast = talib.EMA(close, timeperiod=fast)
        ema_slow = talib.EMA(close, timeperiod=slow)
        macd_line = ema_fast - ema_slow
        signal_line = talib.EMA(macd_line, timeperiod=signal)
        return multiplier * (macd_line - signal_line)

    def _calc_macd_manual(
        self,
        close: np.ndarray,
        fast: int,
        slow: int,
        signal: int,
        multiplier: float,
    ) -> np.ndarray:
        """Calculate MACD histogram without TA-Lib."""
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

        signal_line = np.full(n, np.nan, dtype=np.float64)
        valid_start = slow - 1 + signal - 1
        if n > valid_start:
            alpha = 2.0 / (signal + 1)
            signal_line[valid_start] = np.nanmean(macd_line[slow - 1 : valid_start + 1])
            for i in range(valid_start + 1, n):
                if not np.isnan(macd_line[i]):
                    signal_line[i] = alpha * macd_line[i] + (1 - alpha) * signal_line[i - 1]

        histogram: np.ndarray = multiplier * (macd_line - signal_line)
        return histogram

    @staticmethod
    def _rolling_pctile_rank(arr: np.ndarray, window: int) -> np.ndarray:
        """Causal rolling percentile rank (0-1)."""
        n = len(arr)
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(window - 1, n):
            w = arr[max(0, i - window + 1) : i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) < 2:
                continue
            result[i] = np.sum(valid <= arr[i]) / len(valid)
        return result

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract DualMACD state for rule evaluation."""
        eps = params.get("eps", 1e-3)

        h_slow = self._safe_float(current.get("slow_histogram", 0))
        h_fast = self._safe_float(current.get("fast_histogram", 0))
        dh_slow = self._safe_float(current.get("slow_hist_delta", 0))
        dh_fast = self._safe_float(current.get("fast_hist_delta", 0))
        ddh_fast = self._safe_float(current.get("fast_hist_delta2", 0))
        slow_norm = self._safe_float(current.get("slow_hist_norm", 0))
        fast_norm = self._safe_float(current.get("fast_hist_norm", 0))

        # --- Trend state (override-first priority) ---
        if h_slow > 0 and dh_slow < 0:
            trend_state = TrendState.DETERIORATING
        elif h_slow < 0 and dh_slow > 0:
            trend_state = TrendState.IMPROVING
        elif h_slow > 0:
            trend_state = TrendState.BULLISH
        else:
            trend_state = TrendState.BEARISH

        # --- Tactical signal ---
        tactical_signal = TacticalSignal.NONE
        if h_slow > 0 and h_fast < 0 and abs(dh_fast) > abs(dh_slow) and dh_fast >= 0:
            tactical_signal = TacticalSignal.DIP_BUY
        elif h_slow < 0 and h_fast > 0 and abs(dh_fast) > abs(dh_slow) and dh_fast <= 0:
            tactical_signal = TacticalSignal.RALLY_SELL

        # --- Momentum balance (with freeze zone) ---
        if slow_norm < 0.15 and fast_norm < 0.15:
            momentum_balance = MomentumBalance.BALANCED
        elif fast_norm > slow_norm * 1.5:
            momentum_balance = MomentumBalance.FAST_DOMINANT
        elif slow_norm > fast_norm * 1.5:
            momentum_balance = MomentumBalance.SLOW_DOMINANT
        else:
            momentum_balance = MomentumBalance.BALANCED

        # --- Confidence (curvature-based) ---
        confidence = 0.0
        if tactical_signal == TacticalSignal.DIP_BUY:
            # h_fast < 0, positive ΔΔH = pullback decelerating
            denom = max(abs(h_fast), eps)
            confidence = float(np.clip(ddh_fast / denom, 0.0, 1.0))
        elif tactical_signal == TacticalSignal.RALLY_SELL:
            # h_fast > 0, negative ΔΔH = bounce decelerating
            denom = max(abs(h_fast), eps)
            confidence = float(np.clip(-ddh_fast / denom, 0.0, 1.0))

        return {
            "slow_histogram": h_slow,
            "fast_histogram": h_fast,
            "slow_hist_delta": dh_slow,
            "fast_hist_delta": dh_fast,
            "trend_state": trend_state.value,
            "tactical_signal": tactical_signal.value,
            "momentum_balance": momentum_balance.value,
            "confidence": confidence,
        }

    @staticmethod
    def _safe_float(val: Any) -> float:
        """Convert value to float, defaulting to 0.0 for NaN/None."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        try:
            f = float(val)
            return 0.0 if np.isnan(f) else f
        except (TypeError, ValueError):
            return 0.0
