"""
TrendPulse Indicator - Hybrid swing/trend/top detection system.

Combines five sub-systems into a unified state machine:
1. EMA Stack (14/25/99/144/453): Trend filter and alignment detection
2. Causal ZIG (pivot-only, forward-fill): Swing signal generation
3. DMI/ADX: Trend strength measurement
4. Top Detection (Williams %R-based, gated by trend_strength): Overbought warnings
5. Confidence Scorer: Decomposed score from strength, alignment, and anti-top

State Output (every bar, fixed 8-key schema):
- swing_signal: BUY | SELL | NONE
- trend_filter: BULLISH | BEARISH | NEUTRAL
- trend_strength: 0.0-1.0 (ADX normalized)
- trend_strength_label: STRONG | MODERATE | WEAK
- top_warning: TOP_DETECTED | TOP_ZONE | TOP_PENDING | NONE
- ema_alignment: ALIGNED_BULL | ALIGNED_BEAR | MIXED
- confidence: 0.0-1.0
- score: 0-100
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


class SwingSignal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


class TrendFilter(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TopWarning(str, Enum):
    TOP_DETECTED = "TOP_DETECTED"
    TOP_ZONE = "TOP_ZONE"
    TOP_PENDING = "TOP_PENDING"
    NONE = "NONE"


class EmaAlignment(str, Enum):
    ALIGNED_BULL = "ALIGNED_BULL"
    ALIGNED_BEAR = "ALIGNED_BEAR"
    MIXED = "MIXED"


class TrendStrengthLabel(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


@dataclass(frozen=True)
class TrendPulseConfig:
    """Configuration for TrendPulse indicator. Parameterized per asset class."""

    ema_periods: tuple[int, ...] = (14, 25, 99, 144, 453)
    zig_threshold_pct: float = 5.0
    dmi_period: int = 25
    dmi_smooth: int = 15
    norm_max_adx: float = 50.0
    top_wr_main: int = 34
    top_wr_short: int = 13
    top_wr_smooth: int = 19
    swing_filter_bars: int = 5
    trend_strength_strong: float = 0.6
    trend_strength_moderate: float = 0.3
    trend_strength_weak: float = 0.15
    confidence_weights: tuple[float, ...] = (0.4, 0.3, 0.3)


# Encoding for swing_signal column: 1.0=BUY, -1.0=SELL, 0.0=NONE
_SWING_BUY = 1.0
_SWING_SELL = -1.0
_SWING_NONE = 0.0

# Encoding for top_warning column: 3=DETECTED, 2=ZONE, 1=PENDING, 0=NONE
_TOP_NONE = 0.0
_TOP_PENDING = 1.0
_TOP_ZONE = 2.0
_TOP_DETECTED = 3.0


class TrendPulseIndicator(IndicatorBase):
    """
    Hybrid TrendPulse indicator combining EMA stack, causal ZIG,
    DMI/ADX, top detection, and confidence scoring.

    Swing signals are computed in _calculate() with same-direction cooldown
    and trend filter so that the full bar history is available for tracking.
    Top warnings are also computed in _calculate() with proper CROSS/FILTER logic.

    Default Parameters:
        ema_periods: (14, 25, 99, 144, 453)
        zig_threshold_pct: 5.0 (3.0-8.0)
        dmi_period: 25 (15-35)
        dmi_smooth: 15 (10-20)
        norm_max_adx: 50.0
        swing_filter_bars: 5 (3-10, same-direction suppress)
        trend_strength_strong: 0.6
        trend_strength_moderate: 0.3
        confidence_weights: (0.4, 0.3, 0.3)
    """

    name = "trend_pulse"
    category = SignalCategory.TREND
    required_fields = ["high", "low", "close"]
    warmup_periods = 500

    _default_params: Dict[str, Any] = {
        "ema_periods": (14, 25, 99, 144, 453),
        "zig_threshold_pct": 5.0,
        "dmi_period": 25,
        "dmi_smooth": 15,
        "norm_max_adx": 50.0,
        "top_wr_main": 34,
        "top_wr_short": 13,
        "top_wr_smooth": 19,
        "swing_filter_bars": 5,
        "trend_strength_strong": 0.6,
        "trend_strength_moderate": 0.3,
        "trend_strength_weak": 0.15,
        "confidence_weights": (0.4, 0.3, 0.3),
    }

    def _calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate all TrendPulse components including swing signals and top warnings."""
        n = len(data)
        if n == 0:
            return self._empty_frame(data.index)

        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        close = data["close"].values.astype(np.float64)

        ema_periods: tuple[int, ...] = params["ema_periods"]
        dmi_period: int = params["dmi_period"]
        dmi_smooth: int = params["dmi_smooth"]
        top_wr_main: int = params["top_wr_main"]
        top_wr_short_period: int = params["top_wr_short"]
        top_wr_smooth: int = params["top_wr_smooth"]
        swing_filter_bars: int = params["swing_filter_bars"]

        # --- EMAs ---
        emas: Dict[str, np.ndarray] = {}
        for p in ema_periods:
            if HAS_TALIB:
                emas[f"ema_{p}"] = talib.EMA(close, timeperiod=p)
            else:
                emas[f"ema_{p}"] = self._ema_manual(close, p)

        # --- Causal ZIG ---
        zig_value = self._causal_zig(close, params["zig_threshold_pct"])
        zig_ma = self._simple_ma(zig_value, 2)

        # ZIG/MA crossover detection
        zig_cross_up = np.zeros(n, dtype=np.float64)
        zig_cross_down = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if np.isnan(zig_value[i]) or np.isnan(zig_ma[i]):
                continue
            if np.isnan(zig_value[i - 1]) or np.isnan(zig_ma[i - 1]):
                continue
            if zig_value[i - 1] <= zig_ma[i - 1] and zig_value[i] > zig_ma[i]:
                zig_cross_up[i] = 1.0
            elif zig_value[i - 1] >= zig_ma[i - 1] and zig_value[i] < zig_ma[i]:
                zig_cross_down[i] = 1.0

        # --- DMI/ADX ---
        if HAS_TALIB:
            dmi_plus = talib.PLUS_DI(high, low, close, timeperiod=dmi_period)
            dmi_minus = talib.MINUS_DI(high, low, close, timeperiod=dmi_period)
            adx = talib.ADX(high, low, close, timeperiod=dmi_period)
        else:
            dmi_plus, dmi_minus, adx = self._dmi_manual(high, low, close, dmi_period, dmi_smooth)

        # --- Williams %R per Futu formula ---
        # Raw W%R(34) and W%R(13)
        wr_raw_long = self._williams_r_rescaled(high, low, close, top_wr_main)
        wr_raw_short = self._williams_r_rescaled(high, low, close, top_wr_short_period)
        # long_line = MA(W%R(34), 19)  (simple moving average smoothing)
        wr_long = self._simple_ma(wr_raw_long, top_wr_smooth)
        # mid_line = EMA(W%R(34), 4)   (fast EMA for crossing detection)
        wr_mid = self._ema_arr(wr_raw_long, 4)
        # short_line = W%R(13) raw (for extreme detection)
        wr_short = wr_raw_short

        # --- Trend filter per bar (price vs EMA 453) ---
        ema_453 = emas.get(f"ema_{ema_periods[-1]}", np.full(n, np.nan))
        # 1=BULLISH, -1=BEARISH, 0=NEUTRAL
        trend_filter_arr = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if np.isnan(close[i]) or np.isnan(ema_453[i]):
                trend_filter_arr[i] = 0
            elif close[i] > ema_453[i]:
                trend_filter_arr[i] = 1
            elif close[i] < ema_453[i]:
                trend_filter_arr[i] = -1

        # --- Swing signal with BUY-only cooldown (SELL always fires) ---
        # Per original formula: FILTER(D=1, N) suppresses repeated BUY for N bars.
        # SELL is never suppressed — exits should fire immediately.
        swing_signal_arr = np.full(n, _SWING_NONE)
        bars_since_last_buy = swing_filter_bars  # start allowing

        for i in range(n):
            bars_since_last_buy += 1
            signal = _SWING_NONE

            if zig_cross_up[i] > 0.5 and trend_filter_arr[i] != -1:
                # BUY: suppress if another BUY fired within cooldown
                if bars_since_last_buy >= swing_filter_bars:
                    signal = _SWING_BUY
            elif zig_cross_down[i] > 0.5 and trend_filter_arr[i] != 1:
                # SELL: always fires (no cooldown)
                signal = _SWING_SELL

            if signal != _SWING_NONE:
                swing_signal_arr[i] = signal
                if signal == _SWING_BUY:
                    bars_since_last_buy = 0

        # --- Top warning with CROSS, REF, and FILTER (computed per bar) ---
        top_warning_arr = self._compute_top_warnings(wr_long, wr_mid, wr_short, adx, n)

        # String columns for rule engine state_change detection
        swing_signal_str = np.array(
            ["BUY" if v > 0.5 else ("SELL" if v < -0.5 else "NONE") for v in swing_signal_arr],
            dtype=object,
        )
        top_warning_str = np.array(
            [
                (
                    "TOP_DETECTED"
                    if v >= 2.5
                    else ("TOP_ZONE" if v >= 1.5 else ("TOP_PENDING" if v >= 0.5 else "NONE"))
                )
                for v in top_warning_arr
            ],
            dtype=object,
        )

        # Build result DataFrame
        result: Dict[str, Any] = {
            "trend_pulse_zig_value": zig_value,
            "trend_pulse_zig_ma": zig_ma,
            "trend_pulse_zig_cross_up": zig_cross_up,
            "trend_pulse_zig_cross_down": zig_cross_down,
            "trend_pulse_dmi_plus": dmi_plus,
            "trend_pulse_dmi_minus": dmi_minus,
            "trend_pulse_adx": adx,
            "trend_pulse_wr_long": wr_long,
            "trend_pulse_wr_mid": wr_mid,
            "trend_pulse_wr_short": wr_short,
            "trend_pulse_close": close,
            "trend_pulse_swing_signal_raw": swing_signal_arr,
            "trend_pulse_top_warning_raw": top_warning_arr,
            "trend_pulse_trend_filter": trend_filter_arr,
            # String columns for rule engine state_change detection
            "trend_pulse_swing_signal": swing_signal_str,
            "trend_pulse_top_warning": top_warning_str,
        }
        for key, arr in emas.items():
            result[f"trend_pulse_{key}"] = arr

        return pd.DataFrame(result, index=data.index)

    @staticmethod
    def _compute_top_warnings(
        wr_long: np.ndarray,
        wr_mid: np.ndarray,
        wr_short: np.ndarray,
        adx: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """
        Compute top warnings per bar with proper CROSS, REF(short,2), and FILTER logic.

        CROSS(wr_long, wr_short): wr_long crosses above wr_short (prev <= prev, now >)
        REF(wr_short, 2): wr_short value 2 bars ago
        FILTER(top_zone, N): debounce — fire on first occurrence, suppress for N-1 bars
        """
        top_arr = np.full(n, _TOP_NONE)
        wr_mid_below_60_count = 0
        # FILTER debounce: bars remaining to suppress top_zone re-fires
        top_zone_suppress_remaining = 0

        for i in range(n):
            wl = _sfv(wr_long, i)
            wm = _sfv(wr_mid, i)
            ws = _sfv(wr_short, i)
            cur_adx = _sfv(adx, i)

            prev_wl = _sfv(wr_long, i - 1) if i > 0 else 0.0
            prev_wm = _sfv(wr_mid, i - 1) if i > 0 else 0.0
            prev_ws = _sfv(wr_short, i - 1) if i > 0 else 0.0
            prev_adx = _sfv(adx, i - 1) if i > 0 else 0.0

            # REF(wr_short, 2) = wr_short 2 bars ago
            ref_ws_2 = _sfv(wr_short, i - 2) if i >= 2 else 0.0

            trend_strength_declining = cur_adx < prev_adx

            # --- Exit condition: wr_mid < 60 for 3 consecutive bars ---
            if wm < 60:
                wr_mid_below_60_count += 1
            else:
                wr_mid_below_60_count = 0

            if wr_mid_below_60_count >= 3:
                top_arr[i] = _TOP_NONE
                top_zone_suppress_remaining = 0
                continue

            # --- TOP_PENDING: wr_short > 70 ---
            top_pending = ws > 70

            # --- TOP_ZONE candidate (Futu constraints) ---
            top_zone_raw = (
                wm < prev_wm
                and prev_wm > 80
                and (prev_ws > 95 or ref_ws_2 > 95)
                and wl > 60
                and ws < 83.5
                and ws < wm  # short < mid (Futu: short below mid line)
                and ws < wl + 4  # short < long + 4 (Futu: short near/below long)
            )

            # FILTER(top_zone, 4): debounce — fire first, suppress next 3 bars
            top_zone_filtered = False
            if top_zone_suppress_remaining > 0:
                top_zone_suppress_remaining -= 1
                # Suppressed: don't emit TOP_ZONE again
            elif top_zone_raw:
                top_zone_filtered = True
                top_zone_suppress_remaining = 3  # suppress next 3 bars

            # --- TOP_DETECTED: proper CROSS(wr_long, wr_short) ---
            cross_long_above_short = prev_wl <= prev_ws and wl > ws
            top_detected_candidate = (
                prev_wm > 85 and prev_ws > 85 and prev_wl > 65 and cross_long_above_short
            )

            if top_detected_candidate:
                if trend_strength_declining:
                    top_arr[i] = _TOP_DETECTED
                else:
                    top_arr[i] = _TOP_ZONE
            elif top_zone_filtered:
                top_arr[i] = _TOP_ZONE
            elif top_pending:
                top_arr[i] = _TOP_PENDING
            else:
                top_arr[i] = _TOP_NONE

        return top_arr

    @staticmethod
    def _causal_zig(close: np.ndarray, threshold_pct: float) -> np.ndarray:
        """
        Causal zigzag: pivot-only with forward-fill of last confirmed pivot.

        Compares reversal against the candidate extreme (running high/low),
        not the last confirmed pivot. This ensures reversals register when
        price retraces threshold% from the current swing extreme.

        Initial state (type=0) tracks both directions until the first
        threshold move is detected.
        """
        n = len(close)
        zig = np.full(n, np.nan)
        if n == 0:
            return zig

        last_pivot_val = close[0]
        last_pivot_type = 0  # 0=undecided, 1=tracking high, -1=tracking low
        candidate_high = close[0]
        candidate_low = close[0]
        zig[0] = last_pivot_val

        for i in range(1, n):
            if last_pivot_type == 0:
                # Undecided: track both extremes
                if close[i] > candidate_high:
                    candidate_high = close[i]
                if close[i] < candidate_low:
                    candidate_low = close[i]

                # Check if threshold move from initial point
                up_pct = (candidate_high - close[0]) / close[0] * 100
                down_pct = (close[0] - candidate_low) / close[0] * 100

                if up_pct >= threshold_pct and up_pct >= down_pct:
                    # Confirm initial as low, now tracking high
                    last_pivot_val = candidate_low
                    last_pivot_type = 1
                    candidate_high = close[i] if close[i] > candidate_low else candidate_high
                elif down_pct >= threshold_pct:
                    # Confirm initial as high, now tracking low
                    last_pivot_val = candidate_high
                    last_pivot_type = -1
                    candidate_low = close[i] if close[i] < candidate_high else candidate_low

            elif last_pivot_type == 1:
                # Tracking high: update candidate upward
                if close[i] > candidate_high:
                    candidate_high = close[i]

                # Check reversal DOWN from the candidate high
                pct_from_high = (close[i] - candidate_high) / candidate_high * 100
                if pct_from_high <= -threshold_pct:
                    # Confirm high pivot, switch to tracking low
                    last_pivot_val = candidate_high
                    last_pivot_type = -1
                    candidate_low = close[i]

            elif last_pivot_type == -1:
                # Tracking low: update candidate downward
                if close[i] < candidate_low:
                    candidate_low = close[i]

                # Check reversal UP from the candidate low
                pct_from_low = (close[i] - candidate_low) / candidate_low * 100
                if pct_from_low >= threshold_pct:
                    # Confirm low pivot, switch to tracking high
                    last_pivot_val = candidate_low
                    last_pivot_type = 1
                    candidate_high = close[i]

            zig[i] = last_pivot_val  # forward-fill confirmed pivot

        return zig

    @staticmethod
    def _simple_ma(arr: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        n = len(arr)
        result = np.full(n, np.nan)
        if n < period:
            return result
        cumsum = np.nancumsum(arr)
        result[period - 1] = cumsum[period - 1] / period
        for i in range(period, n):
            if np.isnan(arr[i]) or np.isnan(arr[i - period]):
                continue
            result[i] = (cumsum[i] - cumsum[i - period]) / period
        return result

    @staticmethod
    def _ema_manual(arr: np.ndarray, period: int) -> np.ndarray:
        """Manual EMA calculation."""
        n = len(arr)
        result = np.full(n, np.nan)
        if n < period:
            return result
        alpha = 2.0 / (period + 1)
        result[period - 1] = np.mean(arr[:period])
        for i in range(period, n):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _ema_arr(arr: np.ndarray, period: int) -> np.ndarray:
        """EMA of an array that may contain NaN (skips NaN, starts from first valid)."""
        n = len(arr)
        result = np.full(n, np.nan)
        alpha = 2.0 / (period + 1)
        initialized = False
        for i in range(n):
            if np.isnan(arr[i]):
                if initialized:
                    result[i] = result[i - 1]
                continue
            if not initialized:
                result[i] = arr[i]
                initialized = True
            else:
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _dmi_manual(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        smooth: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Manual DMI/ADX calculation (Wilder's smoothing)."""
        n = len(close)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        adx_arr = np.full(n, np.nan)

        if n < period + smooth:
            return plus_di, minus_di, adx_arr

        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0

        atr_sum = np.sum(tr[1 : period + 1])
        plus_sum = np.sum(plus_dm[1 : period + 1])
        minus_sum = np.sum(minus_dm[1 : period + 1])

        for i in range(period, n):
            if i > period:
                atr_sum = atr_sum - atr_sum / period + tr[i]
                plus_sum = plus_sum - plus_sum / period + plus_dm[i]
                minus_sum = minus_sum - minus_sum / period + minus_dm[i]

            if atr_sum > 0:
                plus_di[i] = 100 * plus_sum / atr_sum
                minus_di[i] = 100 * minus_sum / atr_sum

        dx = np.full(n, np.nan)
        for i in range(period, n):
            if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
                s = plus_di[i] + minus_di[i]
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / s if s > 0 else 0

        start = period + smooth - 1
        if start < n:
            valid_dx = [dx[j] for j in range(period, start + 1) if not np.isnan(dx[j])]
            if valid_dx:
                adx_arr[start] = np.mean(valid_dx)
                for i in range(start + 1, n):
                    if not np.isnan(dx[i]) and not np.isnan(adx_arr[i - 1]):
                        adx_arr[i] = (adx_arr[i - 1] * (smooth - 1) + dx[i]) / smooth

        return plus_di, minus_di, adx_arr

    @staticmethod
    def _williams_r_rescaled(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        """Williams %R rescaled to 0-100 (0=oversold, 100=overbought)."""
        n = len(close)
        wr = np.full(n, np.nan)
        for i in range(period - 1, n):
            hh = np.max(high[i - period + 1 : i + 1])
            ll = np.min(low[i - period + 1 : i + 1])
            if hh != ll:
                wr[i] = (close[i] - ll) / (hh - ll) * 100
        return wr

    @staticmethod
    def _chinese_sma(arr: np.ndarray, n_period: int, m: int) -> np.ndarray:
        """Chinese SMA: SMA(X,N,M) = (M*X + (N-M)*prev) / N."""
        length = len(arr)
        result = np.full(length, np.nan)
        for i in range(length):
            if np.isnan(arr[i]):
                continue
            if i == 0 or np.isnan(result[i - 1]):
                result[i] = arr[i]
            else:
                result[i] = (m * arr[i] + (n_period - m) * result[i - 1]) / n_period
        return result

    def _empty_frame(self, index: pd.Index) -> pd.DataFrame:
        """Return empty DataFrame with correct columns."""
        cols = [
            "trend_pulse_zig_value",
            "trend_pulse_zig_ma",
            "trend_pulse_zig_cross_up",
            "trend_pulse_zig_cross_down",
            "trend_pulse_dmi_plus",
            "trend_pulse_dmi_minus",
            "trend_pulse_adx",
            "trend_pulse_wr_long",
            "trend_pulse_wr_mid",
            "trend_pulse_wr_short",
            "trend_pulse_close",
            "trend_pulse_swing_signal_raw",
            "trend_pulse_top_warning_raw",
            "trend_pulse_trend_filter",
            "trend_pulse_swing_signal",
            "trend_pulse_top_warning",
        ]
        for p in self._default_params["ema_periods"]:
            cols.append(f"trend_pulse_ema_{p}")
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols}, index=index)

    def _get_state(
        self,
        current: pd.Series,
        previous: Optional[pd.Series],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract TrendPulse state — fixed 8-key schema, always present."""
        close = _sf(current.get("trend_pulse_close", np.nan))
        adx = _sf(current.get("trend_pulse_adx", 0))

        norm_max = params.get("norm_max_adx", 50.0)
        strong_th = params.get("trend_strength_strong", 0.6)
        moderate_th = params.get("trend_strength_moderate", 0.3)
        weights = params.get("confidence_weights", (0.4, 0.3, 0.3))
        ema_periods = params.get("ema_periods", (14, 25, 99, 144, 453))

        # --- Read pre-computed swing signal ---
        raw_swing = _sf(current.get("trend_pulse_swing_signal_raw", 0))
        if raw_swing > 0.5:
            swing_signal = SwingSignal.BUY
        elif raw_swing < -0.5:
            swing_signal = SwingSignal.SELL
        else:
            swing_signal = SwingSignal.NONE

        # --- Read pre-computed trend filter ---
        raw_tf = _sf(current.get("trend_pulse_trend_filter", 0))
        if raw_tf > 0.5:
            trend_filter = TrendFilter.BULLISH
        elif raw_tf < -0.5:
            trend_filter = TrendFilter.BEARISH
        else:
            trend_filter = TrendFilter.NEUTRAL

        # --- Read pre-computed top warning ---
        raw_top = _sf(current.get("trend_pulse_top_warning_raw", 0))
        if raw_top >= 2.5:
            top_warning = TopWarning.TOP_DETECTED
        elif raw_top >= 1.5:
            top_warning = TopWarning.TOP_ZONE
        elif raw_top >= 0.5:
            top_warning = TopWarning.TOP_PENDING
        else:
            top_warning = TopWarning.NONE

        # --- EMA alignment ---
        ema_vals = [_sf(current.get(f"trend_pulse_ema_{p}", np.nan)) for p in ema_periods]
        any_nan = any(np.isnan(v) for v in ema_vals)

        if any_nan:
            ema_alignment = EmaAlignment.MIXED
        elif all(ema_vals[i] >= ema_vals[i + 1] for i in range(len(ema_vals) - 1)):
            ema_alignment = EmaAlignment.ALIGNED_BULL
        elif all(ema_vals[i] <= ema_vals[i + 1] for i in range(len(ema_vals) - 1)):
            ema_alignment = EmaAlignment.ALIGNED_BEAR
        else:
            ema_alignment = EmaAlignment.MIXED

        # --- Trend strength ---
        trend_strength = min(adx / norm_max, 1.0) if norm_max > 0 else 0.0
        if trend_strength >= strong_th:
            strength_label = TrendStrengthLabel.STRONG
        elif trend_strength >= moderate_th:
            strength_label = TrendStrengthLabel.MODERATE
        else:
            strength_label = TrendStrengthLabel.WEAK

        # --- Confidence scoring (decomposed) ---
        conf_trend = min(adx / norm_max, 1.0) if norm_max > 0 else 0.0
        if ema_alignment == EmaAlignment.ALIGNED_BULL:
            conf_align = 1.0
        elif ema_alignment == EmaAlignment.MIXED:
            conf_align = 0.7
        else:
            conf_align = 0.4

        top_risk_map = {
            TopWarning.NONE: 0.0,
            TopWarning.TOP_PENDING: 0.3,
            TopWarning.TOP_ZONE: 0.6,
            TopWarning.TOP_DETECTED: 1.0,
        }
        conf_top = max(0.0, 1.0 - top_risk_map[top_warning])

        w1, w2, w3 = weights[0], weights[1], weights[2]
        confidence = w1 * conf_trend + w2 * conf_align + w3 * conf_top

        signal_mult = 1.0 if swing_signal != SwingSignal.NONE else 0.5
        score = confidence * 100 * signal_mult

        return {
            "swing_signal": swing_signal.value,
            "trend_filter": trend_filter.value,
            "trend_strength": round(trend_strength, 4),
            "trend_strength_label": strength_label.value,
            "top_warning": top_warning.value,
            "ema_alignment": ema_alignment.value,
            "confidence": round(confidence, 4),
            "score": round(score, 2),
        }


def _sf(val: Any) -> float:
    """Safe float conversion, NaN/None -> 0.0."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _sfv(arr: np.ndarray, idx: int) -> float:
    """Safe array value access, returns 0.0 for out-of-bounds or NaN."""
    if idx < 0 or idx >= len(arr):
        return 0.0
    v = arr[idx]
    return 0.0 if np.isnan(v) else float(v)
