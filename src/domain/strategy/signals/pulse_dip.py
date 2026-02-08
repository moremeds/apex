"""PulseDip SignalGenerator for VectorBT backtesting.

Upgraded (v2) to match event-driven PulseDipStrategy signal logic using
TrendPulseIndicator + DualMACDIndicator, same as trend_pulse.py.

PulseDip is a dip-buying variant: entry requires RSI < threshold (dip)
PLUS the TrendPulse/DualMACD trend + momentum confirmation that
the event-driven version uses.

Entry conditions (ALL required):
    1. close > EMA-99 (daily trend bullish)
    2. trend_strength >= moderate threshold (ADX-based)
    3. RSI < rsi_entry_threshold (dip condition — PulseDip-specific)
    4. DualMACD trend_state in (BULLISH, IMPROVING, DETERIORATING)
    5. ADX >= adx_entry_min (chop filter)

Exit conditions (priority order, any triggers):
    1. ATR trailing stop (expanding-max proxy for peak)
    2. DualMACD bearish state transition (N consecutive bars)
    3. TrendPulse top_detected
    4. RSI overbought (> 65)
    5. Hard stop (% below entry proxy)

Intentional vectorized-only differences (documented, not matched):
    - No regime gating (VectorBT lacks regime context — tested separately)
    - No position management (VectorBT handles internally)
    - Expanding-max ATR trail instead of per-trade peak tracking
    - No confluence scoring (VectorBT has no ConfluenceProvider)

Signal shift: entries/exits shifted +1 bar for backtest realism.
Same-bar conflict: exit wins.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ....domain.signals.indicators.momentum.dual_macd import DualMACDIndicator
from ....domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator
from ..param_loader import get_strategy_params
from ..signals.indicators import atr as calc_atr
from ..signals.indicators import rsi as calc_rsi

# Canonical defaults loaded from config/strategy/pulse_dip.yaml
_DEFAULTS = get_strategy_params("pulse_dip")


class PulseDipSignalGenerator:
    """
    Vectorized PulseDip signal generation (v2).

    Uses TrendPulseIndicator + DualMACDIndicator for parity with the
    event-driven PulseDipStrategy. Entry requires RSI dip plus trend/momentum
    confirmation.
    """

    @property
    def warmup_bars(self) -> int:
        """DualMACD hist_norm_window(252) + slope(3) + buffer."""
        return 260

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate PulseDip entry/exit signals.

        Args:
            data: OHLCV DataFrame with open, high, low, close, volume.
            params: Strategy parameters (overrides YAML defaults).
            secondary_data: Unused (reserved for MTF).

        Returns:
            (entries, exits): Boolean series, shifted +1 bar for realism.
        """
        n = len(data)
        idx = data.index

        if n == 0:
            empty = pd.Series(False, index=idx, dtype=bool)
            return empty, empty.copy()

        # --- Extract params: YAML defaults merged with caller overrides ---
        effective = {**_DEFAULTS, **params}
        ema_period = int(effective.get("ema_trend_period", 99))
        rsi_period = int(effective.get("rsi_period", 14))
        rsi_thresh = float(effective.get("rsi_entry_threshold", 35.0))
        atr_mult = float(effective.get("atr_stop_mult", 3.0))
        hard_stop_pct = float(effective.get("hard_stop_pct", 0.08))
        adx_entry_min = float(effective.get("adx_entry_min", 15.0))
        trend_strength_moderate = float(effective.get("trend_strength_moderate", 0.15))
        norm_max_adx = float(effective.get("norm_max_adx", 50.0))
        exit_bearish_bars = int(effective.get("exit_bearish_bars", 3))

        warmup = self.warmup_bars

        # --- Run TrendPulse indicator ---
        tp_indicator = TrendPulseIndicator()
        tp_params = {
            "ema_periods": (14, 25, 99, 144, 453),
            "zig_threshold_pct": float(effective.get("zig_threshold_pct", 3.5)),
            "dmi_period": 25,
            "dmi_smooth": 15,
            "norm_max_adx": norm_max_adx,
            "top_wr_main": int(effective.get("top_wr_main", 34)),
            "top_wr_short": 13,
            "top_wr_smooth": 19,
            "swing_filter_bars": int(effective.get("swing_filter_bars", 5)),
            "trend_strength_strong": float(effective.get("trend_strength_strong", 0.6)),
            "trend_strength_moderate": trend_strength_moderate,
            "trend_strength_weak": 0.15,
            "confidence_weights": (0.4, 0.3, 0.3),
        }
        tp_df = tp_indicator._calculate(data, tp_params)

        # --- Run DualMACD indicator ---
        dm_indicator = DualMACDIndicator()
        dm_params = {
            "slow_fast": int(effective.get("slow_fast", 55)),
            "slow_slow": int(effective.get("slow_slow", 89)),
            "slow_signal": int(effective.get("slow_signal", 34)),
            "fast_fast": 13,
            "fast_slow": 21,
            "fast_signal": 9,
            "slope_lookback": int(effective.get("slope_lookback", 3)),
            "hist_norm_window": 252,
            "histogram_multiplier": 2.0,
            "eps": 1e-3,
        }
        dm_df = dm_indicator._calculate(data, dm_params)

        # --- Extract arrays ---
        close_arr = data["close"].values.astype(np.float64)
        close_s = data["close"]
        high_s = data["high"]
        low_s = data["low"]

        adx_vals = tp_df["trend_pulse_adx"].values
        top_warning_raw = tp_df["trend_pulse_top_warning_raw"].values

        slow_hist = dm_df["slow_histogram"].values
        slow_hist_delta = dm_df["slow_hist_delta"].values

        # --- 1. EMA-99 trend filter (same as TrendPulse) ---
        ema_99 = tp_df["trend_pulse_ema_99"].values
        trend_bull = np.where(np.isnan(ema_99), False, close_arr > ema_99)

        # --- 2. Trend strength (ADX-based, same as TrendPulse) ---
        trend_strength = np.where(
            np.isnan(adx_vals), 0.0, np.clip(adx_vals / norm_max_adx, 0.0, 1.0)
        )
        cond_strength = trend_strength >= trend_strength_moderate

        # --- 3. RSI dip condition (PulseDip-specific) ---
        rsi_vals = calc_rsi(close_s, rsi_period).values
        cond_rsi_dip = np.where(np.isnan(rsi_vals), False, rsi_vals < rsi_thresh)

        # --- 4. DualMACD confluence (same as TrendPulse) ---
        sh = np.where(np.isnan(slow_hist), 0.0, slow_hist)
        shd = np.where(np.isnan(slow_hist_delta), 0.0, slow_hist_delta)

        dm_bullish = (sh > 0) & (shd >= 0)
        dm_improving = (sh < 0) & (shd > 0)
        dm_deteriorating = (sh > 0) & (shd < 0)
        dm_entry_ok = dm_bullish | dm_improving | dm_deteriorating

        # --- 5. ADX chop filter (same as TrendPulse) ---
        adx_ok = np.where(np.isnan(adx_vals), False, adx_vals >= adx_entry_min)

        # --- Combined entry: trend + strength + dip + momentum + chop filter ---
        entries_raw = trend_bull & cond_strength & cond_rsi_dip & dm_entry_ok & adx_ok

        # --- Exit conditions ---
        # (a) DualMACD bearish persistence — trigger once at N consecutive (not sticky)
        dm_bearish_raw = (sh <= 0) & (shd <= 0)
        consec = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            consec[i] = (consec[i - 1] + 1) if dm_bearish_raw[i] else 0
        dm_exit_bearish = consec == exit_bearish_bars

        # (b) TrendPulse top_detected
        cond_top_detected = top_warning_raw >= 2.5

        # (c) RSI overbought
        rsi_exit = np.where(np.isnan(rsi_vals), False, rsi_vals > 65)

        # (d) ATR trailing stop (expanding-max proxy — intentional VBT difference)
        atr_vals_series = calc_atr(high_s, low_s, close_s, period=14)
        atr_arr = atr_vals_series.values
        rolling_peak = close_s.expanding().max().values
        trail_stop = rolling_peak - atr_mult * np.where(np.isnan(atr_arr), 0.0, atr_arr)
        trail_exit = close_arr < trail_stop

        # (e) Hard stop (% below entry proxy)
        entry_proxy = np.roll(close_arr, 1)
        entry_proxy[0] = close_arr[0]
        hard_stop_level = entry_proxy * (1.0 - hard_stop_pct)
        hard_stop_exit = close_arr < hard_stop_level

        # --- Combined exits ---
        exits_raw = dm_exit_bearish | cond_top_detected | rsi_exit | trail_exit | hard_stop_exit

        # --- Signal shift +1 bar ---
        entries_raw = np.roll(entries_raw, 1)
        entries_raw[0] = False
        exits_raw = np.roll(exits_raw, 1)
        exits_raw[0] = False

        # --- Warmup gate ---
        entries_raw[:warmup] = False
        exits_raw[:warmup] = False

        # --- Same-bar conflict: exit wins ---
        conflict = entries_raw & exits_raw
        entries_raw[conflict] = False

        entries = pd.Series(entries_raw, index=idx, dtype=bool)
        exits = pd.Series(exits_raw, index=idx, dtype=bool)

        return entries, exits
