"""
TrendPulse v2.2 SignalGenerator for VectorBT backtesting.

Combines TrendPulseIndicator (swing/trend/top detection) with DualMACDIndicator
(momentum confirmation) to produce entry/exit signals with confidence-based sizing.

Entry conditions (ALL required):
    1. zig_cross_up (swing BUY)
    2. close > EMA-99 (daily trend bullish)
    3. trend_strength >= moderate_threshold
    4. dualmacd trend_state in (BULLISH, IMPROVING, DETERIORATING)
    5. (optional) weekly trend bullish (MTF confirmation)
    6. ADX >= adx_entry_min (chop filter — blocks entry in trendless regimes)
    7. cooldown_bars elapsed since last exit

Re-entry: EMA cross-up when trend_bull AND strength_ok AND dm_entry_ok.

Exit conditions (priority order, highest wins):
    1. ATR trailing stop (risk protection)
    2. DM bearish state transition (first bar reaching N consecutive bearish)
    3. zig_cross_down (profit-taking / soft exit)
    4. top_warning == TOP_DETECTED

Sizing:
    4-factor confidence scoring (ZIG strength, DM health, trend alignment, vol quality)
    ATR-based volatility scaler for position sizing.

Signal shift: entries/exits shifted +1 bar for backtest realism.

Same-bar conflict: If exit and entry on same bar, exit wins.

v2.2 changes from v2.1:
    - dm_regime exit: state transition (trigger once at N consecutive) instead of sticky
    - ADX chop filter: blocks entry when ADX < threshold
    - Cooldown: N bars after exit before re-entry allowed
    - ATR stop: default atr_mult lowered to 3.0, search range tightened
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ....domain.signals.indicators.momentum.dual_macd import DualMACDIndicator
from ....domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator
from ..signals.indicators import atr as calc_atr
from ..signals.indicators import ema as calc_ema


class TrendPulseSignalGenerator:
    """
    Vectorized TrendPulse + DualMACD signal generator (v2.2).

    Produces boolean entry/exit Series plus a float entry_sizes Series
    for confidence-based position sizing in VectorBT.
    """

    @property
    def warmup_bars(self) -> int:
        """Max warmup across both indicators."""
        return 500

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry/exit signals from OHLCV data.

        After calling, self.entry_sizes and self.exit_reasons are set.

        Args:
            data: DataFrame with open, high, low, close, volume columns.
            params: Strategy parameters (overrides defaults).
            secondary_data: Optional dict of timeframe -> OHLCV DataFrame (e.g. {"1W": ...}).

        Returns:
            (entries, exits) boolean Series aligned to data.index.
        """
        n = len(data)
        idx = data.index

        if n == 0:
            empty = pd.Series(False, index=idx, dtype=bool)
            self.entry_sizes = pd.Series(0.0, index=idx, dtype=np.float64)
            self.exit_reasons = pd.Series("", index=idx, dtype=object)
            return empty, empty.copy()

        # --- Extract params with defaults ---
        zig_threshold_pct = float(params.get("zig_threshold_pct", 5.0))
        swing_filter_bars = int(params.get("swing_filter_bars", 5))
        trend_strength_moderate = float(params.get("trend_strength_moderate", 0.2))
        trend_strength_strong = float(params.get("trend_strength_strong", 0.6))
        norm_max_adx = float(params.get("norm_max_adx", 50.0))

        # DualMACD params
        slow_fast = int(params.get("slow_fast", 55))
        slow_slow = int(params.get("slow_slow", 89))
        slow_signal = int(params.get("slow_signal", 34))
        slope_lookback = int(params.get("slope_lookback", 3))

        # Sizing params
        min_pct = float(params.get("min_pct", 0.2))
        max_pct = float(params.get("max_pct", 0.8))

        # Top detection params
        top_wr_main = int(params.get("top_wr_main", 34))

        # v2.1 params
        exit_bearish_bars = int(params.get("exit_bearish_bars", 3))
        enable_trend_reentry = bool(params.get("enable_trend_reentry", True))
        ema_reentry_period = int(params.get("ema_reentry_period", 25))
        min_confidence = float(params.get("min_confidence", 0.4))
        atr_stop_mult = float(params.get("atr_stop_mult", 3.0))
        signal_shift_bars = int(params.get("signal_shift_bars", 1))
        enable_mtf_confirm = bool(params.get("enable_mtf_confirm", True))
        weekly_ema_period = int(params.get("weekly_ema_period", 26))

        # v2.2 params
        enable_chop_filter = bool(params.get("enable_chop_filter", True))
        adx_entry_min = float(params.get("adx_entry_min", 20.0))
        cooldown_bars = int(params.get("cooldown_bars", 5))

        warmup = self.warmup_bars

        # --- Run TrendPulse indicator ---
        tp_indicator = TrendPulseIndicator()
        tp_params = {
            "ema_periods": (14, 25, 99, 144, 453),
            "zig_threshold_pct": zig_threshold_pct,
            "dmi_period": 25,
            "dmi_smooth": 15,
            "norm_max_adx": norm_max_adx,
            "top_wr_main": top_wr_main,
            "top_wr_short": 13,
            "top_wr_smooth": 19,
            "swing_filter_bars": swing_filter_bars,
            "trend_strength_strong": trend_strength_strong,
            "trend_strength_moderate": trend_strength_moderate,
            "trend_strength_weak": 0.15,
            "confidence_weights": (0.4, 0.3, 0.3),
        }
        tp_df = tp_indicator._calculate(data, tp_params)

        # --- Run DualMACD indicator ---
        dm_indicator = DualMACDIndicator()
        dm_params = {
            "slow_fast": slow_fast,
            "slow_slow": slow_slow,
            "slow_signal": slow_signal,
            "fast_fast": 13,
            "fast_slow": 21,
            "fast_signal": 9,
            "slope_lookback": slope_lookback,
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

        # --- 0a. Use zig_cross_down for exits (not swing_signal_raw SELL) ---
        zig_cross_up = tp_df["trend_pulse_zig_cross_up"].values
        zig_cross_down = tp_df["trend_pulse_zig_cross_down"].values
        cond_swing_buy = zig_cross_up > 0.5
        cond_swing_sell = zig_cross_down > 0.5

        # --- 0b. Use EMA-99 for trend filter (not the indicator's EMA-453 filter) ---
        ema_99 = tp_df["trend_pulse_ema_99"].values
        trend_bull = np.where(np.isnan(ema_99), False, close_arr > ema_99)

        # --- Compute trend_strength per bar ---
        trend_strength = np.where(
            np.isnan(adx_vals), 0.0, np.clip(adx_vals / norm_max_adx, 0.0, 1.0)
        )
        cond_strength = trend_strength >= trend_strength_moderate

        # --- 1. DualMACD: DETERIORATING now allows entry ---
        sh = np.where(np.isnan(slow_hist), 0.0, slow_hist)
        shd = np.where(np.isnan(slow_hist_delta), 0.0, slow_hist_delta)

        dm_bullish = (sh > 0) & (shd >= 0)
        dm_improving = (sh < 0) & (shd > 0)
        dm_deteriorating = (sh > 0) & (shd < 0)
        dm_entry_ok = dm_bullish | dm_improving | dm_deteriorating

        # --- 2. Exit bearish persistence — state transition (trigger once) ---
        # Only fire on the FIRST bar that reaches N consecutive bearish,
        # not every subsequent bar. This prevents "sticky exit" that turns
        # the strategy into a fast in/out protector instead of a trend holder.
        dm_bearish_raw = (sh <= 0) & (shd <= 0)
        consec = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            consec[i] = (consec[i - 1] + 1) if dm_bearish_raw[i] else 0
        dm_exit_bearish = consec == exit_bearish_bars  # == not >=

        # --- 10. Multi-timeframe confirmation ---
        if enable_mtf_confirm and secondary_data and "1W" in secondary_data:
            weekly = secondary_data["1W"]
            weekly_close = weekly["close"]
            weekly_ema = calc_ema(weekly_close, weekly_ema_period)
            weekly_bull_s = (weekly_close > weekly_ema).reindex(idx, method="ffill").fillna(False)
            weekly_bull = weekly_bull_s.values.astype(bool)
        else:
            weekly_bull = np.ones(n, dtype=bool)

        mtf_confirm = weekly_bull & trend_bull

        # --- v2.2: ADX chop filter (blocks entry in trendless regimes) ---
        if enable_chop_filter:
            adx_ok = np.where(np.isnan(adx_vals), False, adx_vals >= adx_entry_min)
        else:
            adx_ok = np.ones(n, dtype=bool)

        # --- Entry conditions ---
        entries_raw = cond_swing_buy & mtf_confirm & cond_strength & dm_entry_ok & adx_ok

        # --- 3. EMA cross-up re-entry ---
        if enable_trend_reentry:
            ema_reentry = calc_ema(close_s, ema_reentry_period).values
            cross_up = np.zeros(n, dtype=bool)
            cross_up[1:] = (close_arr[1:] > ema_reentry[1:]) & (close_arr[:-1] <= ema_reentry[:-1])
            reentry_raw = cross_up & trend_bull & cond_strength & dm_entry_ok & adx_ok
            entries_raw = entries_raw | reentry_raw

        # --- ATR computation (shared for stop and sizing) ---
        atr_vals = calc_atr(high_s, low_s, close_s, period=20).values

        # --- Exit conditions (before ATR stop, which needs entries_raw) ---
        cond_top_detected = top_warning_raw >= 2.5

        # --- 6. ATR trailing stop (stateful forward scan) ---
        atr_stop_exit = np.zeros(n, dtype=bool)
        in_position = False
        peak_close = 0.0

        for i in range(n):
            if entries_raw[i] and not in_position:
                in_position = True
                peak_close = close_arr[i]
            if in_position:
                if close_arr[i] > peak_close:
                    peak_close = close_arr[i]
                atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
                stop_level = peak_close - atr_stop_mult * atr_val
                if close_arr[i] < stop_level:
                    atr_stop_exit[i] = True
                    in_position = False
                    peak_close = 0.0
            # Any other exit also resets position
            if cond_swing_sell[i] or dm_exit_bearish[i] or cond_top_detected[i]:
                in_position = False
                peak_close = 0.0

        # --- 8. Combined exits with priority + reason attribution ---
        exits_raw = atr_stop_exit | dm_exit_bearish | cond_swing_sell | cond_top_detected

        exit_reason = np.full(n, "", dtype=object)
        exit_reason[cond_top_detected] = "top_detected"
        exit_reason[cond_swing_sell & ~cond_top_detected] = "zig_sell"
        exit_reason[dm_exit_bearish & ~atr_stop_exit] = "dm_regime"
        exit_reason[atr_stop_exit] = "atr_stop"  # highest priority overwrites

        # --- 5. 4-factor confidence scoring ---
        # (a) ZIG_Strength: swing amplitude relative to threshold
        zig_vals = tp_df["trend_pulse_zig_value"].values
        zig_diff = np.abs(np.diff(zig_vals, prepend=zig_vals[0]))
        zig_pct = np.where(zig_vals != 0, zig_diff / np.abs(zig_vals) * 100, 0.0)
        zig_strength = np.clip(zig_pct / zig_threshold_pct, 0.0, 1.0)

        # (b) DM_Health
        dm_health = np.where(
            dm_bullish, 1.0, np.where(dm_improving, 0.7, np.where(dm_deteriorating, 0.5, 0.0))
        )

        # (c) Trend_Alignment
        if enable_mtf_confirm:
            trend_align = np.where(weekly_bull & trend_bull & cond_strength, 1.0, 0.0)
        else:
            trend_align = np.where(trend_bull & cond_strength, 1.0, 0.0)

        # (d) Volatility_Quality (bell-curve preference for moderate vol)
        atr_series = pd.Series(atr_vals)
        atr_rank = atr_series.rolling(252, min_periods=60).rank(pct=True).values
        vol_quality = np.clip(1.0 - 2.0 * np.abs(atr_rank - 0.5), 0.0, 1.0)
        vol_quality = np.where(np.isnan(vol_quality), 0.5, vol_quality)

        # Weighted score
        confidence = (
            0.30 * zig_strength + 0.25 * dm_health + 0.30 * trend_align + 0.15 * vol_quality
        )

        # Filter by min_confidence
        entries_raw = entries_raw & (confidence >= min_confidence)

        # ATR-based vol scaler for sizing
        median_atr = np.nanmedian(atr_vals)
        if median_atr > 0:
            vol_scaler = np.clip(
                median_atr / np.where(atr_vals > 0, atr_vals, median_atr), 0.3, 1.5
            )
        else:
            vol_scaler = np.ones(n)
        entry_sizes_arr = np.clip(confidence * vol_scaler, min_pct, max_pct)
        entry_sizes_arr = np.where(entries_raw, entry_sizes_arr, 0.0)

        # --- 7. Signal shift +1 bar ---
        if signal_shift_bars > 0:
            entries_raw = np.roll(entries_raw, signal_shift_bars)
            entries_raw[:signal_shift_bars] = False
            exits_raw = np.roll(exits_raw, signal_shift_bars)
            exits_raw[:signal_shift_bars] = False
            entry_sizes_arr = np.roll(entry_sizes_arr, signal_shift_bars)
            entry_sizes_arr[:signal_shift_bars] = 0.0
            exit_reason = np.roll(exit_reason, signal_shift_bars)
            exit_reason[:signal_shift_bars] = ""

        # --- Warmup gate ---
        entries_raw[:warmup] = False
        exits_raw[:warmup] = False
        entry_sizes_arr[:warmup] = 0.0

        # --- v2.2: Post-exit cooldown (applied on final signals) ---
        # Block entries within cooldown_bars after any exit.
        # This runs after signal shift so it matches what VBT sees.
        if cooldown_bars > 0:
            bars_since_exit = cooldown_bars + 1
            for i in range(n):
                if exits_raw[i]:
                    bars_since_exit = 0
                if bars_since_exit < cooldown_bars:
                    if entries_raw[i]:
                        entries_raw[i] = False
                        entry_sizes_arr[i] = 0.0
                    bars_since_exit += 1
                else:
                    bars_since_exit += 1

        # --- Same-bar conflict: exit wins ---
        conflict = entries_raw & exits_raw
        entries_raw[conflict] = False
        entry_sizes_arr[conflict] = 0.0

        # Convert to Series
        entries = pd.Series(entries_raw, index=idx, dtype=bool)
        exits = pd.Series(exits_raw, index=idx, dtype=bool)
        self.entry_sizes = pd.Series(entry_sizes_arr, index=idx, dtype=np.float64)
        self.exit_reasons = pd.Series(exit_reason, index=idx, dtype=object)

        return entries, exits
