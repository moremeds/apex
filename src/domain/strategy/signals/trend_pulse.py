"""
TrendPulse SignalGenerator for VectorBT backtesting.

Combines TrendPulseIndicator (swing/trend/top detection) with DualMACDIndicator
(momentum confirmation) to produce entry/exit signals with confidence-based sizing.

Entry conditions (ALL required):
    1. swing_signal == BUY
    2. trend_filter == BULLISH
    3. trend_strength >= moderate_threshold
    4. dualmacd trend_state in (BULLISH, IMPROVING)

Exit conditions (ANY triggers):
    1. swing_signal == SELL
    2. top_warning == TOP_DETECTED
    3. dualmacd trend_state == BEARISH

Sizing:
    entry_sizes = clip(confidence, min_pct, max_pct) on entry bars, 0.0 elsewhere.

Same-bar conflict: If exit and entry on same bar, exit wins (entry_sizes zeroed).

Note: All gating logic lives here in generate(). Signal rules.yaml only binds
swing_signal state changes for TUI alerting — it does NOT gate backtesting entries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ....domain.signals.indicators.momentum.dual_macd import DualMACDIndicator
from ....domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator


class TrendPulseSignalGenerator:
    """
    Vectorized TrendPulse + DualMACD signal generator.

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

        After calling, self.entry_sizes is set to a float Series in [0, 1]
        aligned to data.index, representing confidence-based position sizing.

        Args:
            data: DataFrame with open, high, low, close, volume columns.
            params: Strategy parameters (overrides defaults).

        Returns:
            (entries, exits) boolean Series aligned to data.index.
        """
        n = len(data)
        idx = data.index

        if n == 0:
            empty = pd.Series(False, index=idx, dtype=bool)
            self.entry_sizes = pd.Series(0.0, index=idx, dtype=np.float64)
            return empty, empty.copy()

        # --- Extract params with defaults ---
        zig_threshold_pct = float(params.get("zig_threshold_pct", 5.0))
        swing_filter_bars = int(params.get("swing_filter_bars", 5))
        trend_strength_moderate = float(params.get("trend_strength_moderate", 0.3))
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

        # Confidence weights
        confidence_weights = params.get("confidence_weights", (0.4, 0.3, 0.3))

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
            "confidence_weights": confidence_weights,
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
        swing_raw = tp_df["trend_pulse_swing_signal_raw"].values
        trend_filter = tp_df["trend_pulse_trend_filter"].values
        adx_vals = tp_df["trend_pulse_adx"].values
        top_warning_raw = tp_df["trend_pulse_top_warning_raw"].values

        slow_hist = dm_df["slow_histogram"].values
        slow_hist_delta = dm_df["slow_hist_delta"].values

        # --- Compute trend_strength per bar ---
        trend_strength = np.where(
            np.isnan(adx_vals), 0.0, np.clip(adx_vals / norm_max_adx, 0.0, 1.0)
        )

        # --- Compute DualMACD trend_state per bar ---
        # BULLISH: slow_hist > 0, delta >= 0
        # IMPROVING: slow_hist < 0, delta > 0
        # DETERIORATING: slow_hist > 0, delta < 0
        # BEARISH: slow_hist <= 0, delta <= 0
        sh = np.where(np.isnan(slow_hist), 0.0, slow_hist)
        shd = np.where(np.isnan(slow_hist_delta), 0.0, slow_hist_delta)

        dm_bullish = (sh > 0) & (shd >= 0)
        dm_improving = (sh < 0) & (shd > 0)
        dm_bearish = ~dm_bullish & ~dm_improving & ~((sh > 0) & (shd < 0))
        # For entry: allow BULLISH or IMPROVING
        dm_entry_ok = dm_bullish | dm_improving
        # For exit: BEARISH = slow_hist <= 0 AND delta <= 0
        dm_exit_bearish = (sh <= 0) & (shd <= 0)

        # --- Entry conditions (all 4 must be true) ---
        cond_swing_buy = swing_raw > 0.5  # BUY
        cond_trend_bull = trend_filter > 0.5  # BULLISH
        cond_strength = trend_strength >= trend_strength_moderate
        entries_raw = cond_swing_buy & cond_trend_bull & cond_strength & dm_entry_ok

        # --- Exit conditions (any triggers) ---
        cond_swing_sell = swing_raw < -0.5  # SELL
        cond_top_detected = top_warning_raw >= 2.5  # TOP_DETECTED
        exits_raw = cond_swing_sell | cond_top_detected | dm_exit_bearish

        # --- Confidence-based sizing ---
        # Compute confidence per bar using TrendPulse state logic
        conf_trend = trend_strength  # already [0, 1]

        # EMA alignment component
        ema_cols = [f"trend_pulse_ema_{p}" for p in (14, 25, 99, 144, 453)]
        ema_arrays = [tp_df[c].values for c in ema_cols]
        aligned_bull = np.ones(n, dtype=bool)
        for i in range(len(ema_arrays) - 1):
            a = ema_arrays[i]
            b = ema_arrays[i + 1]
            aligned_bull &= np.where(np.isnan(a) | np.isnan(b), False, a >= b)

        conf_align = np.where(aligned_bull, 1.0, 0.7)

        # Top risk component
        top_risk = np.where(
            top_warning_raw >= 2.5,
            1.0,
            np.where(top_warning_raw >= 1.5, 0.6, np.where(top_warning_raw >= 0.5, 0.3, 0.0)),
        )
        conf_top = np.clip(1.0 - top_risk, 0.0, 1.0)

        w1, w2, w3 = confidence_weights[0], confidence_weights[1], confidence_weights[2]
        confidence = w1 * conf_trend + w2 * conf_align + w3 * conf_top

        # Clip to [min_pct, max_pct] on entry bars, 0.0 elsewhere
        sizing = np.clip(confidence, min_pct, max_pct)
        entry_sizes = np.where(entries_raw, sizing, 0.0)

        # --- Warmup gate: mask first warmup_bars ---
        entries_raw[:warmup] = False
        exits_raw[:warmup] = False
        entry_sizes[:warmup] = 0.0

        # --- Same-bar conflict: exit wins ---
        conflict = entries_raw & exits_raw
        entries_raw[conflict] = False
        entry_sizes[conflict] = 0.0

        # Convert to Series
        entries = pd.Series(entries_raw, index=idx, dtype=bool)
        exits = pd.Series(exits_raw, index=idx, dtype=bool)
        self.entry_sizes = pd.Series(entry_sizes, index=idx, dtype=np.float64)

        return entries, exits
