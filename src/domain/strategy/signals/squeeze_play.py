"""SqueezePlay SignalGenerator for VectorBT backtesting.

Simplified vectorized approximation of SqueezePlay entry/exit logic for
fast screening. Differs from the event-driven SqueezePlayStrategy:
- No regime gating (all regimes treated equally)
- Long-only (no short entries on close < lower BB)
- Simplified ATR trail (uses expanding max, not per-trade peak)

Signal shift: entries are shifted +1 bar (trade at next bar open).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..param_loader import get_strategy_params
from .indicators import adx, atr, bbands, ema

# Canonical defaults loaded from config/strategy/squeeze_play.yaml
_DEFAULTS = get_strategy_params("squeeze_play")


class SqueezePlaySignalGenerator:
    """
    Vectorized SqueezePlay signal generation.

    Entry: Squeeze release with persistence + close outside BB + ADX.
    Exit: ATR trail OR hard stop.
    """

    @property
    def warmup_bars(self) -> int:
        return 50

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate SqueezePlay entry/exit signals.

        Args:
            data: OHLCV DataFrame.
            params: Strategy parameters.

        Returns:
            (entries, exits): Boolean series.
        """
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Parameters: YAML defaults merged with caller overrides
        effective = {**_DEFAULTS, **params}
        bb_period = int(effective.get("bb_period", 20))
        bb_std_val = float(effective.get("bb_std", 2.0))
        kc_mult = float(effective.get("kc_multiplier", 1.5))
        release_persist = int(effective.get("release_persist_bars", 2))
        outside_persist = int(effective.get("close_outside_bars", 2))
        atr_mult = float(effective.get("atr_stop_mult", 2.5))
        adx_min = float(effective.get("adx_min", 20.0))
        hard_stop_pct = float(effective.get("hard_stop_pct", 0.08))

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = bbands(close, bb_period, bb_std_val, bb_std_val)

        # Keltner Channels (EMA + ATR*mult)
        kc_mid = ema(close, bb_period)
        atr_vals = atr(high, low, close, 14)
        kc_upper = kc_mid + kc_mult * atr_vals
        kc_lower = kc_mid - kc_mult * atr_vals

        # Squeeze: BB inside KC
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        squeeze_off = ~squeeze_on

        # Release persistence: consecutive bars with squeeze OFF
        release_count = _consecutive_true(squeeze_off)

        # Close outside BB persistence
        close_above_bb = close > bb_upper
        close_below_bb = close < bb_lower
        close_outside = close_above_bb | close_below_bb
        outside_count = _consecutive_true(close_outside)

        # Persistence filter
        release_ok = release_count >= release_persist
        outside_ok = outside_count >= outside_persist

        # ADX filter
        adx_vals = adx(high, low, close, 14)
        adx_ok = adx_vals >= adx_min

        # Direction: above upper BB = long, below lower BB = short
        # For simple SignalGenerator, use long-only
        long_entry = close_above_bb & release_ok & outside_ok & adx_ok

        # Shift +1 for execution realism
        entries = long_entry.shift(1, fill_value=False)

        # Exits: ATR trail or hard stop
        rolling_peak = close.expanding().max()
        trail_stop = rolling_peak - atr_mult * atr_vals
        trail_exit = close < trail_stop

        entry_proxy = close.shift(1)
        hard_stop_level = entry_proxy * (1.0 - hard_stop_pct)
        hard_exit = close < hard_stop_level

        # Squeeze re-engages (volatility contracts again)
        squeeze_re_enter = squeeze_on & squeeze_off.shift(1, fill_value=False)

        exits = (trail_exit | hard_exit | squeeze_re_enter).fillna(False).astype(bool)

        return entries, exits


def _consecutive_true(series: pd.Series) -> pd.Series:
    """Count consecutive True values, reset on False."""
    groups = (~series).cumsum()
    result = series.groupby(groups).cumsum()
    return result.fillna(0).astype(int)
