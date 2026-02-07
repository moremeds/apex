"""PulseDip SignalGenerator for VectorBT backtesting.

Simplified vectorized approximation of PulseDip entry/exit logic for
fast screening. Differs from the event-driven PulseDipStrategy:
- No regime gating (all regimes treated equally)
- No confluence scoring
- Simplified ATR trail (uses expanding max, not per-trade peak)

Signal shift: entries are shifted +1 bar (trade at next bar open).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import atr, ema, rsi


class PulseDipSignalGenerator:
    """
    Vectorized PulseDip signal generation.

    Entry: Close > EMA(trend) AND RSI < threshold.
    Exit: Hard stop OR ATR trail OR RSI > 65 OR time stop.
    """

    @property
    def warmup_bars(self) -> int:
        return 120  # Max(ema_trend_period=99, rsi=14, atr=14) + buffer

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate PulseDip entry/exit signals.

        Args:
            data: OHLCV DataFrame.
            params: Strategy parameters.

        Returns:
            (entries, exits): Boolean series, entries shifted +1 for realism.
        """
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Parameters with defaults
        ema_period = int(params.get("ema_trend_period", 99))
        rsi_period = int(params.get("rsi_period", 14))
        rsi_thresh = float(params.get("rsi_entry_threshold", 35.0))
        atr_mult = float(params.get("atr_stop_mult", 3.0))
        max_hold = int(params.get("max_hold_bars", 40))
        hard_stop_pct = float(params.get("hard_stop_pct", 0.08))

        # Calculate indicators
        ema_trend = ema(close, ema_period)
        rsi_vals = rsi(close, rsi_period)
        atr_vals = atr(high, low, close, 14)

        # Entry conditions (all must be true)
        cond_ema = close > ema_trend
        cond_rsi = rsi_vals < rsi_thresh

        # Raw entry signal
        raw_entries = cond_ema & cond_rsi

        # Signal shift +1 bar (entry at next bar open)
        entries = raw_entries.shift(1).fillna(False).astype(bool)

        # Exit conditions (any triggers exit)
        # RSI overbought exit
        rsi_exit = rsi_vals > 65

        # ATR trailing stop (simplified: close drops below entry - ATR*mult)
        # For vectorized: use rolling max as proxy for peak
        rolling_peak = close.expanding().max()
        trail_stop = rolling_peak - atr_mult * atr_vals
        trail_exit = close < trail_stop

        # Hard stop
        # Use shifted close as entry proxy
        entry_price_proxy = close.shift(1)
        hard_stop_level = entry_price_proxy * (1.0 - hard_stop_pct)
        hard_stop_exit = close < hard_stop_level

        # Combine exits
        exits = (rsi_exit | trail_exit | hard_stop_exit).fillna(False).astype(bool)

        return entries, exits
