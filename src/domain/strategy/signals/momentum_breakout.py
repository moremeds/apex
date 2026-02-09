"""Momentum Breakout SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..param_loader import get_strategy_params
from .indicators import momentum

_DEFAULTS = get_strategy_params("momentum_breakout")


class MomentumBreakoutSignalGenerator:
    """
    Vectorized momentum breakout signal generation using TA-Lib.

    Generates entry signals when momentum exceeds threshold,
    and exit signals when momentum turns negative.

    Parameters:
        lookback_days: Momentum calculation period (default 20)
        momentum_threshold: Entry threshold for momentum (default 0.0)
    """

    @property
    def warmup_bars(self) -> int:
        """Momentum needs lookback_days + 1 bars."""
        return 21  # Default lookback_days + 1

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate momentum breakout signals.

        Args:
            data: OHLCV DataFrame with 'close' column.
            params: Strategy parameters.
                - lookback_days: Momentum period
                - momentum_threshold: Entry threshold

        Returns:
            (entries, exits): Boolean series for entry/exit signals.
        """
        close = data["close"]

        effective = {**_DEFAULTS, **params}
        lookback = int(effective.get("lookback_days", 20))
        threshold = float(effective.get("momentum_threshold", 0.0))

        # Calculate momentum using TA-Lib
        mom = momentum(close, lookback)

        # Entry: momentum exceeds positive threshold
        entries = mom > threshold

        # Exit: momentum turns negative
        exits = mom < 0

        return entries.fillna(False), exits.fillna(False)
