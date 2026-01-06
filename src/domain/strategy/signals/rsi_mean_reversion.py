"""RSI Mean Reversion SignalGenerator for VectorBT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .indicators import rsi


class RSIMeanReversionSignalGenerator:
    """
    Vectorized RSI mean reversion signal generation using TA-Lib.

    Generates entry signals when RSI drops below oversold threshold,
    and exit signals when RSI rises above overbought threshold.

    Parameters:
        rsi_period: RSI calculation period (default 14)
        rsi_oversold: Oversold threshold for entry (default 30)
        rsi_overbought: Overbought threshold for exit (default 70)
    """

    @property
    def warmup_bars(self) -> int:
        """RSI needs period + 1 bars for first valid value."""
        return 15  # Default rsi_period + 1

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate RSI mean reversion signals.

        Args:
            data: OHLCV DataFrame with 'close' column.
            params: Strategy parameters.
                - rsi_period: RSI calculation period
                - rsi_oversold: Entry threshold
                - rsi_overbought: Exit threshold

        Returns:
            (entries, exits): Boolean series for entry/exit signals.
        """
        close = data["close"]

        # Support both prefixed and non-prefixed param names for compatibility
        period = int(params.get("rsi_period", 14))
        oversold = float(params.get("rsi_oversold", params.get("oversold", 30)))
        overbought = float(params.get("rsi_overbought", params.get("overbought", 70)))

        # Calculate RSI using TA-Lib
        rsi_values = rsi(close, period)

        # Entry: RSI drops into oversold territory
        entries = rsi_values < oversold

        # Exit: RSI rises into overbought territory
        exits = rsi_values > overbought

        return entries.fillna(False), exits.fillna(False)
