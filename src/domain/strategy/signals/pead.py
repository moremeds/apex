"""PEAD (Post-Earnings Announcement Drift) SignalGenerator for VectorBT.

Unlike price-based signal generators, PEAD entries are triggered by
earnings surprise events, not OHLCV price patterns. The generator
receives earnings data via ``params["earnings_data"]`` — a list of dicts
with ``report_date``, ``sue_score``, ``gap_pct``, ``volume_ratio``,
``revenue_beat``, and ``quality_score`` fields.

Look-ahead prevention:
    Only earnings with ``report_date < bar_date`` generate signals.
    Entry is delayed by ``entry_delay_bars`` (default 2) after report.

Exit logic (first-touch, vectorized forward scan):
    1. Profit target: close >= entry_price * (1 + profit_target_pct)
    2. Stop loss: close <= entry_price * (1 + stop_loss_pct)
    3. Timeout: bars_held >= max_hold_bars → exit at close

Intentional difference from live screener:
    - Live screener uses intraday OHLC for first-touch (high/low).
    - VectorBT backtest uses daily close for simplicity and consistency
      with how VectorBT prices exits. This slightly understates wins
      (misses intraday target touches) but is conservative.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..param_loader import get_strategy_params

_DEFAULTS = get_strategy_params("pead")


class PeadSignalGenerator:
    """Vectorized PEAD signal generation for VectorBT backtesting.

    Earnings-event-driven entries with target/stop/timeout exits.

    Required ``params["earnings_data"]``:
        List of dicts, each with at minimum::

            {
                "report_date": "2024-01-25",  # ISO date string or date object
                "sue_score": 3.5,
                "gap_pct": 0.04,
                "volume_ratio": 2.8,
                "revenue_beat": True,
                "quality_score": 72.0,  # 0-100, optional for sizing
            }
    """

    @property
    def warmup_bars(self) -> int:
        """No indicator warmup needed — entries from external events."""
        return 0

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate PEAD entry/exit signals from OHLCV + earnings data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Strategy parameters. Must include ``earnings_data`` key.
            secondary_data: Not used.

        Returns:
            (entries, exits) boolean Series aligned to data.index.
        """
        effective = {**_DEFAULTS, **params}

        # Extract parameters
        min_sue = float(effective.get("min_sue", 2.0))
        min_gap = float(effective.get("min_gap_pct", 0.02))
        min_vol = float(effective.get("min_volume_ratio", 2.0))
        target_pct = float(effective.get("profit_target_pct", 0.06))
        stop_pct = float(effective.get("stop_loss_pct", -0.05))
        max_hold = int(effective.get("max_hold_bars", 25))
        delay = int(effective.get("entry_delay_bars", 2))
        use_sizing = bool(effective.get("use_quality_sizing", True))
        min_quality = float(effective.get("min_quality_for_entry", 0.0))

        # Parse earnings data
        earnings_data: list[dict[str, Any]] = effective.get("earnings_data", [])

        n = len(data)
        close = data["close"].values.astype(np.float64)
        bar_dates = _extract_dates(data.index)

        # Build entry map: bar_index -> quality_score for qualified earnings
        entry_map = self._build_entry_map(
            earnings_data, bar_dates, delay, min_sue, min_gap, min_vol, min_quality
        )

        # Forward scan: generate entries and exits
        entries_arr = np.zeros(n, dtype=bool)
        exits_arr = np.zeros(n, dtype=bool)
        sizing_arr = np.ones(n, dtype=np.float64)

        in_position = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_position:
                bars_held += 1
                pnl_pct = (close[i] - entry_price) / entry_price

                # Check exits (priority: stop > target > timeout)
                if pnl_pct <= stop_pct:
                    exits_arr[i] = True
                    in_position = False
                elif pnl_pct >= target_pct:
                    exits_arr[i] = True
                    in_position = False
                elif bars_held >= max_hold:
                    exits_arr[i] = True
                    in_position = False
            else:
                # Check for entry
                if i in entry_map:
                    entries_arr[i] = True
                    in_position = True
                    entry_price = close[i]
                    bars_held = 0
                    if use_sizing:
                        sizing_arr[i] = entry_map[i] / 100.0  # Normalize 0-100 → 0-1

        entries = pd.Series(entries_arr, index=data.index, dtype=bool)
        exits = pd.Series(exits_arr, index=data.index, dtype=bool)

        if use_sizing:
            self.entry_sizes = pd.Series(np.clip(sizing_arr, 0.1, 1.0), index=data.index)

        return entries, exits

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_entry_map(
        earnings_data: list[dict[str, Any]],
        bar_dates: list[date],
        delay: int,
        min_sue: float,
        min_gap: float,
        min_vol: float,
        min_quality: float,
    ) -> dict[int, float]:
        """Map qualifying earnings events to bar indices.

        For each earning:
            1. Parse report_date
            2. Apply filters (SUE, gap, volume, quality)
            3. Find bar_index = first bar that is >= (report_date + delay trading days)
            4. Look-ahead prevention: only report_date < bar_date

        Returns:
            {bar_index: quality_score} for each qualifying entry.
        """
        if not earnings_data or not bar_dates:
            return {}

        # Sort bar_dates for binary search
        entry_map: dict[int, float] = {}

        for earning in earnings_data:
            # Parse report date
            rdate = earning.get("report_date")
            if rdate is None:
                continue
            if isinstance(rdate, str):
                rdate = date.fromisoformat(rdate)

            # Apply filters
            sue = float(earning.get("sue_score", 0.0))
            gap = float(earning.get("gap_pct", 0.0))
            vol = float(earning.get("volume_ratio", 0.0))
            quality = float(earning.get("quality_score", 50.0))

            if sue < min_sue or gap < min_gap or vol < min_vol:
                continue
            if quality < min_quality:
                continue

            # Find entry bar: first bar >= report_date + delay days
            # This is approximate — uses calendar days, not trading days
            # (trading day arithmetic would require NYSE calendar import
            # which is avoided in the signal generator for portability)
            target_idx = _find_entry_bar(bar_dates, rdate, delay)
            if target_idx is not None and target_idx not in entry_map:
                entry_map[target_idx] = quality

        return entry_map


def _extract_dates(index: pd.DatetimeIndex) -> list[date]:
    """Convert DatetimeIndex to list of date objects."""
    return [d.date() if hasattr(d, "date") else d for d in index]


def _find_entry_bar(bar_dates: list[date], report_date: date, delay: int) -> int | None:
    """Find first bar index that is at least ``delay`` bars after report_date.

    Look-ahead safe: the entry bar is always strictly after the report date,
    and at least ``delay`` trading bars later.

    Returns:
        Bar index, or None if report_date is outside the data range.
    """
    # Find the first bar after report_date
    first_after = None
    for i, d in enumerate(bar_dates):
        if d > report_date:
            first_after = i
            break

    if first_after is None:
        return None

    # Advance by delay bars (these are trading bars since OHLCV only has trading days)
    target = first_after + delay - 1
    if target >= len(bar_dates):
        return None

    return target
