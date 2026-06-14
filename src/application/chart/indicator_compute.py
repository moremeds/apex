"""Compute-on-read indicator series for the chart read surface.

Given a visible window, fetch livewire bars (extended back by a warmup lead so the
window has valid, non-NaN values), run the SAME pure-functional
``indicator.calculate(default_params)`` the live IndicatorEngine uses
(``indicator_engine.py:601``), and return a per-bar ``[{time, state, bar_close}]``
series trimmed to the window. Deterministic: a pure function of bars + default params,
so the lines always match the live signals and the candles.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

# Bar duration per timeframe -- used to derive the warmup lead in calendar time.
# Public so the chart routes can reuse it for default windows.
TF_DELTAS: Dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}
DEFAULT_TF_DELTA = timedelta(days=1)


class UnknownIndicatorError(ValueError):
    """Raised when the requested indicator is not registered."""


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars / NaN to JSON-native types (NaN -> None), recursively."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    item = value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            item = value.item()  # numpy scalar -> python scalar
        except (ValueError, AttributeError):
            item = value
    if isinstance(item, float) and math.isnan(item):
        return None
    return item


def _bars_to_df(bars: List[Any]) -> pd.DataFrame:
    """Build the OHLCV DataFrame exactly as the live IndicatorEngine does."""
    rows = [
        {
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


async def compute_indicator_series(
    provider: Any,
    registry: Any,
    symbol: str,
    timeframe: str,
    indicator: str,
    start: datetime,
    end: datetime,
    *,
    safety: int = 3,
) -> List[Dict[str, Any]]:
    """Return ``[{time, state, bar_close}]`` for ``indicator`` over ``[start, end]``.

    ``provider`` must expose ``async fetch_bars(symbol, timeframe, start, end)``;
    ``registry`` must expose ``get(name)``. ``safety`` multiplies the warmup lead to
    cover non-trading gaps (weekends/holidays) when converting bar-count to time.
    """
    ind = registry.get(indicator)
    if ind is None:
        raise UnknownIndicatorError(indicator)

    delta = TF_DELTAS.get(timeframe, DEFAULT_TF_DELTA)
    warmup = max(int(getattr(ind, "warmup_periods", 0)), 0)
    fetch_start = start - delta * warmup * max(safety, 1)

    bars = await provider.fetch_bars(symbol, timeframe, fetch_start, end)
    if not bars:
        return []

    df = _bars_to_df(bars)
    result = ind.calculate(df, ind.default_params)

    points: List[Dict[str, Any]] = []
    prev: Optional[pd.Series] = None
    for ts, row in result.iterrows():
        py_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if start <= py_ts <= end:
            state = ind.get_state(row, prev, ind.default_params)
            close = df.at[ts, "close"] if ts in df.index else None
            points.append(
                {
                    "time": py_ts,
                    "state": _jsonable(state),
                    "bar_close": _jsonable(close),
                }
            )
        prev = row
    return points
