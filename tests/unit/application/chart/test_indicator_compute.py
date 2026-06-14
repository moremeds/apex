"""compute_indicator_series: compute-on-read indicator series for the chart surface.

The /indicators route hands argon a per-bar value series so it can draw indicator
lines/oscillators over the candles. The service fetches bars from livewire (with a
warmup lead so the visible window has valid values), runs the SAME pure-functional
``indicator.calculate(default_params)`` the live engine uses, and returns
``[{time, state, bar_close}]`` trimmed to the requested window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pytest

from src.application.chart.indicator_compute import (
    UnknownIndicatorError,
    compute_indicator_series,
)
from src.domain.events.domain_events import BarData
from src.domain.signals.indicators.registry import get_indicator_registry

_DAY = timedelta(days=1)
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _series(n: int) -> List[BarData]:
    """Deterministic non-constant closes so RSI is well-defined (not flat/NaN)."""
    bars: List[BarData] = []
    price = 100.0
    for i in range(n):
        price += 1.0 if i % 3 else -0.7  # zig-zag, net rising
        ts = _T0 + i * _DAY
        bars.append(
            BarData(
                symbol="AAPL",
                timeframe="1d",
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000 + i,
                timestamp=ts,
                bar_start=ts,
            )
        )
    return bars


class _FakeProvider:
    """Holds a full series; returns only the slice within [start, end] (like livewire)."""

    def __init__(self, bars: List[BarData]) -> None:
        self._bars = bars
        self.calls: List[Tuple[datetime, datetime]] = []

    async def fetch_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> List[BarData]:
        self.calls.append((start, end))
        return [b for b in self._bars if start <= b.timestamp <= end]


async def test_returns_one_point_per_visible_bar_with_state_and_close() -> None:
    full = _series(200)
    provider = _FakeProvider(full)
    registry = get_indicator_registry()

    start = full[120].timestamp
    end = full[160].timestamp

    points = await compute_indicator_series(provider, registry, "AAPL", "1d", "rsi", start, end)

    visible = [b for b in full if start <= b.timestamp <= end]
    assert len(points) == len(visible)
    assert all(start <= p["time"] <= end for p in points)
    assert [p["time"] for p in points] == sorted(p["time"] for p in points)

    first = points[0]
    assert "value" in first["state"]
    assert isinstance(first["state"]["value"], float)  # JSON-native, not numpy/NaN
    assert first["bar_close"] == visible[0].close


async def test_fetches_warmup_lead_before_requested_start() -> None:
    full = _series(200)
    provider = _FakeProvider(full)
    registry = get_indicator_registry()
    start = full[120].timestamp
    end = full[160].timestamp

    await compute_indicator_series(provider, registry, "AAPL", "1d", "rsi", start, end)

    called_start, called_end = provider.calls[0]
    assert called_start < start  # extended back to cover warmup
    assert called_end == end


async def test_unknown_indicator_raises() -> None:
    provider = _FakeProvider(_series(50))
    registry = get_indicator_registry()
    with pytest.raises(UnknownIndicatorError):
        await compute_indicator_series(
            provider, registry, "AAPL", "1d", "not_an_indicator", _T0, _T0 + 10 * _DAY
        )


async def test_empty_bars_returns_empty() -> None:
    provider = _FakeProvider([])  # livewire has nothing for this symbol
    registry = get_indicator_registry()
    points = await compute_indicator_series(
        provider, registry, "AAPL", "1d", "rsi", _T0, _T0 + 10 * _DAY
    )
    assert points == []
