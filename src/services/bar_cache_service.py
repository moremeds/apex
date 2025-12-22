"""
Bar cache utilities for historical data.

Contains:
- BarPeriod: Period selector for bar requests
- BarCacheStore: In-memory LRU cache for bar data

Note: The actual historical data fetching is done via HistoricalDataService
which uses the monitoring adapter on the main event loop.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Tuple

from ..domain.events.domain_events import BarData


# -----------------------------------------------------------------------------
# BarPeriod - period selector for bar requests
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BarPeriod:
    """Period selector for bar requests."""

    mode: str  # "bars" or "range"
    count: Optional[int] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @staticmethod
    def bars(count: int) -> "BarPeriod":
        return BarPeriod(mode="bars", count=count)

    @staticmethod
    def range(start: datetime | date, end: datetime | date) -> "BarPeriod":
        start_dt = _as_datetime(start)
        end_dt = _as_datetime(end)
        return BarPeriod(mode="range", start=start_dt, end=end_dt)


def _as_datetime(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


# -----------------------------------------------------------------------------
# BarCacheStore - in-memory LRU cache
# -----------------------------------------------------------------------------


class BarCacheStore:
    """In-memory LRU cache for bar data."""

    def __init__(self, max_entries: int = 512) -> None:
        self._max_entries = max_entries
        self._cache: OrderedDict[Tuple[str, str], List[BarData]] = OrderedDict()

    def get_last(self, key: Tuple[str, str], count: int) -> Optional[List[BarData]]:
        """Get last N bars for key. Returns None if not enough bars cached."""
        bars = self._cache.get(key)
        if not bars or len(bars) < count:
            return None
        self._cache.move_to_end(key)
        return bars[-count:]

    def get_range(
        self,
        key: Tuple[str, str],
        start: datetime,
        end: datetime,
    ) -> Optional[List[BarData]]:
        """Get bars in date range. Returns None if range not fully covered."""
        bars = self._cache.get(key)
        if not bars:
            return None
        if not bars[0].timestamp or not bars[-1].timestamp:
            return None
        if bars[0].timestamp > start or bars[-1].timestamp < end:
            return None
        self._cache.move_to_end(key)
        return [bar for bar in bars if bar.timestamp and start <= bar.timestamp <= end]

    def set(self, key: Tuple[str, str], bars: List[BarData]) -> None:
        """Cache bars for key with LRU eviction."""
        self._cache[key] = bars
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def get(
        self,
        symbol: str,
        timeframe: str,
        period: BarPeriod,
    ) -> Optional[List[BarData]]:
        """Get bars matching period. Returns None on cache miss."""
        key = (symbol, timeframe)
        if period.mode == "bars" and period.count:
            return self.get_last(key, period.count)
        elif period.mode == "range" and period.start and period.end:
            return self.get_range(key, period.start, period.end)
        return None

    def put(self, symbol: str, timeframe: str, bars: List[BarData]) -> None:
        """Store bars in cache."""
        key = (symbol, timeframe)
        self.set(key, bars)
