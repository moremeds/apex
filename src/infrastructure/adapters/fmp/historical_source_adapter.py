"""FMP adapter wrapper implementing HistoricalSourcePort.

Wraps FMPHistoricalAdapter to provide the async HistoricalSourcePort interface
used by HistoricalDataManager. Handles chunked fetching for intraday timeframes
where FMP caps per-request responses.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List

from ....domain.events.domain_events import BarData
from ....utils.logging_setup import get_logger
from .historical_adapter import FMPHistoricalAdapter

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Per-request caps (conservative vs empirical ~410/~245 bar limits)
_FMP_CHUNK_DAYS = {"1h": 85, "4h": 170, "1d": 3650}

# Warmup-safe max history (matches Yahoo so manager requests same range)
MAX_HISTORY_DAYS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "4h": 730,
    "1d": 18250,
}

# FMP timeframe support (mirrors _TIMEFRAME_MAP in historical_adapter.py)
_SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}


class FMPHistoricalSourceAdapter:
    """FMP wrapper implementing HistoricalSourcePort for HistoricalDataManager.

    Chunks large date ranges into FMP per-request limits transparently.
    """

    def __init__(self) -> None:
        self._inner = FMPHistoricalAdapter()

    @property
    def source_name(self) -> str:
        return "fmp"

    def supports_timeframe(self, timeframe: str) -> bool:
        return timeframe in _SUPPORTED_TIMEFRAMES

    def get_supported_timeframes(self) -> List[str]:
        return list(_SUPPORTED_TIMEFRAMES)

    def get_max_history_days(self, timeframe: str) -> int:
        return MAX_HISTORY_DAYS.get(timeframe, 365)

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        """Fetch bars with automatic chunking for FMP per-request caps."""
        chunk_days = _FMP_CHUNK_DAYS.get(timeframe, 3650)
        chunks = self._compute_chunks(start, end, chunk_days)

        all_bars: List[BarData] = []
        for chunk_start, chunk_end in chunks:
            df = await asyncio.to_thread(
                self._inner.fetch_bars,
                symbol,
                timeframe,
                chunk_start.date(),
                chunk_end.date(),
            )

            if df.empty:
                continue

            bars = self._dataframe_to_bars(df, symbol, timeframe)
            all_bars.extend(bars)

        # Deduplicate by bar_start
        seen = set()
        unique_bars: List[BarData] = []
        for bar in all_bars:
            key = bar.bar_start
            if key not in seen:
                seen.add(key)
                unique_bars.append(bar)

        return sorted(unique_bars, key=lambda b: b.bar_start or datetime.min)

    @staticmethod
    def _compute_chunks(
        start: datetime, end: datetime, chunk_days: int
    ) -> List[tuple[datetime, datetime]]:
        """Split [start, end] into chunk_days-sized windows (both inclusive)."""
        chunks = []
        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        return chunks

    @staticmethod
    def _dataframe_to_bars(
        df: "pd.DataFrame", symbol: str, timeframe: str  # noqa: F821
    ) -> List[BarData]:
        """Convert FMP DataFrame to BarData list (same pattern as Yahoo adapter)."""
        import pandas as pd

        bars: List[BarData] = []
        for idx, row in df.iterrows():
            if hasattr(idx, "to_pydatetime"):
                bar_time = idx.to_pydatetime()
            else:
                bar_time = idx

            # Ensure timezone-aware
            if bar_time.tzinfo is None:
                import pytz

                bar_time = pytz.UTC.localize(bar_time)

            bar = BarData(
                symbol=symbol,
                timeframe=timeframe,
                open=float(row["open"]) if not pd.isna(row.get("open")) else None,
                high=float(row["high"]) if not pd.isna(row.get("high")) else None,
                low=float(row["low"]) if not pd.isna(row.get("low")) else None,
                close=float(row["close"]) if not pd.isna(row.get("close")) else None,
                volume=(
                    int(row["volume"])
                    if "volume" in row.index and not pd.isna(row["volume"])
                    else None
                ),
                bar_start=bar_time,
                source="fmp",
                timestamp=bar_time,
            )
            bars.append(bar)

        return bars
