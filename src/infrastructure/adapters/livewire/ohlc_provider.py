"""DuckDB-over-parquet provider that reads livewire bronze for subscribed tickers.

Implements apex's HistoricalSourcePort (src/domain/interfaces/historical_source.py).
Reads are on-demand, one (symbol, timeframe, date-range) at a time -- never the
full universe.

UNVERIFIED: COLUMN_MAP below uses the documented/fixture schema (`ts` + OHLCV).
The REAL livewire bronze column names MUST be confirmed against a live parquet
(see Phase 1 plan Task 1) before production wiring.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import duckdb

from ....domain.events.domain_events import BarData
from .paths import SUPPORTED_TIMEFRAMES, parquet_path

# livewire column -> BarData field. CONFIRM the `ts`/OHLCV names against real data.
COLUMN_MAP = {
    "ts": "bar_start",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

# Bar duration per timeframe -- used to derive bar_end (not a zero-width bar).
_TF_DELTAS = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


class LivewireOhlcProvider:
    """Reads historical bars from livewire's bronze parquet via DuckDB.

    Satisfies HistoricalSourcePort (runtime_checkable Protocol).
    """

    def __init__(self, bronze_root: Path) -> None:
        self._root = Path(bronze_root)

    # --- HistoricalSourcePort ---
    @property
    def source_name(self) -> str:
        return "livewire"

    def supports_timeframe(self, timeframe: str) -> bool:
        return timeframe in SUPPORTED_TIMEFRAMES

    def get_supported_timeframes(self) -> List[str]:
        return list(SUPPORTED_TIMEFRAMES)

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        # parquet_path raises ValueError on an unsupported timeframe.
        path = parquet_path(self._root, symbol, timeframe)
        if not path.exists():
            return []
        return await asyncio.to_thread(self._query, path, symbol, timeframe, start, end)

    # --- internals ---
    def _query(
        self,
        path: Path,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        ts_col = next(k for k, v in COLUMN_MAP.items() if v == "bar_start")
        # NOTE: read_parquet path is inlined (NOT a bound parameter) -- DuckDB does
        # not accept a prepared-statement parameter for the parquet path. The path
        # is constructed by us from a validated symbol/timeframe, not user SQL.
        sql = (
            f"SELECT * FROM read_parquet('{path.as_posix()}') "
            f"WHERE {ts_col} >= ? AND {ts_col} <= ? ORDER BY {ts_col} ASC"
        )
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(sql, [start, end]).fetch_arrow_table().to_pylist()
        finally:
            con.close()
        return [self._row_to_bar(r, symbol, timeframe) for r in rows]

    @staticmethod
    def _row_to_bar(row: dict, symbol: str, timeframe: str) -> BarData:
        ts = row[next(k for k, v in COLUMN_MAP.items() if v == "bar_start")]
        end = ts + _TF_DELTAS.get(timeframe, timedelta(0))
        vol = row.get("volume")
        return BarData(
            symbol=symbol,
            timeframe=timeframe,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=int(vol) if vol is not None else None,
            vwap=row.get("vwap"),
            bar_start=ts,
            bar_end=end,
            timestamp=ts,  # event time = bar time, NOT construction-time now()
            source="livewire",
        )
