"""DuckDB-over-parquet provider that reads livewire bronze for subscribed tickers.

Implements apex's HistoricalSourcePort (src/domain/interfaces/historical_source.py).
Reads are on-demand, one (symbol, timeframe, date-range) at a time -- never the
full universe.

Bronze schema matched to livewire's writers (``clients/bronze_client.py`` +
``clients/intraday_bronze_client.py``, 2026-06-14): daily bars are keyed by
``trade_date`` (date32) and carry ``adj_close``; intraday bars are keyed by
``bar_timestamp`` (timestamp us, tz=UTC). Both also carry ``symbol_id`` (ignored
here -- the symbol comes from the partition). OHLCV column names match 1:1.

Smoke-tested 2026-06-14 against the real livewire bronze data lake: AAPL ``1d``
(``trade_date``/DATE) and ``1m`` (``bar_timestamp``/TIMESTAMPTZ) read back correct
OHLCV at the right instants.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Literal

import duckdb

from ....domain.events.domain_events import BarData
from .paths import SUPPORTED_TIMEFRAMES, daily_silver_path, factor_path, parquet_path

# livewire keys daily bars by `trade_date` (a DATE) and intraday bars by
# `bar_timestamp` (a tz-aware UTC TIMESTAMP). OHLCV columns are read by name; extra
# columns (symbol_id, adj_close) are ignored.
_DAILY_TS_COLUMN = "trade_date"
_INTRADAY_TS_COLUMN = "bar_timestamp"

PriceMode = Literal["raw", "adjusted"]


class AdjustedDataUnavailable(RuntimeError):
    """Raised when adjusted mode cannot prove complete Silver coverage."""


def _timestamp_column(timeframe: str) -> str:
    return _DAILY_TS_COLUMN if timeframe == "1d" else _INTRADAY_TS_COLUMN


def _to_utc_datetime(value: Any) -> datetime:
    """Coerce a livewire timestamp to a UTC-*labelled* tz-aware datetime.

    DuckDB returns ``trade_date`` (date32) as ``datetime.date`` and ``bar_timestamp``
    (TIMESTAMPTZ) as a tz-aware ``datetime`` -- but in the *session* timezone, NOT UTC
    (e.g. ``Asia/Hong_Kong`` +08:00 on a HK-locale box). The instant is correct, the
    label is not. We must ``astimezone(UTC)`` so seeded warmup bars (session tz) and
    live tick bars (UTC) share one offset: a mixed-offset column crashes
    ``pd.to_datetime(...)`` in the indicator engine and silently kills live compute.
    datetime is a subclass of date, so check it first.
    """
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc)
            if value.tzinfo is not None
            else value.replace(tzinfo=timezone.utc)
        )
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    raise TypeError(f"unexpected livewire timestamp type: {type(value)!r}")


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

    def __init__(
        self,
        bronze_root: Path,
        silver_root: Path | None = None,
        price_mode: PriceMode = "raw",
    ) -> None:
        if price_mode not in ("raw", "adjusted"):
            raise ValueError(f"unsupported Livewire price mode: {price_mode!r}")
        self._bronze_root = Path(bronze_root)
        self._silver_root = Path(silver_root) if silver_root is not None else None
        self._price_mode = price_mode

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
        bronze_path = parquet_path(self._bronze_root, symbol, timeframe)
        if self._price_mode == "raw":
            if not bronze_path.exists():
                return []
            return await asyncio.to_thread(
                self._query,
                bronze_path,
                symbol,
                timeframe,
                start,
                end,
            )
        if self._silver_root is None:
            raise AdjustedDataUnavailable("Silver root is not configured")
        if timeframe == "1d":
            path = daily_silver_path(self._silver_root, symbol)
            if not path.exists():
                raise AdjustedDataUnavailable(f"Silver daily artifact is missing for {symbol}")
            return await asyncio.to_thread(self._query, path, symbol, timeframe, start, end)

        if not bronze_path.exists():
            return []
        factors = factor_path(self._silver_root, symbol)
        if not factors.exists():
            raise AdjustedDataUnavailable(f"Silver factor artifact is missing for {symbol}")
        return await asyncio.to_thread(
            self._query_adjusted_intraday,
            bronze_path,
            factors,
            symbol,
            timeframe,
            start,
            end,
        )

    # --- internals ---
    def _query(
        self,
        path: Path,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        ts_col = _timestamp_column(timeframe)
        # NOTE: read_parquet path is inlined (NOT a bound parameter) -- DuckDB does
        # not accept a prepared-statement parameter for the parquet path. The path
        # is constructed by us from a validated symbol/timeframe, not user SQL.
        sql = (
            f"SELECT * FROM read_parquet('{path.as_posix()}') "
            f"WHERE {ts_col} >= ? AND {ts_col} <= ? ORDER BY {ts_col} ASC"
        )
        # Daily `trade_date` is a DATE -- bind calendar-date params so the comparison
        # is tz-agnostic (avoids DATE-vs-TIMESTAMPTZ session-tz surprises). Intraday
        # `bar_timestamp` is TIMESTAMPTZ -- bind the tz-aware datetimes directly.
        params = [start.date(), end.date()] if timeframe == "1d" else [start, end]
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(sql, params).fetch_arrow_table().to_pylist()
        finally:
            con.close()
        return [self._row_to_bar(r, symbol, timeframe) for r in rows]

    def _query_adjusted_intraday(
        self,
        bronze_path: Path,
        factors_path: Path,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        sql = """
            WITH raw AS (
                SELECT *, count(*) OVER () AS raw_count
                FROM read_parquet(?)
                WHERE bar_timestamp >= ? AND bar_timestamp <= ?
            )
            SELECT
                b.bar_timestamp,
                b.open * f.price_adjustment_factor AS open,
                b.high * f.price_adjustment_factor AS high,
                b.low * f.price_adjustment_factor AS low,
                b.close * f.price_adjustment_factor AS close,
                CAST(ROUND(b.volume * f.split_volume_factor) AS BIGINT) AS volume,
                b.raw_count,
                f.adjustment_revision
            FROM raw b
            LEFT JOIN read_parquet(?) f
              ON CAST(timezone('America/New_York', b.bar_timestamp) AS DATE)
                 >= COALESCE(f.effective_start, DATE '0001-01-01')
             AND CAST(timezone('America/New_York', b.bar_timestamp) AS DATE)
                 <= COALESCE(f.effective_end, DATE '9999-12-31')
            ORDER BY b.bar_timestamp
        """
        con = duckdb.connect(database=":memory:")
        try:
            rows = (
                con.execute(
                    sql,
                    [bronze_path.as_posix(), start, end, factors_path.as_posix()],
                )
                .fetch_arrow_table()
                .to_pylist()
            )
        finally:
            con.close()
        if not rows:
            return []
        raw_count = int(rows[0]["raw_count"])
        if len(rows) != raw_count or any(row["adjustment_revision"] is None for row in rows):
            raise AdjustedDataUnavailable(
                f"incomplete or overlapping factor coverage for {symbol} {timeframe}"
            )
        return [self._row_to_bar(row, symbol, timeframe) for row in rows]

    @staticmethod
    def _row_to_bar(row: dict, symbol: str, timeframe: str) -> BarData:
        ts = _to_utc_datetime(row[_timestamp_column(timeframe)])
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
