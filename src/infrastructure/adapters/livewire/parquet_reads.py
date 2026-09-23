"""Read rows from already-resolved Livewire parquet artifacts.

``ohlc_provider.py`` decides WHICH artifact a request reads (tree, listing, Silver pin);
this module reads it: the SQL, the DuckDB execution with its deadline, and the
row -> BarData conversion. Every path and value is a bound parameter; the only
interpolated identifiers are the two timestamp-column constants.

Bronze schema (livewire ``clients/bronze_client.py`` + ``intraday_bronze_client.py``):
daily bars key on ``trade_date`` (DATE), intraday on ``bar_timestamp`` (TIMESTAMPTZ).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

import duckdb

from ....domain.events.domain_events import BarData

logger = logging.getLogger(__name__)

_DAILY_TS_COLUMN = "trade_date"
_INTRADAY_TS_COLUMN = "bar_timestamp"
DEFAULT_QUERY_TIMEOUT_SECONDS = 30.0

# Bar duration per timeframe -- used to derive bar_end (not a zero-width bar).
_TF_DELTAS = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}

DateRanges = Optional[Sequence[tuple[date, Optional[date]]]]


class QueryTimeout(RuntimeError):
    """A parquet read exceeded its deadline and was interrupted."""


def timestamp_column(timeframe: str) -> str:
    return _DAILY_TS_COLUMN if timeframe == "1d" else _INTRADAY_TS_COLUMN


def to_utc_datetime(value: Any) -> datetime:
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


def row_to_bar(row: dict, symbol: str, timeframe: str) -> BarData:
    ts = to_utc_datetime(row[timestamp_column(timeframe)])
    vol = row.get("volume")
    oi = row.get("open_interest")
    return BarData(
        symbol=symbol,
        timeframe=timeframe,
        open=row.get("open"),
        high=row.get("high"),
        low=row.get("low"),
        close=row.get("close"),
        volume=int(vol) if vol is not None else None,
        vwap=row.get("vwap"),
        settlement=row.get("settlement"),
        open_interest=int(oi) if oi is not None else None,
        contract_id=row.get("contract_id"),
        root_symbol=row.get("root_symbol"),
        # str(), not the raw value: asdict() would leak a live date into JSON.
        expiry_date=(str(row["expiry_date"]) if row.get("expiry_date") is not None else None),
        source_price_basis=row.get("price_basis"),
        bar_start=ts,
        bar_end=ts + _TF_DELTAS.get(timeframe, timedelta(0)),
        timestamp=ts,  # event time = bar time, NOT construction-time now()
        source="livewire",
    )


# -- SQL --------------------------------------------------------------------------


def _window(timeframe: str, start: datetime, end: datetime) -> tuple[str, List[Any]]:
    # Daily `trade_date` is a DATE: bind calendar dates so the comparison is
    # tz-agnostic. Intraday `bar_timestamp` is TIMESTAMPTZ: bind aware datetimes.
    col = timestamp_column(timeframe)
    params: List[Any] = [start.date(), end.date()] if timeframe == "1d" else [start, end]
    return f"{col} >= ? AND {col} <= ?", params


def _ordered(select: str, ts_col: str, tail: Optional[int]) -> str:
    """Chronological rows; with ``tail``, only the last N (LIMIT bound last)."""
    if tail is None:
        return f"{select} ORDER BY {ts_col} ASC"
    return f"SELECT * FROM ({select} ORDER BY {ts_col} DESC LIMIT ?) ORDER BY {ts_col} ASC"


def bars_sql(
    path: Path,
    timeframe: str,
    start: datetime,
    end: datetime,
    tail: Optional[int] = None,
    date_ranges: DateRanges = None,
) -> tuple[str, List[Any]]:
    """One artifact's bars in a window; ``date_ranges`` keeps ``[from, to)`` spans."""
    where, params = _window(timeframe, start, end)
    params = [path.as_posix(), *params]
    if date_ranges:
        spans = " OR ".join(
            "(trade_date >= ? AND (CAST(? AS DATE) IS NULL OR trade_date < ?))" for _ in date_ranges
        )
        where += f" AND ({spans})"
        for lo, hi in date_ranges:
            params.extend([lo, hi, hi])
    sql = _ordered(
        f"SELECT * FROM read_parquet(?) WHERE {where}",
        timestamp_column(timeframe),
        tail,
    )
    return sql, params + ([tail] if tail is not None else [])


def union_sql(
    live: Path,
    archived: Path,
    timeframe: str,
    start: datetime,
    end: datetime,
    tail: Optional[int],
) -> tuple[str, List[Any]]:
    """Live wins every America/New_York trading date the two trees share.

    Daily ``trade_date`` is already the session date. Intraday dedupes on the NY date
    of ``bar_timestamp``: an archived bar can carry a timestamp the live tree lacks on
    a date the live tree covers, and exact-timestamp dedupe would let it survive
    beside the live session it duplicates. ``UNION ALL BY NAME`` because the live daily
    file carries ``source``/``price_basis`` and the archive does not.
    """
    day = (
        "trade_date"
        if timeframe == "1d"
        else "CAST(timezone('America/New_York', bar_timestamp) AS DATE)"
    )
    where, window = _window(timeframe, start, end)
    body = (
        f"WITH live AS (SELECT * FROM read_parquet(?) WHERE {where}), "
        f"archived AS (SELECT * FROM read_parquet(?) WHERE {where}) "
        "SELECT * FROM live UNION ALL BY NAME "
        f"SELECT * FROM archived WHERE {day} NOT IN (SELECT {day} FROM live)"
    )
    sql = _ordered(f"SELECT * FROM ({body})", timestamp_column(timeframe), tail)
    params = [live.as_posix(), *window, archived.as_posix(), *window]
    return sql, params + ([tail] if tail is not None else [])


def adjusted_intraday_sql(
    bronze: Path, factors: Path, start: datetime, end: datetime, tail: Optional[int]
) -> tuple[str, List[Any]]:
    """Bronze intraday bars joined to the Silver factor interval of their NY date.

    With a tail, the LIMIT applies to the raw bars BEFORE the join and ``raw_count`` is
    taken over that limited set, so the caller's coverage check compares like with like.
    """
    limit = "ORDER BY bar_timestamp DESC LIMIT ?" if tail is not None else ""
    sql = f"""
        WITH selected AS (
            SELECT * FROM read_parquet(?)
            WHERE bar_timestamp >= ? AND bar_timestamp <= ?
            {limit}
        ),
        raw AS (
            SELECT *, count(*) OVER () AS raw_count FROM selected
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
    params: List[Any] = [
        bronze.as_posix(),
        start,
        end,
        *([] if tail is None else [tail]),
    ]
    return sql, [*params, factors.as_posix()]


def rates_sql(
    path: Path, start: datetime, end: datetime, tail: Optional[int]
) -> tuple[str, List[Any]]:
    sql = _ordered(
        "SELECT trade_date, tenor_years, yield_pct FROM read_parquet(?) "
        "WHERE trade_date >= ? AND trade_date <= ?",
        "trade_date",
        tail,
    )
    return sql, [
        path.as_posix(),
        start.date(),
        end.date(),
        *([] if tail is None else [tail]),
    ]


# -- execution --------------------------------------------------------------------


class LakeDb:
    """Executes parquet reads with a per-query deadline.

    One in-memory DuckDB parent per process; every query takes its own cursor from it
    in a worker thread and closes it when done. Cursors are never shared between
    queries. On deadline expiry the awaiting coroutine interrupts exactly that cursor
    (an interrupt never reaches another request's query -- verified against duckdb
    1.5.5 in P0) and raises ``QueryTimeout``; the worker closes the cursor when the
    interrupted execute returns.

    Chosen by measurement (P1.6, mini lake, 2026-09-23): against a fresh connection per
    query, the shared parent cut bulk 50-symbol reads from 432 to 246 ms median and
    held every other workload equal or better. The parent lives for the process and is
    released at interpreter exit.
    """

    _parent: Optional[duckdb.DuckDBPyConnection] = None
    _parent_lock = threading.Lock()

    def __init__(self, timeout: Optional[float] = None) -> None:
        self.timeout = (
            timeout
            if timeout is not None
            else float(
                os.environ.get("APEX_LAKE_QUERY_TIMEOUT_SECONDS", DEFAULT_QUERY_TIMEOUT_SECONDS)
            )
        )

    def _handle(self) -> duckdb.DuckDBPyConnection:
        with LakeDb._parent_lock:
            if LakeDb._parent is None:
                LakeDb._parent = duckdb.connect(database=":memory:")
            return LakeDb._parent.cursor()

    @staticmethod
    def _run(handle: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any]) -> List[dict]:
        try:
            return handle.execute(sql, list(params)).fetch_arrow_table().to_pylist()
        finally:
            handle.close()

    def rows_sync(self, sql: str, params: Sequence[Any]) -> List[dict]:
        """Blocking read for sync callers (health recency, thread-offloaded helpers),
        under the same deadline: a timer thread interrupts the cursor on expiry."""
        handle = self._handle()
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            _interrupt(handle)

        timer = threading.Timer(self.timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            return self._run(handle, sql, params)
        except duckdb.Error as exc:
            if expired.is_set():
                raise QueryTimeout(f"lake query exceeded {self.timeout:.0f}s") from exc
            raise
        finally:
            timer.cancel()

    async def rows(self, sql: str, params: Sequence[Any]) -> List[dict]:
        handle = self._handle()
        worker = asyncio.ensure_future(asyncio.to_thread(self._run, handle, sql, params))
        try:
            return await asyncio.wait_for(asyncio.shield(worker), self.timeout)
        except asyncio.TimeoutError as exc:
            _abandon(handle, worker, f"deadline {self.timeout:.1f}s")
            raise QueryTimeout(f"lake query exceeded {self.timeout:.0f}s") from exc
        except asyncio.CancelledError:
            # The request went away (client disconnect, shutdown): stop its query too.
            _abandon(handle, worker, "request cancelled")
            raise


def _interrupt(handle: duckdb.DuckDBPyConnection) -> None:
    try:
        handle.interrupt()
    except duckdb.Error as exc:  # the query finished and the cursor closed first
        logger.debug("interrupt after completion: %s", exc)


def _abandon(handle: duckdb.DuckDBPyConnection, worker: "asyncio.Future[Any]", why: str) -> None:
    """Interrupt the query and stop waiting for it. The worker closes its cursor when
    the interrupted execute returns; a result already past DuckDB (Arrow -> Python
    conversion) finishes in its thread and is discarded rather than awaited."""
    _interrupt(handle)
    worker.add_done_callback(_discard)
    logger.warning("parquet read abandoned: %s", why)


def _discard(future: "asyncio.Future[Any]") -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.debug("abandoned parquet read ended with %r", future.exception())
