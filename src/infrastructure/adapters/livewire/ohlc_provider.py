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
import logging
from copy import copy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Literal

import duckdb

from ....domain.events.domain_events import BarData
from .asset_classes import DEFAULT_ASSET_CLASS, get_asset_class
from .paths import SUPPORTED_TIMEFRAMES, delisted_bronze_path, parquet_path
from .revisions import (
    ArtifactKind,
    RevisionManifestError,
    RevisionManifestReader,
    SilverRevision,
)

# livewire keys daily bars by `trade_date` (a DATE) and intraday bars by
# `bar_timestamp` (a tz-aware UTC TIMESTAMP). OHLCV columns are read by name; extra
# columns (symbol_id, adj_close) are ignored.
logger = logging.getLogger(__name__)

_DAILY_TS_COLUMN = "trade_date"
_INTRADAY_TS_COLUMN = "bar_timestamp"

PriceMode = Literal["raw", "adjusted"]


class AdjustedDataUnavailable(RuntimeError):
    """Raised when adjusted mode cannot prove complete Silver coverage."""


@dataclass(frozen=True)
class RatePoint:
    """One observation from asset_class=rates. Not a bar -- there is no OHLC."""

    time: datetime
    tenor_years: float
    yield_pct: float


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
        delisted_root: Path | None = None,
    ) -> None:
        if price_mode not in ("raw", "adjusted"):
            raise ValueError(f"unsupported Livewire price mode: {price_mode!r}")
        self._bronze_root = Path(bronze_root)
        self._silver_root = Path(silver_root) if silver_root is not None else None
        self._price_mode = price_mode
        self._snapshot: SilverRevision | None = None
        # bronze-delisted/: the archived tree. Raw only -- livewire publishes no Silver
        # over it -- so adjusted reads that touch it fail loudly rather than mixing bases.
        self._delisted_root = Path(delisted_root) if delisted_root is not None else None

    # --- HistoricalSourcePort ---
    @property
    def source_name(self) -> str:
        return "livewire"

    @property
    def bronze_root(self) -> Path:
        return self._bronze_root

    @property
    def delisted_root(self) -> Path | None:
        return self._delisted_root

    @property
    def silver_root(self) -> Path | None:
        return self._silver_root

    @property
    def price_mode(self) -> PriceMode:
        return self._price_mode

    @property
    def snapshot(self) -> SilverRevision | None:
        return self._snapshot

    def pin_snapshot(self, revision: SilverRevision | None = None) -> LivewireOhlcProvider:
        """Return an operation-local provider; never change the shared provider's revision."""
        if self._silver_root is None:
            raise AdjustedDataUnavailable("Silver root is not configured")
        try:
            snapshot = (
                revision
                or self._snapshot
                or RevisionManifestReader(self._silver_root).read_current()
            )
            if snapshot.root != self._silver_root.resolve():
                raise RevisionManifestError("Silver snapshot belongs to another root")
        except RevisionManifestError as exc:
            raise AdjustedDataUnavailable(str(exc)) from exc
        pinned = copy(self)
        pinned._snapshot = snapshot
        return pinned

    def silver_artifact_path(
        self, symbol: str, kind: ArtifactKind, *, verify: bool = True
    ) -> Path | None:
        """Resolve only a committed reference, never a file omitted by the manifest."""
        pinned = self if self._snapshot is not None else self.pin_snapshot()
        assert pinned._snapshot is not None
        try:
            return pinned._snapshot.artifact_path(symbol, kind, verify=verify)
        except RevisionManifestError as exc:
            raise AdjustedDataUnavailable(str(exc)) from exc

    def supports_timeframe(self, timeframe: str) -> bool:
        return timeframe in SUPPORTED_TIMEFRAMES

    def get_supported_timeframes(self) -> List[str]:
        return list(SUPPORTED_TIMEFRAMES)

    def effective_price_mode(self, asset_class: str = DEFAULT_ASSET_CLASS) -> PriceMode:
        """The mode this provider can actually serve for ``asset_class``.

        Silver exists only under asset_class=equity, so every other class is raw
        regardless of the configured mode. Callers put this in the payload -- the
        consumer must never have to infer the basis.
        """
        if self._price_mode == "adjusted" and get_asset_class(asset_class).supports_adjusted:
            return "adjusted"
        return "raw"

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        asset_class: str = DEFAULT_ASSET_CLASS,
        price_mode: PriceMode | None = None,
        listing: str = "listed",
        tail: int | None = None,
    ) -> List[BarData]:
        """Bars in ``[start, end]``, chronological.

        ``tail`` keeps only the last N rows of the window and is pushed into the
        DuckDB query (``ORDER BY ... DESC LIMIT``) so a bounded read never
        materializes a whole minute file in Python.
        """
        # An explicit price_mode is a per-call override (the route passes the mode it
        # already validated); None falls back to what this provider can serve.
        resolved = price_mode or self.effective_price_mode(asset_class)
        if resolved == "adjusted" and not get_asset_class(asset_class).supports_adjusted:
            raise AdjustedDataUnavailable(f"Silver does not exist for {asset_class}")
        if listing != "listed":
            # There is no Silver over bronze-delisted, so an adjusted read that touches
            # it would have to splice an adjusted segment onto a raw one. Rule 12: fail,
            # never fall back.
            if resolved == "adjusted":
                raise AdjustedDataUnavailable("no Silver for delisted names; use price_mode=raw")
            return await self._fetch_including_delisted(
                symbol, timeframe, start, end, asset_class, listing, tail
            )
        bronze_path = parquet_path(self._bronze_root, symbol, timeframe, asset_class)
        if resolved == "raw":
            if not bronze_path.exists():
                return []
            return await asyncio.to_thread(
                self._query, bronze_path, symbol, timeframe, start, end, tail
            )
        if self._silver_root is None:
            raise AdjustedDataUnavailable("Silver root is not configured")
        if self._snapshot is None:
            pinned = await asyncio.to_thread(self.pin_snapshot)
            return await pinned.fetch_bars(
                symbol, timeframe, start, end, asset_class, resolved, tail=tail
            )
        if timeframe == "1d":
            path = await asyncio.to_thread(self.silver_artifact_path, symbol, "daily")
            if path is None:
                # No Silver AND no Bronze means the symbol does not exist at all -- an
                # unknown ticker, which the route turns into 404. Raising here instead
                # would answer a typo with "retry later" and the caller would retry
                # forever. Only a symbol that HAS bronze is genuinely quarantined.
                if not bronze_path.exists():
                    return []
                raise AdjustedDataUnavailable(f"Silver daily artifact is missing for {symbol}")
            return await asyncio.to_thread(self._query, path, symbol, timeframe, start, end, tail)

        if not bronze_path.exists():
            return []
        factors = await asyncio.to_thread(self.silver_artifact_path, symbol, "factors")
        if factors is None:
            raise AdjustedDataUnavailable(f"Silver factor artifact is missing for {symbol}")
        return await asyncio.to_thread(
            self._query_adjusted_intraday,
            bronze_path,
            factors,
            symbol,
            timeframe,
            start,
            end,
            tail,
        )

    async def fetch_artifact_daily(
        self,
        path: Path,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        tail: int | None = None,
        date_ranges: tuple[tuple[date, date | None], ...] | None = None,
    ) -> List[BarData]:
        """Daily bars from one already-verified artifact (a PIT-served Silver file).

        ``date_ranges`` restricts rows to ``[from, to)`` spans (``to`` None = open),
        pushed into the query together with ``tail``.
        """
        return await asyncio.to_thread(
            self._query, path, symbol, "1d", start, end, tail, date_ranges
        )

    async def _fetch_including_delisted(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        asset_class: str,
        listing: str,
        tail: int | None = None,
    ) -> List[BarData]:
        """Raw bars from bronze-delisted, optionally unioned with the live tree.

        ``delisted`` reads the archived tree alone. ``dual`` reads both and lets the
        live tree win on any bar the two share: the live artifact is the one livewire
        still maintains. "Wins" means per America/New_York trading date, not per exact
        timestamp -- for an intraday timeframe an archived bar can carry a timestamp the
        live tree lacks even on a date the live tree does cover, and deduping by exact
        timestamp alone would let that archived bar survive alongside the live session
        it duplicates. Whether the archived rows behind a ``dual`` symbol are the same
        issuer under a reused ticker or a duplicate copy of the live company's own
        history is not decided here -- see ``/v1/equity/{symbol}/delisting`` for that.
        """
        archived: Path | None = None
        if self._delisted_root is not None:
            candidate = delisted_bronze_path(self._delisted_root, symbol, timeframe, asset_class)
            archived = candidate if candidate.exists() else None
        live_path = parquet_path(self._bronze_root, symbol, timeframe, asset_class)
        live = live_path if listing != "delisted" and live_path.exists() else None
        if archived is None and live is None:
            return []
        if archived is None or live is None:
            only = archived if live is None else live
            assert only is not None
            return await asyncio.to_thread(self._query, only, symbol, timeframe, start, end, tail)
        # Both trees: the union and its live-wins-per-date rule run in one query, so a
        # tail over the merged series is exact rather than a per-source approximation.
        return await asyncio.to_thread(
            self._query_union, live, archived, symbol, timeframe, start, end, tail
        )

    async def fetch_rate_series(
        self, symbol: str, start: datetime, end: datetime, tail: int | None = None
    ) -> List[RatePoint]:
        """Read a yield series. Rates are never adjusted -- a yield has no split."""
        path = parquet_path(self._bronze_root, symbol, "1d", "rates")
        if not path.exists():
            return []
        return await asyncio.to_thread(self._query_rates, path, start, end, tail)

    def _query_rates(
        self, path: Path, start: datetime, end: datetime, tail: int | None = None
    ) -> List[RatePoint]:
        sql = self._ordered(
            "SELECT trade_date, tenor_years, yield_pct FROM read_parquet(?) "
            "WHERE trade_date >= ? AND trade_date <= ?",
            "trade_date",
            tail,
        )
        params: List[Any] = [path.as_posix(), start.date(), end.date()]
        if tail is not None:
            params.append(tail)
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(sql, params).fetch_arrow_table().to_pylist()
        finally:
            con.close()
        return [
            RatePoint(
                time=_to_utc_datetime(r["trade_date"]),
                tenor_years=float(r["tenor_years"]),
                yield_pct=float(r["yield_pct"]),
            )
            for r in rows
        ]

    def fetch_futures_contract(self, symbol: str) -> dict[str, Any]:
        """Identity and bounds of one futures contract file, from one aggregate read
        (parquet min/max statistics make it cheap). Contract columns are constant
        across a contract's rows."""
        path = parquet_path(self._bronze_root, symbol, "1d", "futures")
        con = duckdb.connect(database=":memory:")
        try:
            row = con.execute(
                "SELECT any_value(contract_id), any_value(root_symbol), "
                "any_value(expiry_date), min(trade_date), max(trade_date), count(*) "
                "FROM read_parquet(?)",
                [path.as_posix()],
            ).fetchone()
        finally:
            con.close()
        assert row is not None
        return {
            "contract_id": None if row[0] is None else int(row[0]),
            "root_symbol": row[1],
            "expiry_date": None if row[2] is None else str(row[2]),
            "first_date": None if row[3] is None else str(row[3]),
            "last_date": None if row[4] is None else str(row[4]),
            "rows": int(row[5]),
        }

    def fetch_recency(self, reference_symbol: str = "AAPL") -> dict[str, Any]:
        """Last trade_date in bronze and silver for a liquid reference symbol.

        ``fetch_`` not ``get_``: this touches the disk on every call (two DuckDB reads),
        it is not a cache lookup.

        Read from the ARTIFACTS, not livewire's coverage table -- that table is an
        11:00 UTC snapshot and under-reports by design. AAPL is the default probe
        because it trades every session the market is open.
        """
        bronze_last = self._last_trade_date(parquet_path(self._bronze_root, reference_symbol, "1d"))
        silver_last = None
        if self._silver_root is not None:
            try:
                path = self.silver_artifact_path(reference_symbol, "daily")
                if path is not None:
                    silver_last = self._last_trade_date(path)
            except AdjustedDataUnavailable as exc:
                logger.warning("Silver recency unavailable: %s", exc)
        lag: int | None = None
        if bronze_last is not None and silver_last is not None:
            # Calendar days, NOT trading sessions -- apex has no exchange calendar
            # here, and calling a Friday/Monday gap "3 sessions" would be a lie.
            lag = (date.fromisoformat(bronze_last) - date.fromisoformat(silver_last)).days
        return {
            "bronze_last_trade_date": bronze_last,
            "silver_last_trade_date": silver_last,
            "lag_days": lag,
        }

    @staticmethod
    def _last_trade_date(path: Path) -> str | None:
        if not path.exists():
            return None
        con = duckdb.connect(database=":memory:")
        try:
            row = con.execute(
                "SELECT max(trade_date) FROM read_parquet(?)", [path.as_posix()]
            ).fetchone()
        except duckdb.Error as exc:
            logger.error("recency probe failed for %s: %s", path, exc)
            return None
        finally:
            con.close()
        return str(row[0]) if row and row[0] is not None else None

    # --- internals ---
    def _query(
        self,
        path: Path,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        tail: int | None = None,
        date_ranges: tuple[tuple[date, date | None], ...] | None = None,
    ) -> List[BarData]:
        ts_col = _timestamp_column(timeframe)
        # The parquet path is a BOUND parameter, not interpolated. An earlier comment
        # here claimed DuckDB could not bind it; measured against duckdb 1.4.3 it can,
        # including a path containing a quote. `ts_col` is still interpolated because
        # an identifier cannot be a parameter -- it comes from _timestamp_column, which
        # returns one of two module constants, never caller input.
        where, window = self._window_clause(timeframe, start, end)
        params: List[Any] = [path.as_posix(), *window]
        if date_ranges:
            spans = " OR ".join(
                "(trade_date >= ? AND (CAST(? AS DATE) IS NULL OR trade_date < ?))"
                for _ in date_ranges
            )
            where += f" AND ({spans})"
            for lo, hi in date_ranges:
                params.extend([lo, hi, hi])
        sql = self._ordered(f"SELECT * FROM read_parquet(?) WHERE {where}", ts_col, tail)
        if tail is not None:
            params.append(tail)
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(sql, params).fetch_arrow_table().to_pylist()
        finally:
            con.close()
        return [self._row_to_bar(r, symbol, timeframe) for r in rows]

    def _query_union(
        self,
        live_path: Path,
        archived_path: Path,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        tail: int | None,
    ) -> List[BarData]:
        """Live wins every America/New_York trading date the two trees share.

        Daily ``trade_date`` is already the session date. Intraday dedupes on the NY
        date of ``bar_timestamp``: an archived bar can carry a timestamp the live tree
        lacks on a date the live tree covers, and exact-timestamp dedupe would let it
        survive beside the live session it duplicates. ``UNION ALL BY NAME`` because
        the live daily file carries ``source``/``price_basis`` and the archive does not.
        """
        ts_col = _timestamp_column(timeframe)
        day = (
            "trade_date"
            if timeframe == "1d"
            else "CAST(timezone('America/New_York', bar_timestamp) AS DATE)"
        )
        where, window = self._window_clause(timeframe, start, end)
        body = (
            f"WITH live AS (SELECT * FROM read_parquet(?) WHERE {where}), "
            f"archived AS (SELECT * FROM read_parquet(?) WHERE {where}) "
            "SELECT * FROM live UNION ALL BY NAME "
            f"SELECT * FROM archived WHERE {day} NOT IN (SELECT {day} FROM live)"
        )
        sql = self._ordered(f"SELECT * FROM ({body})", ts_col, tail)
        params: List[Any] = [live_path.as_posix(), *window, archived_path.as_posix(), *window]
        if tail is not None:
            params.append(tail)
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(sql, params).fetch_arrow_table().to_pylist()
        finally:
            con.close()
        return [self._row_to_bar(r, symbol, timeframe) for r in rows]

    @staticmethod
    def _window_clause(timeframe: str, start: datetime, end: datetime) -> tuple[str, List[Any]]:
        # Daily `trade_date` is a DATE -- bind calendar-date params so the comparison
        # is tz-agnostic (avoids DATE-vs-TIMESTAMPTZ session-tz surprises). Intraday
        # `bar_timestamp` is TIMESTAMPTZ -- bind the tz-aware datetimes directly.
        ts_col = _timestamp_column(timeframe)
        window: List[Any] = [start.date(), end.date()] if timeframe == "1d" else [start, end]
        return f"{ts_col} >= ? AND {ts_col} <= ?", window

    @staticmethod
    def _ordered(select: str, ts_col: str, tail: int | None) -> str:
        """Chronological rows; with ``tail``, only the last N (LIMIT bound last)."""
        if tail is None:
            return f"{select} ORDER BY {ts_col} ASC"
        return f"SELECT * FROM ({select} ORDER BY {ts_col} DESC LIMIT ?) ORDER BY {ts_col} ASC"

    def _query_adjusted_intraday(
        self,
        bronze_path: Path,
        factors_path: Path,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        tail: int | None = None,
    ) -> List[BarData]:
        # With a tail, the LIMIT applies to the raw bars BEFORE the factor join, and
        # raw_count is taken over that limited set, so the coverage check below still
        # compares like with like.
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
        con = duckdb.connect(database=":memory:")
        try:
            rows = (
                con.execute(
                    sql,
                    [
                        bronze_path.as_posix(),
                        start,
                        end,
                        *([] if tail is None else [tail]),
                        factors_path.as_posix(),
                    ],
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
            bar_end=end,
            timestamp=ts,  # event time = bar time, NOT construction-time now()
            source="livewire",
        )
