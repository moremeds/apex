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
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, Literal

import duckdb

from ....domain.events.domain_events import BarData
from .asset_classes import DEFAULT_ASSET_CLASS, get_asset_class
from .parquet_reads import (
    DateRanges,
    LakeDb,
    adjusted_intraday_sql,
    bars_sql,
    rates_sql,
    row_to_bar,
    to_utc_datetime,
    union_sql,
)
from .paths import SUPPORTED_TIMEFRAMES, delisted_bronze_path, parquet_path
from .revisions import (
    ArtifactKind,
    RevisionManifestError,
    RevisionManifestReader,
    SilverRevision,
)

logger = logging.getLogger(__name__)

PriceMode = Literal["raw", "adjusted"]


class AdjustedDataUnavailable(RuntimeError):
    """Raised when adjusted mode cannot prove complete Silver coverage."""


@dataclass(frozen=True)
class RatePoint:
    """One observation from asset_class=rates. Not a bar -- there is no OHLC."""

    time: datetime
    tenor_years: float
    yield_pct: float


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
        db: LakeDb | None = None,
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
        self._db = db or LakeDb()

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
    def db(self) -> LakeDb:
        """The executor every lake parquet read goes through (deadline, shared parent)."""
        return self._db

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
            return await self._bars(
                bars_sql(bronze_path, timeframe, start, end, tail), symbol, timeframe
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
            return await self._bars(bars_sql(path, timeframe, start, end, tail), symbol, timeframe)

        if not bronze_path.exists():
            return []
        factors = await asyncio.to_thread(self.silver_artifact_path, symbol, "factors")
        if factors is None:
            raise AdjustedDataUnavailable(f"Silver factor artifact is missing for {symbol}")
        rows = await self._db.rows(*adjusted_intraday_sql(bronze_path, factors, start, end, tail))
        if not rows:
            return []
        raw_count = int(rows[0]["raw_count"])
        if len(rows) != raw_count or any(row["adjustment_revision"] is None for row in rows):
            raise AdjustedDataUnavailable(
                f"incomplete or overlapping factor coverage for {symbol} {timeframe}"
            )
        return [row_to_bar(row, symbol, timeframe) for row in rows]

    async def fetch_artifact_daily(
        self,
        path: Path,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        tail: int | None = None,
        date_ranges: DateRanges = None,
    ) -> List[BarData]:
        """Daily bars from one already-verified artifact (a PIT-served Silver file).

        ``date_ranges`` restricts rows to ``[from, to)`` spans (``to`` None = open),
        pushed into the query together with ``tail``.
        """
        return await self._bars(bars_sql(path, "1d", start, end, tail, date_ranges), symbol, "1d")

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
            return await self._bars(bars_sql(only, timeframe, start, end, tail), symbol, timeframe)
        # Both trees: the union and its live-wins-per-date rule run in one query, so a
        # tail over the merged series is exact rather than a per-source approximation.
        return await self._bars(
            union_sql(live, archived, timeframe, start, end, tail), symbol, timeframe
        )

    async def _bars(
        self, query: tuple[str, List[Any]], symbol: str, timeframe: str
    ) -> List[BarData]:
        rows = await self._db.rows(*query)
        return [row_to_bar(row, symbol, timeframe) for row in rows]

    async def fetch_rate_series(
        self, symbol: str, start: datetime, end: datetime, tail: int | None = None
    ) -> List[RatePoint]:
        """Read a yield series. Rates are never adjusted -- a yield has no split."""
        path = parquet_path(self._bronze_root, symbol, "1d", "rates")
        if not path.exists():
            return []
        rows = await self._db.rows(*rates_sql(path, start, end, tail))
        return [
            RatePoint(
                time=to_utc_datetime(r["trade_date"]),
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
        (row,) = self._db.rows_sync(
            "SELECT any_value(contract_id) AS contract_id, any_value(root_symbol) AS root, "
            "any_value(expiry_date) AS expiry, min(trade_date) AS first, "
            "max(trade_date) AS last, count(*) AS n FROM read_parquet(?)",
            [path.as_posix()],
        )
        return {
            "contract_id": None if row["contract_id"] is None else int(row["contract_id"]),
            "root_symbol": row["root"],
            "expiry_date": None if row["expiry"] is None else str(row["expiry"]),
            "first_date": None if row["first"] is None else str(row["first"]),
            "last_date": None if row["last"] is None else str(row["last"]),
            "rows": int(row["n"]),
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

    def _last_trade_date(self, path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            (row,) = self._db.rows_sync(
                "SELECT max(trade_date) AS last FROM read_parquet(?)", [path.as_posix()]
            )
        except duckdb.Error as exc:
            logger.error("recency probe failed for %s: %s", path, exc)
            return None
        return str(row["last"]) if row["last"] is not None else None
