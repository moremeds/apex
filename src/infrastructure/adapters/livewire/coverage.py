"""Read livewire's coverage catalog to answer "what does apex hold?".

livewire's scheduled coverage job (com.livewire.coverage, 11:00 UTC) writes one row
per (view_name, symbol) into analytics.duckdb. Reading it is the only affordable way
to answer discovery: measured on the production lake 2026-08-23, one scandir of
bronze/asset_class=equity costs 5.5s for 14,756 entries, and descending into symbols
to read date ranges costs 78ms each -- roughly 19 minutes for the full set.

The table is a SNAPSHOT, so first_date/last_date lag reality by up to a day. Callers
must present these as snapshot-derived, not live. Measured 2026-08-23: the table is
also incomplete -- it covers 7 of the 13 views livewire's duckdb_catalog declares, so
equity intraday coverage is invisible here.

An unreadable catalog RAISES rather than returning an empty list. That distinction is
load-bearing: on the mini, binding a catalog path outside colima's VM mount set makes
Docker fabricate an empty directory at that path, and Path.exists() answers True for a
directory. Degrading to [] would make a broken deployment indistinguishable from a
genuinely empty lake.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

import duckdb

logger = logging.getLogger(__name__)

# coverage.view_name -> asset class. Silver views are folded into the equity rows as
# the silver_available flag rather than listed as their own class.
_VIEW_TO_CLASS = {
    "bronze_equity_1d": "equity",
    "bronze_volatility_1d": "volatility",
    "bronze_fx_1d": "fx",
    "bronze_futures_1d": "futures",
    "bronze_cmdty_1d": "cmdty",
    "bronze_rates_1d": "rates",
}
_SILVER_VIEW = "silver_equity_1d"
# Every catalog view livewire's duckdb_catalog can declare: <tier>_<class>_<timeframe>.
# A view that does not parse is reported as unknown, never folded into a class.
_VIEW_RE = re.compile(r"^(bronze|silver)_([a-z_]+)_(1m|5m|30m|1h|1d)$")


class CoverageUnavailable(RuntimeError):
    """Raised when the coverage catalog cannot be read at all.

    Distinct from "the catalog is readable and matched nothing", which is an empty list.
    """


@dataclass(frozen=True)
class InstrumentRow:
    """One instrument's coverage, as of livewire's last snapshot."""

    symbol: str
    asset_class: str
    listing_status: str
    first_date: Optional[str]
    last_date: Optional[str]
    silver_available: bool
    price_mode: str


@dataclass(frozen=True)
class CoverageRow:
    """One raw catalog row, classified by its view name."""

    view_name: str
    tier: Optional[str]
    asset_class: Optional[str]
    timeframe: Optional[str]
    symbol: str
    n_rows: int
    first_date: Optional[str]
    last_date: Optional[str]


@dataclass(frozen=True)
class CatalogIdentity:
    """Which catalog file answered: livewire replaces it atomically, so a paged
    traversal whose identity changes between pages must restart."""

    size_bytes: int
    modified_at: str


def classify_view(view_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """``(tier, asset_class, timeframe)`` for a catalog view, or all None if unknown."""
    match = _VIEW_RE.match(view_name)
    if match is None:
        return None, None, None
    return match.group(1), match.group(2), match.group(3)


class CoverageCatalog:
    """Read-only view over livewire's coverage table."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def list_instruments(
        self,
        asset_class: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 500,
    ) -> List[InstrumentRow]:
        """Return coverage rows, filtered and capped.

        Raises ``CoverageUnavailable`` when the catalog is absent or unreadable.
        """
        # is_file(), NOT exists(): a broken bind mount leaves a DIRECTORY here, and
        # exists() would happily accept it.
        if not self._db_path.is_file():
            raise CoverageUnavailable(f"coverage catalog is not a readable file: {self._db_path}")
        views = (
            [v for v, c in _VIEW_TO_CLASS.items() if c == asset_class]
            if asset_class
            else list(_VIEW_TO_CLASS)
        )
        if not views:
            return []
        placeholders = ", ".join("?" for _ in views)
        sql = f"""
            SELECT b.view_name, b.symbol, b.first_date, b.last_date,
                   (s.symbol IS NOT NULL) AS has_silver
            FROM coverage b
            LEFT JOIN coverage s
              ON s.view_name = '{_SILVER_VIEW}' AND s.symbol = b.symbol
            WHERE b.view_name IN ({placeholders})
              AND (? IS NULL OR b.symbol LIKE ? ESCAPE '\\')
            ORDER BY b.symbol
            LIMIT ?
        """
        # '_' and '%' are LIKE wildcards, and futures symbols contain '_'
        # (BZ_202609). Verified against the real catalog: LIKE 'BRK_B' matches
        # 'BRK.B'. Escape them or a prefix search silently over-matches.
        like = _escape_like(query.upper()) + "%" if query else None
        params: list[object] = [*views, like, like, limit]
        try:
            # read_only: the catalog is bind-mounted :ro in production, and a
            # read-write connect would try to take a lock and fail.
            con = duckdb.connect(str(self._db_path), read_only=True)
        except duckdb.Error as exc:
            logger.error("cannot open coverage catalog %s: %s", self._db_path, exc)
            raise CoverageUnavailable(f"cannot open coverage catalog: {exc}") from exc
        try:
            rows = con.execute(sql, params).fetch_arrow_table().to_pylist()
        except duckdb.Error as exc:
            logger.error("coverage query failed: %s", exc)
            raise CoverageUnavailable(f"coverage query failed: {exc}") from exc
        finally:
            con.close()
        return [self._to_row(r) for r in rows]

    def identity(self) -> CatalogIdentity:
        try:
            stat = self._db_path.stat()
        except OSError as exc:
            raise CoverageUnavailable(f"coverage catalog is not readable: {exc}") from exc
        return CatalogIdentity(
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        )

    def list_coverage(
        self,
        *,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
        include_silver: bool = True,
        symbol_prefix: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[CoverageRow], bool]:
        """One page of raw catalog rows, ordered by (symbol, view_name).

        Returns ``(rows, truncated)``; ``truncated`` comes from selecting one row past
        the page, not from a count. Unknown views appear only when no class filter
        is given, labelled with null tier/class/timeframe.
        """
        views = [row[0] for row in self._execute("SELECT DISTINCT view_name FROM coverage", [])]
        selected = []
        for view in views:
            tier, view_class, _ = classify_view(view)
            if tier == "silver" and not include_silver:
                continue
            if asset_class is not None and view_class != asset_class:
                continue
            selected.append(view)
        if not selected:
            return [], False
        placeholders = ", ".join("?" for _ in selected)
        sql = (
            "SELECT view_name, symbol, n_rows, first_date, last_date FROM coverage "
            f"WHERE view_name IN ({placeholders}) "
            "AND (? IS NULL OR symbol = ?) "
            "AND (? IS NULL OR symbol LIKE ? ESCAPE '\\') "
            "ORDER BY symbol, view_name LIMIT ? OFFSET ?"
        )
        like = None if symbol_prefix is None else _escape_like(symbol_prefix) + "%"
        rows = self._execute(sql, [*selected, symbol, symbol, like, like, limit + 1, offset])
        parsed = [
            CoverageRow(
                view_name=row[0],
                tier=classify_view(row[0])[0],
                asset_class=classify_view(row[0])[1],
                timeframe=classify_view(row[0])[2],
                symbol=row[1],
                n_rows=int(row[2]),
                first_date=None if row[3] is None else str(row[3]),
                last_date=None if row[4] is None else str(row[4]),
            )
            for row in rows
        ]
        return parsed[:limit], len(parsed) > limit

    def _execute(self, sql: str, params: Sequence[Any]) -> List[Any]:
        if not self._db_path.is_file():
            raise CoverageUnavailable(f"coverage catalog is not a readable file: {self._db_path}")
        try:
            con = duckdb.connect(str(self._db_path), read_only=True)
        except duckdb.Error as exc:
            logger.error("cannot open coverage catalog %s: %s", self._db_path, exc)
            raise CoverageUnavailable(f"cannot open coverage catalog: {exc}") from exc
        try:
            return con.execute(sql, list(params)).fetchall()
        except duckdb.Error as exc:
            logger.error("coverage query failed: %s", exc)
            raise CoverageUnavailable(f"coverage query failed: {exc}") from exc
        finally:
            con.close()

    def get_instrument(self, symbol: str, asset_class: str) -> Optional[InstrumentRow]:
        """Exact-match lookup for one instrument.

        Not ``list_instruments(limit=1)``: that filters by PREFIX, so "AA" would
        return whichever of AA/AAL/AAPL sorts first.
        """
        rows = [
            r
            for r in self.list_instruments(asset_class=asset_class, query=symbol, limit=50)
            if r.symbol == symbol
        ]
        return rows[0] if rows else None

    @staticmethod
    def _to_row(row: dict) -> InstrumentRow:
        asset_class = _VIEW_TO_CLASS[row["view_name"]]
        has_silver = bool(row["has_silver"]) and asset_class == "equity"
        return InstrumentRow(
            symbol=row["symbol"],
            asset_class=asset_class,
            listing_status="listed",
            first_date=str(row["first_date"]) if row["first_date"] is not None else None,
            last_date=str(row["last_date"]) if row["last_date"] is not None else None,
            silver_available=has_silver,
            # Silver exists only for equity, so nothing else can be served adjusted.
            price_mode="adjusted" if has_silver else "raw",
        )


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards: futures symbols contain '_' (BZ_202609)."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
