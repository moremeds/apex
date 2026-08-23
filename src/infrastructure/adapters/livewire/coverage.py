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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        like = (
            query.upper().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            if query
            else None
        )
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
