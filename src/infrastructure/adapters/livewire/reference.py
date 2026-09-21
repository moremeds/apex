"""Corporate actions and instrument identity read from the livewire lake.

Two artifacts, two roots, one reader:

- ``<APEX_LIVEWIRE_ROOT>/asset_class=corporate_action/symbol=<enc>/events.parquet`` --
  livewire's per-ticker corporate-action log (15,015 symbols on 2026-09-21, provider
  ``massive``). It is an *event* log, not a table of facts: a correction lands as a new
  ``action_id`` at ``event_revision`` 2 whose ``supersedes_action_id`` names the old row,
  and the old row flips to ``status='corrected'`` (measured on AAA, 2026-09-21). Only
  ``status='active'`` rows are live. Rows are keyed by ticker, not by a permanent security id -- see the reuse
  caveat on the route.
- ``<APEX_LIVEWIRE_LAKE_ROOT>/security_master/events.parquet`` -- the same interval
  records ``membership.py`` reads, queried here by ticker instead of by security id.
  Measured 2026-09-21: it carries NO delisting reason and NO final consideration --
  ``relationship_type`` and ``related_security_id`` are null across the file. The
  identity intervals are all that exists, so they are all this module returns.

Reads are per-request through an in-memory DuckDB session, like every other lake read
in this package. apex only reads; livewire writes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Sequence

import duckdb

from .paths import corporate_action_path

logger = logging.getLogger(__name__)

# The provider livewire ingests US corporate actions and listings from.
ACTION_TYPES = ("split", "cash_dividend")

_ACTIVE = "active"
_VERIFIED = "verified"


class ReferenceDataError(RuntimeError):
    """A corporate-action or security-master artifact could not be read."""


@dataclass(frozen=True)
class CorporateAction:
    """One live corporate action, at its highest ``event_revision``."""

    action_type: str
    ex_date: Optional[str]
    split_from: Optional[float]
    split_to: Optional[float]
    cash_amount: Optional[float]
    currency: Optional[str]
    declaration_date: Optional[str]
    record_date: Optional[str]
    pay_date: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdentityInterval:
    """One ``[effective_from, effective_to)`` identity record for a ticker."""

    security_id: str
    symbol: str
    issuer_name: Optional[str]
    exchange_mic: Optional[str]
    currency: Optional[str]
    effective_from: Optional[str]
    effective_to: Optional[str]
    status: Optional[str]
    continuity_basis: Optional[str]
    relationship_type: Optional[str]
    related_security_id: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso(value: Any) -> Optional[str]:
    """Render a DuckDB date/timestamp as ISO-8601, leaving None alone."""
    return None if value is None else str(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


class LivewireReferenceReader:
    """Read-only corporate-action and security-master access over the livewire lake."""

    def __init__(self, bronze_root: Optional[Path], lake_root: Optional[Path] = None) -> None:
        self.bronze_root = Path(bronze_root) if bronze_root is not None else None
        self.lake_root = Path(lake_root) if lake_root is not None else None

    @classmethod
    def from_env(cls) -> "LivewireReferenceReader":
        """Build from the lake env vars. Unset roots are not an error -- each endpoint
        degrades to 503 on its own root, the way every other apex source does."""
        bronze = os.environ.get("APEX_LIVEWIRE_ROOT", "").strip()
        lake = os.environ.get("APEX_LIVEWIRE_LAKE_ROOT", "").strip()
        return cls(
            Path(bronze).expanduser() if bronze else None,
            Path(lake).expanduser() if lake else None,
        )

    @property
    def security_master_path(self) -> Optional[Path]:
        if self.lake_root is None:
            return None
        return self.lake_root / "security_master" / "events.parquet"

    def actions_path(self, symbol: str) -> Optional[Path]:
        if self.bronze_root is None:
            return None
        return corporate_action_path(self.bronze_root, symbol)

    # -- reads -------------------------------------------------------------

    def fetch_actions(
        self,
        symbol: str,
        *,
        action_type: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> Optional[List[CorporateAction]]:
        """Live actions for ``symbol``, ascending by ex_date.

        ``None`` means there is no artifact for this ticker at all -- a different answer
        from an empty list, which means the ticker has a log and no action matched the
        filter. The route renders the first as 404 and the second as a 200 with zero
        actions; collapsing them would report an unknown ticker as a quiet one.
        """
        path = self.actions_path(symbol)
        if path is None or not path.is_file():
            return None
        # status='active' is what drops corrected rows (a correction is a new action_id
        # superseding the old one). The per-action_id window is a guard against the same
        # id ever appearing twice as active; the lake has no such row today.
        sql = (
            "WITH ranked AS ("
            "  SELECT *, row_number() OVER ("
            "    PARTITION BY action_id ORDER BY event_revision DESC"
            "  ) AS rn"
            "  FROM read_parquet(?) WHERE status = ?"
            ") "
            "SELECT action_type, ex_date, split_from, split_to, cash_amount, currency, "
            "       declaration_date, record_date, pay_date "
            "FROM ranked WHERE rn = 1"
        )
        params: List[Any] = [path.as_posix(), _ACTIVE]
        if action_type is not None:
            sql += " AND action_type = ?"
            params.append(action_type)
        if start is not None:
            sql += " AND ex_date >= ?"
            params.append(start)
        if end is not None:
            sql += " AND ex_date <= ?"
            params.append(end)
        sql += " ORDER BY ex_date ASC, action_type ASC"
        rows = self._query(sql, params)
        return [
            CorporateAction(
                action_type=str(row[0]),
                ex_date=_iso(row[1]),
                split_from=_optional_float(row[2]),
                split_to=_optional_float(row[3]),
                cash_amount=_optional_float(row[4]),
                currency=row[5],
                declaration_date=_iso(row[6]),
                record_date=_iso(row[7]),
                pay_date=_iso(row[8]),
            )
            for row in rows
        ]

    def fetch_provider(self, symbol: str) -> Optional[str]:
        """The provider that supplied this ticker's action log, or None when mixed."""
        path = self.actions_path(symbol)
        if path is None or not path.is_file():
            return None
        rows = self._query(
            "SELECT DISTINCT provider FROM read_parquet(?) WHERE status = ?",
            [path.as_posix(), _ACTIVE],
        )
        providers = sorted({str(row[0]) for row in rows if row[0] is not None})
        return providers[0] if len(providers) == 1 else None

    def fetch_identity(self, symbol: str) -> Optional[List[IdentityInterval]]:
        """Standing identity intervals for a ticker, ascending by ``effective_from``.

        Matches ``membership.py``'s live view exactly: ``status = 'verified'`` AND the
        row is not superseded by any other row. This drops ``candidate`` and
        ``unresolved`` rows outright (not just ``rejected``) -- a candidate interval is
        not yet a standing identity -- and drops a verified row that a later revision
        has since superseded, even if that later row was itself rejected. ``None`` when
        the master is not configured or not on disk.
        """
        path = self.security_master_path
        if path is None or not path.is_file():
            return None
        sql = (
            "WITH src AS (SELECT * FROM read_parquet(?)), "
            # CAST: a livewire generation in which no row supersedes another writes the
            # column as an all-null parquet field, and DuckDB then binds it as INTEGER
            # against a VARCHAR event_id. The cast makes the comparison well-typed
            # whatever the column's physical type is.
            "superseded AS ("
            "  SELECT DISTINCT CAST(supersedes AS VARCHAR) AS supersedes FROM src"
            "  WHERE supersedes IS NOT NULL"
            ") "
            "SELECT security_id, symbol, issuer_name, exchange_mic, currency, "
            "       effective_from, effective_to, status, continuity_basis, "
            "       relationship_type, related_security_id "
            "FROM src WHERE symbol = ? AND status = ? "
            "  AND CAST(event_id AS VARCHAR) NOT IN (SELECT supersedes FROM superseded) "
            "ORDER BY effective_from ASC, security_id ASC"
        )
        rows = self._query(sql, [path.as_posix(), symbol.upper(), _VERIFIED])
        return [
            IdentityInterval(
                security_id=str(row[0]),
                symbol=str(row[1]),
                issuer_name=row[2],
                exchange_mic=row[3],
                currency=row[4],
                effective_from=_iso(row[5]),
                effective_to=_iso(row[6]),
                status=row[7],
                continuity_basis=row[8],
                relationship_type=row[9],
                related_security_id=row[10],
            )
            for row in rows
        ]

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _query(sql: str, params: Sequence[Any]) -> List[Any]:
        con = duckdb.connect(database=":memory:")
        try:
            # Same as membership.py: TIMESTAMPTZ renders in the session zone, and the
            # API must not answer a different effective_from per server timezone.
            con.execute("SET TimeZone='UTC'")
            return con.execute(sql, list(params)).fetchall()
        except duckdb.Error as exc:
            # The lake sits on an external volume; a truncated or unmounted artifact
            # must surface as a typed 503, never as an untyped 500.
            logger.error("livewire reference read failed: %s", exc)
            raise ReferenceDataError(str(exc)) from exc
        finally:
            con.close()
