"""Point-in-time index membership and security-master reads from the livewire lake.

Two sibling trees under ``APEX_LIVEWIRE_LAKE_ROOT`` back this module:

- ``index_membership/<index_id>/events.parquet`` -- an append-only **event log**, one
  file per index. There is no materialised member list: membership as of a date is a
  replay of ``add``/``remove`` events ordered by
  ``(effective_at, known_at, revision, event_id)``.
- ``security_master/events.parquet`` -- interval records ``[effective_from,
  effective_to)`` mapping ``security_id`` to a ticker, so a replayed member set can be
  labelled at the same as-of date rather than with today's ticker.

The lake is written by livewire and replaced atomically with ``os.replace``; apex only
reads, per request, through an in-memory DuckDB session like every other lake read.
There is no manifest and no cache to invalidate.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import duckdb

logger = logging.getLogger(__name__)

# A rejected event is retracted; everything else is at least a candidate. "Verified"
# is the strict reading; the looser one exists because r2k-proxy is never verified.
_VERIFIED = "verified"
_REJECTED = "rejected"

# An index id becomes a directory name, so it may not escape the membership root.
_INDEX_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The security master is keyed by the provider livewire ingests US listings from.
_SYMBOL_PROVIDER = "massive"
_SYMBOL_EXCHANGES = ("XNAS", "XNYS", "ARCX")


class MembershipDataError(RuntimeError):
    """The membership or security-master artifact could not be read."""


@dataclass(frozen=True)
class MembershipEvent:
    """One append-only membership log row, as served by the history endpoint."""

    index_id: str
    security_id: str
    action: str
    announced_at: Optional[datetime]
    effective_at: datetime
    known_at: datetime
    revision: int
    status: str
    event_id: str
    supersedes: Optional[str]


@dataclass(frozen=True)
class Member:
    """A replayed member, labelled with its as-of ticker when one resolves."""

    security_id: str
    symbol: Optional[str]


@dataclass(frozen=True)
class SymbolResolution:
    """Outcome of mapping a ticker to a ``security_id`` at a point in time.

    ``ambiguous`` is a distinct outcome rather than a pick: two live securities sharing
    a ticker at one date is a data condition the caller must see, not one apex guesses
    its way past.
    """

    security_id: Optional[str]
    ambiguous: bool


def _as_utc(value: Any) -> Optional[datetime]:
    """Normalise a DuckDB timestamp to an aware UTC datetime."""
    if value is None:
        return None
    if not isinstance(value, datetime):  # pragma: no cover - defensive
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _end_of_day(day: date) -> datetime:
    """The inclusive upper bound for a day expressed as an instant."""
    return datetime.combine(day, time.max, tzinfo=timezone.utc)


def _start_of_day(day: date) -> datetime:
    """The instant livewire evaluates a security-master interval at for a date."""
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def validate_index_id(index_id: str) -> bool:
    """Reject anything that is not a plain directory-safe index id."""
    return bool(_INDEX_ID_RE.match(index_id))


def resolve_security_ids(security_ids: Sequence[str]) -> SymbolResolution:
    """Reduce the distinct security ids a ticker matched to a single outcome.

    Kept separate from the query so the decision -- one match wins, several are
    ambiguous, none is unknown -- is testable without a lake.
    """
    distinct = sorted(set(security_ids))
    if not distinct:
        return SymbolResolution(security_id=None, ambiguous=False)
    if len(distinct) > 1:
        return SymbolResolution(security_id=None, ambiguous=True)
    return SymbolResolution(security_id=distinct[0], ambiguous=False)


class MembershipReader:
    """Read-only point-in-time membership over a livewire lake root."""

    def __init__(self, lake_root: Path) -> None:
        self.lake_root = lake_root
        self.membership_root = lake_root / "index_membership"
        self.security_master_path = lake_root / "security_master" / "events.parquet"

    @classmethod
    def from_env(cls) -> Optional["MembershipReader"]:
        """Build from ``APEX_LIVEWIRE_LAKE_ROOT``; ``None`` when the source is unset.

        Unset is not an error here -- like every other apex source, the endpoints
        degrade to 503 rather than failing the boot.
        """
        raw = os.environ.get("APEX_LIVEWIRE_LAKE_ROOT", "").strip()
        if not raw:
            return None
        return cls(Path(raw).expanduser())

    # -- discovery ---------------------------------------------------------

    def list_indices(self) -> List[str]:
        """Index ids present on disk, sorted. Never hardcoded -- livewire adds indices."""
        if not self.membership_root.is_dir():
            return []
        try:
            entries = list(self.membership_root.iterdir())
        except OSError as exc:
            logger.error("cannot list %s: %s", self.membership_root, exc)
            raise MembershipDataError("index membership root is not readable") from exc
        found = [
            entry.name
            for entry in entries
            if entry.is_dir()
            and validate_index_id(entry.name)
            and (entry / "events.parquet").is_file()
        ]
        return sorted(found)

    def events_path(self, index_id: str) -> Optional[Path]:
        """The event log for ``index_id``, or ``None`` when the index is unknown."""
        if not validate_index_id(index_id):
            return None
        path = self.membership_root / index_id / "events.parquet"
        return path if path.is_file() else None

    # -- reads -------------------------------------------------------------

    def _query(self, sql: str, params: Sequence[Any]) -> List[Tuple[Any, ...]]:
        try:
            conn = duckdb.connect(database=":memory:")
            try:
                conn.execute("SET TimeZone='UTC'")
                return conn.execute(sql, list(params)).fetchall()
            finally:
                conn.close()
        except duckdb.Error as exc:
            logger.error("membership query failed: %s", exc)
            raise MembershipDataError("membership artifact could not be read") from exc

    def _read_events(
        self,
        path: Path,
        index_id: str,
        *,
        known_at: Optional[date] = None,
        security_id: Optional[str] = None,
    ) -> List[MembershipEvent]:
        """Log rows visible at the ``known_at`` knowledge cutoff, in replay order.

        Only the knowledge cutoff is applied here. ``effective_at`` and ``status`` are
        left to the caller, because the superseded set has to be derived from exactly
        these rows: narrowing first would drop a superseding correction out of the set
        and leave the event it retracts alive.
        """
        clauses = ["TRUE"]
        params: List[Any] = [str(path)]
        if known_at is not None:
            clauses.append("known_at <= ?")
            params.append(_end_of_day(known_at))
        if security_id is not None:
            clauses.append("security_id = ?")
            params.append(security_id)
        sql = (
            "SELECT index_id, security_id, action, announced_at, effective_at, known_at, "
            "revision, status, event_id, supersedes FROM read_parquet(?) WHERE "
            + " AND ".join(clauses)
            + " ORDER BY effective_at, known_at, revision, event_id"
        )
        rows = self._query(sql, params)
        events: List[MembershipEvent] = []
        for row in rows:
            effective_at = _as_utc(row[4])
            known = _as_utc(row[5])
            if effective_at is None or known is None:
                logger.warning("skipping %s event with null timestamp: %s", index_id, row[8])
                continue
            events.append(
                MembershipEvent(
                    index_id=str(row[0] or index_id),
                    security_id=str(row[1]),
                    action=str(row[2]),
                    announced_at=_as_utc(row[3]),
                    effective_at=effective_at,
                    known_at=known,
                    revision=int(row[6] or 0),
                    status=str(row[7]),
                    event_id=str(row[8]),
                    supersedes=None if row[9] is None else str(row[9]),
                )
            )
        return events

    def members_as_of(
        self,
        index_id: str,
        as_of: date,
        *,
        known_at: Optional[date] = None,
        include_candidates: bool = False,
    ) -> List[str]:
        """Replay the log into the set of ``security_id``s in the index at ``as_of``.

        Three gates, in livewire's order: what was known by ``known_at``, what that
        knowledge has not since superseded, and what had taken effect by ``as_of``.
        ``include_candidates`` selects the looser status reading -- everything that is
        not ``rejected`` -- which is the only way r2k-proxy has members at all, since it
        is never verified.

        On the strict reading the survivors are then checked against the security
        master, matching livewire: a member whose identity is no longer verified at
        ``as_of`` is not a member. The loose reading skips that check, because its ids
        are mostly ``unresolved:<TICKER>`` placeholders the master cannot know.
        """
        path = self.events_path(index_id)
        if path is None:
            raise MembershipDataError(f"unknown index {index_id!r}")
        events = self._read_events(path, index_id, known_at=known_at)
        superseded = {event.supersedes for event in events if event.supersedes is not None}
        cutoff = _end_of_day(as_of)
        applicable = [
            event
            for event in events
            if event.event_id not in superseded
            and event.effective_at <= cutoff
            and (event.status != _REJECTED if include_candidates else event.status == _VERIFIED)
        ]
        members: set[str] = set()
        for event in applicable:
            if event.action == "add":
                members.add(event.security_id)
            elif event.action == "remove":
                members.discard(event.security_id)
            else:
                logger.warning("ignoring unknown action %r in %s", event.action, index_id)
        if not include_candidates:
            members &= self.verified_security_ids(sorted(members), as_of, known_at=known_at)
        return sorted(members)

    def has_events_for_status(self, index_id: str, *, include_candidates: bool = False) -> bool:
        """Whether the log holds any row at all under this status reading.

        This is what separates "nothing has been published for this index yet", which
        must fail closed, from "this date is simply before the index had members",
        which is a real point-in-time answer and an honest empty list.
        """
        path = self.events_path(index_id)
        if path is None:
            raise MembershipDataError(f"unknown index {index_id!r}")
        if include_candidates:
            sql = "SELECT 1 FROM read_parquet(?) WHERE status IS DISTINCT FROM ? LIMIT 1"
            status = _REJECTED
        else:
            sql = "SELECT 1 FROM read_parquet(?) WHERE status = ? LIMIT 1"
            status = _VERIFIED
        return bool(self._query(sql, [str(path), status]))

    def history_for_security(
        self, security_id: str, index_id: Optional[str] = None
    ) -> List[MembershipEvent]:
        """Every logged event for a security, all statuses, across one or all indices.

        Deliberately ungated: the history is the audit trail, so superseded rows and
        rejected rows stay visible, each carrying the ``supersedes`` that retracts it.
        """
        if index_id is not None:
            paths = [(index_id, self.events_path(index_id))]
            if paths[0][1] is None:
                raise MembershipDataError(f"unknown index {index_id!r}")
        else:
            paths = [(name, self.events_path(name)) for name in self.list_indices()]
        events: List[MembershipEvent] = []
        for name, path in paths:
            if path is None:  # pragma: no cover - list_indices only yields real files
                continue
            events.extend(self._read_events(path, name, security_id=security_id))
        events.sort(key=lambda e: (e.effective_at, e.known_at, e.revision, e.event_id))
        return events

    # -- security master ---------------------------------------------------

    def _security_master_available(self) -> bool:
        return self.security_master_path.is_file()

    def _current_rows_cte(self, known_at: Optional[date]) -> Tuple[str, List[Any]]:
        """Verified rows that nothing supersedes -- the live view of the master.

        Returns the CTE prefix and its leading bind parameters together so the two
        cannot drift. With a ``known_at`` cutoff, the source rows are cut *before* the
        superseded set is derived, so a correction nobody knew about yet cannot retract
        a row that was still standing then.
        """
        source = "SELECT * FROM read_parquet(?)"
        params: List[Any] = [str(self.security_master_path)]
        if known_at is not None:
            source += " WHERE known_at <= ?"
            params.append(_end_of_day(known_at))
        sql = (
            f"WITH src AS ({source}), "
            "superseded AS (SELECT DISTINCT supersedes FROM src WHERE supersedes IS NOT NULL), "
            "current_rows AS ("
            "  SELECT * FROM src WHERE status = 'verified' "
            "    AND event_id NOT IN (SELECT supersedes FROM superseded)"
            ") "
        )
        return sql, params

    def resolve_symbol(
        self, symbol: str, as_of: date, *, known_at: Optional[date] = None
    ) -> SymbolResolution:
        """Map a ticker to its ``security_id`` at ``as_of``; never guess when ambiguous."""
        if not self._security_master_available():
            return SymbolResolution(security_id=None, ambiguous=False)
        placeholders = ", ".join("?" for _ in _SYMBOL_EXCHANGES)
        cte, cte_params = self._current_rows_cte(known_at)
        sql = (
            cte + "SELECT DISTINCT security_id FROM current_rows "
            "WHERE symbol = ? AND provider = ? "
            f"AND exchange_mic IN ({placeholders}) "
            "AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?)"
        )
        instant = _start_of_day(as_of)
        params: List[Any] = [
            *cte_params,
            symbol.upper(),
            _SYMBOL_PROVIDER,
            *_SYMBOL_EXCHANGES,
            instant,
            instant,
        ]
        rows = self._query(sql, params)
        return resolve_security_ids([str(row[0]) for row in rows])

    def verified_security_ids(
        self, security_ids: Sequence[str], as_of: date, *, known_at: Optional[date] = None
    ) -> set[str]:
        """The subset whose identity the master still calls verified at ``as_of``.

        No provider or venue filter, matching livewire's ``is_verified``: the question
        is whether the identity stands at all, not where it trades.
        """
        if not security_ids or not self._security_master_available():
            return set()
        ids = ", ".join("?" for _ in security_ids)
        cte, cte_params = self._current_rows_cte(known_at)
        sql = (
            cte + f"SELECT DISTINCT security_id FROM current_rows WHERE security_id IN ({ids}) "
            "AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?)"
        )
        instant = _start_of_day(as_of)
        params: List[Any] = [
            *cte_params,
            *security_ids,
            instant,
            instant,
        ]
        return {str(row[0]) for row in self._query(sql, params)}

    def symbols_for(
        self, security_ids: Sequence[str], as_of: date, *, known_at: Optional[date] = None
    ) -> dict[str, str]:
        """As-of tickers for the given securities; ids without one are simply absent.

        The order is pinned so the label is deterministic. livewire's master forbids a
        verified id holding two tickers at one instant (``_validate_append`` rejects the
        collision), so the first row is the only row.
        """
        if not security_ids or not self._security_master_available():
            return {}
        ids = ", ".join("?" for _ in security_ids)
        venues = ", ".join("?" for _ in _SYMBOL_EXCHANGES)
        cte, cte_params = self._current_rows_cte(known_at)
        sql = (
            cte + f"SELECT DISTINCT security_id, symbol FROM current_rows "
            f"WHERE security_id IN ({ids}) "
            f"AND provider = ? AND exchange_mic IN ({venues}) "
            "AND effective_from <= ? AND (effective_to IS NULL OR effective_to > ?) "
            "ORDER BY security_id, exchange_mic, symbol"
        )
        instant = _start_of_day(as_of)
        params: List[Any] = [
            *cte_params,
            *security_ids,
            _SYMBOL_PROVIDER,
            *_SYMBOL_EXCHANGES,
            instant,
            instant,
        ]
        out: dict[str, str] = {}
        for row in self._query(sql, params):
            if row[1] is not None:
                out.setdefault(str(row[0]), str(row[1]))
        return out

    def members_with_symbols(
        self,
        index_id: str,
        as_of: date,
        *,
        known_at: Optional[date] = None,
        include_candidates: bool = False,
    ) -> List[Member]:
        """Replayed members labelled with their as-of ticker where one resolves."""
        ids = self.members_as_of(
            index_id, as_of, known_at=known_at, include_candidates=include_candidates
        )
        labels = self.symbols_for(ids, as_of, known_at=known_at)
        return [Member(security_id=sid, symbol=labels.get(sid)) for sid in ids]
