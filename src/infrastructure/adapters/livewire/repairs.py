"""Read Livewire's gap-engine repair reports as supplementary gap evidence.

Contract (Livewire ``coverage_report.py`` / ``gap_engine.py``; shapes checked against
the mini lake on 2026-09-23), all at the repairs ROOT only:

- ``tier_a_<date>.json``    -- object ``{"repairs": [entry, ...]}``
- ``decisions_<date>.json`` -- list of entries, each with a ``verdict``
  (``terminus`` / ``inconclusive`` / ...)
- ``unresolved.json``       -- list of ``{symbol, asset_class, timeframe, session,
  reason, as_of}``

An entry carries symbol, asset_class, timeframe, ``gap`` and ``sessions`` where
supplied, plus ``heal_by_days`` and ``source``. Dated subdirectories (one-off repair
batches, Shepherd staging/rollback receipts) are not reports and are never read.

These files are historical evidence, not current truth: a repair report says what the
gap engine saw on its report date. Nothing here creates a report or triggers a repair.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

_REPORT_RE = re.compile(r"^(tier_a|decisions)_(\d{4}-\d{2}-\d{2})\.json$")
_UNRESOLVED = "unresolved.json"
_KNOWN_FIELDS = ("symbol", "asset_class", "timeframe", "sessions", "session")

RepairsState = Literal["available", "not_configured", "absent", "degraded"]


@dataclass(frozen=True)
class RepairEntry:
    """One report row, with the report that carried it."""

    report_kind: str
    reports: tuple[str, ...]
    report_date: Optional[str]
    symbol: str
    asset_class: str
    timeframe: str
    sessions: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairsEvidence:
    """Filtered evidence plus the state of the reports behind it.

    ``state == "degraded"`` means at least one report was unreadable or malformed, so
    an empty ``entries`` is not "no repairs"; ``warnings`` says which.
    """

    state: RepairsState
    entries: tuple[RepairEntry, ...]
    warnings: tuple[str, ...]
    reports_read: int


class RepairsReader:
    def __init__(self, root: Optional[Path]) -> None:
        self._root = Path(root) if root is not None else None

    @classmethod
    def from_env(cls) -> "RepairsReader":
        """From ``APEX_LIVEWIRE_REPAIRS_ROOT``; unset is ``not_configured``, not an error."""
        raw = os.environ.get("APEX_LIVEWIRE_REPAIRS_ROOT", "").strip()
        return cls(Path(raw).expanduser() if raw else None)

    @property
    def configured(self) -> bool:
        return self._root is not None

    def evidence(
        self,
        symbol: str,
        asset_class: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> RepairsEvidence:
        """Entries for one series whose sessions intersect ``[start, end]``."""
        state, raw, warnings, count = self._read_all()
        lo, hi = start.isoformat(), end.isoformat()
        merged: dict[str, RepairEntry] = {}
        for entry in raw:
            if (entry.symbol, entry.asset_class, entry.timeframe) != (
                symbol,
                asset_class,
                timeframe,
            ):
                continue
            if not any(lo <= session <= hi for session in entry.sessions):
                continue
            # Exact repeats across daily reports collapse into one entry listing every
            # report; entries that differ in any field are kept side by side.
            key = json.dumps(
                [entry.report_kind, entry.sessions, entry.details],
                sort_keys=True,
                default=str,
            )
            previous = merged.get(key)
            merged[key] = (
                entry
                if previous is None
                else RepairEntry(
                    report_kind=entry.report_kind,
                    reports=previous.reports + entry.reports,
                    report_date=max(previous.report_date or "", entry.report_date or "") or None,
                    symbol=entry.symbol,
                    asset_class=entry.asset_class,
                    timeframe=entry.timeframe,
                    sessions=entry.sessions,
                    details=entry.details,
                )
            )
        return RepairsEvidence(state, tuple(merged.values()), tuple(warnings), count)

    def status(self) -> dict[str, Any]:
        state, _, warnings, count = self._read_all()
        return {"state": state, "reports_read": count, "warnings": warnings}

    def _read_all(self) -> tuple[RepairsState, list[RepairEntry], list[str], int]:
        if self._root is None:
            return "not_configured", [], [], 0
        if not self._root.is_dir():
            return (
                "absent",
                [],
                [f"repairs root is not a directory: {self._root.name}"],
                0,
            )
        entries: list[RepairEntry] = []
        warnings: list[str] = []
        count = 0
        for path in sorted(self._root.iterdir()):
            match = _REPORT_RE.match(path.name)
            if match is None and path.name != _UNRESOLVED:
                continue
            if not path.is_file():
                continue
            kind = match.group(1) if match else "unresolved"
            report_date = match.group(2) if match else None
            try:
                payload = json.loads(path.read_bytes())
                rows = _rows(kind, payload)
                parsed = [_entry(kind, path.name, report_date, row) for row in rows]
                entries.extend(parsed)
                count += 1
            except OSError as exc:
                logger.warning("repairs report %s unreadable: %s", path.name, exc)
                warnings.append(f"{path.name}: {exc.strerror or type(exc).__name__}")
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning("repairs report %s malformed: %s", path.name, exc)
                warnings.append(f"{path.name}: malformed ({type(exc).__name__})")
        if warnings:
            return "degraded", entries, warnings, count
        if count == 0:
            # A readable root with no report is not "no repairs": nothing was published.
            return "absent", [], ["no repair reports in the repairs root"], 0
        return "available", entries, warnings, count


def _rows(kind: str, payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("repairs") if kind == "tier_a" and isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{kind} report must hold a list of objects")
    return rows


def _entry(kind: str, report: str, report_date: Optional[str], row: dict[str, Any]) -> RepairEntry:
    sessions = row["sessions"] if "sessions" in row else [row["session"]]
    if not isinstance(sessions, list) or not all(isinstance(s, str) for s in sessions):
        raise ValueError("sessions must be a list of ISO dates")
    for session in sessions:
        date.fromisoformat(session)
    for name in ("symbol", "asset_class", "timeframe"):
        if not isinstance(row[name], str):
            raise ValueError(f"{name} must be a string")
    details = {key: value for key, value in row.items() if key not in _KNOWN_FIELDS}
    return RepairEntry(
        report_kind=kind,
        reports=(report,),
        report_date=report_date,
        symbol=row["symbol"],
        asset_class=row["asset_class"],
        timeframe=row["timeframe"],
        sessions=tuple(sessions),
        details=details,
    )
