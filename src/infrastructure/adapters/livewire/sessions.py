"""Which session dates do an instrument's artifacts cover?

Gap diagnosis is **session presence**, not minute completeness: a session is present
when at least one bar lands on it. Daily files key by ``trade_date`` (already a
session date); intraday files key by ``bar_timestamp`` (UTC), bucketed to the
America/New_York calendar date -- the session convention of every class the lake
holds intraday (equity, volatility, fx), stated as an approximation for fx, whose
trading day rolls at 17:00 New York.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence

from .parquet_reads import LakeDb


@dataclass(frozen=True)
class SessionPresence:
    present: List[date]
    file_first: Optional[date]
    file_last: Optional[date]


def _day_expr(timeframe: str) -> str:
    if timeframe == "1d":
        return "trade_date"
    return "CAST(timezone('America/New_York', bar_timestamp) AS DATE)"


async def session_presence(
    db: LakeDb, paths: Sequence[Path], timeframe: str, start: date, end: date
) -> SessionPresence:
    """Distinct session dates in ``[start, end]`` across ``paths`` (a listing union),
    plus the overall first/last session of the files themselves. Runs through
    ``LakeDb`` so it shares the per-query deadline of every other lake read."""
    if not paths:
        return SessionPresence([], None, None)
    day = _day_expr(timeframe)
    # Interpolated parts are code constants (the day expression) and one "?" per path;
    # every path and date is a bound parameter.
    union = " UNION ALL ".join(f"SELECT {day} AS day FROM read_parquet(?)" for _ in paths)
    params = [p.as_posix() for p in paths]
    # One statement: the file bounds and the in-window sessions come from the same
    # scan (one snapshot of a mutable Bronze file) under one deadline.
    (row,) = await db.rows(
        f"WITH d AS ({union}) SELECT min(day) AS first, max(day) AS last, "  # nosec B608
        "list(DISTINCT day ORDER BY day) FILTER (WHERE day >= ? AND day <= ?) AS present "
        "FROM d",
        [*params, start, end],
    )
    return SessionPresence(sorted(row["present"] or []), row["first"], row["last"])
