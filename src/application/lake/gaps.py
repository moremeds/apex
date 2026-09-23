"""Session gap diagnosis (design §4).

The assessment is ``session_presence``: which expected sessions have at least one bar,
never whether every minute is there. Expected sessions come from an explicit calendar:

- equity: the XNYS exchange calendar (pandas-market-calendars), ``certainty=exchange``
- fx: Monday-Friday, ``certainty=approximate`` (FX rolls at 17:00 New York; holidays
  are not modelled)
- volatility / cmdty / futures / rates: XNYS as an approximation, labelled so

Edges are never clamped away: sessions before the first and after the last observed
session in the window are reported as leading/trailing unobserved ranges, and sessions
outside the security's known identity lifetime are ``not_expected``. A window with no
observed session is ``no_data``, never "zero gaps".
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import pandas as pd
import pandas_market_calendars as mcal

from src.application.lake.errors import LakeError
from src.application.lake.guards import check_listing, check_timeframe, spec_or_raise
from src.application.lake.identity import today_utc
from src.application.lake.services import LakeServices
from src.infrastructure.adapters.livewire.parquet_reads import QueryTimeout
from src.infrastructure.adapters.livewire.paths import (
    delisted_bronze_path,
    parquet_path,
)
from src.infrastructure.adapters.livewire.reference import ReferenceDataError
from src.infrastructure.adapters.livewire.repairs import RepairsEvidence
from src.infrastructure.adapters.livewire.sessions import session_presence

logger = logging.getLogger(__name__)

CalendarPolicy = Literal["auto", "xnys", "weekdays"]
MAX_GAPS_LIMIT = 2000
_EXCHANGE_CLASSES = ("equity",)


@dataclass(frozen=True)
class SessionRange:
    start: date
    end: date
    sessions: int


@dataclass(frozen=True)
class GapsResult:
    symbol: str
    asset_class: str
    timeframe: str
    listing_status: str
    start: date
    end: date
    calendar: Dict[str, str]
    lifetime: Dict[str, Any]
    file_first: Optional[date]
    file_last: Optional[date]
    expected_sessions: int
    present_sessions: int
    not_expected_sessions: int
    status: str
    leading_unobserved: Optional[SessionRange]
    trailing_unobserved: Optional[SessionRange]
    gaps: List[SessionRange]
    gaps_total: int
    truncated: bool
    repairs: RepairsEvidence


@lru_cache(maxsize=1)
def _xnys() -> Any:
    return mcal.get_calendar("XNYS")


def expected_sessions(policy: str, start: date, end: date) -> Tuple[List[date], Dict[str, str]]:
    """The calendar's sessions in ``[start, end]`` and a label describing it."""
    if policy == "weekdays":
        days = pd.bdate_range(start, end)
        label = {"name": "weekdays", "certainty": "approximate"}
    else:
        days = _xnys().valid_days(start_date=start.isoformat(), end_date=end.isoformat())
        label = {
            "name": "XNYS",
            "version": f"pandas-market-calendars {mcal.__version__}",
        }
    return [ts.date() for ts in days], label


def _runs(sessions: Sequence[date], order: Dict[date, int]) -> List[SessionRange]:
    """Group missing sessions into runs of consecutive EXPECTED sessions."""
    runs: List[SessionRange] = []
    for day in sessions:
        if runs and order[day] == order[runs[-1].end] + 1:
            last = runs[-1]
            runs[-1] = SessionRange(last.start, day, last.sessions + 1)
        else:
            runs.append(SessionRange(day, day, 1))
    return runs


async def find_gaps(
    services: LakeServices,
    *,
    symbol: str,
    asset_class: str,
    timeframe: str,
    start: date,
    end: date,
    listing: str = "listed",
    max_gaps: int = 100,
    calendar: CalendarPolicy = "auto",
) -> GapsResult:
    spec = spec_or_raise(asset_class)
    check_timeframe(spec, timeframe)
    if start > end:
        raise LakeError("invalid_parameter", f"start {start} is after end {end}")
    if not 1 <= max_gaps <= MAX_GAPS_LIMIT:
        raise LakeError("invalid_parameter", f"max_gaps must be between 1 and {MAX_GAPS_LIMIT}")
    if calendar not in ("auto", "xnys", "weekdays"):
        raise LakeError("invalid_parameter", f"unknown calendar {calendar!r}")
    provider = services.require_provider()
    status = check_listing(provider, listing, symbol, spec.name, timeframe, "raw")
    paths = []
    if status in ("listed", "dual"):
        paths.append(parquet_path(provider.bronze_root, symbol, timeframe, spec.name))
    if status in ("delisted", "dual") and provider.delisted_root is not None:
        paths.append(delisted_bronze_path(provider.delisted_root, symbol, timeframe, spec.name))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise LakeError(
            "unknown_symbol",
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )

    policy = calendar
    if policy == "auto":
        policy = "weekdays" if spec.name == "fx" else "xnys"
    expected, label = expected_sessions(policy, start, end)
    label["certainty"] = (
        "exchange" if policy == "xnys" and spec.name in _EXCHANGE_CLASSES else "approximate"
    )
    try:
        presence = await session_presence(provider.db, paths, timeframe, start, end)
    except QueryTimeout as exc:
        raise LakeError("query_timeout", str(exc), symbol=symbol, asset_class=spec.name) from exc
    lifetime, in_life = await _lifetime(services, symbol, spec.name)
    present = set(presence.present)
    considered = [day for day in expected if in_life(day)]
    not_expected = len(expected) - len(considered)
    observed = [day for day in considered if day in present]
    order = {day: index for index, day in enumerate(considered)}
    repairs = await asyncio.to_thread(
        services.repairs.evidence, symbol, spec.name, timeframe, start, end
    )

    common: Dict[str, Any] = dict(
        symbol=symbol,
        asset_class=spec.name,
        timeframe=timeframe,
        listing_status=status,
        start=start,
        end=end,
        calendar=label,
        lifetime=lifetime,
        file_first=presence.file_first,
        file_last=presence.file_last,
        expected_sessions=len(considered),
        present_sessions=len(observed),
        not_expected_sessions=not_expected,
        repairs=repairs,
    )
    if not observed:
        return GapsResult(
            **common,
            status="no_data",
            leading_unobserved=None,
            trailing_unobserved=None,
            gaps=[],
            gaps_total=0,
            truncated=False,
        )
    first, last = observed[0], observed[-1]
    leading = [day for day in considered if day < first]
    trailing = [day for day in considered if day > last]
    interior = [day for day in considered if first < day < last and day not in present]
    runs = _runs(interior, order)
    return GapsResult(
        **common,
        status="gaps" if runs else "complete_sessions",
        leading_unobserved=SessionRange(leading[0], leading[-1], len(leading)) if leading else None,
        trailing_unobserved=(
            SessionRange(trailing[0], trailing[-1], len(trailing)) if trailing else None
        ),
        gaps=runs[:max_gaps],
        gaps_total=len(runs),
        truncated=len(runs) > max_gaps,
    )


async def _lifetime(
    services: LakeServices, symbol: str, asset_class: str
) -> Tuple[Dict[str, Any], Any]:
    """Identity intervals bound the sessions a ticker can be expected to trade.

    Only equity has a security master. Unknown lifetime expects every calendar session
    and says so, rather than calling the result complete.
    """

    def everything(day: date) -> bool:
        return True

    if asset_class != "equity" or services.reference.lake_root is None:
        return {
            "state": "unknown",
            "reason": "no identity source for this class",
        }, everything
    try:
        intervals = await asyncio.to_thread(services.reference.fetch_identity, symbol.upper())
    except ReferenceDataError as exc:
        return {
            "state": "unknown",
            "reason": f"security master unreadable: {exc}",
        }, everything
    if not intervals:
        return {
            "state": "unknown",
            "reason": "no verified identity interval",
        }, everything
    spans = [
        (
            date.fromisoformat(i.effective_from[:10]) if i.effective_from else date.min,
            date.fromisoformat(i.effective_to[:10]) if i.effective_to else date.max,
        )
        for i in intervals
    ]

    def in_life(day: date) -> bool:
        # [effective_from, effective_to): the end date is the first day not covered.
        return any(lo <= day < hi for lo, hi in spans)

    described = [
        {"from": lo.isoformat(), "to": None if hi == date.max else hi.isoformat()}
        for lo, hi in spans
    ]
    return {"state": "known", "intervals": described}, in_life


def gap_window_default(end: Optional[date], start: Optional[date]) -> Tuple[date, date]:
    """No window means the last 365 calendar days ending today (UTC)."""
    stop = end or today_utc()
    return (start or stop - timedelta(days=365)), stop
