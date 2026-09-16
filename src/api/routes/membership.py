"""Point-in-time index membership read surface.

- ``GET /v1/membership/indices``      -- index ids present in the lake
- ``GET /v1/membership/history``      -- one security's effective membership timeline
- ``GET /v1/membership/{index_id}``   -- members as of a date, PIT-gated by ``known_at``

Registration order matters twice: ``/indices`` and ``/history`` are declared before
``/{index_id}`` so they are not swallowed as an index id, and the router itself is
registered before ``instruments``, whose ``/v1/{asset_class}/{symbol}`` would otherwise
match ``/v1/membership/history``.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from src.api.errors import ApiError, ApiErrorCode
from src.infrastructure.adapters.livewire.membership import (
    MembershipDataError,
    MembershipEvent,
    MembershipReader,
)

router = APIRouter(prefix="/v1/membership", tags=["membership"])

# livewire writes these ids for securities it could not map to the security master.
_UNRESOLVED_PREFIX = "unresolved:"


def _reader_or_raise() -> MembershipReader:
    reader = MembershipReader.from_env()
    if reader is None:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            "index membership is not configured; set APEX_LIVEWIRE_LAKE_ROOT",
        )
    return reader


def _parse_day(value: Optional[str], name: str, *, default: Optional[date] = None) -> date:
    if value is None or not value.strip():
        if default is not None:
            return default
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, f"{name} is required (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"malformed {name} {value!r}; expected YYYY-MM-DD",
        ) from exc


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat()


@router.get("/indices")
async def list_indices() -> Dict[str, List[str]]:
    """Index ids discovered on disk. Never a hardcoded list -- livewire adds indices."""
    reader = _reader_or_raise()
    try:
        return {"indices": await asyncio.to_thread(reader.list_indices)}
    except MembershipDataError as exc:
        raise ApiError(ApiErrorCode.MEMBERSHIP_UNAVAILABLE, str(exc)) from exc


@router.get("/history")
async def membership_history(
    symbol: str = Query(..., description="Ticker to resolve through the security master"),
    index_id: Optional[str] = Query(None, description="Restrict to one index"),
    as_of: Optional[str] = Query(None, description="Date used to resolve the ticker"),
) -> Dict[str, Any]:
    """The effective membership timeline for a ticker, across both of its log ids.

    Events are unioned over the resolved ``security_id`` and the placeholder
    ``unresolved:<TICKER>`` -- livewire logs pre-identity-floor events under the
    placeholder even once the ticker resolves -- and the union is reduced to the
    effective timeline: superseded and rejected rows are dropped, not returned as
    raw audit rows.
    """
    reader = _reader_or_raise()
    resolve_day = _parse_day(as_of, "as_of", default=_today_utc())
    ticker = symbol.strip().upper()
    if not ticker:
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, "symbol is required")

    try:
        resolution = await asyncio.to_thread(reader.resolve_symbol, ticker, resolve_day)
    except MembershipDataError as exc:
        raise ApiError(ApiErrorCode.MEMBERSHIP_UNAVAILABLE, str(exc)) from exc

    if resolution.ambiguous:
        raise ApiError(
            ApiErrorCode.AMBIGUOUS_SECURITY,
            f"symbol {ticker!r} maps to more than one security on {resolve_day.isoformat()}; "
            "apex will not guess which",
            symbol=ticker,
        )
    if index_id is not None and await asyncio.to_thread(reader.events_path, index_id) is None:
        raise ApiError(ApiErrorCode.UNKNOWN_INDEX, f"unknown index {index_id!r}")

    security_id = resolution.security_id
    # livewire's identity backfill only reaches back to the provider's identity floor,
    # so one ticker's log is routinely split: events before the floor stay under
    # `unresolved:<TICKER>` while later ones carry the real id. Both ids go into one
    # call so the adapter can retract superseded and rejected rows over the union --
    # the backfill's rejected revision and its replacement sit under different ids.
    placeholder = f"{_UNRESOLVED_PREFIX}{ticker}"
    ids = [placeholder] if security_id is None else [security_id, placeholder]
    try:
        events: List[MembershipEvent] = await asyncio.to_thread(
            reader.history_for_security, ids, index_id
        )
    except MembershipDataError as exc:
        raise ApiError(ApiErrorCode.MEMBERSHIP_UNAVAILABLE, str(exc)) from exc
    if security_id is None and events:
        security_id = placeholder

    if security_id is None:
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"symbol {ticker!r} is not in the security master on {resolve_day.isoformat()}"
            " and has no unresolved membership events",
            symbol=ticker,
        )

    return {
        "symbol": ticker,
        "security_id": security_id,
        "events": [
            {
                "index_id": event.index_id,
                "security_id": event.security_id,
                "action": event.action,
                "effective_at": _iso(event.effective_at),
                "announced_at": _iso(event.announced_at),
                "known_at": _iso(event.known_at),
                "status": event.status,
                "event_id": event.event_id,
                "supersedes": event.supersedes,
            }
            for event in events
        ],
    }


@router.get("/{index_id}")
async def members_as_of(
    index_id: str,
    as_of: Optional[str] = Query(None, description="Membership date (default: today UTC)"),
    known_at: Optional[str] = Query(None, description="Only events known by this date"),
    include_candidates: bool = Query(
        False,
        description="Include every non-rejected event, not just verified ones",
    ),
) -> Dict[str, Any]:
    """Members of ``index_id`` at ``as_of``, replayed from the event log."""
    reader = _reader_or_raise()
    as_of_day = _parse_day(as_of, "as_of", default=_today_utc())
    known_day = None if known_at is None else _parse_day(known_at, "known_at")

    if await asyncio.to_thread(reader.events_path, index_id) is None:
        raise ApiError(ApiErrorCode.UNKNOWN_INDEX, f"unknown index {index_id!r}")

    try:
        members = await asyncio.to_thread(
            reader.members_with_symbols,
            index_id,
            as_of_day,
            known_at=known_day,
            include_candidates=include_candidates,
        )
    except MembershipDataError as exc:
        raise ApiError(ApiErrorCode.MEMBERSHIP_UNAVAILABLE, str(exc)) from exc

    if not members:
        # An empty replay is two different things. If the log holds no row at all under
        # this status reading, nothing has been published yet and the endpoint fails
        # closed. If it holds rows but none applies here, the emptiness is the answer:
        # the date is before the first constituent, or before anything was known.
        try:
            published = await asyncio.to_thread(
                reader.has_events_for_status,
                index_id,
                include_candidates=include_candidates,
            )
        except MembershipDataError as exc:
            raise ApiError(ApiErrorCode.MEMBERSHIP_UNAVAILABLE, str(exc)) from exc
        if not published:
            raise ApiError(
                ApiErrorCode.MEMBERSHIP_UNAVAILABLE,
                f"membership data is not yet available for index {index_id!r} "
                f"as of {as_of_day.isoformat()}",
            )

    return {
        "index_id": index_id,
        "as_of": as_of_day.isoformat(),
        "known_at": None if known_day is None else known_day.isoformat(),
        "members": [
            {"security_id": member.security_id, "symbol": member.symbol} for member in members
        ],
        "unresolved_count": sum(
            1 for member in members if member.security_id.startswith(_UNRESOLVED_PREFIX)
        ),
    }
