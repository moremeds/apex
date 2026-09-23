"""Point-in-time index membership read surface.

- ``GET /v1/membership/indices``      -- index ids present in the lake
- ``GET /v1/membership/history``      -- one security's effective membership timeline
- ``GET /v1/membership/{index_id}``   -- members as of a date, PIT-gated by ``known_at``

Registration order matters twice: ``/indices`` and ``/history`` are declared before
``/{index_id}`` so they are not swallowed as an index id, and the router itself is
registered before ``instruments``, whose ``/v1/{asset_class}/{symbol}`` would otherwise
match ``/v1/membership/history``.

The queries live in ``src/application/lake/identity.py``; these routes parse dates,
keep the original envelopes, and add pagination fields only when ``limit`` or
``offset`` is passed.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.lake import history_payload, indices_payload, members_payload
from src.api.routes._lake import lake_services
from src.application.lake.identity import (
    index_members,
    list_indices,
    membership_history,
    today_utc,
)
from src.application.lake.services import check_page, page_of

router = APIRouter(prefix="/v1/membership", tags=["membership"])


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


def _paged(limit: Optional[int], offset: Optional[int]) -> bool:
    return limit is not None or offset is not None


@router.get("/indices")
async def get_indices(
    request: Request,
    limit: Optional[int] = Query(None, description="opt-in page size (1..2000)"),
    offset: Optional[int] = Query(None, description="opt-in page offset"),
) -> Dict[str, Any]:
    """Index ids discovered on disk. Never a hardcoded list -- livewire adds indices."""
    indices = await list_indices(lake_services(request))
    page = page_of(indices, *check_page(limit, offset)) if _paged(limit, offset) else None
    return indices_payload(indices, page)


@router.get("/history")
async def get_membership_history(
    request: Request,
    symbol: str = Query(..., description="Ticker to resolve through the security master"),
    index_id: Optional[str] = Query(None, description="Restrict to one index"),
    as_of: Optional[str] = Query(None, description="Date used to resolve the ticker"),
    limit: Optional[int] = Query(None, description="opt-in page size (1..2000)"),
    offset: Optional[int] = Query(None, description="opt-in page offset"),
) -> Dict[str, Any]:
    """The effective membership timeline for a ticker, across both of its log ids."""
    result = await membership_history(
        lake_services(request),
        symbol,
        _parse_day(as_of, "as_of", default=today_utc()),
        index_id=index_id,
        limit=limit,
        offset=offset,
        paged=_paged(limit, offset),
    )
    return history_payload(result, _paged(limit, offset))


@router.get("/{index_id}")
async def members_as_of(
    request: Request,
    index_id: str,
    as_of: Optional[str] = Query(None, description="Membership date (default: today UTC)"),
    known_at: Optional[str] = Query(None, description="Only events known by this date"),
    include_candidates: bool = Query(
        False,
        description="Include every non-rejected event, not just verified ones",
    ),
    limit: Optional[int] = Query(None, description="opt-in page size (1..2000)"),
    offset: Optional[int] = Query(None, description="opt-in page offset"),
) -> Dict[str, Any]:
    """Members of ``index_id`` at ``as_of``, replayed from the event log."""
    as_of_day = _parse_day(as_of, "as_of", default=today_utc())
    known_day = None if known_at is None else _parse_day(known_at, "known_at")
    result = await index_members(
        lake_services(request),
        index_id,
        as_of_day,
        known_at=known_day,
        include_candidates=include_candidates,
        limit=limit,
        offset=offset,
        paged=_paged(limit, offset),
    )
    return members_payload(result, _paged(limit, offset))
