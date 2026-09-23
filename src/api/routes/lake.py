"""Lake discovery, revision, identity and gap routes (the REST twins of the MCP tools).

- ``GET /v1/lake/asset-classes``            registry
- ``GET /v1/lake/status``                   source availability and freshness
- ``GET /v1/lake/coverage``                 raw catalog rows, paged
- ``GET /v1/lake/silver-revisions[/{n}]``   retained Silver revisions / one detail
- ``GET /v1/lake/pit-revisions[/{n}]``      published PIT revisions / one detail
- ``GET /v1/security/{symbol}``             ticker -> security_id at a date
- ``GET /v1/futures/{root}/contracts``      contracts under one futures root
- ``GET /v1/{asset_class}/{symbol}/gaps``   session-presence gap diagnosis

Registered in server.py's literal-namespace block, before ``instruments``: the
three-segment paths here would otherwise match ``/v1/{asset_class}/{symbol}``.
Silver detail takes an explicit number; the list names ``current``. PIT detail
always takes a number -- there is no PIT current (design §3.4).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.lake import (
    coverage_payload,
    futures_payload,
    gaps_payload,
    pit_detail_payload,
    pit_list_payload,
    security_payload,
    silver_detail_payload,
    silver_list_payload,
)
from src.api.routes._lake import lake_services
from src.application.lake import catalog, gaps, identity, revisions

router = APIRouter(tags=["lake"])

_LIMIT = Query(None, description="page size (1..2000, default 100)")
_OFFSET = Query(None, description="page offset (default 0)")


def _day(value: Optional[str], name: str) -> Optional[date]:
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER, f"malformed {name} {value!r}; expected YYYY-MM-DD"
        ) from exc


@router.get("/v1/lake/asset-classes")
async def get_asset_classes() -> Dict[str, Any]:
    return {"asset_classes": catalog.asset_classes()}


@router.get("/v1/lake/status")
async def get_lake_status(request: Request) -> Dict[str, Any]:
    return {"sources": await catalog.lake_status(lake_services(request))}


@router.get("/v1/lake/coverage")
async def get_coverage(
    request: Request,
    symbol: Optional[str] = Query(None, description="exact symbol"),
    asset_class: Optional[str] = None,
    include_silver: bool = True,
    limit: Optional[int] = _LIMIT,
    offset: Optional[int] = _OFFSET,
) -> Dict[str, Any]:
    result = await catalog.coverage(
        lake_services(request),
        symbol=symbol,
        asset_class=asset_class,
        include_silver=include_silver,
        limit=limit,
        offset=offset,
    )
    return coverage_payload(result)


@router.get("/v1/lake/silver-revisions")
async def get_silver_revisions(
    request: Request, limit: Optional[int] = _LIMIT, offset: Optional[int] = _OFFSET
) -> Dict[str, Any]:
    result = await revisions.list_silver_revisions(
        lake_services(request), limit=limit, offset=offset
    )
    return silver_list_payload(result)


@router.get("/v1/lake/silver-revisions/{revision}")
async def get_silver_revision(
    request: Request,
    revision: int,
    limit: Optional[int] = _LIMIT,
    offset: Optional[int] = _OFFSET,
) -> Dict[str, Any]:
    detail = await revisions.silver_revision_detail(
        lake_services(request), revision, limit=limit, offset=offset
    )
    return silver_detail_payload(detail)


@router.get("/v1/lake/pit-revisions")
async def get_pit_revisions(
    request: Request,
    index_id: Optional[str] = None,
    limit: Optional[int] = _LIMIT,
    offset: Optional[int] = _OFFSET,
) -> Dict[str, Any]:
    result = await revisions.list_pit_revisions(
        lake_services(request), index_id=index_id, limit=limit, offset=offset
    )
    return pit_list_payload(result)


@router.get("/v1/lake/pit-revisions/{revision}")
async def get_pit_revision(
    request: Request,
    revision: int,
    limit: Optional[int] = _LIMIT,
    offset: Optional[int] = _OFFSET,
) -> Dict[str, Any]:
    detail = await revisions.pit_revision_detail(
        lake_services(request), revision, limit=limit, offset=offset
    )
    return pit_detail_payload(detail)


@router.get("/v1/security/{symbol}")
async def get_security(
    request: Request,
    symbol: str,
    as_of: Optional[str] = Query(None, description="YYYY-MM-DD (default: today UTC)"),
    known_at: Optional[str] = Query(None, description="only identity known by this date"),
) -> Dict[str, Any]:
    resolution = await identity.resolve_security(
        lake_services(request),
        symbol,
        _day(as_of, "as_of") or identity.today_utc(),
        _day(known_at, "known_at"),
    )
    return security_payload(resolution)


@router.get("/v1/futures/{root}/contracts")
async def get_futures_contracts(
    request: Request, root: str, limit: Optional[int] = _LIMIT, offset: Optional[int] = _OFFSET
) -> Dict[str, Any]:
    page = await catalog.futures_contracts(lake_services(request), root, limit=limit, offset=offset)
    return futures_payload(root, page)


@router.get("/v1/{asset_class}/{symbol}/gaps")
async def get_gaps(
    request: Request,
    asset_class: str,
    symbol: str,
    timeframe: str = "1d",
    start: Optional[str] = Query(None, description="first session, YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="last session, YYYY-MM-DD (default today)"),
    listing: str = Query("listed", description="listed | delisted | any"),
    max_gaps: int = Query(100, description="1..2000"),
    calendar: str = Query("auto", description="auto | xnys | weekdays"),
) -> Dict[str, Any]:
    first, last = gaps.gap_window_default(_day(end, "end"), _day(start, "start"))
    result = await gaps.find_gaps(
        lake_services(request),
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        start=first,
        end=last,
        listing=listing,
        max_gaps=max_gaps,
        calendar=calendar,  # type: ignore[arg-type]
    )
    return gaps_payload(result)
