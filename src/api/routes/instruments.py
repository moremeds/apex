"""Discovery: what does apex hold, and which symbols would fail in adjusted mode."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.lake import instrument_payload, page_fields
from src.api.payload.validate import validate_payload
from src.api.routes._lake import lake_services
from src.application.lake import catalog, identity

logger = logging.getLogger(__name__)

router = APIRouter(tags=["instruments"])


def _parse_day(value: Optional[str], name: str) -> Optional[date]:
    """Parse an optional YYYY-MM-DD filter bound; absent is not an error."""
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"malformed {name} {value!r}; expected YYYY-MM-DD",
        ) from exc


@router.get("/v1/instruments")
async def list_instruments(
    request: Request,
    asset_class: Optional[str] = None,
    q: Optional[str] = Query(default=None, description="symbol prefix filter"),
    listing: str = Query(default="listed", description="listed | delisted"),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    if listing not in ("listed", "delisted", "any"):
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"unknown listing filter {listing!r} (have listed, delisted, any)",
        )
    if listing != "listed":
        # The coverage table measures the live tree only; bronze-delisted/ is not in it.
        raise ApiError(
            ApiErrorCode.NOT_YET_AVAILABLE,
            "delisted discovery requires upstream livewire work "
            "(instrument identity, corporate-action backfill, Silver over bronze-delisted)",
        )
    rows = await catalog.search_instruments(
        lake_services(request), q=q, asset_class=asset_class, limit=limit
    )
    payload = {
        "instruments": [
            {
                "symbol": r.symbol,
                "asset_class": r.asset_class,
                "listing_status": r.listing_status,
                "first_date": r.first_date,
                "last_date": r.last_date,
                "silver_available": r.silver_available,
                "price_mode": r.price_mode,
            }
            for r in rows
        ],
        "count": len(rows),
        # These dates come from livewire's 11:00 UTC coverage snapshot, not from the
        # artifacts. Labelled so a consumer does not mistake them for live values.
        "source": "livewire_coverage_snapshot",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_payload(payload, "instruments_payload")
    return payload


# The ticker-reuse caveat, carried verbatim from the era when these endpoints were 501:
# both artifacts are keyed by TICKER, not by a permanent security id. Measured
# 2026-08-23 -- 6,275 delisted symbols have no corporate-action data at all, and the
# 2,345 that appear to are ticker reuses whose actions belong to a different, living
# company. Every response below says ``identity: "ticker"`` so a consumer cannot
# mistake it for security-level truth.
_TICKER_IDENTITY = "ticker"


@router.get("/v1/equity/{symbol}/actions")
async def get_corporate_actions(
    request: Request,
    symbol: str,
    type: Optional[str] = Query(default=None, description="split | cash_dividend"),
    start: Optional[str] = Query(default=None, description="earliest ex_date, YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="latest ex_date, YYYY-MM-DD"),
    limit: Optional[int] = Query(default=None, description="opt-in page size (1..2000)"),
    offset: Optional[int] = Query(default=None, description="opt-in page offset"),
) -> dict:
    """Corporate actions behind a symbol's adjustment, from livewire bronze.

    **Ticker-keyed, not security-keyed.** The log is stored per ticker, so for a symbol
    that was reused the actions may belong to a different, living company; measured
    2026-08-23, 2,345 delisted tickers are reuses of live ones. ``identity`` says so in
    the payload. Resolve the ticker through ``/v1/equity/{symbol}/delisting`` or the
    membership surface before treating a series as one company's.

    Only ``status='active'`` rows count: a correction lands as a new ``action_id`` that
    supersedes the old row, and the old row is re-marked ``corrected``.
    """
    paged = limit is not None or offset is not None
    result = await identity.corporate_actions(
        lake_services(request),
        symbol,
        action_type=type,
        start=_parse_day(start, "start"),
        end=_parse_day(end, "end"),
        limit=limit,
        offset=offset,
        paged=paged,
    )
    payload: Dict[str, Any] = {
        "symbol": result.symbol,
        "identity": _TICKER_IDENTITY,
        "source": "livewire_bronze_corporate_action",
        "provider": result.provider,
        "actions": [action.as_dict() for action in result.page.items],
        "count": len(result.page.items),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if paged:
        payload.update(page_fields(result.page))
    validate_payload(payload, "actions_payload")
    return payload


@router.get("/v1/equity/{symbol}/delisting")
async def get_delisting(
    request: Request,
    symbol: str,
    limit: Optional[int] = Query(default=None, description="opt-in page size (1..2000)"),
    offset: Optional[int] = Query(default=None, description="opt-in page offset"),
) -> dict:
    """Identity intervals for a ticker, from the livewire security master.

    **Not a terminal-state record.** Measured 2026-09-21, the security master carries
    no delisting reason, no delist date as such and no final consideration --
    ``relationship_type`` and ``related_security_id`` are null across the whole file.
    What it does carry is ``[effective_from, effective_to)`` identity intervals, so the
    honest answer is those intervals and the issuer behind them: a closed
    ``effective_to`` tells you the ticker stopped resolving to that issuer, and nothing
    tells you why. Do not read a bankruptcy into a flat exit.

    **Ticker-keyed**, with the same reuse caveat as ``/actions``: two intervals under
    one ticker are two different securities, not one company's history.
    """
    paged = limit is not None or offset is not None
    page = await identity.delisting(
        lake_services(request), symbol, limit=limit, offset=offset, paged=paged
    )
    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "identity": _TICKER_IDENTITY,
        "source": "livewire_security_master",
        # Named so nobody reads the absence of a reason as "still listed".
        "delisting_reason_available": False,
        "intervals": [interval.as_dict() for interval in page.items],
        "count": len(page.items),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if paged:
        payload.update(page_fields(page))
    validate_payload(payload, "delisting_payload")
    return payload


# Route ordering is NOT a constraint here: Starlette matches on the whole path pattern,
# so /v1/{asset_class}/{symbol} (two segments) can never shadow /v1/instruments (one)
# nor /v1/equity/{symbol}/bars (three). Verified empirically in both registration orders.
@router.get("/v1/{asset_class}/{symbol}")
async def get_instrument(asset_class: str, symbol: str, request: Request) -> dict:
    """One instrument's detail, including the timeframes that actually exist on disk."""
    detail = await catalog.get_instrument(lake_services(request), symbol, asset_class)
    return instrument_payload(detail)
