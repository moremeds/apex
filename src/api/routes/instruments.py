"""Discovery: what does apex hold, and which symbols would fail in adjusted mode."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.lake import (
    actions_payload,
    delisting_payload,
    instrument_payload,
    instruments_payload,
)
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
    rows = await catalog.search_instruments(
        lake_services(request), q=q, asset_class=asset_class, limit=limit, listing=listing
    )
    payload = instruments_payload(rows)
    validate_payload(payload, "instruments_payload")
    return payload


# Actions and delisting are ticker-keyed, not security-keyed: see payload/lake.py.


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
    payload = actions_payload(result, paged)
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
    payload = delisting_payload(symbol, page, paged)
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
