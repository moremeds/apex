"""Bulk OHLCV read: one request, many tickers, one adjustment basis.

- ``GET /v1/equity/bars?symbols=A,B,C&timeframe=1d``

The per-symbol route answers one ticker per round trip, which turns a 200-name
watchlist into 200 requests and -- worse in adjusted mode -- 200 independently pinned
Silver revisions. This route pins one revision for the whole table, so every series in
the response is adjusted on the same corporate-action set.

A symbol that cannot be served is reported in ``missing`` with its reason rather than
failing the request: one delisted ticker in a list of 200 must not cost the other 199.

Registration: the path is three segments (``/v1/equity/bars``) and so cannot collide
with ``/v1/{asset_class}/{symbol}/bars`` (four), but it DOES collide with
``/v1/{asset_class}/{symbol}`` in ``instruments.py``, which would match it as
``symbol="bars"``. This router is registered before that one -- same reason
``/v1/equity/returns`` and ``/v1/membership/*`` are.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request

from src.api.payload.chart import bulk_bars_payload_from
from src.api.payload.validate import validate_payload
from src.api.routes._lake import lake_services
from src.application.lake.bars import query_bulk_bars
from src.application.lake.guards import DEFAULT_BARS

router = APIRouter(tags=["chart"])


@router.get("/v1/equity/bars")
async def bulk_equity_bars(
    request: Request,
    symbols: str = Query("", description="Comma-separated tickers, at most 200"),
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
    price_mode: Optional[str] = Query(default=None, description="raw | adjusted"),
    listing: str = Query(default="listed", description="listed | delisted | any"),
    silver_revision: Optional[int] = Query(
        default=None, description="pin one retained numbered Silver revision (1d, listed)"
    ),
) -> Dict[str, Any]:
    result = await query_bulk_bars(
        lake_services(request),
        symbols=symbols.split(","),
        timeframe=timeframe,
        start=start,
        end=end,
        limit=limit,
        price_mode=price_mode,
        listing=listing,
        silver_revision_pin=silver_revision,
        policy="legacy",
    )
    payload = bulk_bars_payload_from(result, generated_at=datetime.now(timezone.utc))
    validate_payload(payload, "bulk_bars_payload")
    return payload
