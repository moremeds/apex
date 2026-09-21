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

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.chart import basis_for, build_bar_rows
from src.api.payload.validate import validate_payload
from src.api.routes._chart_guards import (
    _DEFAULT_BARS,
    _artifact_exists,
    _check_listing,
    _check_timeframe,
    _provider_or_raise,
    _resolve_window,
    _silver_revision,
    _spec_or_raise,
)
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable

router = APIRouter(tags=["chart"])

_MAX_SYMBOLS = 200


def _norm_symbols(raw: str) -> List[str]:
    """Upper-cased, de-duplicated, order-preserving. Mirrors /v1/equity/returns."""
    ordered: Dict[str, None] = {}
    for part in raw.split(","):
        symbol = part.strip().upper()
        if symbol:
            ordered[symbol] = None
    symbols = list(ordered)
    if not symbols:
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, "symbols is required and must be non-empty")
    if len(symbols) > _MAX_SYMBOLS:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"{len(symbols)} symbols requested; at most {_MAX_SYMBOLS} per call",
        )
    return symbols


@router.get("/v1/equity/bars")
async def bulk_equity_bars(
    request: Request,
    symbols: str = Query("", description="Comma-separated tickers, at most 200"),
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=_DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
    price_mode: Optional[str] = Query(default=None, description="raw | adjusted"),
    listing: str = Query(default="listed", description="listed | delisted | any"),
) -> Dict[str, Any]:
    requested = _norm_symbols(symbols)
    spec = _spec_or_raise("equity")
    _check_timeframe(spec, timeframe)
    provider = _provider_or_raise(request)
    if price_mode is not None and price_mode not in ("raw", "adjusted"):
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"unknown price_mode {price_mode!r} (have raw, adjusted)",
        )
    if listing not in ("listed", "delisted", "any"):
        # Validated once, up front: _check_listing raises the same ApiError per symbol,
        # but the per-symbol loop below catches ApiError and files it under `missing`,
        # which would turn a malformed request into a 200 with an empty result map.
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"unknown listing filter {listing!r} (have listed, delisted, any)",
        )
    effective = price_mode or provider.effective_price_mode(spec.name)
    if effective == "adjusted":
        # One Silver revision for the whole table: a revision landing mid-request would
        # adjust some symbols on one corporate-action set and the rest on another.
        try:
            provider = await asyncio.to_thread(provider.pin_snapshot)
        except AdjustedDataUnavailable as exc:
            raise ApiError(ApiErrorCode.ADJUSTED_UNAVAILABLE, str(exc)) from exc
    # Any request that may touch the archived tier reads from the epoch: the listing
    # status is per symbol, but the window is resolved once for the whole table.
    window_start, window_end, tail = _resolve_window(
        timeframe, start, end, limit, from_epoch=listing != "listed"
    )

    served: Dict[str, Any] = {}
    missing: Dict[str, str] = {}
    for symbol in requested:
        try:
            status = _check_listing(provider, listing, symbol, spec.name, timeframe, effective)
            bars = await provider.fetch_bars(
                symbol,
                timeframe,
                window_start,
                window_end,
                asset_class=spec.name,
                price_mode=effective,
                listing=status,
            )
        except ApiError as exc:
            # A per-symbol condition (adjusted over a delisted name, say) is this
            # symbol's problem, not the request's -- raising would drop the other 199.
            missing[symbol] = exc.message
            continue
        except AdjustedDataUnavailable as exc:
            missing[symbol] = str(exc)
            continue
        if not bars and not _artifact_exists(provider, symbol, timeframe, spec, effective, status):
            missing[symbol] = f"no artifact for {symbol} under {spec.partition}"
            continue
        if tail is not None:
            bars = bars[-tail:]
        served[symbol] = {
            "listing_status": status,
            "bars": build_bar_rows(bars, spec.name),
        }

    payload = {
        "price_mode": effective,
        "basis": basis_for(effective),
        "adjustment_revision": _silver_revision(provider) if effective == "adjusted" else None,
        "timeframe": timeframe,
        "symbols": served,
        "missing": missing,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_payload(payload, "bulk_bars_payload")
    return payload
