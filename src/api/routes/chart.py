"""Chart read surface for argon (stateless renderer pulls everything from apex).

- GET /v1/{asset_class}/{symbol}/bars        -> OHLCV candles from livewire
  start/end bound the window inclusively at BOTH ends, intraday included:
  timeframe=1m over 12:25:00Z..12:35:00Z returns 11 bars, not 10.
- GET /v1/{asset_class}/{symbol}/indicators  -> per-bar indicator series, compute-on-read
- GET /v1/rates/{symbol}/series              -> yield series (no OHLC)
- GET /v1/equity/{symbol}/confluence         -> multi-timeframe confluence, DB-backed

The flat routes (/bars/{ticker} etc.) are preserved as deprecated aliases so argon and
signal-lab need no change; they carry Deprecation/Sunset/Link headers.

Mirrors the signal contract: REST backfill + validate-on-egress on every response.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, Request, Response

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.chart import (
    bars_payload_from,
    build_confluence_payload,
    build_indicator_payload,
    rates_payload_from,
)
from src.api.payload.validate import validate_payload
from src.api.routes._lake import lake_services, provider_or_raise
from src.application.chart.indicator_compute import (
    UnknownIndicatorError,
    compute_indicator_series,
)
from src.application.lake.bars import query_bars, query_rates
from src.application.lake.guards import (
    DEFAULT_BARS,
    artifact_exists,
    check_timeframe,
    require_bars_payload,
    resolve_window,
    spec_or_raise,
)
from src.domain.signals.indicators.registry import get_indicator_registry
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable

router = APIRouter(tags=["chart"])

# Default no-arg window: the most recent N bars. We over-fetch in calendar time
# (markets aren't 24/7, so N*delta would under-cover across closures) then tail-slice
# to exactly N. Callers wanting an exact range pass start/end.
_SUNSET = "Wed, 31 Dec 2026 23:59:59 GMT"


async def _bars_response(
    request: Request,
    asset_class: str,
    symbol: str,
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int,
    price_mode: Optional[str],
    listing: str,
    silver_revision: Optional[int] = None,
    pit_revision: Optional[int] = None,
) -> dict:
    result = await query_bars(
        lake_services(request),
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=limit,
        price_mode=price_mode,
        listing=listing,
        silver_revision_pin=silver_revision,
        pit_revision=pit_revision,
        policy="legacy",
    )
    payload = bars_payload_from(result, generated_at=datetime.now(timezone.utc))
    validate_payload(payload, "bars_payload")
    return payload


async def _indicators_response(
    request: Request,
    asset_class: str,
    symbol: str,
    indicator: str,
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int,
) -> dict:
    spec = spec_or_raise(asset_class)
    # A yield has no OHLC. Without this the rates parquet is read into BarData with
    # null prices and the indicator computes over them, returning 200 and null
    # bar_close -- a number-shaped answer to a question that has no answer.
    require_bars_payload(spec, symbol)
    check_timeframe(spec, timeframe)
    provider = provider_or_raise(request)
    registry = getattr(request.app.state, "indicator_registry", None) or get_indicator_registry()
    start, end, tail = resolve_window(timeframe, start, end, limit)
    try:
        if provider.effective_price_mode(spec.name) == "adjusted":
            provider = await asyncio.to_thread(provider.pin_snapshot)
        points = await compute_indicator_series(
            provider,
            registry,
            symbol,
            timeframe,
            indicator,
            start,
            end,
            asset_class=spec.name,
        )
    except UnknownIndicatorError as exc:
        # The symbol is fine; the `indicator` QUERY VALUE is not. Reporting this as
        # unknown_symbol sent callers to check their ticker.
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"unknown indicator: {indicator}",
            symbol=symbol,
            asset_class=spec.name,
        ) from exc
    except AdjustedDataUnavailable as exc:
        raise ApiError(
            ApiErrorCode.ADJUSTED_UNAVAILABLE, str(exc), symbol=symbol, asset_class=spec.name
        ) from exc
    if not points and not artifact_exists(
        provider, symbol, timeframe, spec, provider.effective_price_mode(spec.name)
    ):
        # Same rule as /bars, and probed only on an empty result so the happy path
        # costs no extra stat(): no artifact is a 404, an empty window over a real
        # one is a legitimate 200.
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    if tail is not None:
        points = points[-tail:]
    payload = build_indicator_payload(
        symbol, timeframe, indicator, points, generated_at=datetime.now(timezone.utc)
    )
    validate_payload(payload, "indicator_series_payload")
    return payload


async def _confluence_response(
    request: Request,
    symbol: str,
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: int,
) -> dict:
    repo = getattr(request.app.state, "signal_repo", None)
    if repo is None:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            "signal persistence not configured",
            symbol=symbol,
        )
    # Confluence is PG-backed (not livewire), so it accepts any timeframe the data has.
    start, end, _ = resolve_window(timeframe, start, end)
    rows = await repo.get_confluence_history(symbol, timeframe, start, end, limit)
    payload = build_confluence_payload(
        symbol, timeframe, rows, generated_at=datetime.now(timezone.utc)
    )
    validate_payload(payload, "confluence_payload")
    return payload


def _mark_deprecated(response: Response, successor: str) -> None:
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = _SUNSET
    response.headers["Link"] = f'<{successor}>; rel="successor-version"'


# --- /v1 routes -------------------------------------------------------------------


@router.get("/v1/{asset_class}/{symbol}/bars")
async def get_bars_v1(
    asset_class: str,
    symbol: str,
    request: Request,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
    price_mode: Optional[str] = Query(default=None, description="raw | adjusted"),
    listing: str = Query(default="listed", description="listed | delisted | any"),
    silver_revision: Optional[int] = Query(
        default=None, description="pin a retained numbered Silver revision (equity 1d)"
    ),
    pit_revision: Optional[int] = Query(
        default=None, description="serve through a published PIT revision (equity 1d)"
    ),
) -> dict:
    return await _bars_response(
        request,
        asset_class,
        symbol,
        timeframe,
        start,
        end,
        limit,
        price_mode,
        listing,
        silver_revision,
        pit_revision,
    )


@router.get("/v1/rates/{symbol}/series")
async def get_rates_series_v1(
    symbol: str,
    request: Request,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = Query(
        default=None,
        description="omit for the full window; a positive N returns the last N points",
    ),
) -> dict:
    result = await query_rates(
        lake_services(request), symbol=symbol, start=start, end=end, limit=limit
    )
    payload = rates_payload_from(
        result, generated_at=datetime.now(timezone.utc), bounded=limit is not None
    )
    validate_payload(payload, "rates_series_payload")
    return payload


@router.get("/v1/{asset_class}/{symbol}/indicators")
async def get_indicators_v1(
    asset_class: str,
    symbol: str,
    request: Request,
    indicator: str,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
) -> dict:
    return await _indicators_response(
        request, asset_class, symbol, indicator, timeframe, start, end, limit
    )


@router.get("/v1/equity/{symbol}/confluence")
async def get_confluence_v1(
    symbol: str,
    request: Request,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    return await _confluence_response(request, symbol, timeframe, start, end, limit)


# --- deprecated flat aliases ------------------------------------------------------


@router.get("/bars/{ticker}")
async def get_bars(
    ticker: str,
    request: Request,
    response: Response,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
) -> dict:
    """DEPRECATED alias for /v1/equity/{symbol}/bars."""
    _mark_deprecated(response, f"/v1/equity/{ticker}/bars")
    return await _bars_response(
        request, "equity", ticker, timeframe, start, end, limit, None, "listed"
    )


@router.get("/indicators/{ticker}")
async def get_indicators(
    ticker: str,
    request: Request,
    response: Response,
    indicator: str,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
) -> dict:
    """DEPRECATED alias for /v1/equity/{symbol}/indicators."""
    _mark_deprecated(response, f"/v1/equity/{ticker}/indicators")
    return await _indicators_response(
        request, "equity", ticker, indicator, timeframe, start, end, limit
    )


@router.get("/confluence/{ticker}")
async def get_confluence(
    ticker: str,
    request: Request,
    response: Response,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    """DEPRECATED alias for /v1/equity/{symbol}/confluence."""
    _mark_deprecated(response, f"/v1/equity/{ticker}/confluence")
    return await _confluence_response(request, ticker, timeframe, start, end, limit)
