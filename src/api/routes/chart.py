"""Chart read surface for argon (stateless renderer pulls everything from apex).

- GET /v1/{asset_class}/{symbol}/bars        -> OHLCV candles from livewire
- GET /v1/{asset_class}/{symbol}/indicators  -> per-bar indicator series, compute-on-read
- GET /v1/rates/{symbol}/series              -> yield series (no OHLC)
- GET /v1/equity/{symbol}/confluence         -> multi-timeframe confluence, DB-backed

The flat routes (/bars/{ticker} etc.) are preserved as deprecated aliases so argon and
signal-lab need no change; they carry Deprecation/Sunset/Link headers.

Mirrors the signal contract: REST backfill + validate-on-egress on every response.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, Request, Response

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.chart import (
    build_bars_payload,
    build_confluence_payload,
    build_indicator_payload,
    build_rates_series_payload,
)
from src.api.payload.validate import validate_payload
from src.api.routes._chart_guards import (
    _DEFAULT_BARS,
    _artifact_exists,
    _check_listing,
    _check_timeframe,
    _contract_identity,
    _provider_or_raise,
    _require_bars_payload,
    _resolve_window,
    _silver_revision,
    _spec_or_raise,
)
from src.application.chart.indicator_compute import (
    UnknownIndicatorError,
    compute_indicator_series,
)
from src.domain.signals.indicators.registry import get_indicator_registry
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable
from src.infrastructure.adapters.livewire.paths import parquet_path

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
) -> dict:
    # Request validation first: a malformed request is malformed whether or not the
    # provider happens to be up, and answering it with 503 tells the caller to retry
    # something that can never succeed.
    spec = _spec_or_raise(asset_class)
    _require_bars_payload(spec, symbol)
    _check_timeframe(spec, timeframe)
    provider = _provider_or_raise(request)
    listing_status = _check_listing(provider, listing, symbol, spec.name)
    if price_mode is not None and price_mode not in ("raw", "adjusted"):
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"unknown price_mode {price_mode!r} (have raw, adjusted)",
            symbol=symbol,
            asset_class=spec.name,
        )
    if price_mode == "adjusted" and not spec.supports_adjusted:
        raise ApiError(
            ApiErrorCode.ADJUSTED_NOT_SUPPORTED,
            f"Silver exists only for equity; {spec.name} is served raw",
            symbol=symbol,
            asset_class=spec.name,
        )
    # An explicit price_mode is a REQUEST, not a filter. `raw` is always satisfiable
    # (bronze is the substrate under both modes) so it may override an adjusted-configured
    # provider; `adjusted` is satisfiable only where Silver exists, already checked above.
    # Accepting the parameter and then ignoring it would be worse than not offering it.
    effective = price_mode or provider.effective_price_mode(spec.name)
    start, end, tail = _resolve_window(timeframe, start, end, limit)
    try:
        bars = await provider.fetch_bars(
            symbol, timeframe, start, end, asset_class=spec.name, price_mode=effective
        )
    except AdjustedDataUnavailable as exc:
        raise ApiError(
            ApiErrorCode.ADJUSTED_UNAVAILABLE, str(exc), symbol=symbol, asset_class=spec.name
        ) from exc
    if not bars and not _artifact_exists(provider, symbol, timeframe, spec, effective):
        # Distinguish "no artifact" from "artifact exists, window is empty". The second is a
        # legitimate 200 with zero bars (a quiet window); the first is a 404.
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    if tail is not None:
        bars = bars[-tail:]
    payload = build_bars_payload(
        symbol,
        timeframe,
        bars,
        generated_at=datetime.now(timezone.utc),
        asset_class=spec.name,
        contract=_contract_identity(spec, bars),
        price_mode=effective,
        listing_status=listing_status,
        adjustment_revision=_silver_revision(request) if effective == "adjusted" else None,
    )
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
    spec = _spec_or_raise(asset_class)
    # A yield has no OHLC. Without this the rates parquet is read into BarData with
    # null prices and the indicator computes over them, returning 200 and null
    # bar_close -- a number-shaped answer to a question that has no answer.
    _require_bars_payload(spec, symbol)
    _check_timeframe(spec, timeframe)
    provider = _provider_or_raise(request)
    registry = getattr(request.app.state, "indicator_registry", None) or get_indicator_registry()
    start, end, tail = _resolve_window(timeframe, start, end, limit)
    try:
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
    if not points and not _artifact_exists(
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
    start, end, _ = _resolve_window(timeframe, start, end)
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
    limit: int = Query(default=_DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
    price_mode: Optional[str] = Query(default=None, description="raw | adjusted"),
    listing: str = Query(default="listed", description="listed | delisted | any"),
) -> dict:
    return await _bars_response(
        request, asset_class, symbol, timeframe, start, end, limit, price_mode, listing
    )


@router.get("/v1/rates/{symbol}/series")
async def get_rates_series_v1(
    symbol: str,
    request: Request,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> dict:
    provider = _provider_or_raise(request)
    spec = _spec_or_raise("rates")
    start, end, _ = _resolve_window("1d", start, end, 0)
    points = await provider.fetch_rate_series(symbol, start, end)
    if not points and not parquet_path(provider.bronze_root, symbol, "1d", spec.name).exists():
        # Same rule as /bars: no artifact is a 404, an empty window over a real
        # artifact is a legitimate 200. Without this an unknown series answers 200
        # with zero points, which reads as "this yield had no observations".
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    payload = build_rates_series_payload(symbol, points, generated_at=datetime.now(timezone.utc))
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
    limit: int = Query(default=_DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
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
    limit: int = Query(default=_DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
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
    limit: int = Query(default=_DEFAULT_BARS, description="tail-slice to N bars; <=0 for all"),
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
