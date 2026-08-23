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
from typing import Any, Optional, Tuple

from fastapi import APIRouter, Query, Request, Response

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.chart import (
    build_bars_payload,
    build_confluence_payload,
    build_indicator_payload,
    build_rates_series_payload,
)
from src.api.payload.validate import validate_payload
from src.application.chart.indicator_compute import (
    DEFAULT_TF_DELTA,
    TF_DELTAS,
    UnknownIndicatorError,
    compute_indicator_series,
)
from src.domain.signals.indicators.registry import get_indicator_registry
from src.infrastructure.adapters.livewire.asset_classes import (
    AssetClassSpec,
    UnknownAssetClass,
    get_asset_class,
)
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable
from src.infrastructure.adapters.livewire.paths import delisted_bronze_path, parquet_path

router = APIRouter(tags=["chart"])

# Default no-arg window: the most recent N bars. We over-fetch in calendar time
# (markets aren't 24/7, so N*delta would under-cover across closures) then tail-slice
# to exactly N. Callers wanting an exact range pass start/end.
_DEFAULT_BARS = 2000
_LOOKBACK_FUDGE = 10

_SUNSET = "Wed, 31 Dec 2026 23:59:59 GMT"


def _resolve_window(
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
    bars: int = _DEFAULT_BARS,
) -> Tuple[datetime, datetime, Optional[int]]:
    """Return (start, end, tail_limit). When start is omitted, fetch a generous
    lookback and tail-slice to `bars`; an explicit start is honoured as-is."""
    end = end or datetime.now(timezone.utc)
    if start is None:
        if bars <= 0:  # full history: no tail-slice, fetch from the epoch
            return datetime(1970, 1, 1, tzinfo=timezone.utc), end, None
        delta = TF_DELTAS.get(timeframe, DEFAULT_TF_DELTA)
        start = end - delta * bars * _LOOKBACK_FUDGE
        return start, end, bars
    return start, end, None


def _silver_revision(request: Request) -> Optional[int]:
    """The revision the payload's adjusted prices were built from.

    Sourced from the running watcher's ``last_fully_applied_revision`` -- NOT
    ``observed_revision``, which may be a revision apex has seen but not finished
    applying. There is no ``app.state.silver_revision``; the watcher owns this.
    """
    watcher = getattr(request.app.state, "revision_watcher", None)
    return getattr(watcher, "last_fully_applied_revision", None) if watcher else None


def _contract_identity(spec: AssetClassSpec, bars: list) -> Optional[dict]:
    """Futures instrument identity, lifted off the first bar.

    livewire stores contract_id / root_symbol / expiry_date as per-row columns on
    asset_class=futures and they are constant across a contract's rows. Every other
    class returns None.
    """
    if spec.name != "futures" or not bars:
        return None
    first = bars[0]
    root = getattr(first, "root_symbol", None)
    if root is None:
        return None
    return {
        "contract_id": getattr(first, "contract_id", None),
        "root_symbol": root,
        "expiry_date": getattr(first, "expiry_date", None),  # already an ISO string
    }


def _provider_or_raise(request: Request) -> Any:
    provider = getattr(request.app.state, "ohlc_provider", None)
    if provider is None:
        raise ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, "bar provider not configured")
    return provider


def _spec_or_raise(asset_class: str) -> AssetClassSpec:
    try:
        return get_asset_class(asset_class)
    except UnknownAssetClass as exc:
        raise ApiError(
            ApiErrorCode.UNSUPPORTED_ASSET_CLASS, str(exc), asset_class=asset_class
        ) from exc


def _require_bars_payload(spec: AssetClassSpec, symbol: str) -> None:
    """Reject a class whose payload is not bars.

    `rates` is registered in the asset-class registry (so paths and discovery work) but a
    yield has no OHLC and cannot satisfy bars_payload.schema.json. Without this the generic
    route would read the parquet, build a payload of null prices, and fail egress validation
    as a 500 instead of telling the caller which route to use.
    """
    if spec.payload != "bars":
        raise ApiError(
            ApiErrorCode.UNSUPPORTED_ASSET_CLASS,
            f"{spec.name} is not an OHLCV class; use /v1/{spec.name}/{{symbol}}/series",
            symbol=symbol,
            asset_class=spec.name,
        )


def _check_timeframe(spec: AssetClassSpec, timeframe: str) -> None:
    if timeframe not in spec.timeframes:
        raise ApiError(
            ApiErrorCode.UNSUPPORTED_TIMEFRAME,
            f"unsupported timeframe {timeframe!r} for {spec.name} "
            f"(have {list(spec.timeframes)})",
            asset_class=spec.name,
        )


def _is_dual_resident(provider: Any, symbol: str) -> bool:
    """True when ``symbol`` exists in BOTH the live and delisted bronze trees.

    Prefers a provider-supplied answer so test doubles need no filesystem; falls back to
    probing bronze-delisted/. With no delisted root configured, nothing is dual-resident.
    """
    probe = getattr(provider, "is_dual_resident", None)
    if probe is not None:
        return bool(probe(symbol))
    delisted_root = getattr(provider, "delisted_root", None)
    return delisted_root is not None and delisted_bronze_path(delisted_root, symbol).exists()


def _check_listing(provider: Any, listing: str, symbol: str, asset_class: str) -> str:
    """Resolve the ``listing`` filter to a listing_status, or fail with a typed code.

    ``delisted`` is specified but blocked upstream: livewire has no Silver tree for
    bronze-delisted/, and no delisted symbol has correct corporate-action data
    (measured 2026-08-23). Serving raw delisted bars would trade survivorship bias
    for silent mis-adjustment, so we fail loudly instead.

    ``any`` resolves to ``listed`` unless the ticker is genuinely dual-resident, in which
    case it is a 409: 2,345 tickers exist in BOTH the live and delisted trees (ticker
    reuse), so for those "either" has no single correct answer.
    """
    if listing == "listed":
        return "listed"
    if listing == "any":
        # Only ambiguous when the ticker really is dual-resident. 2,345 of 8,620 delisted
        # symbols are also live; for the other ~12,400 live symbols "any" has exactly one
        # answer, so 409-ing all of them would be noise.
        if _is_dual_resident(provider, symbol):
            raise ApiError(
                ApiErrorCode.AMBIGUOUS_SYMBOL,
                f"{symbol} resolves to both a live and a delisted entity; "
                "request listing=listed or listing=delisted explicitly",
                symbol=symbol,
                asset_class=asset_class,
            )
        return "listed"
    if listing != "delisted":
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"unknown listing filter {listing!r} (have listed, delisted, any)",
            symbol=symbol,
            asset_class=asset_class,
        )
    raise ApiError(
        ApiErrorCode.NOT_YET_AVAILABLE,
        "delisted coverage requires upstream livewire work "
        "(instrument identity, corporate-action backfill, Silver over bronze-delisted)",
        symbol=symbol,
        asset_class=asset_class,
    )


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
    provider = _provider_or_raise(request)
    spec = _spec_or_raise(asset_class)
    _require_bars_payload(spec, symbol)
    _check_timeframe(spec, timeframe)
    listing_status = _check_listing(provider, listing, symbol, spec.name)
    if price_mode is not None and price_mode not in ("raw", "adjusted"):
        raise ApiError(
            ApiErrorCode.UNSUPPORTED_ASSET_CLASS,
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
    if not bars and not parquet_path(provider.bronze_root, symbol, timeframe, spec.name).exists():
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
    provider = _provider_or_raise(request)
    spec = _spec_or_raise(asset_class)
    _check_timeframe(spec, timeframe)
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
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"unknown indicator: {indicator}",
            symbol=symbol,
            asset_class=spec.name,
        ) from exc
    except AdjustedDataUnavailable as exc:
        raise ApiError(
            ApiErrorCode.ADJUSTED_UNAVAILABLE, str(exc), symbol=symbol, asset_class=spec.name
        ) from exc
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
    start, end, _ = _resolve_window("1d", start, end, 0)
    points = await provider.fetch_rate_series(symbol, start, end)
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
