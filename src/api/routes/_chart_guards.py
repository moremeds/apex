"""Validate and resolve a chart read request before anything touches the lake.

Split out of ``chart.py`` when that file crossed the repo's 500-line budget. The seam
is a responsibility, not a layer: everything here answers "is this request coherent,
and which artifact would it read?" -- the questions that must be settled before a read,
and that every chart route asks in the same order. ``chart.py`` keeps the routes and
the response assembly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from fastapi import Request

from src.api.errors import ApiError, ApiErrorCode
from src.application.chart.indicator_compute import DEFAULT_TF_DELTA, TF_DELTAS
from src.infrastructure.adapters.livewire.asset_classes import (
    AssetClassSpec,
    UnknownAssetClass,
    get_asset_class,
)
from src.infrastructure.adapters.livewire.paths import (
    daily_silver_path,
    delisted_bronze_path,
    parquet_path,
)

_DEFAULT_BARS = 2000
_LOOKBACK_FUDGE = 10


def _resolve_window(
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
    bars: int = _DEFAULT_BARS,
) -> Tuple[datetime, datetime, Optional[int]]:
    """Return (start, end, tail_limit). When start is omitted, fetch a generous
    lookback and tail-slice to `bars`; an explicit start is honoured as-is."""
    end = end or datetime.now(timezone.utc)
    if start is not None and start > end:
        # Otherwise this reads a real artifact, matches nothing, and answers 200 with
        # zero rows -- reporting an impossible request as a quiet market.
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"start {start.isoformat()} is after end {end.isoformat()}",
        )
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


def _artifact_exists(
    provider: Any, symbol: str, timeframe: str, spec: AssetClassSpec, price_mode: str
) -> bool:
    """Does an artifact exist for this read, in whichever tree the read would use?

    Adjusted daily is served from Silver and Silver can outlive its Bronze source, so
    probing Bronze alone would answer a real Silver-only symbol with 404 whenever the
    requested window happened to be empty.
    """
    if (
        price_mode == "adjusted"
        and timeframe == "1d"
        and spec.supports_adjusted
        and provider.silver_root is not None
        and daily_silver_path(provider.silver_root, symbol).exists()
    ):
        return True
    return parquet_path(provider.bronze_root, symbol, timeframe, spec.name).exists()


def _is_dual_resident(provider: Any, symbol: str, asset_class: str) -> bool:
    """True when ``symbol`` exists in BOTH the live and delisted bronze trees.

    Probes bronze-delisted/ directly rather than asking the provider: a provider hook
    here would have no production implementation, so every test would exercise the hook
    and nothing would exercise the path construction that actually runs.

    With no delisted root configured, nothing is dual-resident.
    """
    delisted_root = getattr(provider, "delisted_root", None)
    if delisted_root is None:
        return False
    # Pass the class through: bronze-delisted is overwhelmingly equity but not only
    # equity (asset_class=fx holds USDEUR), and defaulting would silently answer the
    # question for the wrong partition.
    return delisted_bronze_path(delisted_root, symbol, asset_class=asset_class).exists()


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
        if _is_dual_resident(provider, symbol, asset_class):
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
            ApiErrorCode.INVALID_PARAMETER,
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
