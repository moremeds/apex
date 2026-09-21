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
    delisted_bronze_path,
    parquet_path,
)

_DEFAULT_BARS = 2000
_LOOKBACK_FUDGE = 10


def _require_aware(name: str, value: Optional[datetime]) -> None:
    """A naive timestamp is a 400, not a 500.

    FastAPI parses both ``2024-01-02`` and ``2024-01-02T00:00:00`` into a *naive*
    datetime without complaint, because ``date-time`` coercion does not require an
    offset. The first tz-aware comparison downstream then raises TypeError, so a
    malformed query surfaces as an opaque ``internal_error`` and sends the caller to
    read server logs for a mistake in their own URL.

    Defaulting the offset to UTC instead would be worse than rejecting it: a caller
    who meant an America/New_York session boundary would silently receive a window
    shifted by four or five hours, and no error anywhere would say so.
    """
    if value is not None and value.tzinfo is None:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"{name} must carry a UTC offset (got {value.isoformat()!r}); "
            f"use e.g. {value.date().isoformat()}T00:00:00Z",
        )


def _resolve_window(
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
    bars: int = _DEFAULT_BARS,
    from_epoch: bool = False,
) -> Tuple[datetime, datetime, Optional[int]]:
    """Return (start, end, tail_limit). When start is omitted, fetch a generous
    lookback and tail-slice to `bars`; an explicit start is honoured as-is.

    ``from_epoch`` drops the now-anchored lookback and reads the whole history before
    tail-slicing: a delisted name's last bar can be years old, so a window measured
    back from today would answer an empty series for it."""
    _require_aware("start", start)
    _require_aware("end", end)
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
        if from_epoch:
            return datetime(1970, 1, 1, tzinfo=timezone.utc), end, bars
        delta = TF_DELTAS.get(timeframe, DEFAULT_TF_DELTA)
        start = end - delta * bars * _LOOKBACK_FUDGE
        return start, end, bars
    return start, end, None


def _silver_revision(provider: Any) -> Optional[int]:
    """The snapshot used by this read, independent of subscription reseed progress."""
    snapshot = provider.snapshot
    return snapshot.revision if snapshot is not None else None


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
    provider: Any,
    symbol: str,
    timeframe: str,
    spec: AssetClassSpec,
    price_mode: str,
    listing: str = "listed",
) -> bool:
    """Does an artifact exist for this read, in whichever tree the read would use?

    Adjusted daily is served from Silver and Silver can outlive its Bronze source, so
    probing Bronze alone would answer a real Silver-only symbol with 404 whenever the
    requested window happened to be empty. A read that may touch bronze-delisted probes
    that tree too, or a delisted-only ticker with an empty window would 404 although
    its artifact is right there.
    """
    if listing != "listed" and _delisted_artifact_exists(provider, symbol, timeframe, spec.name):
        return True
    if listing == "delisted":
        return False
    if (
        price_mode == "adjusted"
        and timeframe == "1d"
        and spec.supports_adjusted
        and provider.silver_root is not None
        and provider.silver_artifact_path(symbol, "daily", verify=False) is not None
    ):
        return True
    return parquet_path(provider.bronze_root, symbol, timeframe, spec.name).exists()


def _delisted_artifact_exists(provider: Any, symbol: str, timeframe: str, asset_class: str) -> bool:
    """Is there an archived artifact for this (symbol, timeframe, class)?

    Probes bronze-delisted/ directly rather than asking the provider: a provider hook
    here would have no production implementation, so every test would exercise the hook
    and nothing would exercise the path construction that actually runs.
    """
    delisted_root = getattr(provider, "delisted_root", None)
    if delisted_root is None:
        return False
    # Pass the class through: bronze-delisted is overwhelmingly equity but not only
    # equity (asset_class=fx holds USDEUR), and defaulting would silently answer the
    # question for the wrong partition.
    return delisted_bronze_path(delisted_root, symbol, timeframe, asset_class).exists()


def _is_dual_resident(provider: Any, symbol: str, asset_class: str, timeframe: str = "1d") -> bool:
    """True when ``symbol`` has an artifact in BOTH the live and delisted bronze trees.

    Timeframe-aware: the archived tree carries 1d/1h/5m/1m and residency is per-file,
    so a ticker dual-resident at 1d need not be at 1m.

    With no delisted root configured, nothing is dual-resident.
    """
    if not _delisted_artifact_exists(provider, symbol, timeframe, asset_class):
        return False
    return parquet_path(provider.bronze_root, symbol, timeframe, asset_class).exists()


def _check_listing(
    provider: Any,
    listing: str,
    symbol: str,
    asset_class: str,
    timeframe: str = "1d",
    price_mode: str = "raw",
) -> str:
    """Resolve the ``listing`` filter to a listing_status, or fail with a typed code.

    ``delisted`` reads bronze-delisted/ raw. livewire publishes no Silver over that
    tree, so ``price_mode=adjusted`` against it is rejected rather than quietly served
    on the raw basis -- the same rule the live path follows for a missing Silver
    artifact.

    ``any`` resolves to the tree that actually holds the symbol: ``listed`` when the
    live tree has it and the archive does not, ``delisted`` when only the archive does,
    and ``dual`` when both do (2,345 tickers on 2026-08-23). A ``dual`` read returns the
    union with the live tree winning every shared America/New_York trading date. Being
    resident in both trees does NOT by itself mean two issuers -- on a genuinely
    dual-resident name the archived rows can be a duplicate copy of the live company's
    own history, while for a reused ticker they belong to a different company.
    ``listing_status: "dual"`` only says the series was merged; whether one or two
    issuers are behind it is answered by ``/v1/equity/{symbol}/delisting`` (the
    security-master intervals), not by this label. Check those intervals before
    computing a return across the seam.
    """
    if listing not in ("listed", "delisted", "any"):
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"unknown listing filter {listing!r} (have listed, delisted, any)",
            symbol=symbol,
            asset_class=asset_class,
        )
    if listing == "listed":
        return "listed"

    if listing == "delisted":
        status = "delisted"
    elif _is_dual_resident(provider, symbol, asset_class, timeframe):
        status = "dual"
    elif _delisted_artifact_exists(provider, symbol, timeframe, asset_class):
        status = "delisted"
    else:
        return "listed"

    if price_mode == "adjusted":
        raise ApiError(
            ApiErrorCode.ADJUSTED_NOT_SUPPORTED,
            "no Silver for delisted names; use price_mode=raw",
            symbol=symbol,
            asset_class=asset_class,
        )
    return status
