"""Discovery: what does apex hold, and which symbols would fail in adjusted mode."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.validate import validate_payload
from src.infrastructure.adapters.livewire.asset_classes import UnknownAssetClass, get_asset_class
from src.infrastructure.adapters.livewire.coverage import CoverageUnavailable
from src.infrastructure.adapters.livewire.paths import daily_silver_path, parquet_path

router = APIRouter(tags=["instruments"])


def _catalog_or_raise(request: Request) -> object:
    catalog = getattr(request.app.state, "coverage_catalog", None)
    if catalog is None:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            "coverage catalog not configured (set APEX_LIVEWIRE_COVERAGE_DB)",
        )
    return catalog


@router.get("/v1/instruments")
async def list_instruments(
    request: Request,
    asset_class: Optional[str] = None,
    q: Optional[str] = Query(default=None, description="symbol prefix filter"),
    listing: str = Query(default="listed", description="listed | delisted"),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    catalog = _catalog_or_raise(request)
    if listing != "listed":
        # The coverage table measures the live tree only; bronze-delisted/ is not in it.
        raise ApiError(
            ApiErrorCode.NOT_YET_AVAILABLE,
            "delisted discovery requires upstream livewire work "
            "(instrument identity, corporate-action backfill, Silver over bronze-delisted)",
        )
    if asset_class is not None:
        try:
            get_asset_class(asset_class)
        except UnknownAssetClass as exc:
            raise ApiError(
                ApiErrorCode.UNSUPPORTED_ASSET_CLASS, str(exc), asset_class=asset_class
            ) from exc
    try:
        rows = catalog.list_instruments(  # type: ignore[attr-defined]
            asset_class=asset_class, query=q, limit=limit
        )
    except CoverageUnavailable as exc:
        # An unreadable catalog is NOT an empty universe. Reporting zero instruments
        # would let a broken deployment masquerade as a correct one.
        raise ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, str(exc)) from exc
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


# Route ordering is NOT a constraint here: Starlette matches on the whole path pattern,
# so /v1/{asset_class}/{symbol} (two segments) can never shadow /v1/instruments (one)
# nor /v1/equity/{symbol}/bars (three). Verified empirically in both registration orders.
@router.get("/v1/{asset_class}/{symbol}")
async def get_instrument(asset_class: str, symbol: str, request: Request) -> dict:
    """One instrument's detail, including the timeframes that actually exist on disk."""
    try:
        spec = get_asset_class(asset_class)
    except UnknownAssetClass as exc:
        raise ApiError(
            ApiErrorCode.UNSUPPORTED_ASSET_CLASS, str(exc), asset_class=asset_class
        ) from exc
    provider = getattr(request.app.state, "ohlc_provider", None)
    if provider is None:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED, "bar provider not configured", symbol=symbol
        )
    # Probe the artifacts: the coverage table measures no equity intraday, so it
    # cannot answer this. Five exists() calls is fine for one symbol; it is exactly
    # why the LIST endpoint does not do it 14,746 times.
    timeframes = [
        tf
        for tf in spec.timeframes
        if parquet_path(provider.bronze_root, symbol, tf, spec.name).exists()
    ]
    if not timeframes:
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    silver_available = False
    if spec.supports_adjusted and provider.silver_root is not None:
        silver_available = daily_silver_path(provider.silver_root, symbol).exists()
    catalog = getattr(request.app.state, "coverage_catalog", None)
    dates = None
    if catalog is not None:
        try:
            dates = catalog.get_instrument(symbol, spec.name)
        except CoverageUnavailable:
            # Detail degrades to artifact-only facts rather than failing: the
            # timeframes above came from disk and are still correct.
            dates = None
    return {
        "symbol": symbol,
        "asset_class": spec.name,
        "listing_status": "listed",
        "timeframes": timeframes,
        "first_date": dates.first_date if dates else None,
        "last_date": dates.last_date if dates else None,
        "silver_available": silver_available,
        "price_mode": provider.effective_price_mode(spec.name),
        "adjustment_revision": (
            getattr(
                getattr(request.app.state, "revision_watcher", None),
                "last_fully_applied_revision",
                None,
            )
            if silver_available
            else None
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
