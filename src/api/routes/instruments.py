"""Discovery: what does apex hold, and which symbols would fail in adjusted mode."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.validate import validate_payload
from src.infrastructure.adapters.livewire.asset_classes import (
    UnknownAssetClass,
    get_asset_class,
)
from src.infrastructure.adapters.livewire.coverage import CoverageUnavailable
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable
from src.infrastructure.adapters.livewire.paths import parquet_path
from src.infrastructure.adapters.livewire.reference import (
    ACTION_TYPES,
    LivewireReferenceReader,
    ReferenceDataError,
)

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
    if asset_class is not None:
        try:
            get_asset_class(asset_class)
        except UnknownAssetClass as exc:
            raise ApiError(
                ApiErrorCode.UNSUPPORTED_ASSET_CLASS, str(exc), asset_class=asset_class
            ) from exc
    try:
        # DuckDB is synchronous and the catalog lives on the same external volume
        # as the lake; running it inline stalls every other request on this worker
        # for the duration of the read. Same treatment the bars path already gets.
        rows = await asyncio.to_thread(
            catalog.list_instruments,  # type: ignore[attr-defined]
            asset_class=asset_class,
            query=q,
            limit=limit,
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


# The ticker-reuse caveat, carried verbatim from the era when these endpoints were 501:
# both artifacts are keyed by TICKER, not by a permanent security id. Measured
# 2026-08-23 -- 6,275 delisted symbols have no corporate-action data at all, and the
# 2,345 that appear to are ticker reuses whose actions belong to a different, living
# company. Every response below says ``identity: "ticker"`` so a consumer cannot
# mistake it for security-level truth.
_TICKER_IDENTITY = "ticker"


def _reference_or_raise(attribute: str, message: str) -> LivewireReferenceReader:
    reader = LivewireReferenceReader.from_env()
    if getattr(reader, attribute) is None:
        raise ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, message)
    return reader


@router.get("/v1/equity/{symbol}/actions")
async def get_corporate_actions(
    symbol: str,
    type: Optional[str] = Query(default=None, description="split | cash_dividend"),
    start: Optional[str] = Query(default=None, description="earliest ex_date, YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="latest ex_date, YYYY-MM-DD"),
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
    if type is not None and type not in ACTION_TYPES:
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"unknown type {type!r} (have {list(ACTION_TYPES)})",
            symbol=symbol,
            asset_class="equity",
        )
    start_date = _parse_day(start, "start")
    end_date = _parse_day(end, "end")
    if start_date is not None and end_date is not None and start_date > end_date:
        # Otherwise this reads a real artifact, matches nothing, and answers 200 with
        # zero actions -- reporting an impossible request as a quiet history.
        raise ApiError(
            ApiErrorCode.INVALID_PARAMETER,
            f"start {start_date.isoformat()} is after end {end_date.isoformat()}",
            symbol=symbol,
            asset_class="equity",
        )
    reader = _reference_or_raise(
        "bronze_root", "corporate actions are not configured; set APEX_LIVEWIRE_ROOT"
    )
    ticker = symbol.upper()
    try:
        actions = await asyncio.to_thread(
            reader.fetch_actions,
            ticker,
            action_type=type,
            start=start_date,
            end=end_date,
        )
        provider = await asyncio.to_thread(reader.fetch_provider, ticker)
    except ReferenceDataError as exc:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            str(exc),
            symbol=symbol,
            asset_class="equity",
        ) from exc
    if actions is None:
        # No log file at all is an unknown ticker (404); a log with nothing matching the
        # filter is a legitimate 200 with zero actions.
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"no corporate-action log for {ticker}",
            symbol=symbol,
            asset_class="equity",
        )
    payload = {
        "symbol": ticker,
        "identity": _TICKER_IDENTITY,
        "source": "livewire_bronze_corporate_action",
        "provider": provider,
        "actions": [action.as_dict() for action in actions],
        "count": len(actions),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_payload(payload, "actions_payload")
    return payload


@router.get("/v1/equity/{symbol}/delisting")
async def get_delisting(symbol: str) -> dict:
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
    reader = _reference_or_raise(
        "lake_root",
        "the security master is not configured; set APEX_LIVEWIRE_LAKE_ROOT",
    )
    ticker = symbol.upper()
    try:
        intervals = await asyncio.to_thread(reader.fetch_identity, ticker)
    except ReferenceDataError as exc:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            str(exc),
            symbol=symbol,
            asset_class="equity",
        ) from exc
    if intervals is None:
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            "security master artifact is missing under APEX_LIVEWIRE_LAKE_ROOT",
            symbol=symbol,
            asset_class="equity",
        )
    if not intervals:
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"no verified security-master record for {ticker}",
            symbol=symbol,
            asset_class="equity",
        )
    payload = {
        "symbol": ticker,
        "identity": _TICKER_IDENTITY,
        "source": "livewire_security_master",
        # Named so nobody reads the absence of a reason as "still listed".
        "delisting_reason_available": False,
        "intervals": [interval.as_dict() for interval in intervals],
        "count": len(intervals),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_payload(payload, "delisting_payload")
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
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            "bar provider not configured",
            symbol=symbol,
        )
    # Probe the artifacts: the coverage table measures no equity intraday, so it
    # cannot answer this. Five exists() calls is fine for one symbol; it is exactly
    # why the LIST endpoint does not do it 14,746 times.
    silver_daily = False
    adjustment_revision = None
    if spec.supports_adjusted and provider.silver_root is not None:
        try:
            pinned_provider = await asyncio.to_thread(provider.pin_snapshot)
            silver_daily = (
                await asyncio.to_thread(pinned_provider.silver_artifact_path, symbol, "daily")
                is not None
            )
            if silver_daily and pinned_provider.snapshot is not None:
                adjustment_revision = pinned_provider.snapshot.revision
        except AdjustedDataUnavailable as exc:
            raise ApiError(
                ApiErrorCode.ADJUSTED_UNAVAILABLE,
                str(exc),
                symbol=symbol,
                asset_class=spec.name,
            ) from exc
    timeframes = [
        tf
        for tf in spec.timeframes
        # Silver can outlive its Bronze source, so a Bronze-only probe would omit "1d"
        # from a symbol that /bars will happily serve in adjusted mode.
        if parquet_path(provider.bronze_root, symbol, tf, spec.name).exists()
        or (tf == "1d" and silver_daily)
    ]
    if not timeframes:
        raise ApiError(
            ApiErrorCode.UNKNOWN_SYMBOL,
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    silver_available = silver_daily
    catalog = getattr(request.app.state, "coverage_catalog", None)
    dates = None
    # Names WHY first_date/last_date are null, so an unreadable catalog cannot hide
    # behind a symbol that genuinely has no recorded coverage. Not fatal here:
    # timeframes and silver_available came from disk and are still correct.
    coverage_source = "not_configured" if catalog is None else "livewire_coverage_snapshot"
    if catalog is not None:
        try:
            dates = await asyncio.to_thread(catalog.get_instrument, symbol, spec.name)
        except CoverageUnavailable as exc:
            # Detail degrades to artifact-only facts rather than failing: the
            # timeframes above came from disk and are still correct. Logged, not
            # swallowed -- silently nulling the dates would hide a broken catalog
            # mount behind a 200.
            logger.warning("coverage catalog unreadable, serving %s without dates: %s", symbol, exc)
            dates = None
            coverage_source = "unavailable"
    return {
        "symbol": symbol,
        "asset_class": spec.name,
        "listing_status": "listed",
        "timeframes": timeframes,
        "coverage_source": coverage_source,
        "first_date": dates.first_date if dates else None,
        "last_date": dates.last_date if dates else None,
        "silver_available": silver_available,
        "price_mode": provider.effective_price_mode(spec.name),
        "adjustment_revision": adjustment_revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
