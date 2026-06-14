"""Chart read surface for argon (stateless renderer pulls everything from apex).

- GET /bars/{ticker}          -> OHLCV candles from livewire (full history)
- GET /indicators/{ticker}    -> per-bar indicator series, compute-on-read (full depth)
- GET /confluence/{ticker}    -> multi-timeframe confluence, DB-backed (persisted depth)

Mirrors the signal contract: REST backfill + validate-on-egress on every response.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.payload.chart import (
    build_bars_payload,
    build_confluence_payload,
    build_indicator_payload,
)
from src.api.payload.validate import validate_payload
from src.application.chart.indicator_compute import (
    DEFAULT_TF_DELTA,
    TF_DELTAS,
    UnknownIndicatorError,
    compute_indicator_series,
)
from src.domain.signals.indicators.registry import get_indicator_registry
from src.infrastructure.adapters.livewire.paths import SUPPORTED_TIMEFRAMES

router = APIRouter(tags=["chart"])

# Default no-arg window: the most recent N bars. We over-fetch in calendar time
# (markets aren't 24/7, so N*delta would under-cover across closures) then tail-slice
# to exactly N. Callers wanting an exact range pass start/end.
_DEFAULT_BARS = 500
_LOOKBACK_FUDGE = 10


def _require_supported_timeframe(timeframe: str) -> None:
    """Livewire warehouses a narrower set than the schema enum -- reject early (400)
    instead of letting parquet_path raise ValueError -> 500."""
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported timeframe: {timeframe} (have {sorted(SUPPORTED_TIMEFRAMES)})",
        )


def _resolve_window(
    timeframe: str, start: Optional[datetime], end: Optional[datetime]
) -> Tuple[datetime, datetime, Optional[int]]:
    """Return (start, end, tail_limit). When start is omitted, fetch a generous
    lookback and tail-slice to _DEFAULT_BARS; an explicit start is honoured as-is."""
    end = end or datetime.now(timezone.utc)
    if start is None:
        delta = TF_DELTAS.get(timeframe, DEFAULT_TF_DELTA)
        start = end - delta * _DEFAULT_BARS * _LOOKBACK_FUDGE
        return start, end, _DEFAULT_BARS
    return start, end, None


@router.get("/bars/{ticker}")
async def get_bars(
    ticker: str,
    request: Request,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> dict:
    provider = getattr(request.app.state, "ohlc_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="bar provider not configured")
    _require_supported_timeframe(timeframe)
    start, end, limit = _resolve_window(timeframe, start, end)
    bars = await provider.fetch_bars(ticker, timeframe, start, end)
    if limit is not None:
        bars = bars[-limit:]
    payload = build_bars_payload(ticker, timeframe, bars, generated_at=datetime.now(timezone.utc))
    validate_payload(payload, "bars_payload")
    return payload


@router.get("/indicators/{ticker}")
async def get_indicators(
    ticker: str,
    request: Request,
    indicator: str,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> dict:
    provider = getattr(request.app.state, "ohlc_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="bar provider not configured")
    _require_supported_timeframe(timeframe)
    registry = getattr(request.app.state, "indicator_registry", None) or get_indicator_registry()
    start, end, limit = _resolve_window(timeframe, start, end)
    try:
        points = await compute_indicator_series(
            provider, registry, ticker, timeframe, indicator, start, end
        )
    except UnknownIndicatorError:
        raise HTTPException(status_code=404, detail=f"unknown indicator: {indicator}")
    if limit is not None:
        points = points[-limit:]
    payload = build_indicator_payload(
        ticker, timeframe, indicator, points, generated_at=datetime.now(timezone.utc)
    )
    validate_payload(payload, "indicator_series_payload")
    return payload


@router.get("/confluence/{ticker}")
async def get_confluence(
    ticker: str,
    request: Request,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    repo = getattr(request.app.state, "signal_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="signal persistence not configured")
    # Confluence is PG-backed (not livewire), so it accepts any timeframe the data has.
    start, end, _ = _resolve_window(timeframe, start, end)
    rows = await repo.get_confluence_history(ticker, timeframe, start, end, limit)
    payload = build_confluence_payload(
        ticker, timeframe, rows, generated_at=datetime.now(timezone.utc)
    )
    validate_payload(payload, "confluence_payload")
    return payload
