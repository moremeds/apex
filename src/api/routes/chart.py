"""Chart read surface for argon (stateless renderer pulls everything from apex).

- GET /bars/{ticker}          -> OHLCV candles from livewire (full history)
- GET /indicators/{ticker}    -> per-bar indicator series, compute-on-read (full depth)
- GET /confluence/{ticker}    -> multi-timeframe confluence, DB-backed (persisted depth)

Mirrors the signal contract: REST backfill + validate-on-egress on every response.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException, Request

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

router = APIRouter(tags=["chart"])

# Default window when the caller omits start/end: the most recent ~500 bars.
_DEFAULT_BARS = 500


def _resolve_window(
    timeframe: str, start: Optional[datetime], end: Optional[datetime]
) -> Tuple[datetime, datetime]:
    end = end or datetime.now(timezone.utc)
    if start is None:
        start = end - TF_DELTAS.get(timeframe, DEFAULT_TF_DELTA) * _DEFAULT_BARS
    return start, end


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
    start, end = _resolve_window(timeframe, start, end)
    bars = await provider.fetch_bars(ticker, timeframe, start, end)
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
    registry = getattr(request.app.state, "indicator_registry", None) or get_indicator_registry()
    start, end = _resolve_window(timeframe, start, end)
    try:
        points = await compute_indicator_series(
            provider, registry, ticker, timeframe, indicator, start, end
        )
    except UnknownIndicatorError:
        raise HTTPException(status_code=404, detail=f"unknown indicator: {indicator}")
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
) -> dict:
    repo = getattr(request.app.state, "signal_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="signal persistence not configured")
    rows = await repo.get_confluence_history(ticker, timeframe, start, end)
    payload = build_confluence_payload(
        ticker, timeframe, rows, generated_at=datetime.now(timezone.utc)
    )
    validate_payload(payload, "confluence_payload")
    return payload
