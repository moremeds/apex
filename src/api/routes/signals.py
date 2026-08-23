"""REST pull endpoint for TA signals (argon backfill on load/reconnect/?asof)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request, Response

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.builder import build_payload
from src.api.payload.validate import validate_payload

router = APIRouter(tags=["signals"])

_SUNSET = "Wed, 31 Dec 2026 23:59:59 GMT"


async def _signals_payload(ticker: str, request: Request, since: Optional[datetime]) -> dict:
    repo = getattr(request.app.state, "signal_repo", None)
    if repo is None:
        # No Postgres configured -> backfill unavailable (the live WS push still
        # works). Be explicit rather than 500 on a None repo.
        raise ApiError(
            ApiErrorCode.PROVIDER_NOT_CONFIGURED,
            "signal persistence not configured",
            symbol=ticker,
        )
    rows = await repo.fetch_signals(ticker, since=since)
    payload = build_payload(rows, generated_at=datetime.now(timezone.utc))
    validate_payload(payload)  # contract guarantee on every REST response
    return payload


@router.get("/v1/equity/{symbol}/signals")
async def get_signals_v1(symbol: str, request: Request, since: Optional[datetime] = None) -> dict:
    return await _signals_payload(symbol, request, since)


@router.get("/signals/{ticker}")
async def get_signals(
    ticker: str, request: Request, response: Response, since: Optional[datetime] = None
) -> dict:
    """DEPRECATED alias for /v1/equity/{symbol}/signals."""
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = _SUNSET
    response.headers["Link"] = f'</v1/equity/{ticker}/signals>; rel="successor-version"'
    return await _signals_payload(ticker, request, since)
