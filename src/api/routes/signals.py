"""REST pull endpoint for TA signals (argon backfill on load/reconnect/?asof)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request

from src.api.payload.builder import build_payload
from src.api.payload.validate import validate_payload

router = APIRouter(tags=["signals"])


@router.get("/signals/{ticker}")
async def get_signals(ticker: str, request: Request, since: Optional[datetime] = None) -> dict:
    repo = request.app.state.signal_repo
    rows = await repo.fetch_signals(ticker, since=since)
    payload = build_payload(rows, generated_at=datetime.now(timezone.utc))
    validate_payload(payload)  # contract guarantee on every REST response
    return payload
