"""Health check endpoint."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health(request: Request) -> dict:
    """Return service health + PG connection status."""
    watcher = getattr(request.app.state, "revision_watcher", None)
    provider = getattr(request.app.state, "ohlc_provider", None)
    # Two small DuckDB reads on the lake volume: off the event loop, under LakeDb's
    # deadline, so a slow mount cannot stall every other request.
    recency = await asyncio.to_thread(provider.fetch_recency) if provider is not None else None
    return {
        "status": "ok",
        "version": request.app.version,
        "uptime": round(time.time() - _start_time, 1),
        "service": "apex-signal-server",
        "pg_connected": getattr(request.app.state, "pg_connected", False),
        "livewire": {
            "configured": provider is not None,
            "configured_price_mode": getattr(request.app.state, "livewire_price_mode", "raw"),
            "effective_price_mode": provider.price_mode if provider is not None else None,
            # From the artifacts, not livewire's 11:00 UTC coverage snapshot -- that
            # under-reports by design and would show a lag that does not exist.
            "recency": recency,
        },
        "silver_revision": watcher.health() if watcher is not None else {"enabled": False},
    }
