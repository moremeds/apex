"""Health check endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health(request: Request) -> dict:
    """Return service health + PG connection status."""
    return {
        "status": "ok",
        "version": request.app.version,
        "uptime": round(time.time() - _start_time, 1),
        "service": "apex-signal-server",
        "pg_connected": getattr(request.app.state, "pg_connected", False),
    }
