"""Health check endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter

router = APIRouter()

_start_time = time.time()


@router.get("/health")
async def health() -> dict:
    """Return service health status."""
    return {
        "status": "ok",
        "uptime": round(time.time() - _start_time, 1),
        "service": "apex-signal-server",
    }
