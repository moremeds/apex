"""Health check endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()

_start_time = time.time()

# The livewire lake lives on a slow external disk that livewire itself rewrites
# every afternoon. Probing it inline blocked the event loop past the 5s healthcheck
# timeout and 500'd on files mid-replace, so /health only ever reads this cache and
# refreshes it in the background.
_RECENCY_TTL_SEC = 60.0
_recency: dict[str, Any] = {"value": None, "as_of": None}
_recency_task: asyncio.Task | None = None


async def _refresh_recency(provider: Any) -> None:
    try:
        value = await asyncio.to_thread(provider.fetch_recency)
    except Exception as exc:  # lake unavailable/mid-rewrite: keep the last known value
        logger.warning("recency refresh failed, serving stale value: %s", exc)
        return
    _recency["value"] = value
    _recency["as_of"] = datetime.now(timezone.utc)


def _maybe_refresh_recency(provider: Any) -> None:
    """Kick off at most one background lake probe; never block the caller."""
    global _recency_task
    if _recency_task is not None and not _recency_task.done():
        return
    as_of = _recency["as_of"]
    if (
        as_of is not None
        and (datetime.now(timezone.utc) - as_of).total_seconds() < _RECENCY_TTL_SEC
    ):
        return
    _recency_task = asyncio.create_task(_refresh_recency(provider))


@router.get("/health")
async def health(request: Request) -> dict:
    """Return service health + PG connection status."""
    watcher = getattr(request.app.state, "revision_watcher", None)
    provider = getattr(request.app.state, "ohlc_provider", None)
    if provider is not None:
        _maybe_refresh_recency(provider)
    as_of = _recency["as_of"]
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
            # Refreshed off-thread; `recency_as_of` is how stale this value is.
            "recency": _recency["value"] if provider is not None else None,
            "recency_as_of": (
                as_of.isoformat() if as_of is not None and provider is not None else None
            ),
        },
        "silver_revision": watcher.health() if watcher is not None else {"enabled": False},
    }
