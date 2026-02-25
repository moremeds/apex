"""REST routes — /api/monitor for system health and data quality."""

from __future__ import annotations

import logging
import time
from typing import Any, List

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

_server_start_time = time.monotonic()


def create_monitor_router(
    hub: Any = None,
    pipeline: Any = None,
    quote_adapter: Any = None,
    r2_client: Any = None,
) -> APIRouter:
    """Create router for monitoring endpoints.

    Dependencies can be passed directly (for tests) or resolved from
    request.app.state at request time (for production with lifespan).
    """
    router = APIRouter(prefix="/api/monitor")

    def _get_hub(request: Request):
        return hub or getattr(request.app.state, "hub", None)

    def _get_pipeline(request: Request):
        return pipeline or getattr(request.app.state, "pipeline", None)

    def _get_quote_adapter(request: Request):
        return quote_adapter or getattr(request.app.state, "quote_adapter", None)

    def _get_r2_client(request: Request):
        return r2_client or getattr(request.app.state, "r2_client", None)

    @router.get("")
    async def get_monitor_status(request: Request) -> dict:
        """Get system health: providers, WS connections, uptime."""
        uptime = round(time.monotonic() - _server_start_time, 1)

        providers: List[dict] = []
        qa = _get_quote_adapter(request)
        if qa is not None:
            connected = qa.is_connected()
            subscribed = qa.get_subscribed_symbols()
            providers.append({
                "name": "longbridge",
                "connected": connected,
                "symbols": len(subscribed),
                "subscribed_symbols": subscribed[:20],
            })

        h = _get_hub(request)
        ws_clients = h.client_count if h else 0

        p = _get_pipeline(request)
        pipeline_running = False
        timeframes: List[str] = []
        if p is not None:
            pipeline_running = getattr(p, "_started", False)
            timeframes = getattr(p, "_timeframes", [])

        return {
            "status": "ok",
            "uptime_sec": uptime,
            "providers": providers,
            "ws_clients": ws_clients,
            "pipeline": {
                "running": pipeline_running,
                "timeframes": timeframes,
            },
        }

    @router.get("/data-quality")
    async def get_data_quality(request: Request) -> dict:
        """Get data quality report (proxied from R2)."""
        r2 = _get_r2_client(request)
        if r2 is None:
            raise HTTPException(status_code=503, detail="R2 client not configured")

        try:
            data = r2.get_json("data_quality.json")
        except Exception as e:
            logger.error("Failed to fetch data_quality.json: %s", e)
            raise HTTPException(status_code=502, detail="Failed to fetch data quality")

        if data is None:
            raise HTTPException(status_code=404, detail="Data quality report not found")

        return data if isinstance(data, dict) else {"data": data}

    return router
