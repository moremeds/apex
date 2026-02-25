"""REST routes — /api/monitor for system health and data quality."""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

_server_start_time = time.monotonic()


def create_monitor_router(
    hub: Any = None,
    pipeline: Any = None,
    quote_adapter: Any = None,
    r2_client: Any = None,
) -> APIRouter:
    """Create router for monitoring endpoints.

    Args:
        hub: WebSocketHub for connection stats.
        pipeline: ServerPipeline for pipeline stats.
        quote_adapter: QuoteProvider for provider status.
        r2_client: R2Client for data quality proxy.
    """
    router = APIRouter(prefix="/api/monitor")

    @router.get("")
    async def get_monitor_status() -> dict:
        """Get system health: providers, WS connections, uptime."""
        uptime = round(time.monotonic() - _server_start_time, 1)

        # Provider status
        providers: List[dict] = []
        if quote_adapter is not None:
            connected = quote_adapter.is_connected()
            subscribed = quote_adapter.get_subscribed_symbols()
            providers.append({
                "name": "longbridge",
                "connected": connected,
                "symbols": len(subscribed),
                "subscribed_symbols": subscribed[:20],  # Cap at 20 for response size
            })

        # WS hub stats
        ws_clients = hub.client_count if hub else 0

        # Pipeline stats
        pipeline_running = False
        timeframes: List[str] = []
        if pipeline is not None:
            pipeline_running = getattr(pipeline, "_started", False)
            timeframes = getattr(pipeline, "_timeframes", [])

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
    async def get_data_quality() -> dict:
        """Get data quality report (proxied from R2)."""
        if r2_client is None:
            raise HTTPException(status_code=503, detail="R2 client not configured")

        try:
            data = r2_client.get_json("data_quality.json")
        except Exception as e:
            logger.error("Failed to fetch data_quality.json: %s", e)
            raise HTTPException(status_code=502, detail="Failed to fetch data quality")

        if data is None:
            raise HTTPException(status_code=404, detail="Data quality report not found")

        return data if isinstance(data, dict) else {"data": data}

    return router
