"""REST routes — /api/monitor for system health and data quality."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, List

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

_STATIC_DATA_URL = "https://moremeds.github.io/apex/data"


_ssl_ctx = ssl.create_default_context()
try:
    import certifi

    _ssl_ctx.load_verify_locations(certifi.where())
except ImportError:
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE


def _fetch_static_dq_sync() -> Any | None:
    """Fetch data_quality.json from GitHub Pages (sync, for asyncio.to_thread)."""
    url = f"{_STATIC_DATA_URL}/data_quality.json"
    req = urllib.request.Request(url, headers={"User-Agent": "APEX-Dashboard/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


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
            providers.append(
                {
                    "name": "longbridge",
                    "connected": connected,
                    "symbols": len(subscribed),
                    "subscribed_symbols": subscribed[:20],
                }
            )

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
        """Get data quality report (R2 → static fallback)."""
        r2 = _get_r2_client(request)
        data = None

        # Try R2 first (stored under meta/ prefix)
        if r2 is not None:
            try:
                data = r2.get_json("meta/data_quality.json")
            except Exception as e:
                logger.error("R2 fetch data_quality.json failed: %s", e)

        # Fallback to GitHub Pages
        if data is None:
            try:
                data = await asyncio.to_thread(_fetch_static_dq_sync)
            except Exception as e:
                logger.error("Static fallback for data_quality.json failed: %s", e)

        if data is None:
            raise HTTPException(status_code=503, detail="Data quality report not available")

        return data if isinstance(data, dict) else {"data": data}

    return router
