"""REST routes — /api/monitor for system health and data quality."""

from __future__ import annotations

import asyncio
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

    def _get_hub(request: Request) -> Any:
        return getattr(request.app.state, "hub", None) or hub

    def _get_pipeline(request: Request) -> Any:
        return pipeline or getattr(request.app.state, "pipeline", None)

    def _get_quote_adapter(request: Request) -> Any:
        return quote_adapter or getattr(request.app.state, "quote_adapter", None)

    def _get_r2_client(request: Request) -> Any:
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

    @router.get("/r2-freshness")
    async def get_r2_freshness(request: Request) -> dict:
        """Check R2 data freshness — last modified timestamps for key files."""
        from datetime import datetime, timezone

        r2 = _get_r2_client(request)
        persistence_obj = getattr(request.app.state, "persistence", None)

        items: list[dict] = []
        key_files = [
            ("summary.json", "Summary"),
            ("screeners.json", "Screeners"),
            ("strategies.json", "Backtest"),
            ("meta/universe.json", "Universe"),
            ("meta/data_quality.json", "Data Quality"),
            ("meta/market_caps.json", "Market Caps"),
        ]
        for key, label in key_files:
            item: dict = {"key": key, "label": label, "last_modified": None, "status": "unknown"}
            if r2 is not None:
                try:
                    lm = await asyncio.to_thread(
                        lambda k=key: r2.get_last_modified(k) if hasattr(r2, "get_last_modified") else None
                    )
                    if lm:
                        item["last_modified"] = lm.isoformat() if hasattr(lm, "isoformat") else str(lm)
                        # Check staleness: > 48h = stale, > 7d = critical
                        age = (datetime.now(timezone.utc) - lm).total_seconds() if hasattr(lm, "tzinfo") and lm.tzinfo else None
                        if age is not None:
                            if age < 48 * 3600:
                                item["status"] = "fresh"
                            elif age < 7 * 24 * 3600:
                                item["status"] = "stale"
                            else:
                                item["status"] = "critical"
                except Exception as e:
                    item["status"] = "error"
                    item["error"] = str(e)
            items.append(item)

        # DuckDB summary freshness
        duckdb_summary = {"available": False, "ticker_count": 0}
        if persistence_obj:
            try:
                summary = await asyncio.to_thread(persistence_obj.get_summary)
                if summary and summary.get("tickers"):
                    duckdb_summary["available"] = True
                    duckdb_summary["ticker_count"] = len(summary["tickers"])
            except Exception:
                pass

        return {
            "r2_available": r2 is not None,
            "files": items,
            "duckdb_summary": duckdb_summary,
        }

    @router.get("/data-quality")
    async def get_data_quality(request: Request) -> dict:
        """Get data quality report (R2 only)."""
        r2 = _get_r2_client(request)
        data = None

        if r2 is not None:
            try:
                data = await asyncio.to_thread(r2.get_json, "meta/data_quality.json")
            except Exception as e:
                logger.error("R2 fetch data_quality.json failed: %s", e)

        if data is None:
            raise HTTPException(status_code=503, detail="Data quality report not available")

        return data if isinstance(data, dict) else {"data": data}

    return router
