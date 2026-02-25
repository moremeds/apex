"""APEX Live Dashboard — FastAPI entrypoint with full lifecycle orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.server.config import load_server_config
from src.server.persistence import ServerPersistence
from src.server.pipeline import ServerPipeline
from src.server.routes.monitor import create_monitor_router
from src.server.routes.screeners import create_screeners_router
from src.server.routes.symbols import create_symbols_router
from src.server.routes.ws import create_ws_router
from src.server.ws_hub import WebSocketHub

logger = logging.getLogger(__name__)

_start_time = time.monotonic()


def _tick_to_dict(tick) -> dict:
    """Convert QuoteTick to JSON-serializable dict for WS broadcast."""
    return {
        "last": tick.last,
        "bid": tick.bid,
        "ask": tick.ask,
        "volume": tick.volume,
        "ts": tick.timestamp.isoformat() if tick.timestamp else None,
    }


async def _periodic_flush(persistence: ServerPersistence, interval_sec: int) -> None:
    """Periodically flush buffered ticks to DuckDB."""
    while True:
        await asyncio.sleep(interval_sec)
        try:
            count = persistence.flush_to_duckdb()
            if count > 0:
                logger.info("Periodic flush: %d ticks → DuckDB", count)
        except Exception:
            logger.exception("Periodic flush failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle — startup and shutdown orchestration."""
    config = load_server_config()

    # ── Core components ────────────────────────────────────
    hub = WebSocketHub()
    persistence = ServerPersistence(config.duckdb_path)
    pipeline = ServerPipeline(hub=hub, timeframes=config.timeframes, config=config)

    # ── Longbridge adapter (optional — only if enabled + env vars present) ──
    quote_adapter = None
    historical_adapter = None
    r2_client = None

    lb_enabled = config.providers.get("longbridge")
    has_lb_creds = bool(os.environ.get("LONGPORT_APP_KEY"))

    # ── R2 client (optional — only if credentials available) ──
    has_r2_creds = bool(os.environ.get("R2_ACCESS_KEY_ID"))
    if has_r2_creds:
        try:
            from src.infrastructure.adapters.r2.client import R2Client
            r2_client = R2Client()
            logger.info("R2 client initialized")
        except Exception:
            logger.exception("Failed to initialize R2 client")

    # ── Longbridge connection (deferred — SDK takes ~13s to connect) ──
    async def _connect_longbridge():
        nonlocal quote_adapter, historical_adapter
        try:
            from src.infrastructure.adapters.longbridge.historical_adapter import (
                LongbridgeHistoricalAdapter,
            )
            from src.infrastructure.adapters.longbridge.quote_adapter import (
                LongbridgeQuoteAdapter,
            )

            qa = LongbridgeQuoteAdapter()
            await qa.connect()

            ha = LongbridgeHistoricalAdapter()
            if hasattr(qa, "_ctx") and qa._ctx:
                ha._ctx = qa._ctx
                ha._connected = True

            # Wire tick callback
            loop = asyncio.get_running_loop()

            def on_tick(tick):
                pipeline.on_tick(tick)
                persistence.buffer_tick(tick)
                coro = hub.broadcast_quote(tick.symbol, _tick_to_dict(tick))
                asyncio.run_coroutine_threadsafe(coro, loop)

            qa.set_quote_callback(on_tick)

            # Subscribe
            symbols = [s.replace(".US", "") for s in config.core_symbols]
            await qa.subscribe_quotes(symbols)

            # Update refs
            quote_adapter = qa
            historical_adapter = ha
            app.state.quote_adapter = qa
            app.state.historical_adapter = ha

            logger.info("Longbridge connected — %d symbols streaming", len(symbols))
        except Exception:
            logger.exception("Failed to connect Longbridge adapter")

    lb_task = None
    if lb_enabled and lb_enabled.enabled and has_lb_creds:
        lb_task = asyncio.create_task(_connect_longbridge())

    # ── Start pipeline ──
    await pipeline.start()

    # ── Periodic flush task ──
    flush_task = asyncio.create_task(
        _periodic_flush(persistence, config.r2_flush_interval_sec)
    )

    # ── Store refs on app.state for routes ──
    app.state.hub = hub
    app.state.pipeline = pipeline
    app.state.persistence = persistence
    app.state.quote_adapter = quote_adapter
    app.state.historical_adapter = historical_adapter
    app.state.r2_client = r2_client

    logger.info(
        "APEX Live Dashboard started (symbols=%d, timeframes=%s, longbridge=%s, r2=%s)",
        len(config.core_symbols),
        config.timeframes,
        quote_adapter is not None,
        r2_client is not None,
    )

    yield

    # ── Shutdown ────────────────────────────────────────────
    flush_task.cancel()
    try:
        await flush_task
    except asyncio.CancelledError:
        pass

    if lb_task and not lb_task.done():
        lb_task.cancel()
        try:
            await lb_task
        except asyncio.CancelledError:
            pass

    await pipeline.stop()

    if quote_adapter is not None:
        await quote_adapter.disconnect()

    persistence.close()
    logger.info("APEX Live Dashboard stopped")


def create_app(config_path: str = "config/server.yaml") -> FastAPI:
    """Create and configure the FastAPI application.

    For testing, call with no lifespan and inject mocks.
    For production, the lifespan handles all wiring.
    """
    app = FastAPI(
        title="APEX Live Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health endpoint (always available, reads from app.state)
    @app.get("/api/health")
    async def health() -> dict:
        hub = getattr(app.state, "hub", None)
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
            "ws_clients": hub.client_count if hub else 0,
        }

    # Routes — dependencies resolved lazily from app.state
    # This allows the lifespan to wire real adapters after startup
    hub_placeholder = WebSocketHub()  # used before lifespan runs (e.g., tests)
    app.include_router(create_ws_router(hub_placeholder))
    app.include_router(create_symbols_router())
    app.include_router(create_screeners_router())
    app.include_router(create_monitor_router(hub=hub_placeholder))

    # Static file mount for production (web/dist/) — must be LAST
    dist_path = Path("web/dist")
    if dist_path.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")

    return app


def create_test_app() -> FastAPI:
    """Create a minimal app without lifespan for unit testing."""
    app = FastAPI(title="APEX Live Dashboard", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    hub = WebSocketHub()
    app.state.hub = hub

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
            "ws_clients": hub.client_count,
        }

    app.include_router(create_ws_router(hub))
    app.include_router(create_symbols_router())
    app.include_router(create_screeners_router())
    app.include_router(create_monitor_router(hub=hub))

    dist_path = Path("web/dist")
    if dist_path.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")

    return app


# Module-level app for `uvicorn src.server.main:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8080, reload=True)
