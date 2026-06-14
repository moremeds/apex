"""APEX API Server — FastAPI application factory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.jobs.manager import JobManager
from src.api.routes.backtest import router as backtest_router
from src.api.routes.health import router as health_router
from src.api.routes.regime import router as regime_router
from src.api.routes.screener import router as screener_router
from src.api.routes.strategy import router as strategy_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the asyncpg pool on startup, close it on shutdown."""
    pg_url = os.environ.get("APEX_PG_URL")
    if pg_url:
        try:
            app.state.pg_pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=5)
            app.state.pg_connected = True
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.warning("PG connection failed: %s", e)
            app.state.pg_pool = None
            app.state.pg_connected = False
    else:
        app.state.pg_pool = None
        app.state.pg_connected = False
        logger.info("No APEX_PG_URL set — running without PG")

    # Streaming TA signal surface (Phase 3). Guarded with `getattr(..., None) is None`
    # so tests can pre-inject fakes that the lifespan must not clobber.
    from src.api.ws.hub import SignalHub

    if getattr(app.state, "signal_hub", None) is None:
        app.state.signal_hub = SignalHub()
    if getattr(app.state, "signal_repo", None) is None:
        app.state.signal_repo = None  # real TASignalRepository(Database) wired when PG configured
    # NOTE (env-gated): the real SubscriptionManager (livewire provider +
    # TASignalService) and the SignalEmitter(bus) are constructed here when their
    # dependencies are present. Left unbuilt in environments without livewire/PG;
    # the WS route requires app.state.subscription_manager to be set before use.

    try:
        yield
    finally:
        if app.state.pg_pool is not None:
            await app.state.pg_pool.close()
            logger.info("PostgreSQL pool closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="APEX Signal Server",
        description="Signal generation, backtesting, and strategy management API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.job_manager = JobManager()

    app.include_router(health_router)
    app.include_router(strategy_router)
    app.include_router(regime_router)
    app.include_router(screener_router)
    app.include_router(backtest_router)

    # Streaming TA signal surface (Phase 3): REST pull + WS push to argon.
    from src.api.routes.signals import router as signals_router
    from src.api.ws.signals_ws import router as signals_ws_router

    app.include_router(signals_router)
    app.include_router(signals_ws_router)

    return app


def main() -> None:
    """Run the API server via uvicorn."""
    import uvicorn

    port = int(os.environ.get("APEX_API_PORT", "8322"))
    workers = int(os.environ.get("APEX_API_WORKERS", "1"))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    logger.info("Starting APEX API server on port %d", port)

    uvicorn.run(
        "src.api.server:create_app",
        host="0.0.0.0",  # nosec B104 - backend service intentionally listens on all interfaces for Xenon consumers
        port=port,
        workers=workers,
        factory=True,
    )


if __name__ == "__main__":
    main()
