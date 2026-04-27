"""APEX API Server — FastAPI application factory."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.jobs.manager import JobManager
from src.api.routes.health import router as health_router
from src.api.routes.regime import router as regime_router
from src.api.routes.screener import router as screener_router
from src.api.routes.strategy import router as strategy_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="APEX Signal Server",
        description="Signal generation, backtesting, and strategy management API",
        version="0.1.0",
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

    return app


def main() -> None:
    """Run the API server via uvicorn."""
    import uvicorn

    port = int(os.environ.get("APEX_API_PORT", "8322"))
    workers = int(os.environ.get("APEX_API_WORKERS", "1"))

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger.info("Starting APEX API server on port %d", port)

    uvicorn.run(
        "src.api.server:create_app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        factory=True,
    )


if __name__ == "__main__":
    main()
