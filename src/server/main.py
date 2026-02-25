"""APEX Live Dashboard — FastAPI entrypoint."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.server.routes.monitor import create_monitor_router
from src.server.routes.screeners import create_screeners_router
from src.server.routes.symbols import create_symbols_router
from src.server.routes.ws import create_ws_router
from src.server.ws_hub import WebSocketHub

_start_time = time.monotonic()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="APEX Live Dashboard", version="0.1.0")

    # CORS — allow all origins in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create shared WebSocket hub
    hub = WebSocketHub()
    app.state.hub = hub

    # Health endpoint
    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
            "ws_clients": hub.client_count,
        }

    # Routes — adapters/pipeline/r2 wired in Task 2.8 (startup orchestration)
    app.include_router(create_ws_router(hub))
    app.include_router(create_symbols_router())
    app.include_router(create_screeners_router())
    app.include_router(create_monitor_router(hub=hub))

    # Static file mount for production (web/dist/) — must be LAST
    dist_path = Path("web/dist")
    if dist_path.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")

    return app


# Module-level app for `uvicorn src.server.main:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8080, reload=True)
