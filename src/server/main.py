"""APEX Live Dashboard — FastAPI entrypoint."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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

    # Health endpoint
    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
        }

    # Static file mount for production (web/dist/)
    dist_path = Path("web/dist")
    if dist_path.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")

    return app


# Module-level app for `uvicorn src.server.main:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8080, reload=True)
