"""Screener trigger and results endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.jobs.models import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/screener", tags=["screener"])


@router.post("/momentum", status_code=202)
async def trigger_momentum(request: Request) -> JSONResponse:
    """Trigger a momentum screener run. Returns run_id for polling."""
    job_manager = request.app.state.job_manager

    async def run_momentum(run_id: str):
        from config.config_manager import ConfigManager
        from src.runners.momentum_runner import cmd_screen

        cm = ConfigManager(config_dir="config", env="dev")
        config = cm.load()
        result = await asyncio.to_thread(
            cmd_screen,
            config,
            signals_dir=None,
            include_recent_ipos=False,
            no_earnings=False,
            no_refresh=True,
        )
        candidates = []
        if result and hasattr(result, "candidates"):
            candidates = [
                {
                    "symbol": c.signal.symbol,
                    "score": getattr(c.signal, "momentum_12_1", None),
                }
                for c in result.candidates
            ]
        return {"candidates": candidates, "count": len(candidates)}

    run_id = await job_manager.submit("momentum", run_momentum)
    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "status": "submitted"},
    )


@router.post("/pead", status_code=202)
async def trigger_pead(request: Request) -> JSONResponse:
    """Trigger a PEAD screener run. Returns run_id for polling."""
    job_manager = request.app.state.job_manager

    async def run_pead(run_id: str):
        from src.runners.pead_runner import cmd_screen

        result = await asyncio.to_thread(
            cmd_screen, signals_dir=None, regime_fallback="R1"
        )
        candidates = []
        if result and hasattr(result, "candidates"):
            candidates = [
                {"symbol": c.symbol, "score": getattr(c, "score", None)}
                for c in result.candidates
            ]
        return {"candidates": candidates, "count": len(candidates)}

    run_id = await job_manager.submit("pead", run_pead)
    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "status": "submitted"},
    )


@router.get("/results/{run_id}")
async def get_results(run_id: str, request: Request) -> dict:
    """Poll screener results by run_id."""
    job_manager = request.app.state.job_manager
    info = job_manager.get_status(run_id)

    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

    response: dict = {"run_id": run_id, "status": info.status.value}
    if info.status == JobStatus.COMPLETED:
        response["results"] = info.result
    elif info.status == JobStatus.FAILED:
        response["error"] = info.error
    return response
