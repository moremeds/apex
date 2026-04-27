"""Backtest trigger and results endpoints."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.jobs.models import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start: str
    end: str
    capital: float = 100_000.0


@router.post("/run", status_code=202)
async def trigger_backtest(req: BacktestRequest, request: Request) -> JSONResponse:
    """Submit a backtest job. Returns run_id immediately for polling."""
    job_manager = request.app.state.job_manager

    async def run_backtest(run_id: str):
        from datetime import date

        from src.backtest.runner import SingleBacktestRunner

        runner = SingleBacktestRunner(
            strategy_name=req.strategy,
            symbols=req.symbols,
            start_date=date.fromisoformat(req.start),
            end_date=date.fromisoformat(req.end),
            initial_capital=req.capital,
            data_source="historical",
        )
        result = await runner.run()
        return {
            "sharpe": getattr(result, "sharpe", None),
            "total_return": getattr(result, "total_return", None),
            "max_drawdown": getattr(result, "max_drawdown", None),
            "trade_count": getattr(result, "trade_count", None),
            "win_rate": getattr(result, "win_rate", None),
        }

    run_id = await job_manager.submit("backtest", run_backtest)
    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "status": "submitted"},
    )


@router.get("/results/{run_id}")
async def get_results(run_id: str, request: Request) -> dict:
    """Poll backtest results by run_id."""
    job_manager = request.app.state.job_manager
    info = job_manager.get_status(run_id)

    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

    response: dict = {"run_id": run_id, "status": info.status.value}
    if info.status == JobStatus.COMPLETED:
        response["metrics"] = info.result
    elif info.status == JobStatus.FAILED:
        response["error"] = info.error
    return response
