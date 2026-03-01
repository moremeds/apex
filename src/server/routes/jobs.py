"""Background job trigger endpoints.

Provides /api/jobs/ routes for triggering compute-heavy tasks
(momentum screener, PEAD screener, strategy comparison) in a
dedicated thread pool, with status polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Dedicated executor — strategy-compare can take 60+ min
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobState:
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_jobs: dict[str, JobState] = {}
_lock = asyncio.Lock()


def _is_job_running(name: str) -> bool:
    return any(j.name == name and j.status == JobStatus.RUNNING for j in _jobs.values())


async def _run_job_in_thread(
    job_id: str,
    name: str,
    func: Any,
    *args: Any,
    on_complete: Any = None,
) -> None:
    """Execute a sync function in the dedicated thread pool."""
    async with _lock:
        _jobs[job_id] = JobState(
            id=job_id,
            name=name,
            status=JobStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_executor, func, *args)
        async with _lock:
            _jobs[job_id].status = JobStatus.COMPLETED
            _jobs[job_id].completed_at = datetime.now(timezone.utc)
            _jobs[job_id].result = result or {}
        logger.info("Job %s (%s) completed", job_id, name)
        if on_complete and result:
            try:
                on_complete(result)
            except Exception:
                logger.debug("on_complete callback failed for %s", job_id, exc_info=True)
    except Exception as e:
        async with _lock:
            _jobs[job_id].status = JobStatus.FAILED
            _jobs[job_id].completed_at = datetime.now(timezone.utc)
            _jobs[job_id].error = str(e)
        logger.exception("Job %s (%s) failed", job_id, name)


def _compute_momentum() -> dict[str, Any]:
    """Run momentum screener compute (sync), return full JSON payload."""
    import json
    from pathlib import Path

    from src.runners.momentum_runner import _load_config, cmd_screen

    config = _load_config()
    cmd_screen(config)
    # Runner writes the exact JSON the frontend expects
    path = Path("out/momentum/data/momentum_watchlist.json")
    if path.exists():
        return json.loads(path.read_text())
    return {"candidates": [], "universe_size": 0}


def _compute_pead() -> dict[str, Any]:
    """Run PEAD screener compute (sync), return full JSON payload."""
    import json
    from pathlib import Path

    from src.runners.pead_runner import cmd_screen

    cmd_screen()
    path = Path("out/pead/data/pead_candidates.json")
    if path.exists():
        return json.loads(path.read_text())
    return {"candidates": [], "screened_count": 0}


def _compute_strategy_compare() -> dict[str, Any]:
    """Run strategy comparison backtest (sync), return full JSON payload."""
    import json
    from pathlib import Path

    from src.runners.strategy_compare_runner import run_comparison

    filepath = run_comparison(symbols=["SPY", "QQQ", "AAPL"], years=3)
    data = json.loads(Path(filepath).read_text())
    return data


async def _require_admin(request: Request) -> None:
    """Simple bearer-token guard for job trigger endpoints.

    Set APEX_ADMIN_TOKEN env var to enable. When unset, all requests pass
    (local dev mode). When set, requests must include Authorization: Bearer <token>.
    """
    token = os.environ.get("APEX_ADMIN_TOKEN")
    if not token:
        return  # No token configured — open access (local dev)
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


def create_jobs_router() -> APIRouter:
    """Create router for background job trigger endpoints.

    Uses module-level state for the executor and job tracking,
    following the same factory pattern as other route modules.
    POST endpoints require APEX_ADMIN_TOKEN when configured.
    """
    router = APIRouter(prefix="/api/jobs", tags=["jobs"])

    @router.post("/momentum", dependencies=[Depends(_require_admin)])
    async def trigger_momentum(request: Request) -> JSONResponse:
        if _is_job_running("momentum"):
            raise HTTPException(status_code=409, detail="Momentum job already running")
        job_id = f"momentum-{int(datetime.now(timezone.utc).timestamp())}"
        cache = getattr(request.app.state, "screener_cache", None)

        def on_complete(result: dict) -> None:
            if cache:
                cache.merge_cache("screeners.json", {"momentum": result})

        asyncio.create_task(
            _run_job_in_thread(job_id, "momentum", _compute_momentum, on_complete=on_complete)
        )
        return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)

    @router.post("/pead", dependencies=[Depends(_require_admin)])
    async def trigger_pead(request: Request) -> JSONResponse:
        if _is_job_running("pead"):
            raise HTTPException(status_code=409, detail="PEAD job already running")
        job_id = f"pead-{int(datetime.now(timezone.utc).timestamp())}"
        cache = getattr(request.app.state, "screener_cache", None)

        def on_complete(result: dict) -> None:
            if cache:
                cache.merge_cache("screeners.json", {"pead": result})

        asyncio.create_task(
            _run_job_in_thread(job_id, "pead", _compute_pead, on_complete=on_complete)
        )
        return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)

    @router.post("/strategy-compare", dependencies=[Depends(_require_admin)])
    async def trigger_strategy_compare(request: Request) -> JSONResponse:
        if _is_job_running("strategy-compare"):
            raise HTTPException(status_code=409, detail="Strategy compare job already running")
        job_id = f"strategy-compare-{int(datetime.now(timezone.utc).timestamp())}"
        cache = getattr(request.app.state, "screener_cache", None)

        def on_complete(result: dict) -> None:
            if cache:
                cache.set_cache("strategies.json", result)

        asyncio.create_task(
            _run_job_in_thread(
                job_id, "strategy-compare", _compute_strategy_compare, on_complete=on_complete
            )
        )
        return JSONResponse({"job_id": job_id, "status": "running"}, status_code=202)

    @router.get("/status")
    async def get_job_status() -> dict:
        async with _lock:
            return {
                "jobs": [
                    {
                        "id": j.id,
                        "name": j.name,
                        "status": j.status.value,
                        "started_at": j.started_at.isoformat() if j.started_at else None,
                        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                        "error": j.error,
                    }
                    for j in _jobs.values()
                ]
            }

    return router
