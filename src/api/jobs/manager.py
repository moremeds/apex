"""In-memory background job manager — tracks asyncio tasks with status."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from src.api.jobs.models import JobInfo, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Tracks background asyncio tasks with run_id-keyed status."""

    def __init__(self, max_history: int = 100) -> None:
        self._jobs: dict[str, JobInfo] = {}
        self._max_history = max_history

    async def submit(
        self,
        job_type: str,
        task_fn: Callable[[str], Awaitable[Any]],
    ) -> str:
        """Schedule a job. Returns run_id immediately; task runs in the background."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{job_type}-{ts}-{uuid.uuid4().hex[:6]}"

        self._jobs[run_id] = JobInfo(run_id=run_id, job_type=job_type, status=JobStatus.RUNNING)
        asyncio.create_task(self._run(run_id, task_fn))
        return run_id

    def get_status(self, run_id: str) -> Optional[JobInfo]:
        """Return JobInfo for run_id, or None if unknown."""
        return self._jobs.get(run_id)

    async def _run(self, run_id: str, task_fn: Callable[[str], Awaitable[Any]]) -> None:
        info = self._jobs[run_id]
        try:
            info.result = await task_fn(run_id)
            info.status = JobStatus.COMPLETED
        except Exception as e:
            info.status = JobStatus.FAILED
            info.error = str(e)
            logger.warning("Job %s failed: %s", run_id, e)
        finally:
            info.completed_at = datetime.now(timezone.utc)
            self._prune_old_jobs()

    def _prune_old_jobs(self) -> None:
        """Drop oldest finished jobs once we exceed max_history."""
        if len(self._jobs) <= self._max_history:
            return
        finished = sorted(
            ((k, v) for k, v in self._jobs.items() if v.status != JobStatus.RUNNING),
            key=lambda kv: kv[1].submitted_at,
        )
        for k, _ in finished[: len(self._jobs) - self._max_history]:
            del self._jobs[k]
