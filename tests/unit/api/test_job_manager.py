"""Tests for background job manager."""

from __future__ import annotations

import asyncio

import pytest

from src.api.jobs.manager import JobManager
from src.api.jobs.models import JobStatus


@pytest.mark.asyncio
async def test_submit_job_returns_run_id():
    """submit() returns a prefixed run_id and the job is registered."""
    mgr = JobManager()

    async def dummy_task(run_id: str):
        await asyncio.sleep(0.01)
        return {"result": "done"}

    run_id = await mgr.submit("test_job", dummy_task)
    assert run_id.startswith("test_job-")
    status = mgr.get_status(run_id)
    assert status is not None
    assert status.status in (JobStatus.RUNNING, JobStatus.COMPLETED)


@pytest.mark.asyncio
async def test_job_completes_with_result():
    """Completed job has status COMPLETED and stored result."""
    mgr = JobManager()

    async def dummy_task(run_id: str):
        return {"candidates": 5}

    run_id = await mgr.submit("screen", dummy_task)
    await asyncio.sleep(0.05)

    status = mgr.get_status(run_id)
    assert status.status == JobStatus.COMPLETED
    assert status.result == {"candidates": 5}


@pytest.mark.asyncio
async def test_job_failure_captured():
    """Failed job has status FAILED with error message."""
    mgr = JobManager()

    async def failing_task(run_id: str):
        raise ValueError("something broke")

    run_id = await mgr.submit("broken", failing_task)
    await asyncio.sleep(0.05)

    status = mgr.get_status(run_id)
    assert status.status == JobStatus.FAILED
    assert "something broke" in status.error


@pytest.mark.asyncio
async def test_unknown_run_id_returns_none():
    """get_status() for unknown run_id returns None."""
    mgr = JobManager()
    assert mgr.get_status("nonexistent-123") is None
