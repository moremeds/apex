"""Tests for /api/jobs routes."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server.routes import jobs as jobs_module
from src.server.routes.jobs import JobState, JobStatus, create_jobs_router


def _make_app():
    app = FastAPI()
    app.include_router(create_jobs_router())
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Reset module-level job state between tests."""
    jobs_module._jobs.clear()
    yield
    jobs_module._jobs.clear()


class TestJobStatus:
    def test_status_empty(self):
        client = _make_app()
        resp = client.get("/api/jobs/status")
        assert resp.status_code == 200
        assert resp.json() == {"jobs": []}

    def test_status_shows_jobs(self):
        from datetime import datetime, timezone

        jobs_module._jobs["test-1"] = JobState(
            id="test-1",
            name="momentum",
            status=JobStatus.COMPLETED,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        )
        client = _make_app()
        resp = client.get("/api/jobs/status")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["name"] == "momentum"
        assert data["jobs"][0]["status"] == "completed"


class TestTriggerMomentum:
    @patch("src.server.routes.jobs._compute_momentum")
    def test_trigger_returns_202(self, mock_compute):
        mock_compute.return_value = {"symbols_screened": 10}
        client = _make_app()
        resp = client.post("/api/jobs/momentum")
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "running"
        assert data["job_id"].startswith("momentum-")

    def test_duplicate_returns_409(self):
        jobs_module._jobs["running-job"] = JobState(
            id="running-job",
            name="momentum",
            status=JobStatus.RUNNING,
        )
        client = _make_app()
        resp = client.post("/api/jobs/momentum")
        assert resp.status_code == 409
        assert "already running" in resp.json()["detail"]


class TestTriggerPead:
    @patch("src.server.routes.jobs._compute_pead")
    def test_trigger_returns_202(self, mock_compute):
        mock_compute.return_value = {"symbols_screened": 5}
        client = _make_app()
        resp = client.post("/api/jobs/pead")
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"].startswith("pead-")

    def test_duplicate_returns_409(self):
        jobs_module._jobs["running-pead"] = JobState(
            id="running-pead",
            name="pead",
            status=JobStatus.RUNNING,
        )
        client = _make_app()
        resp = client.post("/api/jobs/pead")
        assert resp.status_code == 409


class TestTriggerStrategyCompare:
    @patch("src.server.routes.jobs._compute_strategy_compare")
    def test_trigger_returns_202(self, mock_compute):
        mock_compute.return_value = {"strategies": 3}
        client = _make_app()
        resp = client.post("/api/jobs/strategy-compare")
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"].startswith("strategy-compare-")

    def test_duplicate_returns_409(self):
        jobs_module._jobs["running-sc"] = JobState(
            id="running-sc",
            name="strategy-compare",
            status=JobStatus.RUNNING,
        )
        client = _make_app()
        resp = client.post("/api/jobs/strategy-compare")
        assert resp.status_code == 409


class TestCacheInjection:
    """Verify that job completion injects results into screener cache."""

    @pytest.mark.asyncio
    async def test_on_complete_callback_invoked(self):
        """_run_job_in_thread calls on_complete with the result."""
        results_captured: list[dict] = []

        def on_complete(result: dict) -> None:
            results_captured.append(result)

        await jobs_module._run_job_in_thread(
            "test-cb",
            "test",
            lambda: {"data": "hello"},
            on_complete=on_complete,
        )
        assert len(results_captured) == 1
        assert results_captured[0] == {"data": "hello"}

    @pytest.mark.asyncio
    async def test_on_complete_not_called_on_failure(self):
        """on_complete should NOT be called if the compute function raises."""
        results_captured: list[dict] = []

        def on_complete(result: dict) -> None:
            results_captured.append(result)

        def _failing():
            raise RuntimeError("boom")

        await jobs_module._run_job_in_thread(
            "test-fail",
            "test",
            _failing,
            on_complete=on_complete,
        )
        assert len(results_captured) == 0
        assert jobs_module._jobs["test-fail"].status == JobStatus.FAILED


class TestCompletedJobNotBlocking:
    """A completed job should NOT block a new trigger."""

    def test_completed_momentum_allows_new_trigger(self):
        from datetime import datetime, timezone

        jobs_module._jobs["old-job"] = JobState(
            id="old-job",
            name="momentum",
            status=JobStatus.COMPLETED,
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        )
        client = _make_app()
        with patch("src.server.routes.jobs._compute_momentum"):
            resp = client.post("/api/jobs/momentum")
        assert resp.status_code == 202

    def test_failed_job_allows_retry(self):
        jobs_module._jobs["failed-job"] = JobState(
            id="failed-job",
            name="pead",
            status=JobStatus.FAILED,
            error="Connection timeout",
        )
        client = _make_app()
        with patch("src.server.routes.jobs._compute_pead"):
            resp = client.post("/api/jobs/pead")
        assert resp.status_code == 202
