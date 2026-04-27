"""Tests for backtest trigger endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest.mark.asyncio
async def test_trigger_backtest():
    """POST /backtest/run with valid body returns 202 + run_id."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/backtest/run",
            json={
                "strategy": "trend_pulse",
                "symbols": ["AAPL", "SPY"],
                "start": "2025-01-01",
                "end": "2025-06-30",
            },
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "submitted"
    assert data["run_id"].startswith("backtest-")


@pytest.mark.asyncio
async def test_trigger_backtest_missing_fields():
    """POST /backtest/run with missing required fields returns 422."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/backtest/run", json={"strategy": "trend_pulse"})

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_backtest_results_unknown():
    """GET /backtest/results/{run_id} returns 404 for unknown run_id."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/backtest/results/unknown-123")

    assert resp.status_code == 404
