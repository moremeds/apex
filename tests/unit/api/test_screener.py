"""Tests for screener trigger and results endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest.mark.asyncio
async def test_trigger_momentum_screener():
    """POST /screener/momentum returns 202 with run_id."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/screener/momentum")

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "submitted"
    assert data["run_id"].startswith("momentum-")


@pytest.mark.asyncio
async def test_trigger_pead_screener():
    """POST /screener/pead returns 202 with run_id."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/screener/pead")

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "submitted"
    assert data["run_id"].startswith("pead-")


@pytest.mark.asyncio
async def test_get_results_unknown_run_id():
    """GET /screener/results/{run_id} returns 404 for unknown run_id."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/screener/results/nonexistent-123")

    assert resp.status_code == 404
