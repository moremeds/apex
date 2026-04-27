"""Tests for strategy endpoints."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest.mark.asyncio
async def test_list_strategies():
    """GET /strategy/list returns registered strategies with tier metadata."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/strategy/list")

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "name" in first
    assert "tier" in first
    assert "param_count" in first


@pytest.mark.asyncio
async def test_get_strategy_params():
    """GET /strategy/{name}/params returns params for known strategy."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/strategy/trend_pulse/params")

    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "trend_pulse"
    assert isinstance(data["params"], dict)
    assert "history" in data


@pytest.mark.asyncio
async def test_get_unknown_strategy_returns_404():
    """GET /strategy/{name}/params returns 404 for unknown strategy."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/strategy/nonexistent_strategy/params")

    assert resp.status_code == 404
