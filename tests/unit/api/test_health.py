"""Tests for API health endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest.mark.asyncio
async def test_health_returns_ok():
    """GET /health returns status ok with uptime and service name."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime" in data
    assert data["service"] == "apex-signal-server"
    assert data["silver_revision"] == {"enabled": False}


@pytest.mark.asyncio
async def test_health_includes_revision_watcher_state():
    class _Watcher:
        def health(self) -> dict:
            return {
                "enabled": True,
                "observed_revision": 42,
                "last_fully_applied_revision": 41,
            }

    app = create_app()
    app.state.revision_watcher = _Watcher()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.json()["silver_revision"]["observed_revision"] == 42
    assert resp.json()["silver_revision"]["last_fully_applied_revision"] == 41
