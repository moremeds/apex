"""Tests for API health endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import APEX_VERSION, create_app


@pytest.mark.asyncio
async def test_health_returns_ok():
    """GET /health returns status ok with uptime, service name, and running version."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime" in data
    assert data["service"] == "apex-signal-server"
    # version must reflect the real running build, not a hardcoded literal
    assert data["version"] == APEX_VERSION
    assert data["version"] not in ("", "unknown")
