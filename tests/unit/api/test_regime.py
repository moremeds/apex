"""Tests for regime endpoint."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest.mark.asyncio
async def test_regime_no_pg_returns_503():
    """GET /regime/{symbol} returns 503 when PG pool not on app.state."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/regime/SPY")

    assert resp.status_code == 503
