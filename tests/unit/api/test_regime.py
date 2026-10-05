"""The removed /regime/{symbol} route."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest.mark.asyncio
async def test_regime_route_is_gone():
    """/regime/{symbol} read PG score_history, written only by the undeployed legacy
    signal_service, so it served stale rows; the route was removed."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/regime/SPY")

    assert resp.status_code == 404
