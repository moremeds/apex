"""Routes removed from the API stay gone."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.server import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    [
        # read PG score_history, written only by the undeployed legacy signal_service
        ("GET", "/regime/SPY"),
        # frozen strategy/screener surfaces with no caller in the ecosystem
        ("GET", "/strategy/list"),
        ("GET", "/strategy/trend_pulse/params"),
        ("POST", "/screener/momentum"),
        ("POST", "/screener/pead"),
        ("GET", "/screener/results/abc"),
    ],
)
async def test_removed_route_is_404(method: str, path: str) -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.request(method, path)

    assert resp.status_code == 404
