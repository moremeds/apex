"""Tests for API server lifespan (PG connection lifecycle)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_server_connects_to_pg_on_startup():
    """Lifespan opens an asyncpg pool when APEX_PG_URL is set."""
    with patch.dict(os.environ, {"APEX_PG_URL": "postgresql://test@localhost/test"}):
        with patch("src.api.server.asyncpg") as mock_asyncpg:
            mock_pool = AsyncMock()
            mock_pool.close = AsyncMock()
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

            from src.api.server import create_app, lifespan

            app = create_app()

            async with lifespan(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/health")

                assert resp.status_code == 200
                assert resp.json()["pg_connected"] is True

            # On shutdown, the pool should be closed
            mock_pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_server_runs_without_pg():
    """Without APEX_PG_URL, /health reports pg_connected=False but app still serves."""
    env = {k: v for k, v in os.environ.items() if k != "APEX_PG_URL"}
    with patch.dict(os.environ, env, clear=True):
        from src.api.server import create_app, lifespan

        app = create_app()
        async with lifespan(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/health")

            assert resp.status_code == 200
            assert resp.json()["pg_connected"] is False
