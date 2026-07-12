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
    assert data["silver_revision"] == {"enabled": False}
    assert data["livewire"] == {
        "configured": False,
        "configured_price_mode": "raw",
        "effective_price_mode": None,
    }
    # version must reflect the real running build, not a hardcoded literal
    assert data["version"] == APEX_VERSION
    assert data["version"] not in ("", "unknown")


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


@pytest.mark.asyncio
async def test_health_reports_configured_and_effective_price_mode(tmp_path):
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    app = create_app()
    app.state.livewire_price_mode = "adjusted"
    app.state.ohlc_provider = LivewireOhlcProvider(
        bronze_root=tmp_path / "bronze",
        silver_root=tmp_path / "silver",
        price_mode="adjusted",
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.json()["livewire"] == {
        "configured": True,
        "configured_price_mode": "adjusted",
        "effective_price_mode": "adjusted",
    }
