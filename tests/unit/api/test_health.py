"""Tests for API health endpoint."""

from __future__ import annotations

from pathlib import Path

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
        "recency": None,
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
        # A tmp_path provider whose artifacts do not exist: nulls, not an error.
        "recency": {
            "bronze_last_trade_date": None,
            "silver_last_trade_date": None,
            "lag_days": None,
        },
    }


def test_health_reports_zero_lag_when_silver_matches_bronze(tmp_path: Path) -> None:
    """Real production state 2026-08-23: bronze and silver are both at 2026-08-21."""
    import datetime as dt

    import pandas as pd

    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    dates = [dt.date(2026, 8, 19), dt.date(2026, 8, 20), dt.date(2026, 8, 21)]
    ohlc = {
        "open": [310.140, 317.455, 312.050],
        "high": [319.2799, 320.2800, 312.3800],
        "low": [309.60, 310.65, 307.01],
        "close": [316.83, 311.30, 309.35],
        "volume": [51405496, 40959127, 48591536],
    }
    bronze = tmp_path / "bronze" / "asset_class=equity" / "symbol=AAPL"
    silver = tmp_path / "silver" / "asset_class=equity" / "symbol=AAPL"
    for d in (bronze, silver):
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": dates, **ohlc}).to_parquet(d / "1d.parquet")

    provider = LivewireOhlcProvider(
        bronze_root=tmp_path / "bronze", silver_root=tmp_path / "silver", price_mode="adjusted"
    )
    recency = provider.fetch_recency("AAPL")
    assert recency["bronze_last_trade_date"] == "2026-08-21"
    assert recency["silver_last_trade_date"] == "2026-08-21"
    assert recency["lag_days"] == 0


def test_recency_is_null_when_artifacts_are_absent(tmp_path: Path) -> None:
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    provider = LivewireOhlcProvider(bronze_root=tmp_path)
    assert provider.fetch_recency("AAPL") == {
        "bronze_last_trade_date": None,
        "silver_last_trade_date": None,
        "lag_days": None,
    }


@pytest.mark.asyncio
async def test_health_endpoint_exposes_recency() -> None:
    """A consumer must be able to see how stale the lake is without a second call."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        data = (await client.get("/health")).json()
    assert "recency" in data["livewire"]
    # No provider configured in this app, so recency is null rather than absent.
    assert data["livewire"]["recency"] is None
