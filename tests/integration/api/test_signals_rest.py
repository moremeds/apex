"""REST pull endpoint returns a schema-valid payload."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.payload.validate import validate_payload
from src.api.server import create_app


class _FakeRepo:
    async def fetch_signals(self, symbol, since=None, limit=500):
        return [
            {
                "time": datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
                "signal_id": "trend:macd:AAPL:1d",
                "symbol": symbol,
                "timeframe": "1d",
                "category": "trend",
                "indicator": "MACD",
                "direction": "buy",
                "strength": 65,
                "priority": "medium",
                "trigger_rule": "macd_bull_cross",
                "current_value": 1.23,
                "threshold": 0.0,
                "previous_value": -0.4,
                "message": "x",
                "cooldown_until": None,
                "metadata": {},
            }
        ]


@pytest.mark.asyncio
async def test_get_signals_returns_valid_payload() -> None:
    app = create_app()
    app.state.signal_repo = _FakeRepo()  # ASGITransport does not run lifespan
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/signals/AAPL")
    assert resp.status_code == 200
    payload = resp.json()
    validate_payload(payload)
    assert payload["signals"][0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_signals_503_when_repo_unconfigured() -> None:
    """No Postgres -> signal_repo is None -> explicit 503 (not a 500 AttributeError)."""
    app = create_app()
    app.state.signal_repo = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/signals/AAPL")
    assert resp.status_code == 503
