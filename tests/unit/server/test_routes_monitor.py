"""Tests for /api/monitor routes."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server.routes.monitor import create_monitor_router
from src.server.ws_hub import WebSocketHub


def _make_app(**kwargs):
    app = FastAPI()
    app.include_router(create_monitor_router(**kwargs))
    return TestClient(app)


def _make_quote_adapter(connected=True, symbols=None):
    adapter = MagicMock()
    adapter.is_connected.return_value = connected
    adapter.get_subscribed_symbols.return_value = symbols or []
    return adapter


def _make_pipeline(started=True, timeframes=None):
    pipeline = MagicMock()
    pipeline._started = started
    pipeline._timeframes = timeframes or ["1m", "1d"]
    return pipeline


class TestMonitorEndpoint:
    def test_basic_status(self):
        client = _make_app()
        resp = client.get("/api/monitor")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_sec" in data

    def test_provider_status(self):
        adapter = _make_quote_adapter(connected=True, symbols=["AAPL", "SPY"])
        client = _make_app(quote_adapter=adapter)
        resp = client.get("/api/monitor")
        data = resp.json()
        assert len(data["providers"]) == 1
        assert data["providers"][0]["name"] == "longbridge"
        assert data["providers"][0]["connected"] is True
        assert data["providers"][0]["symbols"] == 2

    def test_ws_client_count(self):
        hub = WebSocketHub()
        ws1 = MagicMock()
        ws2 = MagicMock()
        hub.connect(ws1)
        hub.connect(ws2)
        client = _make_app(hub=hub)
        resp = client.get("/api/monitor")
        assert resp.json()["ws_clients"] == 2

    def test_pipeline_status(self):
        pipeline = _make_pipeline(started=True, timeframes=["1m", "5m", "1d"])
        client = _make_app(pipeline=pipeline)
        resp = client.get("/api/monitor")
        data = resp.json()
        assert data["pipeline"]["running"] is True
        assert data["pipeline"]["timeframes"] == ["1m", "5m", "1d"]

    def test_full_status(self):
        hub = WebSocketHub()
        adapter = _make_quote_adapter(connected=True, symbols=["AAPL"])
        pipeline = _make_pipeline(started=True)
        client = _make_app(hub=hub, quote_adapter=adapter, pipeline=pipeline)
        resp = client.get("/api/monitor")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ws_clients"] == 0
        assert len(data["providers"]) == 1
        assert data["pipeline"]["running"] is True


class TestDataQualityEndpoint:
    def test_returns_quality_data(self):
        r2 = MagicMock()
        r2.get_json.return_value = {"total_symbols": 50, "coverage": 0.95}
        client = _make_app(r2_client=r2)
        resp = client.get("/api/monitor/data-quality")
        assert resp.status_code == 200
        assert resp.json()["total_symbols"] == 50

    def test_503_no_r2(self):
        client = _make_app()
        resp = client.get("/api/monitor/data-quality")
        assert resp.status_code == 503

    def test_503_r2_returns_none(self):
        r2 = MagicMock()
        r2.get_json.return_value = None
        client = _make_app(r2_client=r2)
        resp = client.get("/api/monitor/data-quality")
        assert resp.status_code == 503
