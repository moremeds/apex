"""Tests for /api/screeners and /api/backtest routes (R2 proxy)."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server.routes.screeners import _CachedProxy, create_screeners_router


def _make_app(r2_client=None, cache_ttl=300):
    app = FastAPI()
    app.include_router(create_screeners_router(r2_client=r2_client, cache_ttl=cache_ttl))
    return TestClient(app)


def _make_r2_client(data_map: dict):
    """Create mock R2Client that returns canned JSON data."""
    client = MagicMock()

    def get_json(key):
        return data_map.get(key)

    client.get_json = MagicMock(side_effect=get_json)
    return client


class TestScreenersEndpoint:
    def test_returns_screener_data(self):
        data = {"momentum": [{"symbol": "AAPL", "score": 85}], "pead": []}
        r2 = _make_r2_client({"screeners.json": data})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/screeners")
        assert resp.status_code == 200
        assert resp.json()["momentum"][0]["symbol"] == "AAPL"

    def test_503_when_no_data(self, monkeypatch):
        monkeypatch.setattr("src.server.routes.screeners._fetch_static_sync", lambda _: None)
        r2 = _make_r2_client({})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/screeners")
        assert resp.status_code == 503

    def test_503_when_no_r2_client(self, monkeypatch):
        monkeypatch.setattr("src.server.routes.screeners._fetch_static_sync", lambda _: None)
        client = _make_app()
        resp = client.get("/api/screeners")
        assert resp.status_code == 503

    def test_wraps_list_in_dict(self):
        """If R2 returns a list, it should be wrapped."""
        r2 = _make_r2_client({"screeners.json": [{"symbol": "AAPL"}]})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/screeners")
        assert resp.status_code == 200
        assert "data" in resp.json()


class TestBacktestEndpoint:
    def test_returns_backtest_data(self):
        data = {"strategies": [{"name": "trend_pulse", "sharpe": 1.5}]}
        r2 = _make_r2_client({"strategies.json": data})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/backtest")
        assert resp.status_code == 200
        assert resp.json()["strategies"][0]["name"] == "trend_pulse"

    def test_503_when_no_data(self):
        r2 = _make_r2_client({})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/backtest")
        assert resp.status_code == 503


class TestCachedProxy:
    def test_caches_result(self):
        r2 = _make_r2_client({"key.json": {"value": 42}})
        proxy = _CachedProxy(r2, ttl_sec=60)

        result1 = proxy.get("key.json")
        result2 = proxy.get("key.json")

        assert result1 == {"value": 42}
        assert result2 == {"value": 42}
        # Should only call R2 once due to cache
        assert r2.get_json.call_count == 1

    def test_returns_none_without_r2(self):
        proxy = _CachedProxy(None, ttl_sec=60)
        assert proxy.get("anything") is None

    def test_returns_stale_on_error(self):
        r2 = MagicMock()
        call_count = 0

        def get_json(key):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"fresh": True}
            raise RuntimeError("R2 down")

        r2.get_json = get_json
        proxy = _CachedProxy(r2, ttl_sec=0)  # TTL=0 so cache expires immediately

        result1 = proxy.get("key.json")
        assert result1 == {"fresh": True}

        # Second call: R2 fails, returns stale cache
        result2 = proxy.get("key.json")
        assert result2 == {"fresh": True}
