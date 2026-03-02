"""Tests for /api/screeners and /api/backtest routes (R2 proxy)."""

from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server.routes.screeners import (
    _CachedProxy,
    _compute_signal_data,
    create_screeners_router,
)


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

    def test_200_empty_when_no_data(self):
        r2 = _make_r2_client({})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/screeners")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_200_empty_when_no_r2_client(self):
        client = _make_app()
        resp = client.get("/api/screeners")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_wraps_list_in_dict(self):
        """If R2 returns a list, it should be wrapped."""
        r2 = _make_r2_client({"screeners.json": [{"symbol": "AAPL"}]})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/screeners")
        assert resp.status_code == 200
        assert "data" in resp.json()


class TestCacheInjectionServesData:
    """Verify the full loop: inject into cache → GET /api/screeners returns it."""

    def test_injected_momentum_served_by_screeners_endpoint(self):
        """set_cache → GET /api/screeners returns injected data."""
        proxy = _CachedProxy(None, ttl_sec=60)
        app = FastAPI()
        app.include_router(create_screeners_router(proxy=proxy))

        # Inject momentum data (simulating job completion callback)
        proxy.merge_cache("screeners.json", {
            "momentum": {"candidates": [{"symbol": "AAPL"}], "universe_size": 100}
        })

        client = TestClient(app)
        resp = client.get("/api/screeners")
        assert resp.status_code == 200
        data = resp.json()
        assert data["momentum"]["candidates"][0]["symbol"] == "AAPL"
        assert data["momentum"]["universe_size"] == 100

    def test_merge_preserves_other_section(self):
        """Injecting momentum preserves existing pead data."""
        proxy = _CachedProxy(None, ttl_sec=60)
        app = FastAPI()
        app.include_router(create_screeners_router(proxy=proxy))

        # Pre-populate with pead data
        proxy.set_cache("screeners.json", {
            "pead": {"candidates": [{"symbol": "NVDA"}]}
        })

        # Inject momentum — pead should survive
        proxy.merge_cache("screeners.json", {
            "momentum": {"candidates": [{"symbol": "AAPL"}]}
        })

        client = TestClient(app)
        resp = client.get("/api/screeners")
        assert resp.status_code == 200
        data = resp.json()
        assert data["momentum"]["candidates"][0]["symbol"] == "AAPL"
        assert data["pead"]["candidates"][0]["symbol"] == "NVDA"


class TestBacktestEndpoint:
    def test_returns_backtest_data(self):
        data = {"strategies": [{"name": "trend_pulse", "sharpe": 1.5}]}
        r2 = _make_r2_client({"strategies.json": data})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/backtest")
        assert resp.status_code == 200
        assert resp.json()["strategies"][0]["name"] == "trend_pulse"

    def test_200_empty_when_no_data(self):
        r2 = _make_r2_client({})
        client = _make_app(r2_client=r2)
        resp = client.get("/api/backtest")
        assert resp.status_code == 200
        assert resp.json() == {}


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

    def test_set_cache_injects_data(self):
        proxy = _CachedProxy(None, ttl_sec=60)
        proxy.set_cache("key.json", {"injected": True})
        assert proxy.get("key.json") == {"injected": True}

    def test_merge_cache_preserves_existing(self):
        proxy = _CachedProxy(None, ttl_sec=60)
        proxy.set_cache("screeners.json", {"momentum": {"old": 1}, "pead": {"data": 2}})
        proxy.merge_cache("screeners.json", {"momentum": {"new": 3}})
        result = proxy.get("screeners.json")
        assert result["momentum"] == {"new": 3}
        assert result["pead"] == {"data": 2}

    def test_merge_cache_creates_if_missing(self):
        proxy = _CachedProxy(None, ttl_sec=60)
        proxy.merge_cache("screeners.json", {"momentum": {"data": 1}})
        result = proxy.get("screeners.json")
        assert result == {"momentum": {"data": 1}}

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


# ── Helpers for _compute_signal_data tests ────────────


def _make_mock_indicator(name: str, warmup: int = 5, is_strategy: bool = False):
    """Create a mock indicator with calculate() and get_state()."""
    ind = MagicMock()
    ind.name = name
    ind.warmup_periods = warmup
    ind.default_params = {}

    def calculate(df, params):
        """Return a DataFrame with one synthetic column."""
        result = pd.DataFrame(index=df.index)
        result[f"{name}_value"] = range(len(df))
        return result

    ind.calculate = MagicMock(side_effect=calculate)

    if is_strategy:
        def get_state(current, previous, params):
            """Return a state dict matching the indicator type."""
            if name == "dual_macd":
                return {
                    "slow_histogram": float(current.get(f"{name}_value", 0)),
                    "fast_histogram": 0.1,
                    "slow_hist_delta": 0.05,
                    "fast_hist_delta": 0.02,
                    "trend_state": "BULLISH",
                    "tactical_signal": "HOLD",
                    "momentum_balance": 0.6,
                    "confidence": 0.8,
                }
            elif name == "trend_pulse":
                return {
                    "swing_signal": "BUY",
                    "entry_ok": True,
                    "macd_bullish": True,
                    "adx_strong": True,
                    "score": 85.0,
                    "confidence": 0.9,
                }
            elif name == "regime_detector":
                return {
                    "regime": "R0",
                    "regime_changed": False,
                    "previous_regime": "R0",
                    "confidence": 80,
                    "composite_score": 75.0,
                }
            return {"value": float(current.get(f"{name}_value", 0))}

        ind.get_state = MagicMock(side_effect=get_state)
    else:
        ind.get_state = MagicMock(return_value={})

    return ind


def _make_mock_pipeline(
    bar_count: int = 60,
    indicators: list | None = None,
    persistence=None,
):
    """Create a mock pipeline with indicator engine and optional persistence."""
    pipeline = MagicMock()

    # Generate synthetic bar history
    bars = deque()
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(bar_count):
        bars.append({
            "symbol": "SPY",
            "timeframe": "1d",
            "open": 500.0 + i,
            "high": 502.0 + i,
            "low": 499.0 + i,
            "close": 501.0 + i,
            "volume": 1000000 + i * 1000,
            "timestamp": base_ts.replace(day=min(i + 1, 28)),
        })

    engine = MagicMock()
    engine.get_history = MagicMock(return_value=bars)
    engine._indicators = indicators or []
    pipeline._indicator_engine = engine
    pipeline._persistence = persistence
    return pipeline


# ── _compute_signal_data strategy history tests ────────


class TestComputeSignalDataStrategyHistories:
    def test_signal_data_includes_strategy_histories(self):
        """_compute_signal_data returns all 3 strategy history arrays."""
        indicators = [
            _make_mock_indicator("dual_macd", warmup=5, is_strategy=True),
            _make_mock_indicator("trend_pulse", warmup=5, is_strategy=True),
            _make_mock_indicator("regime_detector", warmup=5, is_strategy=True),
            _make_mock_indicator("rsi", warmup=5, is_strategy=False),
        ]
        pipeline = _make_mock_pipeline(bar_count=60, indicators=indicators)
        result = _compute_signal_data(pipeline, "SPY", "1d")

        assert result is not None
        assert "dual_macd_history" in result
        assert "trend_pulse_history" in result
        assert "regime_flex_history" in result
        assert len(result["dual_macd_history"]) > 0
        assert len(result["trend_pulse_history"]) > 0
        assert len(result["regime_flex_history"]) > 0

    def test_dual_macd_history_shape(self):
        """DualMACD history rows have expected keys and types."""
        indicators = [
            _make_mock_indicator("dual_macd", warmup=5, is_strategy=True),
        ]
        pipeline = _make_mock_pipeline(bar_count=60, indicators=indicators)
        result = _compute_signal_data(pipeline, "SPY", "1d")

        assert result is not None
        history = result["dual_macd_history"]
        assert len(history) > 0
        row = history[0]
        assert "date" in row
        assert "slow_histogram" in row
        assert "trend_state" in row
        assert isinstance(row["slow_histogram"], float)
        assert isinstance(row["trend_state"], str)

    def test_regime_flex_history_mapping(self):
        """Regime history rows use short codes and correct exposure values."""
        indicators = [
            _make_mock_indicator("regime_detector", warmup=5, is_strategy=True),
        ]
        pipeline = _make_mock_pipeline(bar_count=60, indicators=indicators)
        result = _compute_signal_data(pipeline, "SPY", "1d")

        assert result is not None
        history = result["regime_flex_history"]
        assert len(history) > 0
        row = history[0]
        assert "date" in row
        assert "regime" in row
        assert "target_exposure" in row
        assert "signal" in row
        assert row["regime"] in ("R0", "R1", "R2", "R3")
        assert row["target_exposure"] in (0.0, 0.25, 0.5, 1.0)

    def test_signal_data_includes_persisted_signals(self):
        """Persisted signals from DuckDB appear in result."""
        from src.server.persistence import ServerPersistence

        persistence = ServerPersistence(duckdb_path=":memory:")
        base = datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc)
        for i in range(5):
            persistence.insert_signal(
                symbol="SPY",
                rule=f"rule_{i}",
                direction="bullish",
                strength=0.8,
                timeframe="1d",
                indicator="rsi",
                ts=base.replace(minute=i),
            )

        pipeline = _make_mock_pipeline(bar_count=60, persistence=persistence)
        result = _compute_signal_data(pipeline, "SPY", "1d")

        assert result is not None
        assert len(result["signals"]) == 5
        sig = result["signals"][0]
        assert "timestamp" in sig
        assert "rule" in sig
        assert "direction" in sig
        assert "indicator" in sig
        persistence.close()

    def test_history_limited_to_50_rows(self):
        """Strategy histories are capped at 50 entries (newest first)."""
        indicators = [
            _make_mock_indicator("dual_macd", warmup=2, is_strategy=True),
        ]
        # 200 bars → ~198 state rows after warmup, but capped to 50
        pipeline = _make_mock_pipeline(bar_count=200, indicators=indicators)
        result = _compute_signal_data(pipeline, "SPY", "1d")

        assert result is not None
        history = result["dual_macd_history"]
        assert len(history) <= 50

    def test_each_history_row_has_date(self):
        """Every strategy history row includes a date field."""
        indicators = [
            _make_mock_indicator("dual_macd", warmup=5, is_strategy=True),
            _make_mock_indicator("trend_pulse", warmup=5, is_strategy=True),
            _make_mock_indicator("regime_detector", warmup=5, is_strategy=True),
        ]
        pipeline = _make_mock_pipeline(bar_count=60, indicators=indicators)
        result = _compute_signal_data(pipeline, "SPY", "1d")

        assert result is not None
        for key in ("dual_macd_history", "trend_pulse_history", "regime_flex_history"):
            for row in result[key]:
                assert "date" in row, f"Missing 'date' in {key} row: {row}"

    def test_too_few_bars_returns_none(self):
        """Fewer than 10 bars returns None (not enough data)."""
        pipeline = _make_mock_pipeline(bar_count=5)
        result = _compute_signal_data(pipeline, "SPY", "1d")
        assert result is None
