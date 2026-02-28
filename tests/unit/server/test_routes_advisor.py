"""Tests for /api/advisor routes."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.services.advisor.models import (
    EquityAdvice,
    MarketContext,
)
from src.server.routes.advisor import create_advisor_router


def _make_mock_service():
    """Create a mock AdvisorService."""
    svc = MagicMock()
    svc.compute_all.return_value = {
        "market_context": MarketContext(
            regime="R0",
            regime_name="Healthy Uptrend",
            regime_confidence=75,
            vix=18.5,
            vix_percentile=42,
            vrp_zscore=0.82,
            term_structure_ratio=0.92,
            term_structure_state="contango",
            timestamp="2026-02-27T14:30:00Z",
        ),
        "premium": [],
        "equity": [],
        "timestamp": "2026-02-27T14:30:00Z",
    }
    svc.compute_symbol.return_value = {
        "market_context": svc.compute_all.return_value["market_context"],
        "equity": EquityAdvice(
            symbol="AAPL",
            sector="Technology",
            action="BUY",
            confidence=72,
            regime="R0",
            signal_summary={"bullish": 3, "bearish": 1, "neutral": 0},
            top_signals=[],
            trend_pulse=None,
            key_levels={},
            reasoning=["3 bullish signals"],
        ),
        "timestamp": "2026-02-27T14:30:00Z",
    }
    return svc


def _make_app(advisor_service=None):
    app = FastAPI()
    app.include_router(create_advisor_router(advisor_service=advisor_service))
    return TestClient(app)


class TestAdvisorListEndpoint:
    def test_returns_200(self):
        svc = _make_mock_service()
        client = _make_app(advisor_service=svc)
        resp = client.get("/api/advisor")
        assert resp.status_code == 200
        data = resp.json()
        assert "market_context" in data
        assert "premium" in data
        assert "equity" in data

    def test_503_when_no_service(self):
        client = _make_app()
        resp = client.get("/api/advisor")
        assert resp.status_code == 503

    def test_market_context_serialized(self):
        svc = _make_mock_service()
        client = _make_app(advisor_service=svc)
        resp = client.get("/api/advisor")
        ctx = resp.json()["market_context"]
        assert ctx["regime"] == "R0"
        assert ctx["vix"] == 18.5
        assert ctx["term_structure_state"] == "contango"

    def test_sector_filter(self):
        svc = _make_mock_service()
        svc.compute_all.return_value["equity"] = [
            EquityAdvice(
                symbol="AAPL",
                sector="Technology",
                action="BUY",
                confidence=72,
                regime="R0",
                signal_summary={"bullish": 3, "bearish": 1, "neutral": 0},
                top_signals=[],
                trend_pulse=None,
                key_levels={},
                reasoning=["bullish"],
            ),
            EquityAdvice(
                symbol="XOM",
                sector="Energy",
                action="HOLD",
                confidence=50,
                regime="R0",
                signal_summary={"bullish": 1, "bearish": 1, "neutral": 2},
                top_signals=[],
                trend_pulse=None,
                key_levels={},
                reasoning=["neutral"],
            ),
        ]
        client = _make_app(advisor_service=svc)
        resp = client.get("/api/advisor?sector=Technology")
        data = resp.json()
        assert len(data["equity"]) == 1
        assert data["equity"][0]["symbol"] == "AAPL"

    def test_action_filter(self):
        svc = _make_mock_service()
        svc.compute_all.return_value["equity"] = [
            EquityAdvice(
                symbol="AAPL",
                sector="Technology",
                action="BUY",
                confidence=72,
                regime="R0",
                signal_summary={"bullish": 3, "bearish": 1, "neutral": 0},
                top_signals=[],
                trend_pulse=None,
                key_levels={},
                reasoning=["bullish"],
            ),
            EquityAdvice(
                symbol="XOM",
                sector="Energy",
                action="HOLD",
                confidence=50,
                regime="R0",
                signal_summary={"bullish": 1, "bearish": 1, "neutral": 2},
                top_signals=[],
                trend_pulse=None,
                key_levels={},
                reasoning=["neutral"],
            ),
        ]
        client = _make_app(advisor_service=svc)
        resp = client.get("/api/advisor?action=HOLD")
        data = resp.json()
        assert len(data["equity"]) == 1
        assert data["equity"][0]["symbol"] == "XOM"


class TestAdvisorSymbolEndpoint:
    def test_returns_200(self):
        svc = _make_mock_service()
        client = _make_app(advisor_service=svc)
        resp = client.get("/api/advisor/AAPL")
        assert resp.status_code == 200

    def test_uppercases_symbol(self):
        svc = _make_mock_service()
        client = _make_app(advisor_service=svc)
        client.get("/api/advisor/aapl")
        svc.compute_symbol.assert_called_once_with("AAPL")

    def test_503_when_no_service(self):
        client = _make_app()
        resp = client.get("/api/advisor/AAPL")
        assert resp.status_code == 503

    def test_equity_serialized(self):
        svc = _make_mock_service()
        client = _make_app(advisor_service=svc)
        resp = client.get("/api/advisor/AAPL")
        data = resp.json()
        eq = data["equity"]
        assert eq["symbol"] == "AAPL"
        assert eq["action"] == "BUY"
        assert eq["confidence"] == 72
        assert eq["reasoning"] == ["3 bullish signals"]
