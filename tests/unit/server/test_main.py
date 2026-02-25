"""Tests for FastAPI app shell."""

from fastapi.testclient import TestClient

from src.server.main import create_app


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "uptime" in body


def test_static_fallback_404():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/nonexistent.xyz")
    assert resp.status_code == 404


def test_cors_headers():
    app = create_app()
    client = TestClient(app)
    resp = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
