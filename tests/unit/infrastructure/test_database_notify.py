"""Tests for Database NOTIFY/LISTEN helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.persistence.database import Database


@pytest.fixture
def mock_db():
    """Create a Database instance with mocked pool."""
    from config.models import DatabaseConfig, DatabasePoolConfig

    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="test",
        user="test",
        password="",
        pool=DatabasePoolConfig(),
    )
    db = Database(config)
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = mock_ctx
    db._pool = mock_pool
    db._connected = True
    db._mock_conn = mock_conn
    return db


@pytest.mark.asyncio
async def test_notify_sends_pg_notify(mock_db):
    """notify() executes pg_notify with channel and JSON payload."""
    await mock_db.notify("apex_signal", {"symbol": "AAPL", "rule": "macd_cross"})

    mock_db._mock_conn.execute.assert_called_once()
    call_args = mock_db._mock_conn.execute.call_args[0]
    assert "pg_notify" in call_args[0]
    assert call_args[1] == "apex_signal"


@pytest.mark.asyncio
async def test_notify_truncates_large_payload(mock_db):
    """notify() truncates payload exceeding 7900 bytes."""
    await mock_db.notify("apex_signal", {"data": "x" * 8000})

    call_args = mock_db._mock_conn.execute.call_args[0]
    sent_payload = call_args[2]
    assert len(sent_payload) <= 7900
