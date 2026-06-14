"""fetch_signals: the read method the API snapshot/REST path needs.

The routes call ``repo.fetch_signals(ticker, since=...)`` and feed the rows to
``build_payload`` (which maps ``time``->``timestamp`` and normalises contract
fields). fetch_signals must therefore return plain dict rows straight off
ta_signals, newest first, optionally bounded by ``since``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.infrastructure.persistence.repositories.ta_signal_repository import (
    TASignalRepository,
)


class _FakeDB:
    """Records the query+params and returns canned rows (as asyncpg would)."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls: list[tuple] = []

    async def fetch(self, query: str, *args):
        self.calls.append((query, args))
        return self._rows


def _row(symbol: str = "AAPL") -> dict:
    return {
        "time": datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc),
        "signal_id": f"momentum:rsi:{symbol}:1m",
        "symbol": symbol,
        "timeframe": "1m",
        "category": "momentum",
        "indicator": "rsi",
        "direction": "LONG",
        "strength": 72,
        "priority": "high",
        "trigger_rule": "rsi_oversold_exit",
        "current_value": 31.4,
        "threshold": 30.0,
        "previous_value": 28.0,
        "message": "RSI exits oversold",
        "cooldown_until": None,
        "metadata": None,
    }


@pytest.mark.asyncio
async def test_fetch_signals_filters_by_symbol_newest_first() -> None:
    db = _FakeDB([_row()])
    repo = TASignalRepository(db)

    rows = await repo.fetch_signals("AAPL")

    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["symbol"] == "AAPL"
    query, params = db.calls[0]
    assert "FROM ta_signals" in query
    assert "symbol = $1" in query
    assert "ORDER BY time DESC" in query  # snapshot: most recent first
    assert params[0] == "AAPL"


@pytest.mark.asyncio
async def test_fetch_signals_applies_since_bound() -> None:
    db = _FakeDB([_row()])
    repo = TASignalRepository(db)
    since = datetime(2026, 6, 14, 0, 0, tzinfo=timezone.utc)

    await repo.fetch_signals("AAPL", since=since)

    query, params = db.calls[0]
    assert "time > $2" in query
    assert "ORDER BY time ASC" in query  # backfill: contiguous, oldest-first from cursor
    assert params[0] == "AAPL"
    assert params[1] == since
