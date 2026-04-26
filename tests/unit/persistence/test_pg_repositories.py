"""Tests for PostgreSQL repositories.

Uses a real PostgreSQL connection. Skip if PG unavailable.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("APEX_PG_URL") and not os.environ.get("CI"),
    reason="No APEX_PG_URL set — skip PG tests locally unless CI",
)


@pytest.fixture
async def db():
    from config.models import DatabaseConfig, DatabasePoolConfig
    from src.infrastructure.persistence.database import Database
    from src.infrastructure.persistence.pg_schema import ensure_schema

    url = os.environ.get("APEX_PG_URL", "postgresql://chenxi@localhost:5432/apex")
    parts = url.replace("postgresql://", "").split("@")
    user_part = parts[0]
    host_part = parts[1]
    user = user_part.split(":")[0] if ":" in user_part else user_part
    password = user_part.split(":")[1] if ":" in user_part else ""
    host_db = host_part.split("/")
    host_port = host_db[0].split(":")
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 5432
    database = host_db[1] if len(host_db) > 1 else "apex"

    config = DatabaseConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        pool=DatabasePoolConfig(min_connections=1, max_connections=2),
    )
    db = Database(config)
    await db.connect()
    await ensure_schema(db)

    for table in (
        "bars",
        "signals",
        "summary",
        "score_history",
        "screener_results",
        "backtest_results",
    ):
        await db.execute(f"DELETE FROM {table}")  # noqa: S608

    yield db
    await db.close()


@pytest.fixture
def repo(db):
    from src.infrastructure.persistence.pg_repositories import PgRepositories

    return PgRepositories(db)


class TestBarRepository:
    async def test_insert_and_query_bar(self, repo):
        ts = datetime(2025, 1, 15, 16, 0, tzinfo=timezone.utc)
        await repo.insert_bar("AAPL", "1d", 150.0, 155.0, 149.0, 153.0, 1000000, ts)
        rows = await repo.query_bars("AAPL", "1d", limit=10)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["c"] == 153.0

    async def test_insert_bar_upsert_on_conflict(self, repo):
        ts = datetime(2025, 1, 15, 16, 0, tzinfo=timezone.utc)
        await repo.insert_bar("AAPL", "1d", 150.0, 155.0, 149.0, 153.0, 1000000, ts)
        await repo.insert_bar("AAPL", "1d", 151.0, 156.0, 150.0, 154.0, 2000000, ts)
        rows = await repo.query_bars("AAPL", "1d", limit=10)
        assert len(rows) == 1
        assert rows[0]["c"] == 154.0

    async def test_bulk_insert_bars(self, repo):
        bars = [
            {
                "timestamp": datetime(2025, 1, i, 16, 0, tzinfo=timezone.utc),
                "open": 150.0 + i,
                "high": 155.0 + i,
                "low": 149.0 + i,
                "close": 153.0 + i,
                "volume": 1000000 + i * 100,
            }
            for i in range(1, 6)
        ]
        count = await repo.bulk_insert_bars("AAPL", "1d", bars)
        assert count == 5
        rows = await repo.query_bars("AAPL", "1d", limit=10)
        assert len(rows) == 5


class TestSignalRepository:
    async def test_insert_and_query_signal(self, repo):
        ts = datetime(2025, 1, 15, 16, 0, tzinfo=timezone.utc)
        await repo.insert_signal("AAPL", "rsi_oversold", "bullish", 0.8, ts, "1d", "rsi")
        rows = await repo.query_signals(symbol="AAPL", limit=10)
        assert len(rows) == 1
        assert rows[0]["rule"] == "rsi_oversold"
        assert rows[0]["direction"] == "bullish"


class TestSummaryRepository:
    async def test_save_and_get_summary(self, repo):
        summary = {
            "tickers": [
                {
                    "symbol": "AAPL",
                    "close": 153.0,
                    "prev_close": 150.0,
                    "daily_change_pct": 2.0,
                    "regime": "R0",
                    "regime_name": "Healthy Uptrend",
                    "confidence": 80,
                    "composite_score_avg": 75.5,
                    "sector": "Technology",
                    "component_states": {"trend": "up"},
                }
            ],
            "generated_at": "2025-01-15T16:00:00",
        }
        await repo.save_summary(summary)
        result = await repo.get_summary()
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["regime"] == "R0"


class TestScoreHistoryRepository:
    async def test_save_and_get_score_snapshot(self, repo):
        ts = datetime(2025, 1, 15, 16, 0, tzinfo=timezone.utc)
        await repo.save_score_snapshot("AAPL", ts, 75.5, "up", "R0")
        history = await repo.get_score_history(days=30)
        assert len(history) >= 1


class TestScreenerResultsRepository:
    async def test_insert_and_query_screener_results(self, repo):
        results = [
            {"symbol": "AAPL", "score": 0.85, "metadata": {"momentum_12_1": 0.15}},
            {"symbol": "MSFT", "score": 0.72, "metadata": {"momentum_12_1": 0.10}},
        ]
        await repo.insert_screener_results("run-001", "momentum", results)
        rows = await repo.query_screener_results("run-001")
        assert len(rows) == 2


class TestBacktestResultsRepository:
    async def test_insert_and_query_backtest_results(self, repo):
        await repo.insert_backtest_results(
            "run-001",
            "trend_pulse",
            ["AAPL", "MSFT"],
            {"sharpe": 1.5, "max_dd": -0.12},
        )
        rows = await repo.query_backtest_results("run-001")
        assert len(rows) == 1
        assert rows[0]["strategy"] == "trend_pulse"
