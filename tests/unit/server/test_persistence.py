"""Tests for DuckDB persistence — tick/bar/signal buffering + flush."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.domain.events.domain_events import QuoteTick
from src.server.persistence import ServerPersistence


@pytest.fixture
def store():
    """In-memory DuckDB store."""
    s = ServerPersistence(duckdb_path=":memory:")
    yield s
    s.close()


class TestTickBuffering:
    def test_buffer_tick(self, store):
        tick = QuoteTick(
            symbol="AAPL",
            last=185.5,
            volume=1000,
            source="test",
            timestamp=datetime(2026, 2, 25, 14, 30, 0, tzinfo=timezone.utc),
        )
        store.buffer_tick(tick)
        assert store.pending_tick_count == 1

    def test_flush_to_duckdb(self, store):
        for i in range(5):
            store.buffer_tick(
                QuoteTick(
                    symbol="AAPL",
                    last=185.0 + i * 0.1,
                    volume=1000,
                    source="test",
                    timestamp=datetime(2026, 2, 25, 14, 30, i, tzinfo=timezone.utc),
                )
            )
        assert store.pending_tick_count == 5
        store.flush_to_duckdb()
        assert store.pending_tick_count == 0

    def test_query_ticks(self, store):
        for i in range(3):
            store.buffer_tick(
                QuoteTick(
                    symbol="AAPL",
                    last=185.0 + i,
                    volume=1000,
                    source="test",
                    timestamp=datetime(2026, 2, 25, 14, 30, i, tzinfo=timezone.utc),
                )
            )
        store.flush_to_duckdb()
        rows = store.query_ticks("AAPL", limit=10)
        assert len(rows) == 3
        assert rows[0]["price"] == 185.0

    def test_query_ticks_empty(self, store):
        rows = store.query_ticks("AAPL", limit=10)
        assert len(rows) == 0


class TestBarStorage:
    def test_insert_and_query_bars(self, store):
        store.insert_bar(
            symbol="AAPL",
            tf="1d",
            o=184.0,
            h=186.0,
            l=183.5,
            c=185.5,
            v=50000,
            ts=datetime(2026, 2, 25, tzinfo=timezone.utc),
        )
        store.insert_bar(
            symbol="AAPL",
            tf="1d",
            o=185.5,
            h=187.0,
            l=185.0,
            c=186.0,
            v=45000,
            ts=datetime(2026, 2, 26, tzinfo=timezone.utc),
        )
        bars = store.query_bars("AAPL", "1d", limit=10)
        assert len(bars) == 2

    def test_query_bars_filtered(self, store):
        store.insert_bar(
            "AAPL", "1d", 184, 186, 183, 185, 50000, datetime(2026, 2, 25, tzinfo=timezone.utc)
        )
        store.insert_bar(
            "AAPL",
            "1h",
            184,
            185,
            183,
            184.5,
            10000,
            datetime(2026, 2, 25, 14, 0, tzinfo=timezone.utc),
        )
        bars_1d = store.query_bars("AAPL", "1d", limit=10)
        bars_1h = store.query_bars("AAPL", "1h", limit=10)
        assert len(bars_1d) == 1
        assert len(bars_1h) == 1


class TestSignalStorage:
    def test_insert_and_query_signals(self, store):
        store.insert_signal(
            symbol="AAPL",
            rule="rsi_oversold",
            direction="long",
            strength=0.8,
            ts=datetime(2026, 2, 25, 14, 30, tzinfo=timezone.utc),
        )
        signals = store.query_signals(limit=10)
        assert len(signals) == 1
        assert signals[0]["symbol"] == "AAPL"
        assert signals[0]["rule"] == "rsi_oversold"

    def test_query_signals_by_symbol(self, store):
        store.insert_signal(
            "AAPL", "rsi_oversold", "long", 0.8, datetime(2026, 2, 25, 14, 30, tzinfo=timezone.utc)
        )
        store.insert_signal(
            "SPY", "macd_cross", "short", 0.6, datetime(2026, 2, 25, 14, 31, tzinfo=timezone.utc)
        )
        aapl_signals = store.query_signals(symbol="AAPL", limit=10)
        assert len(aapl_signals) == 1
        all_signals = store.query_signals(limit=10)
        assert len(all_signals) == 2


class TestLifecycle:
    def test_close_and_reopen(self, tmp_path):
        db_path = str(tmp_path / "test.duckdb")
        store = ServerPersistence(duckdb_path=db_path)
        store.buffer_tick(
            QuoteTick(
                symbol="AAPL",
                last=185.5,
                volume=1000,
                source="test",
                timestamp=datetime(2026, 2, 25, tzinfo=timezone.utc),
            )
        )
        store.flush_to_duckdb()
        store.close()

        # Reopen — data should persist
        store2 = ServerPersistence(duckdb_path=db_path)
        rows = store2.query_ticks("AAPL", limit=10)
        assert len(rows) == 1
        store2.close()
