"""Tests for DuckDB persistence — tick/bar/signal buffering + flush."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

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

    def test_new_db_has_timeframe_indicator_columns(self, store):
        """Fresh DB includes timeframe and indicator columns on signals table."""
        store.insert_signal(
            symbol="AAPL",
            rule="rsi_oversold",
            direction="bullish",
            strength=0.8,
            timeframe="1d",
            indicator="rsi",
            ts=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
        )
        signals = store.query_signals(symbol="AAPL")
        assert len(signals) == 1
        assert signals[0]["timeframe"] == "1d"
        assert signals[0]["indicator"] == "rsi"

    def test_migration_adds_columns_to_existing_db(self, tmp_path):
        """Opening a DB with old signals schema (no timeframe/indicator) should auto-migrate."""
        import duckdb

        db_path = str(tmp_path / "old.duckdb")

        # Simulate an OLD database: signals table without timeframe/indicator columns
        conn = duckdb.connect(db_path)
        conn.execute(
            "CREATE TABLE signals (symbol VARCHAR, rule VARCHAR, "
            "direction VARCHAR, strength DOUBLE, ts TIMESTAMPTZ)"
        )
        conn.execute(
            "INSERT INTO signals VALUES ('SPY', 'macd_cross', 'bearish', 0.6, "
            "'2026-03-02T16:00:00+00:00')"
        )
        conn.close()

        # Open with ServerPersistence — should migrate without crashing
        store = ServerPersistence(duckdb_path=db_path)

        # Verify old data survived and new columns exist with defaults
        signals = store.query_signals(symbol="SPY")
        assert len(signals) == 1
        assert signals[0]["timeframe"] == ""
        assert signals[0]["indicator"] == ""

        # Verify we can insert with the new columns
        store.insert_signal(
            symbol="AAPL",
            rule="rsi_oversold",
            direction="bullish",
            strength=0.8,
            timeframe="1d",
            indicator="rsi",
            ts=datetime(2026, 3, 2, 16, 0, tzinfo=timezone.utc),
        )
        aapl = store.query_signals(symbol="AAPL")
        assert len(aapl) == 1
        assert aapl[0]["timeframe"] == "1d"
        assert aapl[0]["indicator"] == "rsi"

        store.close()

    def test_migration_is_idempotent(self, store):
        """Calling _migrate_signals_columns multiple times does not crash."""
        store._migrate_signals_columns()
        store._migrate_signals_columns()
        # Should still work fine
        store.insert_signal(
            symbol="AAPL",
            rule="test",
            direction="bullish",
            strength=0.5,
            timeframe="1d",
            indicator="rsi",
            ts=datetime(2026, 3, 2, tzinfo=timezone.utc),
        )
        assert len(store.query_signals(symbol="AAPL")) == 1

    def test_query_signals_filters_by_timeframe(self, store):
        """Timeframe filtering returns only matching signals."""
        base = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        store.insert_signal("AAPL", "rsi", "bullish", 0.8, base, timeframe="1d", indicator="rsi")
        store.insert_signal(
            "AAPL", "macd", "bearish", 0.6, base.replace(minute=1), timeframe="1d", indicator="macd"
        )
        store.insert_signal(
            "AAPL", "rsi", "bullish", 0.7, base.replace(minute=2), timeframe="1h", indicator="rsi"
        )

        daily = store.query_signals(symbol="AAPL", timeframe="1d")
        assert len(daily) == 2

        hourly = store.query_signals(symbol="AAPL", timeframe="1h")
        assert len(hourly) == 1

        all_signals = store.query_signals(symbol="AAPL")
        assert len(all_signals) == 3


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


class TestConcurrentAccess:
    """Stress test: concurrent inserts + queries must not SIGSEGV."""

    def test_concurrent_insert_and_query(self, store):
        """Spawn threads that simultaneously insert and query — no crash."""
        errors: list = []
        symbols = ["AAPL", "SPY", "QQQ", "TSLA"]

        def writer(sym: str, n: int) -> None:
            for i in range(n):
                try:
                    store.insert_bar(
                        sym,
                        "1d",
                        100 + i,
                        101 + i,
                        99 + i,
                        100.5 + i,
                        1000 * (i + 1),
                        datetime(2026, 1, 1 + (i % 28), tzinfo=timezone.utc),
                    )
                    store.insert_signal(
                        sym,
                        "rsi",
                        "bullish",
                        0.5 + i * 0.01,
                        datetime(2026, 1, 1 + (i % 28), 14, i % 60, tzinfo=timezone.utc),
                        timeframe="1d",
                        indicator="rsi",
                    )
                except Exception as exc:
                    errors.append(exc)

        def reader(sym: str, n: int) -> None:
            for _ in range(n):
                try:
                    store.query_bars(sym, "1d", limit=50)
                    store.query_signals(symbol=sym, limit=50)
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = []
            for sym in symbols:
                futs.append(pool.submit(writer, sym, 20))
                futs.append(pool.submit(reader, sym, 20))
            for f in as_completed(futs):
                f.result()  # re-raises if thread crashed

        assert not errors, f"Concurrent access errors: {errors}"
        # Verify data was actually written
        for sym in symbols:
            assert len(store.query_bars(sym, "1d", limit=100)) > 0

    def test_concurrent_tick_buffer_and_flush(self, store):
        """Concurrent tick buffering + flushes must not corrupt data."""
        errors: list = []

        def buffer_ticks(n: int) -> None:
            for i in range(n):
                try:
                    store.buffer_tick(
                        QuoteTick(
                            symbol="AAPL",
                            last=180 + i * 0.01,
                            volume=100,
                            source="test",
                            timestamp=datetime(2026, 1, 1, 10, 0, i % 60, tzinfo=timezone.utc),
                        )
                    )
                except Exception as exc:
                    errors.append(exc)

        def flusher(n: int) -> None:
            for _ in range(n):
                try:
                    store.flush_to_duckdb()
                except Exception as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [
                pool.submit(buffer_ticks, 50),
                pool.submit(buffer_ticks, 50),
                pool.submit(flusher, 20),
                pool.submit(flusher, 20),
            ]
            for f in as_completed(futs):
                f.result()

        store.flush_to_duckdb()  # drain remaining
        assert not errors
        rows = store.query_ticks("AAPL", limit=200)
        assert len(rows) == 100
