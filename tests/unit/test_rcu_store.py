"""
Tests for OPT-014: RCU (Read-Copy-Update) pattern for lock-free reads.

Tests the RCUDict and RCUList classes that provide:
- Lock-free reads (no contention)
- Copy-on-write updates (atomic reference swap)
- Thread-safe concurrent access
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

import pytest

from src.infrastructure.stores.rcu_store import RCUDict, RCUList


class TestRCUDict:
    """Tests for RCUDict."""

    def test_basic_get_set(self):
        """Test basic get/set operations."""
        store = RCUDict[str, int]()

        store.set("a", 1)
        store.set("b", 2)

        assert store.get("a") == 1
        assert store.get("b") == 2
        assert store.get("c") is None
        assert store.get("c", 0) == 0

    def test_update_batch(self):
        """Test batch update is atomic."""
        store = RCUDict[str, int]()
        store.set("a", 1)

        store.update({"b": 2, "c": 3, "a": 10})

        assert store.get("a") == 10
        assert store.get("b") == 2
        assert store.get("c") == 3

    def test_empty_update(self):
        """Test empty update does nothing."""
        store = RCUDict[str, int]()
        store.set("a", 1)
        initial_version = store.version

        store.update({})

        assert store.version == initial_version
        assert store.get("a") == 1

    def test_delete(self):
        """Test delete operation."""
        store = RCUDict[str, int]()
        store.set("a", 1)
        store.set("b", 2)

        assert store.delete("a") is True
        assert store.delete("c") is False
        assert store.get("a") is None
        assert store.get("b") == 2

    def test_clear(self):
        """Test clear operation."""
        store = RCUDict[str, int]()
        store.update({"a": 1, "b": 2, "c": 3})

        store.clear()

        assert len(store) == 0
        assert store.get("a") is None

    def test_len(self):
        """Test length operation."""
        store = RCUDict[str, int]()

        assert len(store) == 0

        store.set("a", 1)
        assert len(store) == 1

        store.update({"b": 2, "c": 3})
        assert len(store) == 3

    def test_contains(self):
        """Test membership test."""
        store = RCUDict[str, int]()
        store.set("a", 1)

        assert "a" in store
        assert "b" not in store

    def test_iter(self):
        """Test iteration over keys."""
        store = RCUDict[str, int]()
        store.update({"a": 1, "b": 2, "c": 3})

        keys = list(store)
        assert set(keys) == {"a", "b", "c"}

    def test_keys_values_items(self):
        """Test keys, values, items methods."""
        store = RCUDict[str, int]()
        store.update({"a": 1, "b": 2})

        assert set(store.keys()) == {"a", "b"}
        assert set(store.values()) == {1, 2}
        assert set(store.items()) == {("a", 1), ("b", 2)}

    def test_get_all(self):
        """Test get_all returns reference to internal dict."""
        store = RCUDict[str, int]()
        store.update({"a": 1, "b": 2})

        all_data = store.get_all()
        assert all_data == {"a": 1, "b": 2}

    def test_get_many(self):
        """Test get_many for multiple keys."""
        store = RCUDict[str, int]()
        store.update({"a": 1, "b": 2, "c": 3})

        result = store.get_many(["a", "c", "d"])
        assert result == {"a": 1, "c": 3}

    def test_compute_if_absent(self):
        """Test compute_if_absent for lazy initialization."""
        store = RCUDict[str, int]()
        factory = MagicMock(return_value=42)

        # First call creates value
        result = store.compute_if_absent("a", factory)
        assert result == 42
        factory.assert_called_once()

        # Second call returns existing value
        factory.reset_mock()
        result = store.compute_if_absent("a", factory)
        assert result == 42
        factory.assert_not_called()

    def test_version_tracking(self):
        """Test version increments on writes."""
        store = RCUDict[str, int]()
        assert store.version == 0

        store.set("a", 1)
        assert store.version == 1

        store.update({"b": 2})
        assert store.version == 2

        store.delete("a")
        assert store.version == 3

        store.clear()
        assert store.version == 4

    def test_initial_data(self):
        """Test initialization with data."""
        initial = {"a": 1, "b": 2}
        store = RCUDict[str, int](initial)

        assert store.get("a") == 1
        assert store.get("b") == 2

        # Verify initial data is copied
        initial["c"] = 3
        assert store.get("c") is None

    def test_concurrent_reads_no_contention(self):
        """Test concurrent reads don't block each other."""
        store = RCUDict[str, int]()
        store.update({f"key{i}": i for i in range(1000)})

        results: List[int] = []
        errors: List[Exception] = []

        def reader(key: str):
            try:
                for _ in range(100):
                    value = store.get(key)
                    if value is not None:
                        results.append(value)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(f"key{i % 100}",)) for i in range(20)]

        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        assert not errors, f"Errors during concurrent reads: {errors}"
        assert len(results) == 2000  # 20 threads * 100 iterations
        assert elapsed < 2.0  # Should complete quickly with no contention

    def test_concurrent_read_write(self):
        """Test concurrent reads during writes see consistent state."""
        store = RCUDict[str, int]()
        store.set("counter", 0)

        read_values: List[int] = []
        errors: List[Exception] = []

        def writer():
            try:
                for i in range(100):
                    store.set("counter", i)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    value = store.get("counter")
                    if value is not None:
                        read_values.append(value)
                    time.sleep(0.0005)
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(5)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        assert not errors, f"Errors during concurrent read/write: {errors}"
        # All read values should be valid (0-99 range)
        assert all(0 <= v <= 99 for v in read_values)


class TestRCUList:
    """Tests for RCUList."""

    def test_basic_operations(self):
        """Test basic list operations."""
        store = RCUList[int]()

        assert len(store) == 0

        store.append(1)
        store.append(2)

        assert len(store) == 2
        assert list(store) == [1, 2]

    def test_get_all(self):
        """Test get_all returns reference."""
        store = RCUList[int]()
        store.append(1)
        store.append(2)

        all_data = store.get_all()
        assert all_data == [1, 2]

    def test_set_all(self):
        """Test set_all replaces data atomically."""
        store = RCUList[int]()
        store.append(1)

        store.set_all([10, 20, 30])

        assert list(store) == [10, 20, 30]

    def test_clear(self):
        """Test clear operation."""
        store = RCUList[int]()
        store.set_all([1, 2, 3])

        store.clear()

        assert len(store) == 0

    def test_version_tracking(self):
        """Test version increments on writes."""
        store = RCUList[int]()
        assert store.version == 0

        store.append(1)
        assert store.version == 1

        store.set_all([1, 2, 3])
        assert store.version == 2

        store.clear()
        assert store.version == 3

    def test_initial_data(self):
        """Test initialization with data."""
        initial = [1, 2, 3]
        store = RCUList[int](initial)

        assert list(store) == [1, 2, 3]

        # Verify initial data is copied
        initial.append(4)
        assert list(store) == [1, 2, 3]


class TestMarketDataStoreRCU:
    """Tests for MarketDataStore with RCU pattern."""

    @pytest.fixture
    def store(self):
        """Create a test store."""
        from src.infrastructure.stores.market_data_store import MarketDataStore
        return MarketDataStore(price_ttl_seconds=5, greeks_ttl_seconds=60)

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        from src.models.market_data import MarketData
        from src.utils.timezone import now_utc

        return MarketData(
            symbol="AAPL",
            bid=150.0,
            ask=150.10,
            mid=150.05,
            last=150.05,
            timestamp=now_utc(),
        )

    def test_upsert_and_get(self, store, mock_market_data):
        """Test upsert and get operations."""
        store.upsert([mock_market_data])

        result = store.get("AAPL")
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.bid == 150.0

    def test_get_nonexistent(self, store):
        """Test get returns None for nonexistent symbol."""
        result = store.get("NONEXISTENT")
        assert result is None

    def test_get_all(self, store, mock_market_data):
        """Test get_all returns all data."""
        from src.models.market_data import MarketData
        from src.utils.timezone import now_utc

        md2 = MarketData(symbol="GOOG", bid=100.0, ask=100.10, timestamp=now_utc())
        store.upsert([mock_market_data, md2])

        all_data = store.get_all()
        assert len(all_data) == 2
        assert "AAPL" in all_data
        assert "GOOG" in all_data

    def test_count(self, store, mock_market_data):
        """Test count operation."""
        assert store.count() == 0

        store.upsert([mock_market_data])
        assert store.count() == 1

    def test_clear(self, store, mock_market_data):
        """Test clear operation."""
        store.upsert([mock_market_data])
        assert store.count() == 1

        store.clear()
        assert store.count() == 0

    def test_has_fresh_data(self, store, mock_market_data):
        """Test has_fresh_data checks price TTL."""
        assert store.has_fresh_data("AAPL") is False

        store.upsert([mock_market_data])
        assert store.has_fresh_data("AAPL") is True

    def test_concurrent_upsert_and_get(self, store):
        """Test concurrent upsert and get are thread-safe."""
        from src.models.market_data import MarketData
        from src.utils.timezone import now_utc

        errors: List[Exception] = []
        read_count = [0]

        def writer():
            try:
                for i in range(50):
                    md = MarketData(
                        symbol=f"SYM{i % 10}",
                        bid=float(i),
                        ask=float(i) + 0.1,
                        timestamp=now_utc(),
                    )
                    store.upsert([md])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    for j in range(10):
                        result = store.get(f"SYM{j}")
                        if result is not None:
                            read_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(2)]
        threads += [threading.Thread(target=reader) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"
        assert read_count[0] > 0  # Should have read some data


class TestPositionStoreRCU:
    """Tests for PositionStore with RCU pattern."""

    @pytest.fixture
    def store(self):
        """Create a test store."""
        from src.infrastructure.stores.position_store import PositionStore
        return PositionStore()

    @pytest.fixture
    def mock_position(self):
        """Create mock position."""
        from src.models.position import Position
        return Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            asset_type="STK",
            underlying="AAPL",
            source="test",
        )

    def test_upsert_and_get_all(self, store, mock_position):
        """Test upsert and get_all operations."""
        store.upsert_positions([mock_position])

        positions = store.get_all()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

    def test_get_by_key(self, store, mock_position):
        """Test get_by_key operation."""
        store.upsert_positions([mock_position])

        key = mock_position.key()
        result = store.get_by_key(key)
        assert result is not None
        assert result.symbol == "AAPL"

    def test_get_by_underlying(self, store):
        """Test get_by_underlying filters correctly."""
        from src.models.position import Position

        p1 = Position(symbol="AAPL", quantity=100, avg_price=150.0, asset_type="STK", underlying="AAPL", source="test")
        p2 = Position(symbol="AAPL_OPT", quantity=10, avg_price=5.0, asset_type="OPT", underlying="AAPL", source="test")
        p3 = Position(symbol="GOOG", quantity=50, avg_price=100.0, asset_type="STK", underlying="GOOG", source="test")

        store.upsert_positions([p1, p2, p3])

        aapl_positions = store.get_by_underlying("AAPL")
        assert len(aapl_positions) == 2
        assert all(p.underlying == "AAPL" for p in aapl_positions)

    def test_count(self, store, mock_position):
        """Test count operation."""
        assert store.count() == 0

        store.upsert_positions([mock_position])
        assert store.count() == 1

    def test_clear(self, store, mock_position):
        """Test clear operation."""
        store.upsert_positions([mock_position])
        store.clear()
        assert store.count() == 0

    def test_refresh_flag(self, store):
        """Test refresh flag operations."""
        assert store.needs_refresh() is False

        # Simulate position update event
        store._on_position_updated({"symbol": "AAPL"})
        assert store.needs_refresh() is True

        store.clear_refresh_flag()
        assert store.needs_refresh() is False

    def test_concurrent_access(self, store):
        """Test concurrent access is thread-safe."""
        from src.models.position import Position

        errors: List[Exception] = []
        read_count = [0]

        def writer():
            try:
                for i in range(50):
                    p = Position(
                        symbol=f"SYM{i % 10}",
                        quantity=float(i),
                        avg_price=100.0,
                        asset_type="STK",
                        underlying=f"SYM{i % 10}",
                        source="test",
                    )
                    store.upsert_positions([p])
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    positions = store.get_all()
                    read_count[0] += len(positions)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(2)]
        threads += [threading.Thread(target=reader) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"
