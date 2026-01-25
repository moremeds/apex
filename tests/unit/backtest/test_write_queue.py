"""Tests for thread-safe write queue."""

from __future__ import annotations

import threading
import time
from typing import Any

from src.backtest.data.storage.write_queue import (
    WriteOperation,
    WriteQueue,
    WriterConfig,
    WriteRequest,
    WriterStats,
)


class MockDatabaseManager:
    """Mock database manager for testing."""

    def __init__(self) -> None:
        self.inserted_records: list[tuple[str, Any]] = []
        self.conn = self  # Self-reference for conn.execute

    def execute(self, query: str, params: Any = None) -> None:
        """Track executed queries."""

    def insert_batch(self, table: str, records: list) -> int:
        """Track inserted records."""
        self.inserted_records.extend([(table, r) for r in records])
        return len(records)


class TestWriterConfig:
    """Tests for WriterConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = WriterConfig()
        assert config.batch_size == 100
        assert config.batch_timeout_seconds == 1.0
        assert config.max_queue_size == 10000
        assert config.retry_on_error is True
        assert config.max_retries == 3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WriterConfig(
            batch_size=50,
            batch_timeout_seconds=0.5,
            max_retries=5,
        )
        assert config.batch_size == 50
        assert config.batch_timeout_seconds == 0.5
        assert config.max_retries == 5


class TestWriterStats:
    """Tests for WriterStats."""

    def test_default_values(self) -> None:
        """Test default stats values."""
        stats = WriterStats()
        assert stats.total_writes == 0
        assert stats.total_batches == 0
        assert stats.total_errors == 0
        assert stats.records_written == 0


class TestWriteRequest:
    """Tests for WriteRequest."""

    def test_insert_request(self) -> None:
        """Test creating insert request."""
        request = WriteRequest(
            operation=WriteOperation.INSERT,
            table="runs",
            data={"run_id": "run_123", "sharpe": 1.5},
        )
        assert request.operation == WriteOperation.INSERT
        assert request.table == "runs"
        assert request.data["run_id"] == "run_123"

    def test_upsert_request(self) -> None:
        """Test creating upsert request."""
        request = WriteRequest(
            operation=WriteOperation.UPSERT,
            table="trials",
            data={"trial_id": "trial_456", "score": 0.8},
            key_columns=["trial_id"],
        )
        assert request.operation == WriteOperation.UPSERT
        assert request.key_columns == ["trial_id"]


class TestWriteQueue:
    """Tests for WriteQueue."""

    def test_start_stop(self) -> None:
        """Test starting and stopping the queue."""
        db = MockDatabaseManager()
        config = WriterConfig(batch_timeout_seconds=0.1)

        queue = WriteQueue(db, config)
        queue.start()

        # Should be running
        assert queue._writer_thread is not None
        assert queue._writer_thread.is_alive()

        queue.stop(timeout=2.0)

        # Should be stopped
        assert queue._writer_thread is None or not queue._writer_thread.is_alive()

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        db = MockDatabaseManager()
        config = WriterConfig(batch_timeout_seconds=0.1)

        with WriteQueue(db, config) as queue:
            assert queue._writer_thread is not None
            assert queue._writer_thread.is_alive()

        # After context, should be stopped
        time.sleep(0.2)

    def test_insert_single_record(self) -> None:
        """Test inserting a single record."""
        db = MockDatabaseManager()
        config = WriterConfig(
            batch_size=10,
            batch_timeout_seconds=0.1,
        )

        with WriteQueue(db, config) as queue:
            queue.insert("runs", {"run_id": "run_1", "sharpe": 1.0})
            queue.flush()

        assert len(db.inserted_records) == 1
        assert db.inserted_records[0][0] == "runs"
        assert db.inserted_records[0][1]["run_id"] == "run_1"

    def test_insert_multiple_records(self) -> None:
        """Test inserting multiple records."""
        db = MockDatabaseManager()
        config = WriterConfig(
            batch_size=100,
            batch_timeout_seconds=0.1,
        )

        with WriteQueue(db, config) as queue:
            for i in range(10):
                queue.insert("runs", {"run_id": f"run_{i}", "sharpe": float(i)})
            queue.flush()

        assert len(db.inserted_records) == 10
        assert queue.stats.records_written == 10

    def test_batch_flush_by_size(self) -> None:
        """Test that batch flushes when size threshold is reached."""
        db = MockDatabaseManager()
        config = WriterConfig(
            batch_size=5,  # Small batch size
            batch_timeout_seconds=60.0,  # Long timeout (won't trigger)
        )

        with WriteQueue(db, config) as queue:
            # Insert 6 records (more than batch size of 5)
            for i in range(6):
                queue.insert("runs", {"run_id": f"run_{i}"})

            # Give time for batch to be processed
            time.sleep(0.3)

            # At least first batch of 5 should be flushed
            assert len(db.inserted_records) >= 5

            queue.flush()

        # All 6 should be flushed now
        assert len(db.inserted_records) == 6

    def test_batch_flush_by_timeout(self) -> None:
        """Test that batch flushes when timeout is reached."""
        db = MockDatabaseManager()
        config = WriterConfig(
            batch_size=1000,  # Large batch size (won't trigger by size)
            batch_timeout_seconds=0.2,  # Short timeout
        )

        with WriteQueue(db, config) as queue:
            queue.insert("runs", {"run_id": "run_1"})

            # Wait for timeout
            time.sleep(0.5)

            # Should have flushed by timeout
            assert len(db.inserted_records) >= 1

    def test_pending_count(self) -> None:
        """Test pending count tracking."""
        db = MockDatabaseManager()
        config = WriterConfig(
            batch_size=100,
            batch_timeout_seconds=60.0,  # Long timeout
        )

        queue = WriteQueue(db, config)
        queue.start()

        try:
            # Add records directly to pending (bypassing queue for test)
            queue._add_to_batch(WriteRequest(WriteOperation.INSERT, "runs", {"id": "1"}))
            queue._add_to_batch(WriteRequest(WriteOperation.INSERT, "runs", {"id": "2"}))

            assert queue.pending_count == 2

            queue.flush()
            assert queue.pending_count == 0

        finally:
            queue.stop()

    def test_stats_tracking(self) -> None:
        """Test that statistics are tracked correctly."""
        db = MockDatabaseManager()
        config = WriterConfig(batch_timeout_seconds=0.1)

        with WriteQueue(db, config) as queue:
            for i in range(20):
                queue.insert("runs", {"run_id": f"run_{i}"})
            queue.flush()

        assert queue.stats.total_writes == 20
        assert queue.stats.records_written == 20
        assert queue.stats.total_batches >= 1

    def test_json_serialization(self) -> None:
        """Test that dict/list values are serialized to JSON."""
        db = MockDatabaseManager()
        config = WriterConfig(batch_timeout_seconds=0.1)

        with WriteQueue(db, config) as queue:
            queue.insert(
                "runs",
                {
                    "run_id": "run_1",
                    "params": {"fast": 10, "slow": 50},  # Dict should be JSON serialized
                    "symbols": ["AAPL", "MSFT"],  # List should be JSON serialized
                },
            )
            queue.flush()

        assert len(db.inserted_records) == 1
        record = db.inserted_records[0][1]
        # Check that dicts/lists are serialized
        assert isinstance(record["params"], str)
        assert isinstance(record["symbols"], str)

    def test_multiple_tables(self) -> None:
        """Test writing to multiple tables."""
        db = MockDatabaseManager()
        config = WriterConfig(batch_timeout_seconds=0.1)

        with WriteQueue(db, config) as queue:
            queue.insert("runs", {"run_id": "run_1"})
            queue.insert("trials", {"trial_id": "trial_1"})
            queue.insert("experiments", {"experiment_id": "exp_1"})
            queue.flush()

        # Check records from different tables
        tables = [r[0] for r in db.inserted_records]
        assert "runs" in tables
        assert "trials" in tables
        assert "experiments" in tables

    def test_concurrent_inserts(self) -> None:
        """Test that concurrent inserts are handled safely."""
        db = MockDatabaseManager()
        config = WriterConfig(
            batch_size=50,
            batch_timeout_seconds=0.5,
        )

        with WriteQueue(db, config) as queue:
            # Simulate concurrent inserts from multiple threads
            def insert_records(thread_id: int, count: int) -> None:
                for i in range(count):
                    queue.insert(
                        "runs",
                        {
                            "run_id": f"run_{thread_id}_{i}",
                            "thread": thread_id,
                        },
                    )

            threads = []
            for t in range(4):
                thread = threading.Thread(target=insert_records, args=(t, 25))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            queue.flush()

        # All 100 records (4 threads x 25 each) should be written
        assert len(db.inserted_records) == 100
