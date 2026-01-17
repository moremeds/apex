"""
Thread-safe write queue for parallel database operations.

When running parallel backtests with ProcessPoolExecutor, multiple workers
may try to write results simultaneously. This module provides a thread-safe
queue-based writer that batches inserts for efficiency and prevents
database contention.

Key features:
- Background thread processes writes asynchronously
- Transaction batching for performance (configurable batch size and timeout)
- Thread-safe interface for multi-process scenarios
- Graceful shutdown with flush support
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class WriteOperation(Enum):
    """Types of write operations."""

    INSERT = "insert"
    UPDATE = "update"
    UPSERT = "upsert"
    FLUSH = "flush"  # Control message to trigger immediate flush
    SHUTDOWN = "shutdown"  # Control message to stop the writer


@dataclass
class WriteRequest:
    """A single write request to be queued."""

    operation: WriteOperation
    table: str
    data: Dict[str, Any]
    # For updates/upserts
    key_columns: Optional[List[str]] = None


@dataclass
class WriterConfig:
    """Configuration for the write queue."""

    batch_size: int = 100  # Maximum records per batch
    batch_timeout_seconds: float = 1.0  # Max wait time before flushing batch
    max_queue_size: int = 10000  # Maximum pending writes
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 0.5


@dataclass
class WriterStats:
    """Statistics for the write queue."""

    total_writes: int = 0
    total_batches: int = 0
    total_errors: int = 0
    total_retries: int = 0
    records_written: int = 0
    avg_batch_size: float = 0.0
    last_flush_time: Optional[float] = None


class WriteQueue:
    """
    Thread-safe write queue for parallel database operations.

    Uses a background thread to process writes asynchronously,
    batching them for efficiency.

    Example:
        from backtest.data.storage import DatabaseManager, WriteQueue

        db = DatabaseManager("results.db")
        db.initialize_schema()

        queue = WriteQueue(db)
        queue.start()

        # In parallel workers:
        queue.insert("runs", {"run_id": "run_123", "sharpe": 1.5})

        # When done:
        queue.flush()
        queue.stop()
    """

    def __init__(
        self,
        db_manager: Any,  # DatabaseManager - avoid circular import
        config: Optional[WriterConfig] = None,
    ):
        """
        Initialize write queue.

        Args:
            db_manager: DatabaseManager instance (must be initialized)
            config: Writer configuration
        """
        self.db = db_manager
        self.config = config or WriterConfig()
        self.stats = WriterStats()

        self._queue: queue.Queue[WriteRequest] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._lock = threading.Lock()

        # Pending batches by table
        self._pending: Dict[str, List[Dict[str, Any]]] = {}
        self._last_batch_time = time.time()

    def start(self) -> None:
        """Start the background writer thread."""
        if self._writer_thread is not None and self._writer_thread.is_alive():
            return

        self._stop_event.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="WriteQueue-Writer",
            daemon=True,
        )
        self._writer_thread.start()
        logger.info("Write queue started")

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the writer thread, flushing pending writes.

        Args:
            timeout: Maximum seconds to wait for thread to stop
        """
        if self._writer_thread is None:
            return

        # Send shutdown signal
        self._queue.put(WriteRequest(WriteOperation.SHUTDOWN, "", {}))
        self._stop_event.set()

        self._writer_thread.join(timeout=timeout)
        if self._writer_thread.is_alive():
            logger.warning("Write queue thread did not stop gracefully")
        else:
            logger.info(
                f"Write queue stopped. Stats: {self.stats.records_written} records, "
                f"{self.stats.total_batches} batches, {self.stats.total_errors} errors"
            )

        self._writer_thread = None

    def insert(self, table: str, data: Dict[str, Any]) -> None:
        """
        Queue an insert operation.

        Args:
            table: Target table name
            data: Record data as dictionary
        """
        request = WriteRequest(WriteOperation.INSERT, table, data)
        try:
            self._queue.put(request, timeout=5.0)
        except queue.Full:
            logger.error(f"Write queue full, dropping insert for {table}")
            self.stats.total_errors += 1

    def upsert(
        self, table: str, data: Dict[str, Any], key_columns: List[str]
    ) -> None:
        """
        Queue an upsert (insert or update) operation.

        Args:
            table: Target table name
            data: Record data as dictionary
            key_columns: Columns that form the primary key for conflict resolution
        """
        request = WriteRequest(
            WriteOperation.UPSERT, table, data, key_columns=key_columns
        )
        try:
            self._queue.put(request, timeout=5.0)
        except queue.Full:
            logger.error(f"Write queue full, dropping upsert for {table}")
            self.stats.total_errors += 1

    def flush(self, timeout: float = 30.0) -> bool:
        """
        Flush all pending writes to database.

        Blocks until flush is complete or timeout.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if flush completed successfully
        """
        self._flush_event.clear()
        self._queue.put(WriteRequest(WriteOperation.FLUSH, "", {}))

        # Wait for flush to complete
        completed = self._flush_event.wait(timeout=timeout)
        if not completed:
            logger.warning(f"Flush did not complete within {timeout}s")
        return completed

    def _writer_loop(self) -> None:
        """Main writer loop - runs in background thread."""
        logger.debug("Writer loop started")

        while not self._stop_event.is_set():
            try:
                # Get next request with timeout for batch flushing
                try:
                    request = self._queue.get(
                        timeout=self.config.batch_timeout_seconds
                    )
                except queue.Empty:
                    # Timeout - check if we need to flush pending batches
                    self._maybe_flush_batches()
                    continue

                # Handle control messages
                if request.operation == WriteOperation.SHUTDOWN:
                    self._flush_all_batches()
                    break

                if request.operation == WriteOperation.FLUSH:
                    self._flush_all_batches()
                    self._flush_event.set()
                    continue

                # Add to pending batch
                self._add_to_batch(request)

                # Check if we should flush
                self._maybe_flush_batches()

            except Exception as e:
                logger.error(f"Error in writer loop: {e}")
                self.stats.total_errors += 1

        logger.debug("Writer loop stopped")

    def _add_to_batch(self, request: WriteRequest) -> None:
        """Add a request to the pending batch for its table."""
        with self._lock:
            if request.table not in self._pending:
                self._pending[request.table] = []
            self._pending[request.table].append(request.data)
            self.stats.total_writes += 1

    def _maybe_flush_batches(self) -> None:
        """Flush batches if size or time threshold is reached."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_batch_time

            for table, records in list(self._pending.items()):
                should_flush = (
                    len(records) >= self.config.batch_size
                    or elapsed >= self.config.batch_timeout_seconds
                )
                if should_flush and records:
                    self._flush_batch(table, records)
                    self._pending[table] = []

            if elapsed >= self.config.batch_timeout_seconds:
                self._last_batch_time = now

    def _flush_all_batches(self) -> None:
        """Flush all pending batches immediately."""
        with self._lock:
            for table, records in list(self._pending.items()):
                if records:
                    self._flush_batch(table, records)
                    self._pending[table] = []
            self._last_batch_time = time.time()
            self.stats.last_flush_time = self._last_batch_time

    def _flush_batch(self, table: str, records: List[Dict[str, Any]]) -> None:
        """
        Write a batch of records to the database.

        Implements retry logic for transient failures.
        """
        if not records:
            return

        for attempt in range(self.config.max_retries + 1):
            try:
                # Serialize any dict/list values to JSON strings
                processed_records = []
                for record in records:
                    processed = {}
                    for key, value in record.items():
                        if isinstance(value, (dict, list)):
                            processed[key] = json.dumps(value)
                        else:
                            processed[key] = value
                    processed_records.append(processed)

                # Use transaction for atomicity
                self.db.conn.execute("BEGIN TRANSACTION")
                try:
                    count = self.db.insert_batch(table, processed_records)
                    self.db.conn.execute("COMMIT")

                    self.stats.records_written += count
                    self.stats.total_batches += 1
                    self.stats.avg_batch_size = (
                        self.stats.records_written / self.stats.total_batches
                    )

                    logger.debug(f"Flushed {count} records to {table}")
                    return

                except Exception as e:
                    self.db.conn.execute("ROLLBACK")
                    raise

            except Exception as e:
                self.stats.total_retries += 1
                if attempt < self.config.max_retries and self.config.retry_on_error:
                    logger.warning(
                        f"Batch write failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                    time.sleep(self.config.retry_delay_seconds * (2**attempt))
                else:
                    logger.error(
                        f"Failed to write batch to {table} after {attempt + 1} attempts: {e}"
                    )
                    self.stats.total_errors += 1
                    # Don't lose the records - could implement dead letter queue
                    return

    @property
    def pending_count(self) -> int:
        """Get count of pending writes across all tables."""
        with self._lock:
            return sum(len(records) for records in self._pending.values())

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def __enter__(self) -> "WriteQueue":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.flush()
        self.stop()
