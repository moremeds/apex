"""
Storage layer for systematic backtesting.

Uses DuckDB for fast analytical queries and Parquet for persistence.
Provides thread-safe write queue for parallel operations.
"""

from .database import DatabaseManager
from .repositories import (
    ExperimentRepository,
    RunRepository,
    TrialRepository,
)
from .write_queue import (
    WriteOperation,
    WriteQueue,
    WriterConfig,
    WriteRequest,
    WriterStats,
)

__all__ = [
    "DatabaseManager",
    "ExperimentRepository",
    "TrialRepository",
    "RunRepository",
    "WriteQueue",
    "WriterConfig",
    "WriterStats",
    "WriteRequest",
    "WriteOperation",
]
