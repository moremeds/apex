"""
DuckDB database manager for backtest storage.

Provides:
- Schema initialization
- Connection management
- Batch operations
- Parquet export/import
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for DuckDB storage.

    Features:
    - Automatic schema creation
    - Connection pooling (single connection for now)
    - Batch inserts for performance
    - Parquet partitioning for large datasets
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """
        Initialize database manager.

        Args:
            db_path: Path to DuckDB file or ":memory:" for in-memory
        """
        if duckdb is None:
            raise ImportError("duckdb is required for storage. Install with: pip install duckdb")

        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._initialized = False

    @property
    def conn(self) -> "duckdb.DuckDBPyConnection":
        """Get database connection, creating if needed."""
        if self._conn is None:
            if self.db_path != ":memory:" and isinstance(self.db_path, Path):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def initialize_schema(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        # Experiments table - create with original schema first for backwards compatibility
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                strategy VARCHAR NOT NULL,
                parameters JSON,
                universe JSON,
                temporal JSON,
                optimization JSON,
                profiles JSON,
                reproducibility JSON,
                created_at TIMESTAMP DEFAULT current_timestamp,
                completed_at TIMESTAMP,
                status VARCHAR DEFAULT 'pending'
            )
        """
        )

        # Schema migration: add version columns if they don't exist (for existing DBs)
        # MUST run BEFORE index creation that depends on these columns
        # Use ADD COLUMN IF NOT EXISTS and always backfill NULLs
        self.conn.execute(
            """
            ALTER TABLE experiments ADD COLUMN IF NOT EXISTS base_experiment_id VARCHAR
        """
        )
        self.conn.execute(
            """
            UPDATE experiments SET base_experiment_id = experiment_id
            WHERE base_experiment_id IS NULL
        """
        )

        self.conn.execute(
            """
            ALTER TABLE experiments ADD COLUMN IF NOT EXISTS run_version INTEGER DEFAULT 1
        """
        )
        self.conn.execute(
            """
            UPDATE experiments SET run_version = 1
            WHERE run_version IS NULL
        """
        )

        # Index for version lookups (AFTER migration ensures column exists)
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_experiments_base_id
            ON experiments(base_experiment_id)
        """
        )

        # Trials table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                trial_id VARCHAR PRIMARY KEY,
                experiment_id VARCHAR NOT NULL,
                params JSON NOT NULL,
                trial_index INTEGER,
                suggested_by VARCHAR,
                -- Aggregates
                median_sharpe DOUBLE,
                median_return DOUBLE,
                median_max_dd DOUBLE,
                median_win_rate DOUBLE,
                median_profit_factor DOUBLE,
                mad_sharpe DOUBLE,
                p10_sharpe DOUBLE,
                p90_sharpe DOUBLE,
                p10_max_dd DOUBLE,
                p90_max_dd DOUBLE,
                stability_score DOUBLE,
                degradation_ratio DOUBLE,
                is_median_sharpe DOUBLE,
                oos_median_sharpe DOUBLE,
                trial_score DOUBLE,
                -- Counts
                total_runs INTEGER,
                successful_runs INTEGER,
                failed_runs INTEGER,
                -- Constraints
                constraints_met BOOLEAN,
                constraint_violations JSON,
                -- Timing
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds DOUBLE,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """
        )

        # Runs table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                trial_id VARCHAR NOT NULL,
                experiment_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                window_id VARCHAR NOT NULL,
                profile_version VARCHAR,
                data_version VARCHAR,
                -- Status
                status VARCHAR NOT NULL,
                error VARCHAR,
                -- Flags
                is_train BOOLEAN,
                is_oos BOOLEAN,
                -- Metrics
                total_return DOUBLE,
                cagr DOUBLE,
                annualized_return DOUBLE,
                sharpe DOUBLE,
                sortino DOUBLE,
                calmar DOUBLE,
                max_drawdown DOUBLE,
                avg_drawdown DOUBLE,
                max_dd_duration_days INTEGER,
                total_trades INTEGER,
                win_rate DOUBLE,
                profit_factor DOUBLE,
                expectancy DOUBLE,
                sqn DOUBLE,
                best_trade_pct DOUBLE,
                worst_trade_pct DOUBLE,
                avg_win_pct DOUBLE,
                avg_loss_pct DOUBLE,
                exposure_pct DOUBLE,
                avg_trade_duration_days DOUBLE,
                total_commission DOUBLE,
                total_slippage DOUBLE,
                total_costs DOUBLE,
                -- Timing
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds DOUBLE,
                -- Params stored for convenience
                params JSON,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """
        )

        # Create indexes for common queries
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_trial_id ON runs(trial_id)
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id)
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trials_experiment_id ON trials(experiment_id)
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_symbol ON runs(symbol)
        """
        )

        self._initialized = True
        logger.info(f"Database schema initialized: {self.db_path}")

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query."""
        if params:
            return self.conn.execute(query, params)
        return self.conn.execute(query)

    def executemany(self, query: str, params_list: List[tuple]) -> None:
        """Execute query with multiple parameter sets."""
        self.conn.executemany(query, params_list)

    def fetchall(self, query: str, params: Optional[tuple] = None) -> List[tuple[Any, ...]]:
        """Execute query and fetch all results."""
        result = self.execute(query, params)
        return list(result.fetchall())

    def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[tuple[Any, ...]]:
        """Execute query and fetch one result."""
        result = self.execute(query, params)
        row = result.fetchone()
        return tuple(row) if row is not None else None

    def insert_batch(self, table: str, records: List[Dict[str, Any]]) -> int:
        """
        Insert batch of records efficiently.

        Args:
            table: Table name
            records: List of dictionaries to insert

        Returns:
            Number of records inserted
        """
        if not records:
            return 0

        # Get columns from first record
        columns = list(records[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_str = ", ".join(columns)

        query = f"INSERT INTO {table} ({column_str}) VALUES ({placeholders})"

        # Convert dicts to tuples
        params_list = [tuple(r.get(c) for c in columns) for r in records]

        self.executemany(query, params_list)
        return len(records)

    def export_to_parquet(self, table: str, path: Union[str, Path]) -> None:
        """Export table to Parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET)")
        logger.info(f"Exported {table} to {path}")

    def import_from_parquet(self, table: str, path: Union[str, Path]) -> int:
        """Import Parquet file into table."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        result = self.conn.execute(f"INSERT INTO {table} SELECT * FROM read_parquet('{path}')")
        row = result.fetchone() if result else None
        count: int = int(row[0]) if row else 0
        logger.info(f"Imported {count} records from {path} to {table}")
        return count

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
