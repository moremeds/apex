"""DuckDB adapter for persistence layer."""

from __future__ import annotations
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import duckdb

logger = logging.getLogger(__name__)


class DuckDBAdapter:
    """
    DuckDB database adapter.

    Provides connection management and schema initialization for the persistence layer.
    DuckDB is columnar and optimized for analytical queries on historical data.

    Thread Safety:
        This adapter uses a threading.RLock to ensure thread-safe access to the
        DuckDB connection. All database operations acquire the lock before executing.
        DuckDB connections are not thread-safe by default, so this is essential
        when the dashboard or other components access the database from multiple threads.
    """

    SCHEMA_VERSION = 2

    def __init__(self, db_path: str = "./data/apex_risk.duckdb"):
        """
        Initialize DuckDB adapter.

        Args:
            db_path: Path to database file. Use ":memory:" for in-memory database.
        """
        self.db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Ensure data directory exists
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._connect()
        self._init_schema()
        logger.info(f"DuckDB adapter initialized: {db_path}")

    def _connect(self) -> None:
        """Establish database connection."""
        self._conn = duckdb.connect(self.db_path)
        # Enable parallel execution for analytical queries
        self._conn.execute("SET threads TO 4")

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get active connection, reconnecting if needed."""
        if self._conn is None:
            self._connect()
        return self._conn

    @contextmanager
    def transaction(self):
        """Context manager for transactions (thread-safe)."""
        with self._lock:
            try:
                self.conn.begin()
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self.conn

        # Schema version tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Check current version
        result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = result[0] if result[0] else 0

        if current_version < self.SCHEMA_VERSION:
            self._apply_migrations(current_version)

    def _apply_migrations(self, from_version: int) -> None:
        """Apply schema migrations."""
        conn = self.conn

        if from_version < 1:
            # Position snapshots - track position state over time
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY,
                    snapshot_time TIMESTAMP NOT NULL,
                    symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    asset_type VARCHAR NOT NULL,
                    quantity DOUBLE NOT NULL,
                    avg_price DOUBLE,
                    mark_price DOUBLE,
                    unrealized_pnl DOUBLE,
                    daily_pnl DOUBLE,
                    delta DOUBLE,
                    gamma DOUBLE,
                    vega DOUBLE,
                    theta DOUBLE,
                    iv DOUBLE,
                    expiry VARCHAR,
                    strike DOUBLE,
                    option_type VARCHAR,
                    source VARCHAR,
                    has_market_data BOOLEAN
                )
            """)

            # Create sequence for position_snapshots
            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS position_snapshots_id_seq START 1
            """)

            # Portfolio snapshots - aggregated metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY,
                    snapshot_time TIMESTAMP NOT NULL,
                    total_unrealized_pnl DOUBLE,
                    total_daily_pnl DOUBLE,
                    portfolio_delta DOUBLE,
                    portfolio_gamma DOUBLE,
                    portfolio_vega DOUBLE,
                    portfolio_theta DOUBLE,
                    total_gross_notional DOUBLE,
                    total_net_notional DOUBLE,
                    margin_utilization DOUBLE,
                    total_net_liquidation DOUBLE,
                    total_positions INTEGER,
                    positions_with_missing_md INTEGER,
                    concentration_pct DOUBLE,
                    max_underlying_symbol VARCHAR
                )
            """)

            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS portfolio_snapshots_id_seq START 1
            """)

            # Position changes - audit trail
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_changes (
                    id INTEGER PRIMARY KEY,
                    change_time TIMESTAMP NOT NULL,
                    change_type VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    quantity_before DOUBLE,
                    quantity_after DOUBLE,
                    avg_price_before DOUBLE,
                    avg_price_after DOUBLE,
                    source VARCHAR,
                    notes VARCHAR
                )
            """)

            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS position_changes_id_seq START 1
            """)

            # Risk alerts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id INTEGER PRIMARY KEY,
                    alert_time TIMESTAMP NOT NULL,
                    alert_type VARCHAR NOT NULL,
                    severity VARCHAR NOT NULL,
                    trigger_rule VARCHAR,
                    symbol VARCHAR,
                    current_value DOUBLE,
                    threshold DOUBLE,
                    breach_pct DOUBLE,
                    message VARCHAR,
                    suggested_action VARCHAR,
                    context_json VARCHAR,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TIMESTAMP,
                    acknowledged_by VARCHAR
                )
            """)

            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS risk_alerts_id_seq START 1
            """)

            # Daily P&L summary
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    id INTEGER PRIMARY KEY,
                    trade_date DATE NOT NULL UNIQUE,
                    realized_pnl DOUBLE DEFAULT 0,
                    unrealized_pnl_open DOUBLE,
                    unrealized_pnl_close DOUBLE,
                    daily_pnl DOUBLE,
                    max_drawdown DOUBLE,
                    peak_pnl DOUBLE,
                    total_positions INTEGER,
                    positions_opened INTEGER DEFAULT 0,
                    positions_closed INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS daily_pnl_id_seq START 1
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pos_snap_time ON position_snapshots(snapshot_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pos_snap_symbol ON position_snapshots(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pos_snap_underlying ON position_snapshots(underlying)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_port_snap_time ON portfolio_snapshots(snapshot_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_changes_time ON position_changes(change_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time ON risk_alerts(alert_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON risk_alerts(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_pnl(trade_date)")

            # Record migration
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            logger.info("Applied schema migration v1")

        if from_version < 2:
            # Orders table - track order history from all brokers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    order_id VARCHAR NOT NULL,
                    account_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    asset_type VARCHAR NOT NULL,
                    side VARCHAR NOT NULL,
                    order_type VARCHAR NOT NULL,
                    quantity DOUBLE NOT NULL,
                    limit_price DOUBLE,
                    stop_price DOUBLE,
                    status VARCHAR NOT NULL,
                    filled_quantity DOUBLE DEFAULT 0,
                    avg_fill_price DOUBLE,
                    commission DOUBLE DEFAULT 0,
                    created_time TIMESTAMP,
                    submitted_time TIMESTAMP,
                    filled_time TIMESTAMP,
                    updated_time TIMESTAMP,
                    expiry VARCHAR,
                    strike DOUBLE,
                    option_right VARCHAR,
                    broker_order_id VARCHAR,
                    exchange VARCHAR,
                    time_in_force VARCHAR,
                    notes VARCHAR
                )
            """)

            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS orders_id_seq START 1
            """)

            # Trades table - track executions/fills from all brokers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    trade_id VARCHAR NOT NULL,
                    order_id VARCHAR NOT NULL,
                    account_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    asset_type VARCHAR NOT NULL,
                    side VARCHAR NOT NULL,
                    quantity DOUBLE NOT NULL,
                    price DOUBLE NOT NULL,
                    commission DOUBLE DEFAULT 0,
                    trade_time TIMESTAMP NOT NULL,
                    expiry VARCHAR,
                    strike DOUBLE,
                    option_right VARCHAR,
                    exchange VARCHAR,
                    liquidity VARCHAR
                )
            """)

            conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS trades_id_seq START 1
            """)

            # Create indexes for orders and trades
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_key ON orders(source, order_id, account_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_key ON trades(source, trade_id, account_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(trade_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_order ON trades(source, order_id, account_id)")

            # Record migration
            conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            logger.info("Applied schema migration v2 (orders and trades tables)")

    def execute(self, query: str, params: Optional[tuple] = None) -> duckdb.DuckDBPyRelation:
        """Execute a query (thread-safe)."""
        with self._lock:
            if params:
                return self.conn.execute(query, params)
            return self.conn.execute(query)

    def executemany(self, query: str, params_list: List[tuple]) -> None:
        """Execute a query with multiple parameter sets (thread-safe)."""
        with self._lock:
            self.conn.executemany(query, params_list)

    def fetch_df(self, query: str, params: Optional[tuple] = None):
        """Execute query and return as pandas DataFrame (thread-safe)."""
        with self._lock:
            result = self.conn.execute(query, params) if params else self.conn.execute(query)
            return result.df()

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute query and return all rows as dicts (thread-safe)."""
        with self._lock:
            result = self.conn.execute(query, params) if params else self.conn.execute(query)
            columns = [desc[0] for desc in result.description] if result.description else []
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows] if rows else []

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Execute query and return first row as dict (thread-safe)."""
        with self._lock:
            result = self.conn.execute(query, params) if params else self.conn.execute(query)
            columns = [desc[0] for desc in result.description] if result.description else []
            row = result.fetchone()
            return dict(zip(columns, row)) if row else None

    def close(self) -> None:
        """Close database connection (thread-safe)."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                logger.info("DuckDB connection closed")

    def vacuum(self) -> None:
        """Optimize database storage (thread-safe)."""
        with self._lock:
            self.conn.execute("VACUUM")
            logger.info("Database vacuumed")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics (thread-safe)."""
        stats = {}
        for table in ["position_snapshots", "portfolio_snapshots", "position_changes", "risk_alerts", "daily_pnl", "orders", "trades"]:
            try:
                result = self.fetch_one(f"SELECT COUNT(*) as cnt FROM {table}")
                stats[f"{table}_count"] = result["cnt"] if result else 0
            except Exception:
                stats[f"{table}_count"] = 0  # Table may not exist yet

        # Get database file size
        if self.db_path != ":memory:":
            path = Path(self.db_path)
            if path.exists():
                stats["db_size_mb"] = path.stat().st_size / (1024 * 1024)

        return stats

    def __repr__(self) -> str:
        return f"DuckDBAdapter(path={self.db_path})"
