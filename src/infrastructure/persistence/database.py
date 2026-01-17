"""
Database connection manager for PostgreSQL/TimescaleDB.

Provides async connection pooling using asyncpg with:
- Automatic connection pool management
- Health checks and reconnection
- Query execution helpers
- Transaction support
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, TypeVar

import asyncpg
from asyncpg import Connection, Pool, Record

from config.models import DatabaseConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseError(Exception):
    """Base exception for database operations."""


class ConnectionError(DatabaseError):
    """Failed to establish database connection."""


class QueryError(DatabaseError):
    """Query execution failed."""


class Database:
    """
    Async database connection manager using asyncpg.

    Usage:
        db = Database(config)
        await db.connect()

        # Execute queries
        rows = await db.fetch("SELECT * FROM users WHERE id = $1", user_id)
        await db.execute("INSERT INTO users (name) VALUES ($1)", name)

        # Use transactions
        async with db.transaction() as conn:
            await conn.execute("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from_id)
            await conn.execute("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to_id)

        await db.close()
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager.

        Args:
            config: Database configuration with connection details and pool settings.
        """
        self._config = config
        self._pool: Optional[Pool] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self._pool is not None

    @property
    def pool(self) -> Pool:
        """Get the connection pool, raising if not connected."""
        if self._pool is None:
            raise ConnectionError("Database not connected. Call connect() first.")
        return self._pool

    async def connect(self) -> None:
        """
        Establish connection pool to the database.

        Raises:
            ConnectionError: If connection fails after retries.
        """
        if self._connected:
            logger.warning("Database already connected")
            return

        dsn = self._config.dsn
        min_size = self._config.pool.min_connections
        max_size = self._config.pool.max_connections

        logger.info(
            "Connecting to database",
            extra={
                "host": self._config.host,
                "port": self._config.port,
                "database": self._config.database,
                "pool_min": min_size,
                "pool_max": max_size,
            },
        )

        try:
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=min_size,
                max_size=max_size,
                command_timeout=60,
                # Enable prepared statement caching
                statement_cache_size=100,
            )
            self._connected = True
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise ConnectionError(f"Failed to connect to database: {e}") from e

    async def close(self) -> None:
        """Close the connection pool gracefully."""
        if self._pool is not None:
            logger.info("Closing database connection pool")
            await self._pool.close()
            self._pool = None
            self._connected = False
            logger.info("Database connection pool closed")

    async def health_check(self) -> bool:
        """
        Check database health by executing a simple query.

        Returns:
            True if database is healthy, False otherwise.
        """
        if not self.is_connected:
            return False

        try:
            await self.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Query Execution Methods
    # -------------------------------------------------------------------------

    async def execute(self, query: str, *args: Any, timeout: Optional[float] = None) -> str:
        """
        Execute a query and return the status.

        Args:
            query: SQL query string with $1, $2, etc. placeholders.
            *args: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            Status string (e.g., "INSERT 0 1", "UPDATE 5").

        Raises:
            QueryError: If query execution fails.
        """
        try:
            async with self.pool.acquire() as conn:
                return await conn.execute(query, *args, timeout=timeout)
        except Exception as e:
            logger.error(f"Query execution failed: {e}", extra={"query": query[:200]})
            raise QueryError(f"Query execution failed: {e}") from e

    async def executemany(
        self, query: str, args: List[Tuple[Any, ...]], timeout: Optional[float] = None
    ) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query string with $1, $2, etc. placeholders.
            args: List of parameter tuples.
            timeout: Query timeout in seconds.

        Raises:
            QueryError: If query execution fails.
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(query, args, timeout=timeout)
        except Exception as e:
            logger.error(f"Batch execution failed: {e}", extra={"query": query[:200]})
            raise QueryError(f"Batch execution failed: {e}") from e

    async def fetch(self, query: str, *args: Any, timeout: Optional[float] = None) -> List[Record]:
        """
        Execute a query and return all rows.

        Args:
            query: SQL query string with $1, $2, etc. placeholders.
            *args: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            List of Record objects.

        Raises:
            QueryError: If query execution fails.
        """
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetch(query, *args, timeout=timeout)
        except Exception as e:
            logger.error(f"Fetch failed: {e}", extra={"query": query[:200]})
            raise QueryError(f"Fetch failed: {e}") from e

    async def fetchrow(
        self, query: str, *args: Any, timeout: Optional[float] = None
    ) -> Optional[Record]:
        """
        Execute a query and return a single row.

        Args:
            query: SQL query string with $1, $2, etc. placeholders.
            *args: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            Single Record object or None if no rows.

        Raises:
            QueryError: If query execution fails.
        """
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchrow(query, *args, timeout=timeout)
        except Exception as e:
            logger.error(f"Fetchrow failed: {e}", extra={"query": query[:200]})
            raise QueryError(f"Fetchrow failed: {e}") from e

    async def fetchval(
        self, query: str, *args: Any, column: int = 0, timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a query and return a single value.

        Args:
            query: SQL query string with $1, $2, etc. placeholders.
            *args: Query parameters.
            column: Column index to return (default: 0).
            timeout: Query timeout in seconds.

        Returns:
            Single value from the specified column.

        Raises:
            QueryError: If query execution fails.
        """
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, *args, column=column, timeout=timeout)
        except Exception as e:
            logger.error(f"Fetchval failed: {e}", extra={"query": query[:200]})
            raise QueryError(f"Fetchval failed: {e}") from e

    # -------------------------------------------------------------------------
    # Transaction Support
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        """
        Context manager for database transactions.

        Usage:
            async with db.transaction() as conn:
                await conn.execute("INSERT INTO ...")
                await conn.execute("UPDATE ...")
                # Commits on success, rolls back on exception

        Yields:
            Connection object with active transaction.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Connection]:
        """
        Acquire a connection from the pool.

        Usage:
            async with db.acquire() as conn:
                await conn.execute("...")

        Yields:
            Connection object.
        """
        async with self.pool.acquire() as conn:
            yield conn

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = $1
            )
        """
        return await self.fetchval(query, table_name)

    async def get_table_row_count(self, table_name: str) -> int:
        """
        Get approximate row count for a table.

        Args:
            table_name: Name of the table.

        Returns:
            Approximate row count.
        """
        # Use reltuples for fast approximate count
        query = """
            SELECT reltuples::bigint AS count
            FROM pg_class
            WHERE relname = $1
        """
        result = await self.fetchval(query, table_name)
        return result if result and result > 0 else 0

    async def get_pool_stats(self) -> Dict[str, int]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool stats (size, free, used).
        """
        if not self.is_connected:
            return {"size": 0, "free": 0, "used": 0}

        return {
            "size": self.pool.get_size(),
            "free": self.pool.get_idle_size(),
            "used": self.pool.get_size() - self.pool.get_idle_size(),
        }


# Global database instance (optional singleton pattern)
_db_instance: Optional[Database] = None


async def get_database(config: Optional[DatabaseConfig] = None) -> Database:
    """
    Get or create the global database instance.

    Args:
        config: Database configuration. Required on first call.

    Returns:
        Database instance.

    Raises:
        ValueError: If config is not provided on first call.
    """
    global _db_instance

    if _db_instance is None:
        if config is None:
            raise ValueError("Database config required for first initialization")
        _db_instance = Database(config)
        await _db_instance.connect()

    return _db_instance


async def close_database() -> None:
    """Close the global database instance."""
    global _db_instance

    if _db_instance is not None:
        await _db_instance.close()
        _db_instance = None
