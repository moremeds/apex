"""
Persistence layer for PostgreSQL/TimescaleDB.

Provides:
- Database connection management with asyncpg
- Repository pattern for data access
- Migration support for schema versioning
"""

from src.infrastructure.persistence.database import (
    ConnectionError,
    Database,
    DatabaseError,
    QueryError,
    close_database,
    get_database,
)
from src.infrastructure.persistence.pg_repositories import PgRepositories
from src.infrastructure.persistence.pg_schema import ensure_schema, reset_schema
from src.infrastructure.persistence.signal_listener import SignalListener

__all__ = [
    "Database",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "get_database",
    "close_database",
    "SignalListener",
    "PgRepositories",
    "ensure_schema",
    "reset_schema",
]
