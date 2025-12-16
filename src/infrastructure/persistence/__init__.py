"""
Persistence layer for PostgreSQL/TimescaleDB.

Provides:
- Database connection management with asyncpg
- Repository pattern for data access
- Migration support for schema versioning
"""

from src.infrastructure.persistence.database import (
    Database,
    DatabaseError,
    ConnectionError,
    QueryError,
    get_database,
    close_database,
)

__all__ = [
    "Database",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "get_database",
    "close_database",
]
