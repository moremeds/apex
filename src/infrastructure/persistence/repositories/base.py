"""
Base repository pattern for database operations.

Provides common CRUD operations and query helpers that concrete
repositories can extend.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

from asyncpg import Record

from src.infrastructure.persistence.database import Database

logger = logging.getLogger(__name__)

# Type variable for entity types
T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with common database operations.

    Subclasses must implement:
    - table_name: The database table name
    - _to_entity: Convert a database record to an entity
    - _to_row: Convert an entity to a database row dict

    Provides:
    - UPSERT operations (insert or update on conflict)
    - Batch inserts with conflict handling
    - Common query methods
    """

    def __init__(self, db: Database):
        """
        Initialize repository with database connection.

        Args:
            db: Database connection manager.
        """
        self._db = db

    @property
    @abstractmethod
    def table_name(self) -> str:
        """The database table name for this repository."""
        pass

    @property
    def primary_key_columns(self) -> List[str]:
        """
        Primary key column(s) for the table.

        Override in subclass if different from default.
        Default assumes single 'id' column.
        """
        return ["id"]

    @property
    def conflict_columns(self) -> List[str]:
        """
        Columns to check for conflicts in UPSERT operations.

        Override in subclass. Typically the business key, not surrogate key.
        """
        return self.primary_key_columns

    @abstractmethod
    def _to_entity(self, record: Record) -> T:
        """
        Convert a database record to an entity object.

        Args:
            record: asyncpg Record from query result.

        Returns:
            Entity object.
        """
        pass

    @abstractmethod
    def _to_row(self, entity: T) -> Dict[str, Any]:
        """
        Convert an entity to a dictionary for database insertion.

        Args:
            entity: Entity object.

        Returns:
            Dictionary with column names as keys.
        """
        pass

    # -------------------------------------------------------------------------
    # Common Query Methods
    # -------------------------------------------------------------------------

    async def find_by_id(self, id_value: Any) -> Optional[T]:
        """
        Find entity by primary key.

        Args:
            id_value: Primary key value.

        Returns:
            Entity if found, None otherwise.
        """
        pk_col = self.primary_key_columns[0]
        query = f"SELECT * FROM {self.table_name} WHERE {pk_col} = $1"
        record = await self._db.fetchrow(query, id_value)
        return self._to_entity(record) if record else None

    async def find_all(self, limit: int = 1000, offset: int = 0) -> List[T]:
        """
        Find all entities with pagination.

        Args:
            limit: Maximum number of records.
            offset: Number of records to skip.

        Returns:
            List of entities.
        """
        query = f"SELECT * FROM {self.table_name} ORDER BY id LIMIT $1 OFFSET $2"
        records = await self._db.fetch(query, limit, offset)
        return [self._to_entity(r) for r in records]

    async def count(self) -> int:
        """
        Count total records in table.

        Returns:
            Total record count.
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        return await self._db.fetchval(query)

    async def exists(self, **conditions: Any) -> bool:
        """
        Check if a record exists matching the conditions.

        Args:
            **conditions: Column=value conditions.

        Returns:
            True if record exists.
        """
        where_clause, params = self._build_where_clause(conditions)
        query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE {where_clause})"
        return await self._db.fetchval(query, *params)

    async def find_where(
        self,
        limit: int = 1000,
        order_by: Optional[str] = None,
        **conditions: Any,
    ) -> List[T]:
        """
        Find entities matching conditions.

        Args:
            limit: Maximum number of records.
            order_by: Column to order by (e.g., "created_at DESC").
            **conditions: Column=value conditions.

        Returns:
            List of matching entities.
        """
        where_clause, params = self._build_where_clause(conditions)
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {where_clause}
            {order_clause}
            LIMIT ${len(params) + 1}
        """
        records = await self._db.fetch(query, *params, limit)
        return [self._to_entity(r) for r in records]

    async def find_one_where(self, **conditions: Any) -> Optional[T]:
        """
        Find a single entity matching conditions.

        Args:
            **conditions: Column=value conditions.

        Returns:
            Entity if found, None otherwise.
        """
        results = await self.find_where(limit=1, **conditions)
        return results[0] if results else None

    # -------------------------------------------------------------------------
    # Insert/Update Operations
    # -------------------------------------------------------------------------

    async def insert(self, entity: T) -> T:
        """
        Insert a new entity.

        Args:
            entity: Entity to insert.

        Returns:
            Inserted entity (with generated ID if applicable).
        """
        row = self._to_row(entity)
        columns = list(row.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = list(row.values())

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING *
        """
        record = await self._db.fetchrow(query, *values)
        return self._to_entity(record)

    async def upsert(self, entity: T) -> T:
        """
        Insert or update entity based on conflict columns.

        Uses ON CONFLICT ... DO UPDATE for idempotent writes.

        Args:
            entity: Entity to upsert.

        Returns:
            Upserted entity.
        """
        row = self._to_row(entity)
        columns = list(row.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = list(row.values())

        # Build update clause (exclude conflict columns)
        update_cols = [c for c in columns if c not in self.conflict_columns]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        conflict_cols = ", ".join(self.conflict_columns)

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}
            RETURNING *
        """
        record = await self._db.fetchrow(query, *values)
        return self._to_entity(record)

    async def upsert_many(self, entities: List[T]) -> int:
        """
        Batch upsert multiple entities.

        Args:
            entities: List of entities to upsert.

        Returns:
            Number of entities processed.
        """
        if not entities:
            return 0

        # Get columns from first entity
        row = self._to_row(entities[0])
        columns = list(row.keys())

        # Build update clause
        update_cols = [c for c in columns if c not in self.conflict_columns]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        conflict_cols = ", ".join(self.conflict_columns)

        # Build parameterized query
        placeholders = [f"${i+1}" for i in range(len(columns))]
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}
        """

        # Convert all entities to tuples
        values = [tuple(self._to_row(e).values()) for e in entities]

        await self._db.executemany(query, values)
        return len(entities)

    async def delete(self, id_value: Any) -> bool:
        """
        Delete entity by primary key.

        Args:
            id_value: Primary key value.

        Returns:
            True if deleted, False if not found.
        """
        pk_col = self.primary_key_columns[0]
        query = f"DELETE FROM {self.table_name} WHERE {pk_col} = $1"
        result = await self._db.execute(query, id_value)
        return "DELETE 1" in result

    async def delete_where(self, **conditions: Any) -> int:
        """
        Delete entities matching conditions.

        Args:
            **conditions: Column=value conditions.

        Returns:
            Number of deleted records.
        """
        where_clause, params = self._build_where_clause(conditions)
        query = f"DELETE FROM {self.table_name} WHERE {where_clause}"
        result = await self._db.execute(query, *params)
        # Parse result like "DELETE 5"
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_where_clause(
        self, conditions: Dict[str, Any]
    ) -> Tuple[str, List[Any]]:
        """
        Build WHERE clause from conditions dictionary.

        Args:
            conditions: Column=value mapping.

        Returns:
            Tuple of (where_clause_string, parameter_list).
        """
        if not conditions:
            return "1=1", []

        clauses = []
        params = []
        for i, (col, val) in enumerate(conditions.items(), 1):
            if val is None:
                clauses.append(f"{col} IS NULL")
            else:
                clauses.append(f"{col} = ${i}")
                params.append(val)

        return " AND ".join(clauses), params

    @staticmethod
    def _to_json(data: Any) -> Optional[str]:
        """
        Convert data to JSON string for JSONB columns.

        Args:
            data: Data to serialize.

        Returns:
            JSON string or None.
        """
        if data is None:
            return None
        return json.dumps(data, default=str)

    @staticmethod
    def _from_json(data: Any) -> Any:
        """
        Convert JSONB column data to Python object.

        Args:
            data: JSON data from database.

        Returns:
            Python object (dict, list, etc.).
        """
        if data is None:
            return None
        if isinstance(data, (dict, list)):
            return data
        return json.loads(data)

    @staticmethod
    def _decimal_to_float(value: Any) -> Optional[float]:
        """
        Convert Decimal to float for numeric columns.

        Args:
            value: Decimal or numeric value.

        Returns:
            Float value or None.
        """
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        return value
