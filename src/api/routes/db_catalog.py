"""Read-only PostgreSQL catalog and the allowlist used by table reads."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast

import asyncpg
from fastapi import APIRouter, Depends, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.validate import validate_payload
from src.api.routes._db_auth import require_db_token
from src.api.routes._db_errors import map_driver_error

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/db", tags=["database"], dependencies=[Depends(require_db_token)])

_CATALOG_SQL = """
SELECT n.nspname AS schema_name, c.relname AS table_name,
       a.attname AS column_name, format_type(a.atttypid, a.atttypmod) AS data_type,
       COALESCE(bt.typname, t.typname) AS value_type, NOT a.attnotnull AS nullable,
       a.attnum AS ordinal
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
JOIN pg_catalog.pg_type t ON t.oid = a.atttypid
LEFT JOIN pg_catalog.pg_type bt ON bt.oid = NULLIF(t.typbasetype, 0)
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND a.attnum > 0 AND NOT a.attisdropped
  AND has_schema_privilege(n.oid, 'USAGE')
  AND has_table_privilege(c.oid, 'SELECT')
  AND n.nspname <> 'information_schema'
  AND n.nspname !~ '^(pg_|_timescaledb|timescaledb_)'
ORDER BY n.nspname, c.relname, a.attnum
"""

_CONSTRAINT_SQL = """
SELECT con.conname AS constraint_name, con.contype AS constraint_type,
       sn.nspname AS schema_name, sc.relname AS table_name,
       sa.attname AS column_name, keys.ordinality,
       tn.nspname AS target_schema, tc.relname AS target_table,
       ta.attname AS target_column
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class sc ON sc.oid = con.conrelid
JOIN pg_catalog.pg_namespace sn ON sn.oid = sc.relnamespace
CROSS JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS keys(attnum, ordinality)
JOIN pg_catalog.pg_attribute sa
  ON sa.attrelid = con.conrelid AND sa.attnum = keys.attnum
LEFT JOIN pg_catalog.pg_class tc ON tc.oid = con.confrelid
LEFT JOIN pg_catalog.pg_namespace tn ON tn.oid = tc.relnamespace
LEFT JOIN pg_catalog.pg_attribute ta
  ON ta.attrelid = con.confrelid AND ta.attnum = con.confkey[keys.ordinality]
WHERE con.contype IN ('p', 'u', 'f')
ORDER BY sn.nspname, sc.relname, con.conname, keys.ordinality
"""

_EXACT_EXCLUDED_TABLES = frozenset(
    {
        "api_request_audit",
        "raw_payloads",
        "external_api_requests",
        "jobs",
        "job_failures",
        "worker_heartbeat",
        "pipeline_benchmark_snapshots",
        "data_freshness_snapshots",
        "volatility_backfill_status",
        "ws_consumer_state",
        "macro_source_status",
        "uw_fetch_memo",
    }
)
_EXCLUDED_PREFIXES = ("data_gap_", "pg_stat_statements")


def _table_allowed(schema: str, table: str) -> bool:
    return not (
        schema == "information_schema"
        or schema.startswith(("pg_", "_timescaledb", "timescaledb_"))
        or table in _EXACT_EXCLUDED_TABLES
        or table.startswith(_EXCLUDED_PREFIXES)
    )


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    value_type: str
    nullable: bool


@dataclass(frozen=True)
class ForeignKeyInfo:
    columns: tuple[str, ...]
    referenced_schema: str
    referenced_table: str
    referenced_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableInfo:
    schema: str
    name: str
    columns: tuple[ColumnInfo, ...]
    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()
    foreign_keys: tuple[ForeignKeyInfo, ...] = ()

    @property
    def columns_by_name(self) -> dict[str, ColumnInfo]:
        return {column.name: column for column in self.columns}


@dataclass(frozen=True)
class DatabaseCatalog:
    database: str
    tables: Mapping[tuple[str, str], TableInfo]


@dataclass
class CatalogCache:
    ttl_seconds: float = 600.0
    _entries: dict[str, tuple[float, DatabaseCatalog]] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    async def get(self, database: str, pool: Any) -> DatabaseCatalog:
        now = time.monotonic()
        entry = self._entries.get(database)
        if entry is not None and now - entry[0] < self.ttl_seconds:
            return entry[1]
        lock = self._locks.setdefault(database, asyncio.Lock())
        async with lock:
            entry = self._entries.get(database)
            if entry is not None and time.monotonic() - entry[0] < self.ttl_seconds:
                return entry[1]
            catalog = await fetch_database_catalog(database, pool)
            self._entries[database] = (time.monotonic(), catalog)
            return catalog


def _record(row: Any, name: str) -> Any:
    return row[name]


def build_database_catalog(
    database: str, column_rows: list[Any], constraint_rows: list[Any]
) -> DatabaseCatalog:
    columns: dict[tuple[str, str], list[ColumnInfo]] = defaultdict(list)
    for row in column_rows:
        schema, table = str(_record(row, "schema_name")), str(_record(row, "table_name"))
        if _table_allowed(schema, table):
            columns[(schema, table)].append(
                ColumnInfo(
                    name=str(_record(row, "column_name")),
                    type=str(_record(row, "data_type")),
                    value_type=str(_record(row, "value_type")),
                    nullable=bool(_record(row, "nullable")),
                )
            )

    grouped: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for row in constraint_rows:
        key = (str(row["schema_name"]), str(row["table_name"]))
        if key not in columns:
            continue
        if row["constraint_type"] == "f":
            target = (str(row["target_schema"]), str(row["target_table"]))
            if target not in columns:
                continue
        grouped[(key[0], key[1], str(row["constraint_name"]), str(row["constraint_type"]))].append(
            row
        )

    tables: dict[tuple[str, str], TableInfo] = {}
    for key, table_columns in columns.items():
        primary_key: tuple[str, ...] = ()
        unique_keys: list[tuple[str, ...]] = []
        foreign_keys: list[ForeignKeyInfo] = []
        for (schema, table, _name, kind), rows in grouped.items():
            if (schema, table) != key:
                continue
            ordered = sorted(rows, key=lambda row: int(row["ordinality"]))
            names = tuple(str(row["column_name"]) for row in ordered)
            if kind == "p":
                primary_key = names
            elif kind == "u":
                unique_keys.append(names)
            else:
                foreign_keys.append(
                    ForeignKeyInfo(
                        columns=names,
                        referenced_schema=str(ordered[0]["target_schema"]),
                        referenced_table=str(ordered[0]["target_table"]),
                        referenced_columns=tuple(str(row["target_column"]) for row in ordered),
                    )
                )
        tables[key] = TableInfo(
            schema=key[0],
            name=key[1],
            columns=tuple(table_columns),
            primary_key=primary_key,
            unique_keys=tuple(unique_keys),
            foreign_keys=tuple(foreign_keys),
        )
    return DatabaseCatalog(database=database, tables=tables)


async def fetch_database_catalog(database: str, pool: Any) -> DatabaseCatalog:
    async with pool.acquire(timeout=30.0) as connection:
        async with connection.transaction(readonly=True):
            column_rows = await connection.fetch(_CATALOG_SQL)
            constraint_rows = await connection.fetch(_CONSTRAINT_SQL)
    return build_database_catalog(database, list(column_rows), list(constraint_rows))


def get_read_pools(request: Request) -> Mapping[str, Any]:
    pools = getattr(request.app.state, "pg_read_pools", None)
    if not pools or all(pool is None for pool in pools.values()):
        raise ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, "PostgreSQL read API not configured")
    return cast(Mapping[str, Any], pools)


async def get_catalog(request: Request, database: str) -> DatabaseCatalog:
    pools = get_read_pools(request)
    if database not in pools:
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, f"unknown database {database!r}")
    pool = pools[database]
    if pool is None:  # configured, but its pool failed to start
        raise ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, "database unavailable")
    cache = getattr(request.app.state, "pg_catalog_cache", None)
    if cache is None:
        cache = CatalogCache()
        request.app.state.pg_catalog_cache = cache
    try:
        return await cache.get(database, pool)
    except asyncpg.QueryCanceledError as exc:
        logger.warning("catalog query timed out for database %s", database)
        raise ApiError(ApiErrorCode.QUERY_TIMEOUT, "database query timed out") from exc
    except TimeoutError as exc:
        logger.warning("catalog connection timed out for database %s", database)
        raise ApiError(ApiErrorCode.QUERY_TIMEOUT, "database query timed out") from exc
    except Exception as exc:
        logger.warning("catalog read failed for database %s (%s)", database, type(exc).__name__)
        mapped = map_driver_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


def build_catalog_payload(
    catalogs: list[DatabaseCatalog], unavailable: list[str] | None = None
) -> dict[str, Any]:
    databases: list[dict[str, Any]] = []
    for catalog in catalogs:
        schemas: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for table in catalog.tables.values():
            schemas[table.schema].append(
                {
                    "name": table.name,
                    "columns": [
                        {"name": column.name, "type": column.type, "nullable": column.nullable}
                        for column in table.columns
                    ],
                    "primary_key": list(table.primary_key),
                    "unique_keys": [list(key) for key in table.unique_keys],
                    "foreign_keys": [
                        {
                            "columns": list(key.columns),
                            "referenced_schema": key.referenced_schema,
                            "referenced_table": key.referenced_table,
                            "referenced_columns": list(key.referenced_columns),
                        }
                        for key in table.foreign_keys
                    ],
                }
            )
        schema_payloads: list[dict[str, Any]] = []
        for name, tables in sorted(schemas.items()):
            tables.sort(key=lambda item: str(item["name"]))
            schema_payloads.append({"name": name, "tables": tables})
        databases.append({"name": catalog.database, "schemas": schema_payloads})
    payload = {
        "databases": sorted(databases, key=lambda database: database["name"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unavailable": sorted(unavailable or []),
    }
    validate_payload(payload, "db_catalog_payload")
    return payload


@router.get("/catalog")
async def db_catalog(request: Request) -> dict[str, Any]:
    unknown = set(request.query_params) - {"database"}
    if unknown:
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, f"unknown query parameter {min(unknown)!r}")
    if len(request.query_params.getlist("database")) > 1:
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, "database may only be specified once")
    pools = get_read_pools(request)
    database = request.query_params.get("database")
    if database is not None:
        return build_catalog_payload([await get_catalog(request, database)])
    # Unfiltered listing is best-effort: one database being down must not hide the others.
    catalogs: list[DatabaseCatalog] = []
    unavailable: list[str] = []
    for name in sorted(pools):
        try:
            catalogs.append(await get_catalog(request, name))
        except ApiError as exc:
            if exc.code is not ApiErrorCode.PROVIDER_NOT_CONFIGURED:
                raise
            unavailable.append(name)
    return build_catalog_payload(catalogs, unavailable)
