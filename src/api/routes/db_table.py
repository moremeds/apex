"""Bounded generic reads over tables allowlisted by the PostgreSQL catalog."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.tabular import build_tabular
from src.api.routes._db_auth import require_db_token
from src.api.routes._db_errors import map_driver_error
from src.api.routes.db_catalog import ColumnInfo, TableInfo, get_catalog

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/db", tags=["database"], dependencies=[Depends(require_db_token)])

_OPS = {"eq", "ne", "lt", "le", "gt", "ge", "in", "like", "isnull"}
_INT64_MAX = 9_223_372_036_854_775_807
_SQL_OPS = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_STRING_TYPES = {"text", "varchar", "bpchar", "char", "name", "citext"}
_INTEGER_TYPES = {"int2", "int4", "int8", "smallint", "integer", "bigint", "oid"}
_DECIMAL_TYPES = {"numeric", "decimal"}
_FLOAT_TYPES = {"float4", "float8", "real", "double precision"}
_BOOL_TYPES = {"bool", "boolean"}
_DATE_TYPES = {"date"}
_DATETIME_TYPES = {
    "timestamp",
    "timestamptz",
    "timestamp without time zone",
    "timestamp with time zone",
}
_TIME_TYPES = {"time", "timetz", "time without time zone", "time with time zone"}


@dataclass(frozen=True)
class QueryPlan:
    sql: str
    params: tuple[Any, ...]
    columns: tuple[ColumnInfo, ...]
    limit: int


def _invalid(message: str) -> ApiError:
    return ApiError(ApiErrorCode.INVALID_PARAMETER, message)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _parse_int(
    name: str, raw: str | None, default: int, *, minimum: int, maximum: int | None = None
) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise _invalid(f"{name} must be an integer") from exc
    if value < minimum:
        raise _invalid(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise _invalid(f"{name} is too large")
    return value


def _parse_bool(raw: str) -> bool:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise _invalid("isnull value must be true or false")


def _convert_value(column: ColumnInfo, raw: str) -> Any:
    value_type = column.value_type.lower()
    try:
        if value_type in _INTEGER_TYPES:
            return int(raw)
        if value_type in _DECIMAL_TYPES:
            return Decimal(raw)
        if value_type in _FLOAT_TYPES:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
            return value
        if value_type in _BOOL_TYPES:
            return _parse_bool(raw)
        if value_type in _DATE_TYPES:
            return date.fromisoformat(raw)
        if value_type in _DATETIME_TYPES:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if value_type in _TIME_TYPES:
            return time.fromisoformat(raw.replace("Z", "+00:00"))
        if value_type == "uuid":
            return UUID(raw)
        if value_type in {"json", "jsonb"}:
            json.loads(raw)
            return raw
        if value_type == "bytea":
            if not raw.startswith("\\x"):
                raise ValueError
            return bytes.fromhex(raw[2:])
        if value_type.startswith("_"):
            raise _invalid(f"filters are not supported for array column {column.name!r}")
        return raw
    except (ValueError, TypeError, InvalidOperation, json.JSONDecodeError) as exc:
        raise _invalid(f"invalid value for column {column.name!r}") from exc


def _one(query: Any, name: str) -> str | None:
    values = query.getlist(name)
    if len(values) > 1:
        raise _invalid(f"{name} may only be specified once")
    return values[0] if values else None


def _select_columns(table: TableInfo, raw: str | None) -> tuple[ColumnInfo, ...]:
    if raw is None:
        return table.columns
    names = raw.split(",")
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise _invalid("columns must be a comma-separated list of unique column names")
    by_name = table.columns_by_name
    try:
        return tuple(by_name[name] for name in names)
    except KeyError as exc:
        raise _invalid(f"unknown column {exc.args[0]!r}") from exc


def _filter_sql(raw: str, table: TableInfo, params: list[Any]) -> str:
    parts = raw.split(":", 2)
    if len(parts) != 3 or not all(parts[:2]):
        raise _invalid("where must have the form column:operator:value")
    name, op, value = parts
    column = table.columns_by_name.get(name)
    if column is None:
        raise _invalid(f"unknown filter column {name!r}")
    if op not in _OPS:
        raise _invalid(f"unknown filter operator {op!r}")
    quoted = _quote(name)
    if op == "isnull":
        return f"{quoted} IS {'NULL' if _parse_bool(value) else 'NOT NULL'}"
    if op == "like":
        if column.value_type.lower() not in _STRING_TYPES:
            raise _invalid(f"like is not supported for column {name!r}")
        params.append(value)
        return f"{quoted} LIKE ${len(params)}"
    if op == "in":
        values = value.split(",")
        if not values or any(item == "" for item in values):
            raise _invalid("in requires one or more comma-separated values")
        placeholders = []
        for item in values:
            params.append(_convert_value(column, item))
            placeholders.append(f"${len(params)}")
        return f"{quoted} IN ({', '.join(placeholders)})"
    params.append(_convert_value(column, value))
    return f"{quoted} {_SQL_OPS[op]} ${len(params)}"


def build_query(table: TableInfo, query: Any) -> QueryPlan:
    unknown = set(query) - {"columns", "where", "order", "limit", "offset"}
    if unknown:
        raise _invalid(f"unknown query parameter {min(unknown)!r}")
    selected = _select_columns(table, _one(query, "columns"))
    limit = min(_parse_int("limit", _one(query, "limit"), 500, minimum=1), 5000)
    offset = _parse_int("offset", _one(query, "offset"), 0, minimum=0, maximum=_INT64_MAX)

    params: list[Any] = []
    filters = [_filter_sql(raw, table, params) for raw in query.getlist("where")]
    sql = (
        f"SELECT {', '.join(_quote(column.name) for column in selected)} "
        f"FROM {_quote(table.schema)}.{_quote(table.name)}"
    )
    if filters:
        sql += " WHERE " + " AND ".join(filters)

    raw_order = _one(query, "order")
    if offset and raw_order is None and not table.primary_key:
        raise _invalid("offset requires order for a table without a primary key")
    if raw_order is not None:
        order_parts = raw_order.split(":")
        if len(order_parts) > 2 or not order_parts[0]:
            raise _invalid("order must have the form column[:asc|desc]")
        column = order_parts[0]
        if column not in table.columns_by_name:
            raise _invalid(f"unknown order column {column!r}")
        direction = order_parts[1].lower() if len(order_parts) == 2 else "asc"
        if direction not in {"asc", "desc"}:
            raise _invalid("order direction must be asc or desc")
        keys = [f"{_quote(column)} {direction.upper()}"]
        # A nonunique sort column alone makes offset pages unstable; the PK
        # columns disambiguate ties so rows cannot repeat or skip pages.
        unique = table.primary_key == (column,) or (column,) in table.unique_keys
        if table.primary_key and not unique:
            keys += [f"{_quote(pk)} ASC" for pk in table.primary_key if pk != column]
        sql += " ORDER BY " + ", ".join(keys)
    elif table.primary_key:
        sql += " ORDER BY " + ", ".join(_quote(column) for column in table.primary_key)

    params.extend((limit + 1, offset))
    sql += f" LIMIT ${len(params) - 1} OFFSET ${len(params)}"
    return QueryPlan(sql=sql, params=tuple(params), columns=selected, limit=limit)


@router.get("/{database}/{schema}/{table}")
async def db_table(request: Request, database: str, schema: str, table: str) -> dict[str, Any]:
    catalog = await get_catalog(request, database)
    table_info = catalog.tables.get((schema, table))
    if table_info is None:
        raise _invalid(f"unknown table {schema}.{table}")
    plan = build_query(table_info, request.query_params)
    pool = request.app.state.pg_read_pools[database]
    try:
        async with pool.acquire(timeout=30.0) as connection:
            async with connection.transaction(readonly=True):
                rows = await connection.fetch(plan.sql, *plan.params)
    except asyncpg.QueryCanceledError as exc:
        logger.warning("read query timed out for %s.%s.%s", database, schema, table)
        raise ApiError(ApiErrorCode.QUERY_TIMEOUT, "database query timed out") from exc
    except TimeoutError as exc:
        logger.warning("read connection timed out for %s.%s.%s", database, schema, table)
        raise ApiError(ApiErrorCode.QUERY_TIMEOUT, "database query timed out") from exc
    except (asyncpg.DataError, asyncpg.UndefinedFunctionError) as exc:
        logger.warning(
            "invalid read filter for %s.%s.%s (%s)",
            database,
            schema,
            table,
            type(exc).__name__,
        )
        raise _invalid("filter value or operator is invalid for its column type") from exc
    except Exception as exc:
        logger.warning(
            "read query failed for %s.%s.%s (%s)", database, schema, table, type(exc).__name__
        )
        mapped = map_driver_error(exc)
        if mapped is None:
            raise
        raise mapped from exc
    return build_tabular(
        database,
        schema,
        table,
        [{"name": column.name, "type": column.type} for column in plan.columns],
        rows,
        plan.limit,
    )
