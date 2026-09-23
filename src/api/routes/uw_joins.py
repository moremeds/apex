"""Read-only routes for the curated ``option_wizard.uw_scan`` joins."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, Request

from src.api.errors import ApiError, ApiErrorCode
from src.api.payload.tabular import build_tabular
from src.api.routes._db_auth import require_db_token
from src.api.uw_join_registry import JoinQuery, build_join_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/uw", tags=["database"], dependencies=[Depends(require_db_token)])


def _coverage(plan: JoinQuery, rows: Sequence[Any]) -> dict[str, Any]:
    page = rows[: plan.limit]
    return {
        "scope": "returned_rows",
        "tables": {
            table: {"matched": sum(bool(row[flag]) for row in page), "total": len(page)}
            for table, flag in plan.coverage
        },
    }


def _columns(attributes: Sequence[Any]) -> list[dict[str, str]]:
    # Prepared-statement attributes expose native type names without typmods
    # (e.g. "numeric", not "numeric(10,2)"); catalog reads carry format_type.
    return [
        {"name": attribute.name, "type": getattr(attribute.type, "name", str(attribute.type))}
        for attribute in attributes
    ]


@router.get("/{join_name}")
async def uw_join(request: Request, join_name: str) -> dict[str, Any]:
    pools: Mapping[str, Any] | None = getattr(request.app.state, "pg_read_pools", None)
    pool = pools.get("option_wizard") if pools else None
    if pool is None:
        raise ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, "PostgreSQL read API not configured")
    plan = build_join_query(join_name, request.query_params)
    try:
        async with pool.acquire(timeout=30.0) as connection, connection.transaction(readonly=True):
            await connection.execute("SET LOCAL statement_timeout = '30s'")
            statement = await connection.prepare(plan.sql)
            columns = _columns(statement.get_attributes())
            rows = await statement.fetch(*plan.params)
    except (asyncpg.QueryCanceledError, TimeoutError) as exc:
        logger.warning("UW join query timed out for %s", join_name)
        raise ApiError(ApiErrorCode.QUERY_TIMEOUT, "database query timed out") from exc
    except (asyncpg.DataError, asyncpg.UndefinedFunctionError) as exc:
        logger.warning("invalid UW join filter for %s (%s)", join_name, type(exc).__name__)
        raise ApiError(ApiErrorCode.INVALID_PARAMETER, "join filter value is invalid") from exc
    except (
        asyncpg.PostgresConnectionError,
        asyncpg.CannotConnectNowError,
        ConnectionError,
        OSError,
    ) as exc:
        logger.warning("UW database unavailable for %s (%s)", join_name, type(exc).__name__)
        raise ApiError(ApiErrorCode.PROVIDER_NOT_CONFIGURED, "database unavailable") from exc
    except Exception as exc:
        logger.warning("UW join query failed for %s (%s)", join_name, type(exc).__name__)
        raise
    return build_tabular(
        "option_wizard",
        "uw_scan",
        join_name,
        columns,
        rows,
        plan.limit,
        coverage=_coverage(plan, rows),
    )
