"""Dedicated, least-privilege PostgreSQL pools for the generic read API."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

import asyncpg

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _encode_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=_json_default)


def _decode_json(value: str) -> Any:
    return json.loads(value, parse_float=Decimal)


async def _init_read_connection(connection: asyncpg.Connection[Any]) -> None:
    for type_name in ("json", "jsonb"):
        await connection.set_type_codec(
            type_name,
            schema="pg_catalog",
            encoder=_encode_json,
            decoder=_decode_json,
            format="text",
        )


def parse_read_urls(raw: str | None) -> dict[str, str]:
    """Return database-name keyed DSNs, rejecting ambiguous database overrides."""
    urls: dict[str, str] = {}
    if not raw:
        return urls
    for dsn in (item.strip() for item in raw.split(",")):
        if not dsn:
            continue
        parsed = urlsplit(dsn)
        database = unquote(parsed.path.removeprefix("/"))
        query_names = {name.lower() for name, _ in parse_qsl(parsed.query, keep_blank_values=True)}
        if (
            parsed.scheme not in {"postgres", "postgresql"}
            or not parsed.netloc
            or not database
            or "/" in database
            or query_names.intersection({"database", "dbname"})
        ):
            raise ValueError("invalid PostgreSQL read URL")
        if database in urls:
            raise ValueError(f"duplicate PostgreSQL read database: {database}")
        urls[database] = dsn
    return urls


async def create_read_pools(raw: str | None = None) -> dict[str, asyncpg.Pool[Any]]:
    """Create isolated pools; one bad database cannot prevent the others starting."""
    try:
        urls = parse_read_urls(raw if raw is not None else os.environ.get("APEX_PG_READ_URLS"))
    except ValueError as exc:
        logger.warning("PostgreSQL read pool configuration rejected: %s", exc)
        return {}

    pools: dict[str, asyncpg.Pool[Any]] = {}
    for database, dsn in urls.items():
        try:
            pools[database] = await asyncpg.create_pool(
                dsn,
                min_size=0,
                max_size=3,
                server_settings={
                    "default_transaction_read_only": "on",
                    "statement_timeout": "30s",
                },
                init=_init_read_connection,
            )
        except Exception as exc:  # asyncpg exposes several connection exception types
            logger.warning(
                "PostgreSQL read pool unavailable for database %s (%s)",
                database,
                type(exc).__name__,
            )
    return pools


async def close_read_pools(pools: Mapping[str, asyncpg.Pool[Any]]) -> None:
    """Close every successfully-created read pool."""
    for database, pool in pools.items():
        try:
            await pool.close()
        except Exception as exc:  # pragma: no cover - best-effort lifespan teardown
            logger.warning(
                "PostgreSQL read pool close failed for database %s (%s)",
                database,
                type(exc).__name__,
            )
