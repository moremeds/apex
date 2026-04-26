"""Optional PG publish for CLI runners.

Writes results to PostgreSQL if APEX_PG_URL is set. Silently skips otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


def publish_screener_results(
    run_id: str, screener_type: str, results: list[dict]
) -> None:
    """Write screener results to PG. No-op if APEX_PG_URL not set."""
    pg_url = os.environ.get("APEX_PG_URL")
    if not pg_url:
        return
    try:
        asyncio.run(_publish_screener(pg_url, run_id, screener_type, results))
    except Exception:
        logger.warning(
            "PG publish failed for %s/%s", screener_type, run_id, exc_info=True
        )


def publish_backtest_results(
    run_id: str, strategy: str, symbols: list[str], metrics: dict
) -> None:
    """Write backtest results to PG. No-op if APEX_PG_URL not set."""
    pg_url = os.environ.get("APEX_PG_URL")
    if not pg_url:
        return
    try:
        asyncio.run(_publish_backtest(pg_url, run_id, strategy, symbols, metrics))
    except Exception:
        logger.warning("PG publish failed for %s/%s", strategy, run_id, exc_info=True)


class _PoolDB:
    """Minimal Database-compatible wrapper around raw asyncpg pool."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def execute(self, sql: str, *args: Any) -> Any:
        async with self._pool.acquire() as conn:
            return await conn.execute(sql, *args)

    async def fetch(self, sql: str, *args: Any) -> Any:
        async with self._pool.acquire() as conn:
            return await conn.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(sql, *args)

    @asynccontextmanager
    async def transaction(self) -> Any:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def close(self) -> None:
        await self._pool.close()


async def _get_repo(pg_url: str) -> tuple[Any, _PoolDB]:
    import asyncpg

    from src.infrastructure.persistence.pg_repositories import PgRepositories
    from src.infrastructure.persistence.pg_schema import ensure_schema

    pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=2)
    db: Any = _PoolDB(pool)
    await ensure_schema(db)
    return PgRepositories(db), db


async def _publish_screener(
    pg_url: str, run_id: str, screener_type: str, results: list[dict]
) -> None:
    repo, db = await _get_repo(pg_url)
    await repo.insert_screener_results(run_id, screener_type, results)
    await db.close()
    logger.info(
        "Published %d %s results to PG (run=%s)", len(results), screener_type, run_id
    )


async def _publish_backtest(
    pg_url: str, run_id: str, strategy: str, symbols: list[str], metrics: dict
) -> None:
    repo, db = await _get_repo(pg_url)
    await repo.insert_backtest_results(run_id, strategy, symbols, metrics)
    await db.close()
    logger.info("Published %s backtest results to PG (run=%s)", strategy, run_id)
