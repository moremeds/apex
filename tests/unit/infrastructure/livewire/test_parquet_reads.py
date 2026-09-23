"""LakeDb deadlines: an expired or cancelled query is interrupted alone.

The queries are synthetic CPU loads over ``range`` (not market data); they exist
only to outlive a deadline or overlap one.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.infrastructure.adapters.livewire.parquet_reads import LakeDb, QueryTimeout

_SLOW = "SELECT count(*) AS n FROM range(20000000000) t(i) WHERE i % 7 = 3"
_MEDIUM_N = 1_500_000_000
_MEDIUM = f"SELECT count(*) AS n FROM range({_MEDIUM_N}) t(i) WHERE i % 7 = 3"
_MEDIUM_ANSWER = (_MEDIUM_N - 1 - 3) // 7 + 1


async def test_deadline_interrupts_only_the_expired_query() -> None:
    """The sibling query is still running inside DuckDB when the interrupt fires."""
    slow, sibling = LakeDb(timeout=0.3), LakeDb(timeout=30)
    finished: dict[str, float] = {}

    async def timed(name: str, db: LakeDb, sql: str) -> object:
        try:
            return await db.rows(sql, [])
        finally:
            finished[name] = time.perf_counter()

    results = await asyncio.gather(
        timed("slow", slow, _SLOW), timed("sibling", sibling, _MEDIUM), return_exceptions=True
    )
    assert isinstance(results[0], QueryTimeout)
    assert results[1] == [{"n": _MEDIUM_ANSWER}]
    # Not vacuous: the sibling was still executing when the slow query was interrupted.
    assert finished["sibling"] > finished["slow"]


async def test_cancelled_request_interrupts_its_query() -> None:
    db = LakeDb(timeout=30)
    task = asyncio.create_task(db.rows(_SLOW, []))
    await asyncio.sleep(0.2)
    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.perf_counter() - started < 1  # returned without waiting for the scan
    assert await db.rows("SELECT 1 AS one", []) == [{"one": 1}]


def test_sync_rows_bind_parameters_and_honour_the_deadline() -> None:
    assert LakeDb().rows_sync("SELECT ? AS v", ["x"]) == [{"v": "x"}]
    with pytest.raises(QueryTimeout):
        LakeDb(timeout=0.3).rows_sync(_SLOW, [])
