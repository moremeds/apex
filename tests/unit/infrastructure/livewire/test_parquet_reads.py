"""LakeDb deadlines: an expired query is interrupted alone and reported as a timeout.

The slow query is a synthetic CPU load (``range`` over 20 billion integers), not
market data; it exists only to outlive the deadline.
"""

from __future__ import annotations

import asyncio

import pytest

from src.infrastructure.adapters.livewire.parquet_reads import LakeDb, QueryTimeout

_SLOW = "SELECT count(*) AS n FROM range(20000000000) t(i) WHERE i % 7 = 3"


@pytest.mark.parametrize("mode", ["per_call", "cursor"])
async def test_deadline_interrupts_only_the_expired_query(mode: str) -> None:
    slow, fast = LakeDb(mode=mode, timeout=0.3), LakeDb(mode=mode, timeout=5)

    results = await asyncio.gather(
        slow.rows(_SLOW, []), fast.rows("SELECT 42 AS answer", []), return_exceptions=True
    )

    assert isinstance(results[0], QueryTimeout)
    assert results[1] == [{"answer": 42}]
    # The handle is usable again afterwards: nothing was left interrupted or open.
    assert await fast.rows("SELECT 1 AS one", []) == [{"one": 1}]


def test_sync_rows_bind_parameters() -> None:
    assert LakeDb(mode="per_call").rows_sync("SELECT ? AS v", ["x"]) == [{"v": "x"}]
