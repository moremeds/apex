"""save_indicator: persist indicator state as JSONB without choking on numpy scalars.

Live indicators (e.g. bollinger's ``squeeze``) carry numpy bools in their ``state``.
stdlib ``json.dumps`` raises ``TypeError: Object of type bool is not JSON serializable``
on those, which previously killed the best-effort persist in a background task. The repo
must coerce numpy scalars to JSON-native types (and the persisted JSON must contain a real
boolean, not the string ``"True"``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from src.infrastructure.persistence.repositories.ta_signal_repository import (
    TASignalRepository,
)


class _FakeDB:
    """Records execute() params, like asyncpg would receive them."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def execute(self, query: str, *args):
        self.calls.append((query, args))
        return "INSERT 0 1"


@pytest.mark.asyncio
async def test_save_indicator_coerces_numpy_bool_in_state() -> None:
    db = _FakeDB()
    repo = TASignalRepository(db)

    state = {
        "value": np.float64(208.7),
        "squeeze": np.bool_(True),  # numpy bool -> would raise TypeError on json.dumps
        "upper": np.float64("nan"),  # NaN -> must become null (invalid JSON otherwise)
    }

    # Previously raised: "Object of type bool is not JSON serializable".
    await repo.save_indicator(
        symbol="AAPL",
        timeframe="1d",
        indicator="bollinger",
        timestamp=datetime(2026, 6, 14, tzinfo=timezone.utc),
        state=state,
    )

    assert len(db.calls) == 1
    _query, args = db.calls[0]
    state_json = args[4]  # 5th INSERT param is state (json string)
    decoded = json.loads(state_json)
    assert decoded["squeeze"] is True  # real JSON boolean, not "True"
    assert decoded["value"] == 208.7
    assert decoded["upper"] is None  # NaN -> null


@pytest.mark.asyncio
async def test_save_indicator_coerces_numpy_in_previous_state() -> None:
    db = _FakeDB()
    repo = TASignalRepository(db)

    await repo.save_indicator(
        symbol="AAPL",
        timeframe="1d",
        indicator="bollinger",
        timestamp=datetime(2026, 6, 14, tzinfo=timezone.utc),
        state={"squeeze": np.bool_(False)},
        previous_state={"squeeze": np.bool_(True)},
    )

    _query, args = db.calls[0]
    assert json.loads(args[4])["squeeze"] is False  # state
    assert json.loads(args[5])["squeeze"] is True  # previous_state
