"""Tests for momentum runner helper functions."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from src.domain.screeners.momentum.models import MomentumScreenResult


class TestWriteWatchlistJson:
    def test_errors_serialized_as_list(self, tmp_path: Path) -> None:
        """dict[str, str] errors should become a list of 'key: value' strings."""
        from src.runners.momentum_runner import _write_watchlist_json

        result = MomentumScreenResult(
            candidates=[],
            universe_size=100,
            passed_filters=50,
            regime="R0",
            generated_at=datetime(2026, 2, 18, 12, 0, 0),
            errors={"TSLA": "insufficient data", "GME": "delisted"},
        )

        out_path = _write_watchlist_json(result, tmp_path)
        data = json.loads(out_path.read_text())

        errors = data["errors"]
        assert isinstance(errors, list)
        assert len(errors) == 2
        assert "TSLA: insufficient data" in errors
        assert "GME: delisted" in errors

    def test_data_as_of_included_when_provided(self, tmp_path: Path) -> None:
        from src.runners.momentum_runner import _write_watchlist_json

        result = MomentumScreenResult(
            candidates=[],
            universe_size=100,
            passed_filters=50,
            regime="R0",
            generated_at=datetime(2026, 2, 18, 12, 0, 0),
            errors={},
        )

        out_path = _write_watchlist_json(result, tmp_path, data_as_of=date(2026, 2, 17))
        data = json.loads(out_path.read_text())

        assert data["data_as_of"] == "2026-02-17"

    def test_data_as_of_absent_when_none(self, tmp_path: Path) -> None:
        from src.runners.momentum_runner import _write_watchlist_json

        result = MomentumScreenResult(
            candidates=[],
            universe_size=100,
            passed_filters=50,
            regime="R0",
            generated_at=datetime(2026, 2, 18, 12, 0, 0),
            errors={},
        )

        out_path = _write_watchlist_json(result, tmp_path, data_as_of=None)
        data = json.loads(out_path.read_text())

        assert "data_as_of" not in data
