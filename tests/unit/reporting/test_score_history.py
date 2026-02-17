"""
Tests for ScoreHistoryManager — load/save roundtrip, append_from_summary,
trim, trend detection, sparklines, and change detection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.infrastructure.reporting.package.score_history import (
    MAX_SNAPSHOTS,
    ScoreHistoryManager,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_summary(
    scores: Dict[str, float],
    trend_states: Dict[str, str] | None = None,
    momentum: Dict[str, float] | None = None,
    tactical: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Build a minimal summary dict for append_from_summary."""
    tickers = []
    for symbol, score in scores.items():
        t: Dict[str, Any] = {"symbol": symbol, "composite_score_avg": score}
        if trend_states and symbol in trend_states:
            t["component_states"] = {"trend_state": trend_states[symbol]}
        if momentum and symbol in momentum:
            t["component_values"] = {"ma50_slope": momentum[symbol]}
        tickers.append(t)

    result: Dict[str, Any] = {"tickers": tickers}

    if tactical:
        result["dual_macd"] = {
            "1d": {"trends": [{"symbol": s, "tactical_signal": sig} for s, sig in tactical.items()]}
        }

    return result


# =============================================================================
# Construction
# =============================================================================


class TestScoreHistoryConstruction:
    def test_default_state(self) -> None:
        mgr = ScoreHistoryManager()
        assert mgr.history.snapshots == []
        assert mgr.max_entries == MAX_SNAPSHOTS

    def test_custom_max_entries(self) -> None:
        mgr = ScoreHistoryManager(max_entries=5)
        assert mgr.max_entries == 5


# =============================================================================
# Load / Save roundtrip
# =============================================================================


class TestLoadSave:
    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save then load produces identical data."""
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 72.5, "SPY": 55.0}))
        mgr.append_from_summary(_make_summary({"AAPL": 75.0, "SPY": 52.0}))

        path = tmp_path / "score_history.json"
        mgr.save(path)

        mgr2 = ScoreHistoryManager()
        mgr2.load(path)

        assert len(mgr2.history.snapshots) == 2
        assert mgr2.history.snapshots[0].scores["AAPL"] == 72.5
        assert mgr2.history.snapshots[1].scores["SPY"] == 52.0

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Loading non-existent file leaves empty history."""
        mgr = ScoreHistoryManager()
        mgr.load(tmp_path / "nonexistent.json")
        assert len(mgr.history.snapshots) == 0

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        """Loading corrupt JSON resets to empty history."""
        path = tmp_path / "bad.json"
        path.write_text("{{{invalid", encoding="utf-8")
        mgr = ScoreHistoryManager()
        mgr.load(path)
        assert len(mgr.history.snapshots) == 0

    def test_load_empty_snapshots(self, tmp_path: Path) -> None:
        """Loading file with no snapshots key works."""
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"snapshots": []}), encoding="utf-8")
        mgr = ScoreHistoryManager()
        mgr.load(path)
        assert len(mgr.history.snapshots) == 0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Save creates parent directories if missing."""
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}))
        path = tmp_path / "nested" / "deep" / "history.json"
        mgr.save(path)
        assert path.exists()

    def test_to_json_data(self) -> None:
        """to_json_data returns serializable dict."""
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 60.0}))
        data = mgr.to_json_data()
        assert "snapshots" in data
        assert len(data["snapshots"]) == 1
        # Must be JSON-serializable
        json.dumps(data)


# =============================================================================
# append_from_summary
# =============================================================================


class TestAppendFromSummary:
    def test_basic_append(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 72.5}))
        assert len(mgr.history.snapshots) == 1
        assert mgr.history.snapshots[0].scores["AAPL"] == 72.5

    def test_scores_rounded(self) -> None:
        """Composite scores are rounded to 1 decimal."""
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 72.555}))
        assert mgr.history.snapshots[0].scores["AAPL"] == 72.6

    def test_skip_none_score(self) -> None:
        """Symbols with None score are excluded."""
        summary: Dict[str, Any] = {
            "tickers": [
                {"symbol": "AAPL", "composite_score_avg": 50.0},
                {"symbol": "BAD", "composite_score_avg": None},
            ]
        }
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(summary)
        assert "AAPL" in mgr.history.snapshots[0].scores
        assert "BAD" not in mgr.history.snapshots[0].scores

    def test_skip_when_no_scores(self) -> None:
        """Empty scores dict skips snapshot creation."""
        mgr = ScoreHistoryManager()
        mgr.append_from_summary({"tickers": []})
        assert len(mgr.history.snapshots) == 0

    def test_custom_timestamp(self) -> None:
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 60.0}), timestamp=ts)
        assert "2024-06-15" in mgr.history.snapshots[0].timestamp

    def test_extracts_trend_states(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 60.0}, trend_states={"AAPL": "uptrend"}))
        assert mgr.history.snapshots[0].trend_states["AAPL"] == "uptrend"

    def test_extracts_momentum_values(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 60.0}, momentum={"AAPL": 0.0234}))
        assert mgr.history.snapshots[0].momentum_values["AAPL"] == 0.0234

    def test_extracts_tactical_signals(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 60.0}, tactical={"AAPL": "DIP_BUY"}))
        assert mgr.history.snapshots[0].tactical_signals["AAPL"] == "DIP_BUY"

    def test_tactical_signal_none_filtered(self) -> None:
        """NONE tactical signals are excluded."""
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 60.0}, tactical={"AAPL": "NONE"}))
        assert "AAPL" not in mgr.history.snapshots[0].tactical_signals


# =============================================================================
# Trim
# =============================================================================


class TestTrim:
    def test_trims_at_max(self) -> None:
        mgr = ScoreHistoryManager(max_entries=3)
        for i in range(5):
            mgr.append_from_summary(_make_summary({"AAPL": float(i)}))
        assert len(mgr.history.snapshots) == 3
        # Should keep the last 3 (scores 2, 3, 4)
        assert mgr.history.snapshots[0].scores["AAPL"] == 2.0

    def test_no_trim_under_max(self) -> None:
        mgr = ScoreHistoryManager(max_entries=10)
        for i in range(5):
            mgr.append_from_summary(_make_summary({"AAPL": float(i)}))
        assert len(mgr.history.snapshots) == 5


# =============================================================================
# Trend detection
# =============================================================================


class TestTrend:
    def test_up_trend(self) -> None:
        mgr = ScoreHistoryManager()
        for s in [50.0, 52.0, 55.0]:
            mgr.append_from_summary(_make_summary({"AAPL": s}))
        assert mgr.get_trend("AAPL") == "up"

    def test_down_trend(self) -> None:
        mgr = ScoreHistoryManager()
        for s in [60.0, 55.0, 52.0]:
            mgr.append_from_summary(_make_summary({"AAPL": s}))
        assert mgr.get_trend("AAPL") == "down"

    def test_stable_trend(self) -> None:
        mgr = ScoreHistoryManager()
        for s in [50.0, 51.0, 52.0]:
            mgr.append_from_summary(_make_summary({"AAPL": s}))
        assert mgr.get_trend("AAPL") == "stable"

    def test_insufficient_data(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}))
        assert mgr.get_trend("AAPL") == "stable"

    def test_missing_symbol(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}))
        assert mgr.get_trend("MISSING") == "stable"

    def test_get_all_trends(self) -> None:
        mgr = ScoreHistoryManager()
        for s in [50.0, 55.0, 60.0]:
            mgr.append_from_summary(_make_summary({"AAPL": s, "SPY": 50.0}))
        trends = mgr.get_all_trends()
        assert trends["AAPL"] == "up"
        assert trends["SPY"] == "stable"

    def test_get_all_trends_empty(self) -> None:
        mgr = ScoreHistoryManager()
        assert mgr.get_all_trends() == {}


# =============================================================================
# Sparklines
# =============================================================================


class TestSparklines:
    def test_sparkline_points(self) -> None:
        mgr = ScoreHistoryManager()
        for s in [50.0, 55.0, 60.0]:
            mgr.append_from_summary(_make_summary({"AAPL": s}))
        points = mgr.get_sparkline_points("AAPL")
        assert points == [50.0, 55.0, 60.0]

    def test_missing_symbol_returns_empty(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}))
        assert mgr.get_sparkline_points("MISSING") == []

    def test_get_all_sparklines(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0, "SPY": 60.0}))
        sparklines = mgr.get_all_sparklines()
        assert "AAPL" in sparklines
        assert "SPY" in sparklines

    def test_get_all_sparklines_empty(self) -> None:
        mgr = ScoreHistoryManager()
        assert mgr.get_all_sparklines() == {}


# =============================================================================
# Change detection
# =============================================================================


class TestChangeDetection:
    def test_score_changes(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0, "SPY": 60.0}))
        mgr.append_from_summary(_make_summary({"AAPL": 60.0, "SPY": 62.0}))
        changes = mgr.get_score_changes(min_delta=5.0)
        assert len(changes) == 1
        assert changes[0]["symbol"] == "AAPL"
        assert changes[0]["delta"] == 10.0

    def test_score_changes_sorted_by_abs_delta(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"A": 50.0, "B": 50.0}))
        mgr.append_from_summary(_make_summary({"A": 56.0, "B": 40.0}))
        changes = mgr.get_score_changes(min_delta=5.0)
        assert changes[0]["symbol"] == "B"  # |delta|=10 > |delta|=6

    def test_score_changes_single_snapshot(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}))
        assert mgr.get_score_changes() == []

    def test_trend_state_changes(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(
            _make_summary(
                {"AAPL": 50.0, "SPY": 50.0},
                trend_states={"AAPL": "uptrend", "SPY": "downtrend"},
            )
        )
        mgr.append_from_summary(
            _make_summary(
                {"AAPL": 50.0, "SPY": 50.0},
                trend_states={"AAPL": "downtrend", "SPY": "downtrend"},
            )
        )
        changes = mgr.get_trend_state_changes()
        assert len(changes) == 1
        assert changes[0]["symbol"] == "AAPL"
        assert changes[0]["prev"] == "uptrend"
        assert changes[0]["curr"] == "downtrend"

    def test_momentum_changes(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}, momentum={"AAPL": 0.01}))
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}, momentum={"AAPL": 0.02}))
        changes = mgr.get_momentum_changes(min_delta=0.005)
        assert len(changes) == 1
        assert changes[0]["symbol"] == "AAPL"

    def test_tactical_signal_changes(self) -> None:
        mgr = ScoreHistoryManager()
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}, tactical={"AAPL": "DIP_BUY"}))
        mgr.append_from_summary(_make_summary({"AAPL": 50.0}, tactical={"AAPL": "RALLY_SELL"}))
        changes = mgr.get_tactical_signal_changes()
        symbols = {c["symbol"] for c in changes}
        assert "AAPL" in symbols
