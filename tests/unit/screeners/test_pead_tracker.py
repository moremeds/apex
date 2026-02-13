"""Unit tests for PEAD tracker service — OHLC first-touch resolution."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.domain.screeners.pead.tracker import TrackedCandidate, TrackerStats
from src.services.pead_tracker_service import PEADTrackerService


def _make_tracked(
    symbol: str = "AAPL",
    entry_date: date = date(2025, 1, 29),
    entry_price: float = 200.0,
    profit_target_pct: float = 0.06,
    stop_loss_pct: float = -0.05,
    max_hold_days: int = 25,
    quality_score: float = 75.0,
    quality_label: str = "STRONG",
    sue_score: float = 4.5,
    regime: str = "R0",
) -> TrackedCandidate:
    return TrackedCandidate(
        symbol=symbol,
        entry_date=entry_date,
        entry_price=entry_price,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
        max_hold_days=max_hold_days,
        quality_score=quality_score,
        quality_label=quality_label,
        sue_score=sue_score,
        multi_quarter_sue=None,
        regime=regime,
    )


def _make_ohlc_df(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """Build yfinance-style OHLC DataFrame: [(date, open, high, low, close), ...]"""
    dates = pd.to_datetime([r[0] for r in rows])
    data = {
        "Open": [r[1] for r in rows],
        "High": [r[2] for r in rows],
        "Low": [r[3] for r in rows],
        "Close": [r[4] for r in rows],
        "Volume": [1_000_000] * len(rows),
    }
    return pd.DataFrame(data, index=dates)


class TestTrackedCandidate:
    def test_target_price(self) -> None:
        c = _make_tracked(entry_price=100.0, profit_target_pct=0.06)
        assert c.target_price == 106.0

    def test_stop_price(self) -> None:
        c = _make_tracked(entry_price=100.0, stop_loss_pct=-0.05)
        assert c.stop_price == 95.0

    def test_round_trip_serialization(self) -> None:
        c = _make_tracked()
        d = c.to_dict()
        c2 = TrackedCandidate.from_dict(d)
        assert c2.symbol == c.symbol
        assert c2.entry_date == c.entry_date
        assert c2.entry_price == c.entry_price
        assert c2.status == "open"


class TestTrackerPersistence:
    def test_add_and_load(self) -> None:
        """Add candidates, save, reload, verify persistence."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        c = _make_tracked()

        # Manually add via internal API
        tracker._load()
        tracker._candidates = [c]
        tracker._save()

        # Reload
        tracker2 = PEADTrackerService(tracker_path=path)
        loaded = tracker2._load()
        assert len(loaded) == 1
        assert loaded[0].symbol == "AAPL"

        path.unlink(missing_ok=True)

    def test_dedup_by_symbol_date(self) -> None:
        """No duplicate tracking for same (symbol, entry_date)."""
        from src.domain.screeners.pead.models import (
            EarningsSurprise,
            LiquidityTier,
            PEADCandidate,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)

        surprise = EarningsSurprise(
            symbol="AAPL",
            report_date=date(2025, 1, 27),
            actual_eps=2.5,
            consensus_eps=2.0,
            surprise_pct=25.0,
            sue_score=4.5,
            earnings_day_return=0.06,
            earnings_day_gap=0.05,
            earnings_day_volume_ratio=3.0,
            revenue_beat=True,
            at_52w_high=False,
            analyst_downgrade=False,
            liquidity_tier=LiquidityTier.LARGE_CAP,
        )
        candidate = PEADCandidate(
            symbol="AAPL",
            surprise=surprise,
            entry_date=date(2025, 1, 29),
            entry_price=200.0,
            profit_target_pct=0.06,
            stop_loss_pct=-0.05,
            trailing_stop_atr=2.0,
            trailing_activation_pct=0.03,
            max_hold_days=25,
            position_size_factor=1.0,
            quality_score=75.0,
            quality_label="STRONG",
            regime="R0",
            gap_held=True,
            estimated_slippage_bps=10,
        )

        # Add twice
        added1 = tracker.add_candidates([candidate])
        added2 = tracker.add_candidates([candidate])
        assert added1 == 1
        assert added2 == 0  # Duplicate

        path.unlink(missing_ok=True)


class TestFirstTouchResolution:
    @patch("src.services.pead_tracker_service.yf")
    def test_first_touch_stop_hit(self, mock_yf: MagicMock) -> None:
        """Stop loss hit before target."""
        candidate = _make_tracked(entry_price=100.0, profit_target_pct=0.06, stop_loss_pct=-0.05)
        # Day 1: price drops to 94 (below stop of 95)
        hist = _make_ohlc_df(
            [
                ("2025-01-29", 100.0, 101.0, 94.0, 96.0),
            ]
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        tracker._candidates = [candidate]

        result = tracker._resolve_candidate(candidate)
        assert result is True
        assert candidate.status == "lost"
        assert candidate.exit_reason == "stop_loss"
        assert candidate.pnl_pct == -0.05

        path.unlink(missing_ok=True)

    @patch("src.services.pead_tracker_service.yf")
    def test_first_touch_target_hit(self, mock_yf: MagicMock) -> None:
        """Profit target hit before stop."""
        candidate = _make_tracked(entry_price=100.0, profit_target_pct=0.06, stop_loss_pct=-0.05)
        # Day 1: price rises to 107 (above target of 106)
        hist = _make_ohlc_df(
            [
                ("2025-01-29", 100.0, 107.0, 99.0, 106.0),
            ]
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        tracker._candidates = [candidate]

        result = tracker._resolve_candidate(candidate)
        assert result is True
        assert candidate.status == "won"
        assert candidate.exit_reason == "profit_target"
        assert candidate.pnl_pct == 0.06

        path.unlink(missing_ok=True)

    @patch("src.services.pead_tracker_service.yf")
    def test_same_bar_stop_wins(self, mock_yf: MagicMock) -> None:
        """Both stop and target triggered on same bar — stop wins (conservative)."""
        candidate = _make_tracked(entry_price=100.0, profit_target_pct=0.06, stop_loss_pct=-0.05)
        # Extreme volatility: low=94 (stop hit), high=107 (target hit)
        hist = _make_ohlc_df(
            [
                ("2025-01-29", 100.0, 107.0, 94.0, 102.0),
            ]
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        tracker._candidates = [candidate]

        result = tracker._resolve_candidate(candidate)
        assert result is True
        assert candidate.status == "lost"
        assert candidate.exit_reason == "stop_loss"

        path.unlink(missing_ok=True)

    @patch("src.services.pead_tracker_service.yf")
    def test_timeout_at_max_hold(self, mock_yf: MagicMock) -> None:
        """Neither target nor stop hit within max_hold_days — timeout."""
        candidate = _make_tracked(
            entry_price=100.0, profit_target_pct=0.06, stop_loss_pct=-0.05, max_hold_days=3
        )
        # 3 days of sideways action — neither triggered
        hist = _make_ohlc_df(
            [
                ("2025-01-29", 100.0, 103.0, 97.0, 101.0),
                ("2025-01-30", 101.0, 104.0, 98.0, 102.0),
                ("2025-01-31", 102.0, 104.0, 98.0, 103.0),
            ]
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        tracker._candidates = [candidate]

        result = tracker._resolve_candidate(candidate)
        assert result is True
        assert candidate.status == "timeout"
        assert candidate.exit_reason == "timeout"
        assert candidate.pnl_pct is not None
        assert abs(candidate.pnl_pct - 0.03) < 0.001  # (103-100)/100

        path.unlink(missing_ok=True)

    @patch("src.services.pead_tracker_service.yf")
    def test_ohlc_data_unavailable(self, mock_yf: MagicMock) -> None:
        """Candidate stays open when OHLC data unavailable."""
        candidate = _make_tracked()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty
        mock_yf.Ticker.return_value = mock_ticker

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        tracker._candidates = [candidate]

        result = tracker._resolve_candidate(candidate)
        assert result is False
        assert candidate.status == "open"

        path.unlink(missing_ok=True)


class TestTrackerStats:
    def test_stats_with_zero_resolved(self) -> None:
        """Empty tracker returns zero stats."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        stats = tracker.get_stats()
        assert stats.total == 0
        assert stats.win_rate is None
        assert stats.avg_pnl_pct is None

        path.unlink(missing_ok=True)

    def test_stats_with_mixed_outcomes(self) -> None:
        """Stats computed correctly with mixed won/lost/timeout."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        tracker = PEADTrackerService(tracker_path=path)
        candidates = [
            _make_tracked(symbol="WIN1", quality_label="STRONG"),
            _make_tracked(symbol="WIN2", quality_label="STRONG"),
            _make_tracked(symbol="LOSE", quality_label="MODERATE"),
            _make_tracked(symbol="OPEN", quality_label="MARGINAL"),
        ]
        # Resolve manually
        candidates[0].status = "won"
        candidates[0].pnl_pct = 0.06
        candidates[0].exit_date = date(2025, 2, 10)
        candidates[1].status = "won"
        candidates[1].pnl_pct = 0.06
        candidates[1].exit_date = date(2025, 2, 12)
        candidates[2].status = "lost"
        candidates[2].pnl_pct = -0.05
        candidates[2].exit_date = date(2025, 2, 5)

        tracker._candidates = candidates
        stats = tracker.get_stats()

        assert stats.total == 4
        assert stats.open == 1
        assert stats.won == 2
        assert stats.lost == 1
        assert stats.timeout == 0
        # Win rate = 2/3 resolved
        assert stats.win_rate is not None
        assert abs(stats.win_rate - 2 / 3) < 0.01
        # Avg P&L = (0.06 + 0.06 - 0.05) / 3
        assert stats.avg_pnl_pct is not None
        expected_pnl = (0.06 + 0.06 - 0.05) / 3
        assert abs(stats.avg_pnl_pct - expected_pnl) < 0.001

        path.unlink(missing_ok=True)
