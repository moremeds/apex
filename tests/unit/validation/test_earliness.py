"""Tests for Earliness Metrics."""

from datetime import date

import pytest

from src.domain.signals.validation.earliness import (
    EarlinessResult,
    TrendEpisode,
    compute_earliness,
    compute_multi_tf_earliness,
    detect_trend_episodes,
    find_first_signal_date,
)


class TestTrendEpisode:
    """Tests for TrendEpisode dataclass."""

    def test_duration_days(self):
        """Test duration calculation."""
        episode = TrendEpisode(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 15),
            symbol="AAPL",
        )
        assert episode.duration_days == 14


class TestDetectTrendEpisodes:
    """Tests for detect_trend_episodes function."""

    def test_single_episode(self):
        """Test detection of single episode."""
        signals = {
            date(2024, 1, 1): False,
            date(2024, 1, 2): True,
            date(2024, 1, 3): True,
            date(2024, 1, 4): True,
            date(2024, 1, 5): True,
            date(2024, 1, 6): True,
            date(2024, 1, 7): True,
            date(2024, 1, 8): True,
            date(2024, 1, 9): True,
            date(2024, 1, 10): True,
            date(2024, 1, 11): True,
            date(2024, 1, 12): True,
            date(2024, 1, 13): True,
            date(2024, 1, 14): False,
        }

        episodes = detect_trend_episodes(signals, min_episode_days=10)

        assert len(episodes) == 1
        assert episodes[0].start_date == date(2024, 1, 2)

    def test_multiple_episodes(self):
        """Test detection of multiple episodes."""
        signals = {}
        # First episode: Jan 1-15
        for i in range(1, 16):
            signals[date(2024, 1, i)] = True
        # Gap: Jan 16-20
        for i in range(16, 21):
            signals[date(2024, 1, i)] = False
        # Second episode: Jan 21-31 + Feb 1-4 (14 days)
        for i in range(21, 32):
            signals[date(2024, 1, i)] = True
        for i in range(1, 5):
            signals[date(2024, 2, i)] = True

        episodes = detect_trend_episodes(signals, min_episode_days=10)

        assert len(episodes) == 2

    def test_short_episode_filtered(self):
        """Test that short episodes are filtered out."""
        signals = {date(2024, 1, i): True for i in range(1, 6)}  # 5 days
        for i in range(6, 10):
            signals[date(2024, 1, i)] = False

        episodes = detect_trend_episodes(signals, min_episode_days=10)

        assert len(episodes) == 0

    def test_empty_signals(self):
        """Test handling of empty signals."""
        episodes = detect_trend_episodes({}, min_episode_days=5)
        assert len(episodes) == 0


class TestFindFirstSignalDate:
    """Tests for find_first_signal_date function."""

    def test_finds_early_signal(self):
        """Test finding early signal before episode."""
        signals = {
            date(2024, 1, 1): False,
            date(2024, 1, 2): True,  # Early signal
            date(2024, 1, 3): False,
            date(2024, 1, 4): True,
            date(2024, 1, 5): True,
        }
        episode_start = date(2024, 1, 4)

        first = find_first_signal_date(signals, episode_start, lookback_days=5)

        assert first == date(2024, 1, 2)

    def test_no_signal_found(self):
        """Test when no signal is found."""
        signals = {
            date(2024, 1, 1): False,
            date(2024, 1, 2): False,
        }
        episode_start = date(2024, 1, 5)

        first = find_first_signal_date(signals, episode_start, lookback_days=3)

        assert first is None


class TestComputeEarliness:
    """Tests for compute_earliness function."""

    @pytest.fixture
    def baseline_signals(self):
        """Create baseline (1d) signals with a trending episode."""
        signals = {}
        # Non-trending: Jan 1-10
        for i in range(1, 11):
            signals[date(2024, 1, i)] = False
        # Trending: Jan 11-30
        for i in range(11, 31):
            signals[date(2024, 1, i)] = True
        return signals

    @pytest.fixture
    def faster_signals_early(self):
        """Create faster TF signals that detect 3 days earlier."""
        signals = {}
        # Non-trending: Jan 1-7
        for i in range(1, 8):
            signals[date(2024, 1, i)] = False
        # Trending: Jan 8-30 (3 days earlier than baseline)
        for i in range(8, 31):
            signals[date(2024, 1, i)] = True
        return signals

    @pytest.fixture
    def faster_signals_late(self):
        """Create faster TF signals that detect later."""
        signals = {}
        # Non-trending: Jan 1-14
        for i in range(1, 15):
            signals[date(2024, 1, i)] = False
        # Trending: Jan 15-30 (4 days later than baseline)
        for i in range(15, 31):
            signals[date(2024, 1, i)] = True
        return signals

    def test_faster_tf_earlier(self, baseline_signals, faster_signals_early):
        """Test when faster TF fires earlier."""
        result = compute_earliness(
            signals_baseline=baseline_signals,
            signals_faster=faster_signals_early,
            min_episode_days=10,
        )

        assert result.n_episodes == 1
        assert result.median_earliness_days > 0  # Positive = earlier
        assert result.pct_earlier_than_baseline == 1.0  # 100% earlier

    def test_faster_tf_later(self, baseline_signals, faster_signals_late):
        """Test when faster TF fires later (negative earliness)."""
        result = compute_earliness(
            signals_baseline=baseline_signals,
            signals_faster=faster_signals_late,
            min_episode_days=10,
        )

        assert result.n_episodes == 1
        assert result.median_earliness_days < 0  # Negative = later
        assert result.pct_earlier_than_baseline == 0.0

    def test_empty_baseline(self):
        """Test with empty baseline signals."""
        result = compute_earliness(
            signals_baseline={},
            signals_faster={date(2024, 1, i): True for i in range(1, 20)},
            min_episode_days=10,
        )

        assert result.n_episodes == 0
        assert result.median_earliness_days == 0.0


class TestEarlinessResult:
    """Tests for EarlinessResult dataclass."""

    def test_passes_gates_success(self):
        """Test when all gates pass."""
        result = EarlinessResult(
            tf_pair="4h_vs_1d",
            median_earliness_days=1.5,
            p75_earliness_days=2.5,
            pct_earlier_than_baseline=0.70,
            ci_95=(1.0, 2.0),
            n_episodes=30,
            earliness_values=[1.5] * 30,
        )

        passes, failures = result.passes_gates(
            min_median_earliness=1.0,
            min_pct_earlier=0.60,
        )

        assert passes
        assert len(failures) == 0

    def test_passes_gates_fails_median(self):
        """Test when median earliness gate fails."""
        result = EarlinessResult(
            tf_pair="4h_vs_1d",
            median_earliness_days=0.5,  # Below threshold
            p75_earliness_days=1.0,
            pct_earlier_than_baseline=0.70,
            ci_95=(0.3, 0.8),
            n_episodes=30,
            earliness_values=[0.5] * 30,
        )

        passes, failures = result.passes_gates(min_median_earliness=1.0)

        assert not passes
        assert any("median_earliness_days" in f for f in failures)

    def test_passes_gates_fails_pct(self):
        """Test when pct_earlier gate fails."""
        result = EarlinessResult(
            tf_pair="4h_vs_1d",
            median_earliness_days=1.5,
            p75_earliness_days=2.5,
            pct_earlier_than_baseline=0.50,  # Below threshold
            ci_95=(1.0, 2.0),
            n_episodes=30,
            earliness_values=[1.5] * 30,
        )

        passes, failures = result.passes_gates(min_pct_earlier=0.60)

        assert not passes
        assert any("pct_earlier_than_baseline" in f for f in failures)

    def test_to_dict(self):
        """Test serialization."""
        result = EarlinessResult(
            tf_pair="4h_vs_1d",
            median_earliness_days=1.5,
            p75_earliness_days=2.5,
            pct_earlier_than_baseline=0.70,
            ci_95=(1.0, 2.0),
            n_episodes=30,
            earliness_values=[],
        )

        d = result.to_dict()

        assert d["tf_pair"] == "4h_vs_1d"
        assert d["median_earliness_days"] == 1.5
        assert d["ci_95"] == [1.0, 2.0]


class TestComputeMultiTfEarliness:
    """Tests for compute_multi_tf_earliness function."""

    def test_computes_for_all_tfs(self):
        """Test computation for multiple timeframes."""
        signals_by_tf = {
            "1d": {date(2024, 1, i): (i >= 10) for i in range(1, 31)},
            "4h": {date(2024, 1, i): (i >= 8) for i in range(1, 31)},
            "2h": {date(2024, 1, i): (i >= 7) for i in range(1, 31)},
        }

        results = compute_multi_tf_earliness(
            signals_by_tf=signals_by_tf,
            baseline_tf="1d",
            faster_tfs=["4h", "2h"],
            min_episode_days=10,
        )

        assert "4h_vs_1d" in results
        assert "2h_vs_1d" in results
        assert results["4h_vs_1d"].median_earliness_days >= 0
        assert results["2h_vs_1d"].median_earliness_days >= 0

    def test_missing_baseline(self):
        """Test handling of missing baseline."""
        signals_by_tf = {
            "4h": {date(2024, 1, i): (i >= 8) for i in range(1, 31)},
        }

        results = compute_multi_tf_earliness(
            signals_by_tf=signals_by_tf,
            baseline_tf="1d",
        )

        assert len(results) == 0

    def test_missing_faster_tf(self):
        """Test handling of missing faster TF."""
        signals_by_tf = {
            "1d": {date(2024, 1, i): (i >= 10) for i in range(1, 31)},
        }

        results = compute_multi_tf_earliness(
            signals_by_tf=signals_by_tf,
            baseline_tf="1d",
            faster_tfs=["4h", "2h"],
        )

        assert len(results) == 0
