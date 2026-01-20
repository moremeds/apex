"""
Earliness Metrics for Multi-Timeframe Validation.

Measures whether faster timeframes (4h, 2h) detect regime changes
earlier than slower timeframes (1d), without higher false positive rate.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .statistics import block_bootstrap_ci


@dataclass
class TrendEpisode:
    """A contiguous trending episode."""

    start_date: date
    end_date: date
    symbol: str

    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days


@dataclass
class SignalEvent:
    """A signal event at a specific date."""

    date: date
    signal: bool  # True = R0/TRENDING detected


@dataclass
class EarlinessResult:
    """Earliness comparison result between two timeframes."""

    tf_pair: str  # e.g., "4h_vs_1d"
    median_earliness_days: float
    p75_earliness_days: float
    pct_earlier_than_baseline: float  # % of episodes where faster TF fires first
    ci_95: Tuple[float, float]  # Bootstrap CI for median
    n_episodes: int
    earliness_values: List[float]  # Individual earliness for each episode

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "tf_pair": self.tf_pair,
            "median_earliness_days": self.median_earliness_days,
            "p75_earliness_days": self.p75_earliness_days,
            "pct_earlier_than_baseline": self.pct_earlier_than_baseline,
            "ci_95": list(self.ci_95),
            "n_episodes": self.n_episodes,
        }

    def passes_gates(
        self,
        min_median_earliness: float = 1.0,
        min_pct_earlier: float = 0.60,
    ) -> Tuple[bool, List[str]]:
        """
        Check if earliness result passes gates.

        Args:
            min_median_earliness: Minimum median earliness in days
            min_pct_earlier: Minimum percentage earlier than baseline

        Returns:
            Tuple of (passes, list_of_failures)
        """
        failures = []

        if self.median_earliness_days < min_median_earliness:
            failures.append(
                f"median_earliness_days ({self.median_earliness_days:.2f}) "
                f"< {min_median_earliness}"
            )

        if self.pct_earlier_than_baseline < min_pct_earlier:
            failures.append(
                f"pct_earlier_than_baseline ({self.pct_earlier_than_baseline:.2%}) "
                f"< {min_pct_earlier:.2%}"
            )

        return len(failures) == 0, failures


def detect_trend_episodes(
    signals: Dict[date, bool],
    min_episode_days: int = 10,
) -> List[TrendEpisode]:
    """
    Detect contiguous trending episodes from signal history.

    Args:
        signals: Dict mapping date -> bool (True = trending)
        min_episode_days: Minimum days for episode to count

    Returns:
        List of TrendEpisode objects
    """
    if not signals:
        return []

    # Sort dates
    sorted_dates = sorted(signals.keys())
    episodes = []

    episode_start: Optional[date] = None

    for d in sorted_dates:
        is_trending = signals[d]

        if is_trending and episode_start is None:
            # Start new episode
            episode_start = d
        elif not is_trending and episode_start is not None:
            # End episode
            duration = (d - episode_start).days
            if duration >= min_episode_days:
                episodes.append(
                    TrendEpisode(
                        start_date=episode_start,
                        end_date=d - timedelta(days=1),
                        symbol="",  # Set by caller
                    )
                )
            episode_start = None

    # Handle episode that extends to end
    if episode_start is not None:
        duration = (sorted_dates[-1] - episode_start).days
        if duration >= min_episode_days:
            episodes.append(
                TrendEpisode(
                    start_date=episode_start,
                    end_date=sorted_dates[-1],
                    symbol="",
                )
            )

    return episodes


def find_first_signal_date(
    signals: Dict[date, bool],
    episode_start: date,
    lookback_days: int = 5,
) -> Optional[date]:
    """
    Find the first signal date for an episode.

    Looks for the first True signal within lookback_days before episode_start.

    Args:
        signals: Dict mapping date -> bool
        episode_start: Start date of the episode
        lookback_days: Days to look back for early signal

    Returns:
        First signal date, or None if not found
    """
    search_start = episode_start - timedelta(days=lookback_days)

    for d in sorted(signals.keys()):
        if d < search_start:
            continue
        if d > episode_start + timedelta(days=lookback_days):
            break
        if signals.get(d, False):
            return d

    return None


def compute_earliness(
    signals_baseline: Dict[date, bool],
    signals_faster: Dict[date, bool],
    min_episode_days: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> EarlinessResult:
    """
    Compute earliness of faster timeframe compared to baseline.

    Earliness = baseline_signal_date - faster_signal_date (in days)
    Positive earliness means faster TF fired first.

    Args:
        signals_baseline: Baseline TF signals (e.g., 1d)
        signals_faster: Faster TF signals (e.g., 4h)
        min_episode_days: Minimum episode length
        n_bootstrap: Bootstrap samples for CI
        seed: Random seed

    Returns:
        EarlinessResult with statistics
    """
    # Detect episodes from baseline
    episodes = detect_trend_episodes(signals_baseline, min_episode_days)

    if not episodes:
        return EarlinessResult(
            tf_pair="faster_vs_baseline",
            median_earliness_days=0.0,
            p75_earliness_days=0.0,
            pct_earlier_than_baseline=0.0,
            ci_95=(0.0, 0.0),
            n_episodes=0,
            earliness_values=[],
        )

    earliness_values = []
    earlier_count = 0

    for episode in episodes:
        # Find first signal in each TF for this episode
        baseline_first = find_first_signal_date(signals_baseline, episode.start_date)
        faster_first = find_first_signal_date(signals_faster, episode.start_date)

        if baseline_first is None or faster_first is None:
            continue

        # Earliness = baseline - faster (positive if faster is earlier)
        earliness = (baseline_first - faster_first).days
        earliness_values.append(float(earliness))

        if earliness > 0:
            earlier_count += 1

    if not earliness_values:
        return EarlinessResult(
            tf_pair="faster_vs_baseline",
            median_earliness_days=0.0,
            p75_earliness_days=0.0,
            pct_earlier_than_baseline=0.0,
            ci_95=(0.0, 0.0),
            n_episodes=len(episodes),
            earliness_values=[],
        )

    median_earliness = float(np.median(earliness_values))
    p75_earliness = float(np.percentile(earliness_values, 75))
    pct_earlier = earlier_count / len(earliness_values)

    # Bootstrap CI for median
    ci = block_bootstrap_ci(
        earliness_values, block_size=5, n_samples=n_bootstrap, seed=seed
    )

    return EarlinessResult(
        tf_pair="faster_vs_baseline",
        median_earliness_days=median_earliness,
        p75_earliness_days=p75_earliness,
        pct_earlier_than_baseline=pct_earlier,
        ci_95=ci,
        n_episodes=len(episodes),
        earliness_values=earliness_values,
    )


def compute_multi_tf_earliness(
    signals_by_tf: Dict[str, Dict[date, bool]],
    baseline_tf: str = "1d",
    faster_tfs: Optional[List[str]] = None,
    min_episode_days: int = 10,
) -> Dict[str, EarlinessResult]:
    """
    Compute earliness for multiple timeframe pairs.

    Args:
        signals_by_tf: Dict mapping TF -> (date -> bool)
        baseline_tf: Baseline timeframe (default "1d")
        faster_tfs: Faster timeframes to compare (default ["4h", "2h"])
        min_episode_days: Minimum episode length

    Returns:
        Dict mapping "faster_vs_baseline" -> EarlinessResult
    """
    if faster_tfs is None:
        faster_tfs = ["4h", "2h"]

    if baseline_tf not in signals_by_tf:
        return {}

    baseline_signals = signals_by_tf[baseline_tf]
    results = {}

    for tf in faster_tfs:
        if tf not in signals_by_tf:
            continue

        result = compute_earliness(
            signals_baseline=baseline_signals,
            signals_faster=signals_by_tf[tf],
            min_episode_days=min_episode_days,
        )
        result = EarlinessResult(
            tf_pair=f"{tf}_vs_{baseline_tf}",
            median_earliness_days=result.median_earliness_days,
            p75_earliness_days=result.p75_earliness_days,
            pct_earlier_than_baseline=result.pct_earlier_than_baseline,
            ci_95=result.ci_95,
            n_episodes=result.n_episodes,
            earliness_values=result.earliness_values,
        )
        results[f"{tf}_vs_{baseline_tf}"] = result

    return results
