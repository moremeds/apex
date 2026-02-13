"""SUE (Standardized Unexpected Earnings) computation.

Single-quarter SUE normalizes the current earnings surprise by historical
surprise volatility. Multi-quarter SUE captures the trajectory of past
surprises using exponential decay weighting.
"""

from __future__ import annotations

import statistics


def compute_sue(
    actual_eps: float,
    consensus_eps: float,
    historical_surprises: list[float],
) -> float:
    """Compute Standardized Unexpected Earnings (SUE).

    When >= 4 quarters of historical surprise data:
        SUE = raw_surprise / stdev(historical_surprises)
    Fallback (< 4 quarters):
        SUE = raw_surprise / (|consensus| * 0.05 + 0.01)

    Args:
        actual_eps: Actual reported EPS.
        consensus_eps: Analyst consensus EPS estimate.
        historical_surprises: List of past (actual - consensus) values.
            Must already exclude current quarter (see Phase 0.3).

    Returns:
        SUE score (positive = beat, negative = miss).
    """
    raw = actual_eps - consensus_eps

    if len(historical_surprises) >= 4:
        std = statistics.stdev(historical_surprises)
        if std > 0:
            return raw / std
        # Zero std means all identical — use fallback
    # Fallback: proxy scaling
    denom = abs(consensus_eps) * 0.05 + 0.01
    return raw / denom


def compute_multi_quarter_sue(
    historical_surprises: list[float],
    decay_lambda: float = 0.75,
    min_quarters: int = 6,
) -> float | None:
    """Weighted multi-quarter SUE using exponential decay.

    Captures the trajectory of past surprises. Does NOT include the
    current quarter (that's the single-Q SUE). Uses historical_surprises
    which already exclude current quarter (Phase 0.3).

    Formula:
        weighted_mean = sum(w_i * s_i) / sum(w_i)
        where w_i = decay_lambda ^ i, s_i = surprise[i] (most recent first)

    Args:
        historical_surprises: Past (actual - consensus) values, most recent first.
        decay_lambda: Decay factor per quarter (0.75 = 25% decay each quarter).
        min_quarters: Minimum quarters required. Returns None if fewer.

    Returns:
        Multi-quarter SUE score, or None if insufficient data.
    """
    if len(historical_surprises) < min_quarters:
        return None

    total_weight = 0.0
    weighted_sum = 0.0
    for i, surprise in enumerate(historical_surprises):
        weight = decay_lambda**i
        weighted_sum += weight * surprise
        total_weight += weight

    if total_weight == 0:
        return None

    weighted_mean = weighted_sum / total_weight

    # Normalize by stdev of historical surprises for scale-invariance
    if len(historical_surprises) >= 4:
        std = statistics.stdev(historical_surprises)
        if std > 0:
            return weighted_mean / std

    # Fallback: return raw weighted mean (unnormalized)
    return weighted_mean
