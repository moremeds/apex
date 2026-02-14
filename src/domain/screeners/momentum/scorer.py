"""Momentum scoring and ranking functions.

Pure functions for cross-sectional percentile ranking, composite score
computation, and quality classification.
"""

from __future__ import annotations

import numpy as np


def compute_percentile_ranks(values: list[float]) -> list[float]:
    """Compute cross-sectional percentile ranks.

    Uses average rank for ties. Result is in [0.0, 1.0] where 1.0 is the
    highest value in the cross-section.

    Args:
        values: List of raw values to rank.

    Returns:
        List of percentile ranks in same order as input.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]

    arr = np.array(values)
    # argsort twice gives ranks (0-based)
    temp = arr.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(n, dtype=float)

    # Handle ties: assign average rank
    sorted_vals = arr[temp]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[temp[k]] = avg_rank
        i = j

    # Normalize to [0, 1]
    return [float(r / (n - 1)) for r in ranks]


def compute_composite_rank(
    momentum_percentile: float,
    fip_percentile: float,
    momentum_weight: float = 0.5,
    fip_weight: float = 0.5,
) -> float:
    """Compute weighted composite rank from percentile inputs.

    Args:
        momentum_percentile: Momentum percentile rank (0-1).
        fip_percentile: FIP percentile rank (0-1).
        momentum_weight: Weight for momentum component.
        fip_weight: Weight for FIP component.

    Returns:
        Composite rank in [0.0, 1.0].
    """
    total_weight = momentum_weight + fip_weight
    if total_weight <= 0:
        return 0.0
    return (momentum_percentile * momentum_weight + fip_percentile * fip_weight) / total_weight


def classify_quality(
    composite: float, strong_threshold: float = 0.80, moderate_threshold: float = 0.60
) -> str:
    """Classify composite rank into quality labels.

    Args:
        composite: Composite rank in [0, 1].
        strong_threshold: Minimum for STRONG classification.
        moderate_threshold: Minimum for MODERATE classification.

    Returns:
        "STRONG", "MODERATE", or "MARGINAL".
    """
    if composite >= strong_threshold:
        return "STRONG"
    if composite >= moderate_threshold:
        return "MODERATE"
    return "MARGINAL"
