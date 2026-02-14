"""Momentum and FIP computation functions.

Pure numpy functions for computing:
- 12-1 momentum (Jegadeesh & Titman 1993): 12-month return skipping last month
- FIP (Frog-In-Pan, Da/Gurun/Warachka 2014): fraction of positive vs negative days
- Adaptive momentum for recent IPOs (optional)
"""

from __future__ import annotations

import numpy as np


def compute_momentum_12_1(
    daily_closes: np.ndarray,
    skip: int = 21,
    lookback: int = 252,
) -> float | None:
    """Standard 12-1 momentum return.

    Computes cumulative return from T-lookback-skip to T-skip,
    skipping the most recent `skip` trading days to avoid
    short-term mean reversion contamination.

    Args:
        daily_closes: Array of daily closing prices, chronologically ordered.
        skip: Trading days to skip at end (typically 21 = 1 month).
        lookback: Total lookback trading days (typically 252 = 12 months).

    Returns:
        Momentum return as a decimal (e.g., 0.25 = +25%), or None if
        insufficient data (need at least lookback + skip days).
    """
    required = lookback + skip
    if len(daily_closes) < required:
        return None

    # Price at start of momentum window
    p_start = daily_closes[-(required)]
    # Price at end of momentum window (skip most recent month)
    p_end = daily_closes[-(skip + 1)]

    if p_start <= 0:
        return None

    return float((p_end - p_start) / p_start)


def compute_fip(
    daily_closes: np.ndarray,
    skip: int = 21,
    lookback: int = 252,
) -> float | None:
    """Frog-In-Pan indicator over the momentum window.

    FIP = (# positive return days - # negative return days) / total days
    in the momentum window (excluding the skip period).

    High FIP (close to 1.0) means smooth, gradual appreciation.
    Low FIP (close to -1.0) means smooth, gradual decline.
    FIP near 0 means roughly equal up/down days.

    Args:
        daily_closes: Array of daily closing prices, chronologically ordered.
        skip: Trading days to skip at end.
        lookback: Total lookback trading days.

    Returns:
        FIP value in [-1.0, 1.0], or None if insufficient data.
    """
    required = lookback + skip
    if len(daily_closes) < required:
        return None

    # Extract the momentum window (exclude skip period)
    window = daily_closes[-(required):-(skip)]

    # Daily returns within the window
    returns = np.diff(window) / window[:-1]

    # Count positive and negative days (exclude zero-return days)
    n_pos = int(np.sum(returns > 0))
    n_neg = int(np.sum(returns < 0))
    total = n_pos + n_neg

    if total == 0:
        return 0.0

    return float((n_pos - n_neg) / total)


def compute_adaptive_momentum(
    daily_closes: np.ndarray,
    skip: int = 21,
    target: int = 252,
    floor: int = 126,
) -> tuple[float, int] | None:
    """Adaptive momentum for stocks with limited history.

    Uses available history (min 6 months) when full 12-month history
    is unavailable (e.g., recent IPOs). Annualizes the return for
    cross-sectional comparison.

    Args:
        daily_closes: Array of daily closing prices.
        skip: Trading days to skip at end.
        target: Target lookback in trading days (252 = 12 months).
        floor: Minimum lookback in trading days (126 = 6 months).

    Returns:
        Tuple of (annualized_return, actual_lookback_days) or None if
        insufficient data (< floor + skip days).
    """
    available = len(daily_closes) - skip
    if available < floor:
        return None

    actual_lookback = min(available, target)
    required = actual_lookback + skip

    p_start = daily_closes[-(required)]
    p_end = daily_closes[-(skip + 1)]

    if p_start <= 0:
        return None

    raw_return = (p_end - p_start) / p_start

    # Annualize: scale to 252-day equivalent
    if actual_lookback == target:
        return (float(raw_return), actual_lookback)

    annualized = raw_return * (target / actual_lookback)
    return (float(annualized), actual_lookback)
