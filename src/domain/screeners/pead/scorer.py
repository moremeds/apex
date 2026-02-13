"""PEAD quality scorer — pure scoring functions.

Scores candidates on SUE magnitude, gap quality, volume confirmation,
and revenue beat. Additive modifiers for multi-quarter SUE and attention.

No hard filters here (52w high, regime, analyst downgrade
are all handled upstream in the screener filter pipeline).
"""

from __future__ import annotations


def score_pead_quality(
    sue: float,
    gap_return: float,
    volume_ratio: float,
    revenue_beat: bool,
) -> float:
    """Score a PEAD candidate from 0-100.

    Components:
        SUE magnitude:        0-30 pts (min(sue/5, 1) * 30)
        Gap quality:          0-30 pts (>5%=30, >3%=25, >2%=20)
        Volume confirmation:  0-25 pts (>3x=25, >2.5x=20, >2x=15)
        Revenue beat bonus:   0-15 pts

    Args:
        sue: Standardized Unexpected Earnings score.
        gap_return: Earnings day gap as decimal (0.05 = 5%).
        volume_ratio: Earnings day volume / 20-day avg volume.
        revenue_beat: Whether revenue also beat estimates.

    Returns:
        Quality score 0-100.
    """
    # SUE magnitude: 0-30
    sue_pts = min(sue / 5.0, 1.0) * 30.0

    # Gap quality: 0-30 (gap_return should be positive by the time scorer is called)
    gap_pct = gap_return * 100
    if gap_pct > 5:
        gap_pts = 30.0
    elif gap_pct > 3:
        gap_pts = 25.0
    elif gap_pct > 2:
        gap_pts = 20.0
    else:
        gap_pts = max(0.0, gap_pct / 2.0 * 10.0)

    # Volume confirmation: 0-25
    if volume_ratio > 3.0:
        vol_pts = 25.0
    elif volume_ratio > 2.5:
        vol_pts = 20.0
    elif volume_ratio > 2.0:
        vol_pts = 15.0
    else:
        vol_pts = max(0.0, (volume_ratio - 1.0) * 15.0)

    # Revenue beat bonus: 0-15
    rev_pts = 15.0 if revenue_beat else 0.0

    return min(100.0, sue_pts + gap_pts + vol_pts + rev_pts)


def classify_quality(
    score: float, strong_threshold: float = 70.0, moderate_threshold: float = 45.0
) -> str:
    """Classify quality score into STRONG / MODERATE / MARGINAL."""
    if score >= strong_threshold:
        return "STRONG"
    if score >= moderate_threshold:
        return "MODERATE"
    return "MARGINAL"


def apply_multi_quarter_modifier(
    base_score: float,
    multi_quarter_sue: float | None,
    max_bonus: float = 10.0,
    max_penalty: float = -5.0,
) -> float:
    """Apply additive multi-quarter SUE modifier to base quality score.

    Strong historical trajectory = bonus, deteriorating = penalty.
    NOT a blend with single-Q SUE — a separate additive modifier on 0-100 score.

    Args:
        base_score: Base quality score (0-100).
        multi_quarter_sue: Multi-quarter SUE score (None = no data).
        max_bonus: Maximum bonus for strong trajectory.
        max_penalty: Maximum penalty for deteriorating trajectory (negative).

    Returns:
        Adjusted quality score, clamped to [0, 100].
    """
    if multi_quarter_sue is None:
        return base_score

    # Map multi-Q SUE to modifier: positive SUE → bonus, negative → penalty
    # Scale: SUE of 2.0 gets full bonus, SUE of -2.0 gets full penalty
    if multi_quarter_sue > 0:
        modifier = min(multi_quarter_sue / 2.0, 1.0) * max_bonus
    else:
        modifier = max(multi_quarter_sue / 2.0, -1.0) * abs(max_penalty)

    return max(0.0, min(100.0, base_score + modifier))


def apply_attention_modifier(
    base_score: float,
    attention_level: str | None,
    low_bonus: float = 5.0,
    high_penalty: float = -5.0,
) -> float:
    """Apply additive attention modifier to base quality score.

    Low attention (under-followed) = bonus (PEAD more likely to persist).
    High attention (well-covered) = penalty (market reacts faster).

    Args:
        base_score: Base quality score (0-100).
        attention_level: "low" / "medium" / "high" / None.
        low_bonus: Bonus for low-attention stocks.
        high_penalty: Penalty for high-attention stocks (negative).

    Returns:
        Adjusted quality score, clamped to [0, 100].
    """
    if attention_level is None:
        return base_score

    modifier = 0.0
    if attention_level == "low":
        modifier = low_bonus
    elif attention_level == "high":
        modifier = high_penalty
    # "medium" = no modifier

    return max(0.0, min(100.0, base_score + modifier))
