"""PEAD quality scorer — pure scoring function.

Scores candidates on SUE magnitude, gap quality, volume confirmation,
and revenue beat. No hard filters here (52w high, regime, analyst downgrade
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
