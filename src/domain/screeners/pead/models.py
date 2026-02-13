"""PEAD screener domain models.

Typed dataclasses for earnings surprise data, PEAD candidates, and screen results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class LiquidityTier(str, Enum):
    """Market-cap-based liquidity classification for slippage estimation."""

    LARGE_CAP = "large_cap"  # >$50B
    MID_CAP = "mid_cap"  # $2B-$50B
    SMALL_CAP = "small_cap"  # <$2B


@dataclass
class EarningsSurprise:
    """Raw earnings surprise data for a single reporting event."""

    symbol: str
    report_date: date
    actual_eps: float
    consensus_eps: float
    surprise_pct: float
    sue_score: float
    earnings_day_return: float
    earnings_day_gap: float
    earnings_day_volume_ratio: float
    revenue_beat: bool
    at_52w_high: bool
    analyst_downgrade: bool
    liquidity_tier: LiquidityTier
    forward_pe: float | None = None
    multi_quarter_sue: float | None = None


@dataclass
class PEADCandidate:
    """Scored PEAD candidate with trade parameters."""

    symbol: str
    surprise: EarningsSurprise
    entry_date: date
    entry_price: float
    profit_target_pct: float
    stop_loss_pct: float
    trailing_stop_atr: float
    trailing_activation_pct: float
    max_hold_days: int
    position_size_factor: float
    quality_score: float
    quality_label: str  # STRONG / MODERATE / MARGINAL
    regime: str
    gap_held: bool
    estimated_slippage_bps: int


@dataclass
class PEADScreenResult:
    """Complete result of a PEAD screening run."""

    candidates: list[PEADCandidate]
    screened_count: int
    passed_filters: int
    regime: str
    generated_at: datetime
    skipped_count: int = 0
    errors: dict[str, str] = field(default_factory=dict)
