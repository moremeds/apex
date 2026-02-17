"""Momentum screener domain models.

Typed dataclasses for momentum signals, candidates, and screen results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.screeners.pead.models import LiquidityTier


@dataclass
class MomentumSignal:
    """Computed momentum signal for a single stock."""

    symbol: str
    momentum_12_1: float
    fip: float
    momentum_percentile: float  # 0.0-1.0
    fip_percentile: float  # 0.0-1.0
    composite_rank: float  # 0.0-1.0
    last_close: float
    market_cap: float
    avg_daily_dollar_volume: float
    liquidity_tier: LiquidityTier
    estimated_slippage_bps: int
    lookback_days: int  # actual lookback used (252 standard, shorter if adaptive)


@dataclass
class MomentumCandidate:
    """Scored momentum candidate with regime-adjusted parameters."""

    signal: MomentumSignal
    rank: int
    quality_label: str  # STRONG / MODERATE / MARGINAL
    position_size_factor: float
    regime: str


@dataclass
class MomentumScreenResult:
    """Complete result of a momentum screening run."""

    candidates: list[MomentumCandidate]
    universe_size: int
    passed_filters: int
    regime: str
    generated_at: datetime
    errors: dict[str, str] = field(default_factory=dict)
