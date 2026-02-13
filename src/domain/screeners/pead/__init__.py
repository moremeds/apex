"""PEAD (Post-Earnings Announcement Drift) screener domain."""

from .config import PEADConfig
from .models import (
    EarningsSurprise,
    LiquidityTier,
    PEADCandidate,
    PEADScreenResult,
)
from .scorer import (
    apply_attention_modifier,
    apply_multi_quarter_modifier,
    classify_quality,
    score_pead_quality,
)
from .screener import PEADScreener
from .sue import compute_multi_quarter_sue, compute_sue
from .tracker import TrackedCandidate, TrackerStats

__all__ = [
    "EarningsSurprise",
    "LiquidityTier",
    "PEADCandidate",
    "PEADConfig",
    "PEADScreenResult",
    "PEADScreener",
    "apply_attention_modifier",
    "apply_multi_quarter_modifier",
    "classify_quality",
    "compute_multi_quarter_sue",
    "compute_sue",
    "score_pead_quality",
    "TrackedCandidate",
    "TrackerStats",
]
