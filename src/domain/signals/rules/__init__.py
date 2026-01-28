"""Pre-built trading rules for the signal engine."""

from .momentum_rules import MOMENTUM_RULES
from .pattern_rules import PATTERN_RULES
from .short_timeframe_rules import SHORT_TIMEFRAME_RULES
from .trend_rules import TREND_RULES
from .volatility_rules import VOLATILITY_RULES
from .volume_rules import VOLUME_RULES

# All pre-built rules combined
ALL_RULES = (
    MOMENTUM_RULES
    + TREND_RULES
    + VOLATILITY_RULES
    + VOLUME_RULES
    + PATTERN_RULES
    + SHORT_TIMEFRAME_RULES
)

__all__ = [
    "MOMENTUM_RULES",
    "TREND_RULES",
    "VOLATILITY_RULES",
    "VOLUME_RULES",
    "PATTERN_RULES",
    "SHORT_TIMEFRAME_RULES",
    "ALL_RULES",
]
