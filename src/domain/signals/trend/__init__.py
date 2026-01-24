"""
Trend Health Analysis Module.

Provides weighted trend quality scoring by combining:
- HH/HL (Higher-High/Higher-Low) swing patterns
- MA alignment (SMA20 > SMA50 > SMA200)
- ADX trend strength
- MA slope acceleration
- RSI health (not at extremes)
"""

from .health_analyzer import TrendHealthAnalyzer, TrendHealthResult
from .hh_hl_detector import HHHLDetector, HHHLResult, SwingPoint
from .ma_alignment import MAAlignmentResult, MAAlignmentScorer

__all__ = [
    "TrendHealthAnalyzer",
    "TrendHealthResult",
    "HHHLDetector",
    "HHHLResult",
    "SwingPoint",
    "MAAlignmentScorer",
    "MAAlignmentResult",
]
