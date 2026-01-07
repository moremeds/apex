"""
Divergence Detection Package.

Provides tools for detecting:
- Price vs indicator divergences (bullish/bearish/hidden)
- Cross-indicator divergences
- Multi-timeframe alignment
- Confluence scoring
"""

from .price_divergence import PriceDivergenceDetector
from .cross_divergence import CrossIndicatorAnalyzer
from .confluence import MTFDivergenceAnalyzer

__all__ = [
    "PriceDivergenceDetector",
    "CrossIndicatorAnalyzer",
    "MTFDivergenceAnalyzer",
]
