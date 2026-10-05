"""
Divergence Detection Package.

Provides tools for detecting:
- Price vs indicator divergences (bullish/bearish/hidden)
- Cross-indicator divergences
- Multi-timeframe alignment
- Confluence scoring
"""

from .confluence import MTFDivergenceAnalyzer
from .cross_divergence import CrossIndicatorAnalyzer

__all__ = [
    "CrossIndicatorAnalyzer",
    "MTFDivergenceAnalyzer",
]
