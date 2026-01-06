"""
Indicators package for technical analysis.

Provides:
- Indicator: Protocol for all indicator implementations
- IndicatorBase: Base class with common functionality
- IndicatorRegistry: Auto-discovery and management of indicators
"""

from .base import Indicator, IndicatorBase
from .registry import IndicatorRegistry

__all__ = [
    "Indicator",
    "IndicatorBase",
    "IndicatorRegistry",
]
