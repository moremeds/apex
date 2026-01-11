"""
Signal Pipeline Components.

Extracted from SignalCoordinator for single responsibility:
- BarPreloader: Historical bar loading for indicator warmup
"""

from .bar_preloader import BarPreloader

__all__ = ["BarPreloader"]
