"""
Temporal splitters for walk-forward and CPCV validation.

Splitters divide data into train/test windows while respecting:
- Purge gap: Prevents look-ahead bias
- Embargo: Allows model decay between periods
- Trading calendar: Uses trading days, not calendar days
"""

from .cpcv import CPCVConfig, CPCVSplitter
from .walk_forward import SplitConfig, WalkForwardSplitter

__all__ = [
    "WalkForwardSplitter",
    "SplitConfig",
    "CPCVSplitter",
    "CPCVConfig",
]
