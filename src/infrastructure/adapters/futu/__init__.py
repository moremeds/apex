"""
Futu OpenD adapter package.

Provides FutuAdapter for connecting to Futu OpenD gateway.
"""

from .adapter import FutuAdapter
from .trade_handler import create_trade_handler

__all__ = ["FutuAdapter", "create_trade_handler"]
