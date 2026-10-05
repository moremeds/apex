"""
Interactive Brokers historical bar adapter, kept only for the frozen backtest data feeds.
"""

from .base import IbBaseAdapter
from .historical_adapter import IbHistoricalAdapter

__all__ = [
    "IbBaseAdapter",
    "IbHistoricalAdapter",
]
