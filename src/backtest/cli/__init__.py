"""
Backtest CLI Module.

Command-line interface for backtest runner:
- parser.py: Argument parsing
- commands.py: Command handlers
"""

from .commands import main, main_async
from .parser import create_parser

__all__ = ["create_parser", "main", "main_async"]
