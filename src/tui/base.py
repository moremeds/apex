"""
Base definitions for the Terminal Dashboard.

Contains enums, constants, and base types used across dashboard modules.
"""

from __future__ import annotations
from enum import Enum


class DashboardView(Enum):
    """Available dashboard views."""
    ACCOUNT_SUMMARY = "account_summary"       # Tab 1: Consolidated view
    RISK_SIGNALS = "risk_signals"             # Tab 2: Risk signals only
    IB_POSITIONS = "ib_positions"             # Tab 3: IB detailed positions
    FUTU_POSITIONS = "futu_positions"         # Tab 4: Futu detailed positions
    LAB = "lab"                               # Tab 5: Strategy lab (backtest strategies)
    TRADING_SIGNALS = "trading_signals"       # Tab 6: Trading signals + confluence


# View tabs configuration for header rendering
VIEW_TABS = [
    ("1", "Summary", "summary", DashboardView.ACCOUNT_SUMMARY),
    ("2", "Signals", "signals", DashboardView.RISK_SIGNALS),
    ("3", "IB", "ib", DashboardView.IB_POSITIONS),
    ("4", "Futu", "futu", DashboardView.FUTU_POSITIONS),
    ("5", "Lab", "lab", DashboardView.LAB),
    ("6", "Trading", "trading", DashboardView.TRADING_SIGNALS),
]
