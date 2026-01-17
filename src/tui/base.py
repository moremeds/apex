"""
Base definitions for the Terminal Dashboard.

Contains enums, constants, and base types used across dashboard modules.
"""

from __future__ import annotations

from enum import Enum


class DashboardView(Enum):
    """Available dashboard views."""

    ACCOUNT_SUMMARY = "account_summary"  # Tab 1: Consolidated view
    SIGNALS = "signals"  # Tab 2: Unified signals (risk + trading)
    IB_POSITIONS = "ib_positions"  # Tab 3: IB detailed positions
    FUTU_POSITIONS = "futu_positions"  # Tab 4: Futu detailed positions
    LAB = "lab"  # Tab 5: Strategy lab (backtest strategies)
    DATA = "data"  # Tab 6: Historical coverage + indicator DB
    SIGNAL_INTROSPECTION = "signal_introspection"  # Tab 7: Signal pipeline introspection


# View tabs configuration for header rendering
VIEW_TABS = [
    ("1", "Summary", "summary", DashboardView.ACCOUNT_SUMMARY),
    ("2", "Signals", "signals", DashboardView.SIGNALS),
    ("3", "IB", "ib", DashboardView.IB_POSITIONS),
    ("4", "Futu", "futu", DashboardView.FUTU_POSITIONS),
    ("5", "Lab", "lab", DashboardView.LAB),
    ("6", "Data", "data", DashboardView.DATA),
    ("7", "Intro", "introspection", DashboardView.SIGNAL_INTROSPECTION),
]
