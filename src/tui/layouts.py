"""
Layout definitions for the Terminal Dashboard.

Creates the various layout structures for different dashboard views.
"""

from __future__ import annotations
from rich.layout import Layout


def create_layout_account_summary() -> Layout:
    """Create account summary view layout (Tab 1)."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),  # Main body - expands to fill available space
        Layout(name="footer", size=5),  # Health status at bottom
    )
    # Split body into left (consolidated positions) and right (summary + alerts)
    layout["body"].split_row(
        Layout(name="positions", ratio=3),  # Left: Consolidated positions (60%)
        Layout(name="right", ratio=2),  # Right: Summary + Alerts (40%)
    )
    # Split right side into summary and alerts
    layout["right"].split_column(
        Layout(name="summary", size=18),  # Upper: Portfolio summary
        Layout(name="alerts"),  # Lower: Market alerts
    )
    return layout


def create_layout_risk_signals() -> Layout:
    """Create risk signals view layout (Tab 2) - full screen signals only."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="signals"),  # Full screen for risk signals
    )
    return layout


def create_layout_broker_positions() -> Layout:
    """Create broker positions view layout (Tab 3 & 4) - positions with history on right."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    # Split body into positions (left) and history (right)
    layout["body"].split_row(
        Layout(name="positions", ratio=3),  # Current positions (left 60%)
        Layout(name="history_panel", ratio=2),  # Position history panel (right 40%)
    )
    # Split history panel into today's changes, open orders, and stored positions
    layout["body"]["history_panel"].split_column(
        Layout(name="history_today"),       # Today's changes (top)
        Layout(name="open_orders"),         # Open/pending orders (middle)
        Layout(name="history_recent"),      # Stored positions (bottom)
    )
    return layout
