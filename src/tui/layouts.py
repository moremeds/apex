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
    """
    Create broker positions view layout (Tab 3 & 4).

    Positions on left, ATR levels + history on right.

    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │ Header                                                   │
    ├───────────────────────────┬─────────────────────────────┤
    │ Positions (60%)           │ Right Panel (40%)           │
    │ > AAPL  100  $150.25     │ ┌─────────────────────────┐ │
    │   TSLA   50  $245.80     │ │ ATR Analysis (40%)      │ │
    │   MSFT  200  $380.50     │ └─────────────────────────┘ │
    │                           │ Today's Trades              │
    │                           │ Open Orders                 │
    └───────────────────────────┴─────────────────────────────┘
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    # Split body into positions (left) and right panel
    layout["body"].split_row(
        Layout(name="positions", ratio=3),  # Current positions (left 60%)
        Layout(name="history_panel", ratio=2),  # Right panel (40%)
    )
    # Split right panel into ATR/history sections
    # ATR panel takes 40% of right panel height (ratio=4), rest split between trades/orders
    layout["body"]["history_panel"].split_column(
        Layout(name="atr_levels", ratio=4),     # ATR analysis panel (40% of right panel)
        Layout(name="history_today", ratio=3),  # Today's trades
        Layout(name="open_orders", ratio=3),    # Open/pending orders
    )
    return layout


def create_layout_lab() -> Layout:
    """Create strategy lab view layout (Tab 5).

    Shows:
    - Left: Backtest strategies with parameters and key metrics
    - Right: Strategy details and last backtest performance
    """
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    # Split body into strategies list (left) and details (right)
    layout["body"].split_row(
        Layout(name="strategies", ratio=3),    # Left: Strategy list (60%)
        Layout(name="details", ratio=2),       # Right: Details panel (40%)
    )
    # Split details into strategy info and performance
    layout["body"]["details"].split_column(
        Layout(name="params"),          # Top: Strategy parameters
        Layout(name="performance"),     # Bottom: Last backtest performance
    )
    return layout
