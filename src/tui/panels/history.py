"""
Position history panels rendering for the Terminal Dashboard.

Note: Persistence layer removed - these panels show placeholder content.
"""

from __future__ import annotations
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def render_position_history_today(broker: str) -> Panel:
    """
    Render today's executed trades for a broker.

    Shows placeholder - persistence layer not configured.

    Args:
        broker: Broker name ("IB" or "FUTU").

    Returns:
        Panel with placeholder content.
    """
    # Create table structure (columns preserved for future use)
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Time", style="dim", no_wrap=True, width=8)
    table.add_column("Side", style="bold", no_wrap=True, width=4)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Qty", justify="right", width=8)
    table.add_column("Price", justify="right", width=10)

    return Panel(
        Text("No trades data", style="dim"),
        title=f"Today's Trades ({broker})",
        border_style="dim",
    )


def render_open_orders(broker: str) -> Panel:
    """
    Render open/pending orders for a broker.

    Shows placeholder - persistence layer not configured.

    Args:
        broker: Broker name ("IB" or "FUTU").

    Returns:
        Panel with placeholder content.
    """
    # Create table structure (columns preserved for future use)
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Time", style="dim", no_wrap=True, width=8)
    table.add_column("Side", style="bold", no_wrap=True, width=4)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Qty", justify="right", width=6)
    table.add_column("Price", justify="right", width=8)
    table.add_column("Status", style="yellow", width=8)

    return Panel(
        Text("No orders data", style="dim"),
        title=f"Open Orders ({broker})",
        border_style="dim",
    )


def render_position_history_recent(broker: str) -> Panel:
    """
    Render stored positions for a broker.

    Shows placeholder - persistence layer not configured.

    Args:
        broker: Broker name ("IB" or "FUTU").

    Returns:
        Panel with placeholder content.
    """
    # Create table structure (columns preserved for future use)
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Qty", justify="right", width=8)
    table.add_column("AvgPx", justify="right", width=10)
    table.add_column("Mark", justify="right", width=10)
    table.add_column("P&L", justify="right", width=10)
    table.add_column("Updated", style="dim", width=10)

    return Panel(
        Text("No stored positions", style="dim"),
        title=f"Stored Positions ({broker})",
        border_style="dim",
    )
