"""
Position history panels rendering for the Terminal Dashboard.

Note: These panels currently show placeholder content.
Full history functionality requires a persistence layer.
"""

from __future__ import annotations
import logging
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)


def render_position_history_today(broker: str) -> Panel:
    """
    Render today's executed trades for a broker.

    Args:
        broker: Broker name ("IB" or "FUTU").

    Returns:
        Panel containing today's trades placeholder.
    """
    return Panel(
        Text("Trade history not available", style="dim"),
        title=f"Today's Trades ({broker})",
        border_style="dim",
    )


def render_open_orders(broker: str) -> Panel:
    """
    Render open/pending orders for a broker.

    Args:
        broker: Broker name ("IB" or "FUTU").

    Returns:
        Panel containing open orders placeholder.
    """
    return Panel(
        Text("Open orders not available", style="dim"),
        title=f"Open Orders ({broker})",
        border_style="dim",
    )


def render_position_history_recent(broker: str) -> Panel:
    """
    Render recent position history for a broker.

    Args:
        broker: Broker name ("IB" or "FUTU").

    Returns:
        Panel containing position history placeholder.
    """
    return Panel(
        Text("Position history not available", style="dim"),
        title=f"Position History ({broker})",
        border_style="dim",
    )
