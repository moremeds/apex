"""
Header panel rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from rich.panel import Panel
from rich.text import Text

from ..base import DashboardView, VIEW_TABS
from ...utils.market_hours import MarketHours


def render_header(env: str, current_view: DashboardView) -> Panel:
    """
    Render header panel with market status, environment, and view tabs.

    Args:
        env: Environment name (dev, demo, prod).
        current_view: Currently active dashboard view.

    Returns:
        Panel containing the header.
    """
    # Get market status
    market_status = MarketHours.get_market_status()

    # Create header text
    header = Text("Live Risk Management System", style="bold cyan")

    # Add environment indicator
    header.append("  |  ", style="dim")
    env_upper = env.upper()
    if env == "prod":
        header.append(env_upper, style="bold red")
    elif env == "demo":
        header.append(env_upper, style="bold magenta")
    else:  # dev
        header.append(env_upper, style="bold yellow")

    # Add market status indicator
    header.append("  |  ", style="dim")
    if market_status == "OPEN":
        header.append("Market: ", style="dim")
        header.append("OPEN", style="bold green")
    elif market_status == "EXTENDED":
        header.append("Market: ", style="dim")
        header.append("EXTENDED HOURS", style="bold yellow")
    else:
        header.append("Market: ", style="dim")
        header.append("CLOSED", style="bold red")

    # Add view tabs
    header.append("  |  ", style="dim")
    for i, (key, label, view) in enumerate(VIEW_TABS):
        if i > 0:
            header.append(" ", style="dim")
        if current_view == view:
            header.append(f"[{key}]{label}", style="bold white on blue")
        else:
            header.append(f"[{key}]{label}", style="dim")

    header.justify = "center"
    return Panel(header, style="bold")
