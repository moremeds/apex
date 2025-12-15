"""
Header panel rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from typing import Optional

from rich.panel import Panel
from rich.text import Text

from ..base import DashboardView, VIEW_TABS
from ...utils.market_hours import MarketHours
from ...utils.timezone import DisplayTimezone

# Global display timezone instance (default: Asia/Hong_Kong)
_display_tz: Optional[DisplayTimezone] = None


def set_display_timezone(tz_name: str) -> None:
    """
    Set the display timezone for the dashboard header.

    Args:
        tz_name: IANA timezone name (e.g., "Asia/Hong_Kong", "America/New_York").
    """
    global _display_tz
    _display_tz = DisplayTimezone(tz_name)


def get_display_timezone() -> DisplayTimezone:
    """Get the current display timezone instance."""
    global _display_tz
    if _display_tz is None:
        _display_tz = DisplayTimezone("Asia/Hong_Kong")
    return _display_tz


def render_header(env: str, current_view: DashboardView) -> Panel:
    """
    Render header panel with market status, environment, and view tabs.

    Args:
        env: Environment name (dev, demo, prod).
        current_view: Currently active dashboard view.

    Returns:
        Panel containing the header.
    """
    # Get market status and current time
    market_status = MarketHours.get_market_status()
    display_tz = get_display_timezone()
    current_time = display_tz.current_time("%H:%M:%S %Z")

    # Create header text
    header = Text("Live Risk Management System", style="bold cyan")

    # Add current time in display timezone
    header.append("  |  ", style="dim")
    header.append(current_time, style="white")

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
