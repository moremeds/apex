"""
Health status panel rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from typing import List
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...infrastructure.monitoring import ComponentHealth, HealthStatus


def render_health(health: List[ComponentHealth]) -> Panel:
    """
    Render health status panel horizontally.

    Args:
        health: List of component health statuses.

    Returns:
        Panel containing the health status display.
    """
    if not health:
        return Panel(Text("No health data", style="dim"), title="Health", border_style="dim")

    # Create table with horizontal layout - each component is a column
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)

    # Add columns for each health component
    for h in health:
        table.add_column(justify="center", no_wrap=True)

    # First row: Status icons
    icons = []
    styles = []
    for h in health:
        if h.status == HealthStatus.HEALTHY:
            style = "green"
            icon = "[OK]"
        elif h.status == HealthStatus.DEGRADED:
            style = "yellow"
            icon = "[W]"
        elif h.status == HealthStatus.UNHEALTHY:
            style = "red"
            icon = "[X]"
        else:  # UNKNOWN
            style = "dim"
            icon = "[?]"
        icons.append(Text(icon, style=style))
        styles.append(style)

    table.add_row(*icons)

    # Second row: Component names
    names = []
    for h in health:
        names.append(Text(h.component_name, style="cyan"))
    table.add_row(*names)

    # Third row: Details
    details_list = []
    for h in health:
        details = h.message if h.message else ""

        # Add metadata info for market data coverage (show for all statuses)
        if h.component_name == "market_data_coverage" and h.metadata:
            if "missing_count" in h.metadata and "total" in h.metadata:
                missing = h.metadata['missing_count']
                total = h.metadata['total']
                if missing > 0:
                    details = f"{missing}/{total} missing MD"
                else:
                    details = f"All {total} OK" if total > 0 else "No positions"
        # Add metadata info for other degraded/unhealthy components
        elif h.status != HealthStatus.HEALTHY and h.metadata and details == "":
            if isinstance(h.metadata, dict):
                details = str(h.metadata)
            else:
                details = str(h.metadata) if h.metadata else ""

        details = str(details) if details is not None else ""
        details_list.append(Text(details, style="dim"))

    table.add_row(*details_list)

    # Set border color based on worst status
    has_unhealthy = any(h.status == HealthStatus.UNHEALTHY for h in health)
    has_degraded = any(h.status == HealthStatus.DEGRADED for h in health)

    if has_unhealthy:
        border_style = "red"
    elif has_degraded:
        border_style = "yellow"
    else:
        border_style = "green"

    return Panel(table, title="Component Health", border_style=border_style)
