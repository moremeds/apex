"""
Market alerts panel rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from typing import Dict, List, Any
from datetime import datetime
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def update_persistent_alerts(
    current_alerts: List[Dict[str, Any]],
    persistent_alerts: Dict[str, Dict],
    alert_retention_seconds: int = 300,
) -> List[Dict[str, Any]]:
    """
    Update persistent alert tracking and return alerts to display.

    Active alerts are updated with last_seen timestamp.
    Cleared alerts are kept for alert_retention_seconds with is_active=False.

    Args:
        current_alerts: List of currently active alerts.
        persistent_alerts: Mutable dict to track persistent alerts (modified in place).
        alert_retention_seconds: How long to keep cleared alerts visible.

    Returns:
        List of alerts to display (active + recently cleared).
    """
    now = datetime.now()

    # Build set of current alert keys
    current_keys = set()
    for alert in current_alerts:
        alert_key = f"{alert.get('type', 'UNKNOWN')}_{alert.get('severity', 'INFO')}"
        current_keys.add(alert_key)

        if alert_key in persistent_alerts:
            # Update existing alert
            persistent_alerts[alert_key]["alert_data"] = alert
            persistent_alerts[alert_key]["last_seen"] = now
            persistent_alerts[alert_key]["is_active"] = True
        else:
            # New alert
            persistent_alerts[alert_key] = {
                "alert_data": alert,
                "first_seen": now,
                "last_seen": now,
                "is_active": True,
            }

    # Mark alerts not in current set as inactive
    for alert_key in persistent_alerts:
        if alert_key not in current_keys:
            persistent_alerts[alert_key]["is_active"] = False

    # Build display list and cleanup expired alerts
    display_alerts = []
    expired_keys = []

    for alert_key, alert_info in persistent_alerts.items():
        age_seconds = (now - alert_info["last_seen"]).total_seconds()

        if alert_info["is_active"]:
            # Active alert - always display
            display_alerts.append({
                **alert_info["alert_data"],
                "first_seen": alert_info["first_seen"],
                "last_seen": alert_info["last_seen"],
                "is_active": True,
            })
        elif age_seconds <= alert_retention_seconds:
            # Recently cleared - display with dimmed style
            display_alerts.append({
                **alert_info["alert_data"],
                "first_seen": alert_info["first_seen"],
                "last_seen": alert_info["last_seen"],
                "is_active": False,
            })
        else:
            # Expired - mark for cleanup
            expired_keys.append(alert_key)

    # Cleanup expired alerts
    for key in expired_keys:
        del persistent_alerts[key]

    return display_alerts


def render_market_alerts(display_alerts: List[Dict[str, Any]]) -> Panel:
    """
    Render market-wide alerts panel.

    Args:
        display_alerts: List of alerts to display (from update_persistent_alerts).
            Each alert should have: type, message, severity, is_active, last_seen.

    Returns:
        Panel containing the market alerts.
    """
    if not display_alerts:
        text = Text("No market alerts", style="dim")
        return Panel(text, title="Market Alerts", border_style="dim")

    table = Table(show_header=False, box=None)
    table.add_column("Alert", style="bold")
    table.add_column("Details", justify="left")
    table.add_column("Time", justify="right", style="dim")

    for alert in display_alerts:
        alert_type = alert.get("type", "UNKNOWN")
        message = alert.get("message", "")
        severity = alert.get("severity", "INFO")
        is_active = alert.get("is_active", True)
        last_seen = alert.get("last_seen")

        # Format time display
        time_str = ""
        if last_seen:
            time_str = last_seen.strftime("%H:%M:%S")

        # Set style based on severity and active status
        if not is_active:
            # Cleared alert - dimmed style
            style = "dim"
            icon = "o"
            status_suffix = " [cleared]"
        elif severity == "CRITICAL":
            style = "bold red"
            icon = "[!]"
            status_suffix = ""
        elif severity == "WARNING":
            style = "bold yellow"
            icon = "[W]"
            status_suffix = ""
        else:
            style = "cyan"
            icon = "[i]"
            status_suffix = ""

        table.add_row(
            Text(f"{icon} {alert_type}", style=style),
            Text(f"{message}{status_suffix}", style=style),
            Text(time_str, style="dim")
        )

    # Set border color based on highest severity of ACTIVE alerts
    active_alerts = [a for a in display_alerts if a.get("is_active", True)]
    has_critical = any(a.get("severity") == "CRITICAL" for a in active_alerts)
    has_warning = any(a.get("severity") == "WARNING" for a in active_alerts)

    active_count = len(active_alerts)
    cleared_count = len(display_alerts) - active_count

    if has_critical:
        border_style = "red"
        title = f"[!] Market Alerts ({active_count} active"
    elif has_warning:
        border_style = "yellow"
        title = f"[W] Market Alerts ({active_count} active"
    elif active_count > 0:
        border_style = "cyan"
        title = f"Market Alerts ({active_count} active"
    else:
        border_style = "dim"
        title = f"Market Alerts (0 active"

    if cleared_count > 0:
        title += f", {cleared_count} cleared)"
    else:
        title += ")"

    return Panel(table, title=title, border_style=border_style)
