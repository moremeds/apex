"""
Alert ViewModel for incremental updates.

OPT-011: Provides diff computation to avoid full rebuilds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AlertDisplayRow:
    """Immutable alert row for comparison."""

    alert_key: str
    alert_type: str
    message: str
    severity: str
    is_active: bool
    time_str: str

    def format_line(self) -> str:
        """Format alert for display."""
        if not self.is_active:
            icon = "o"
            style = "dim"
            suffix = " [cleared]"
        elif self.severity == "CRITICAL":
            icon = "[!]"
            style = "bold red"
            suffix = ""
        elif self.severity == "WARNING":
            icon = "[W]"
            style = "bold yellow"
            suffix = ""
        else:
            icon = "[i]"
            style = "cyan"
            suffix = ""

        return (
            f"[{style}]{icon} {self.alert_type}: {self.message}{suffix}[/] [dim]{self.time_str}[/]"
        )


class AlertViewModel:
    """
    ViewModel for alerts with incremental update detection.

    OPT-011: Only triggers re-render when content actually changes.
    """

    def __init__(self) -> None:
        self._previous_rows: List[AlertDisplayRow] = []
        self._previous_content: Optional[str] = None

    def compute_display(
        self, display_alerts: List[Dict[str, Any]]
    ) -> tuple[List[AlertDisplayRow], str]:
        """
        Compute display rows and formatted content.

        Returns:
            Tuple of (rows, formatted_content)
        """
        if not display_alerts:
            return [], "[dim]No market alerts[/]"

        rows = []
        for alert in display_alerts:
            alert_key = f"{alert.get('type', 'UNKNOWN')}_{alert.get('severity', 'INFO')}"
            last_seen = alert.get("last_seen")
            time_str = last_seen.strftime("%H:%M:%S") if isinstance(last_seen, datetime) else ""

            row = AlertDisplayRow(
                alert_key=alert_key,
                alert_type=alert.get("type", "UNKNOWN"),
                message=alert.get("message", ""),
                severity=alert.get("severity", "INFO"),
                is_active=alert.get("is_active", True),
                time_str=time_str,
            )
            rows.append(row)

        content = "\n".join(row.format_line() for row in rows)
        return rows, content

    def needs_update(self, display_alerts: List[Dict[str, Any]]) -> tuple[bool, str]:
        """
        Check if update is needed and return new content.

        OPT-011: Avoids re-render if content unchanged.

        Returns:
            Tuple of (needs_update, new_content)
        """
        rows, content = self.compute_display(display_alerts)

        # Compare with previous
        if content == self._previous_content:
            return False, content

        # Update cache
        self._previous_rows = rows
        self._previous_content = content
        return True, content

    def invalidate(self) -> None:
        """Clear cache, forcing update on next call."""
        self._previous_rows = []
        self._previous_content = None
