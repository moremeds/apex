"""
Market alerts list widget.

Displays market-wide alerts with persistence tracking:
- Active alerts with severity icons
- Recently cleared alerts (dimmed)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult


class AlertsList(Widget):
    """Market alerts display with persistence."""

    # Reactive state
    alerts: reactive[List[Dict[str, Any]]] = reactive([], init=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._persistent_alerts: Dict[str, Dict] = {}
        self._alert_retention_seconds = 300

    def compose(self) -> ComposeResult:
        """Compose the alerts list layout."""
        with Vertical(id="alerts-content"):
            yield Static("[bold]Market Alerts[/]", id="alerts-title")
            yield Static("[dim]No market alerts[/]", id="alerts-list")

    def watch_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Update display when alerts change."""
        display_alerts = self._update_persistent_alerts(alerts)
        self._render_alerts(display_alerts)

    def _update_persistent_alerts(self, current_alerts: List[Dict[str, Any]]) -> List[Dict]:
        """Update persistent alert tracking."""
        now = datetime.now()
        current_keys = set()

        for alert in current_alerts:
            alert_key = f"{alert.get('type', 'UNKNOWN')}_{alert.get('severity', 'INFO')}"
            current_keys.add(alert_key)

            if alert_key in self._persistent_alerts:
                self._persistent_alerts[alert_key]["alert_data"] = alert
                self._persistent_alerts[alert_key]["last_seen"] = now
                self._persistent_alerts[alert_key]["is_active"] = True
            else:
                self._persistent_alerts[alert_key] = {
                    "alert_data": alert,
                    "first_seen": now,
                    "last_seen": now,
                    "is_active": True,
                }

        # Mark inactive
        for alert_key in self._persistent_alerts:
            if alert_key not in current_keys:
                self._persistent_alerts[alert_key]["is_active"] = False

        # Build display list
        display_alerts = []
        expired_keys = []

        for alert_key, alert_info in self._persistent_alerts.items():
            age_seconds = (now - alert_info["last_seen"]).total_seconds()

            if alert_info["is_active"]:
                display_alerts.append({
                    **alert_info["alert_data"],
                    "first_seen": alert_info["first_seen"],
                    "last_seen": alert_info["last_seen"],
                    "is_active": True,
                })
            elif age_seconds <= self._alert_retention_seconds:
                display_alerts.append({
                    **alert_info["alert_data"],
                    "first_seen": alert_info["first_seen"],
                    "last_seen": alert_info["last_seen"],
                    "is_active": False,
                })
            else:
                expired_keys.append(alert_key)

        for key in expired_keys:
            del self._persistent_alerts[key]

        return display_alerts

    def _render_alerts(self, display_alerts: List[Dict]) -> None:
        """Render alert list."""
        try:
            alerts_list = self.query_one("#alerts-list", Static)

            if not display_alerts:
                alerts_list.update("[dim]No market alerts[/]")
                return

            lines = []
            for alert in display_alerts:
                alert_type = alert.get("type", "UNKNOWN")
                message = alert.get("message", "")
                severity = alert.get("severity", "INFO")
                is_active = alert.get("is_active", True)
                last_seen = alert.get("last_seen")

                time_str = last_seen.strftime("%H:%M:%S") if last_seen else ""

                if not is_active:
                    icon = "o"
                    style = "dim"
                    suffix = " [cleared]"
                elif severity == "CRITICAL":
                    icon = "[!]"
                    style = "bold red"
                    suffix = ""
                elif severity == "WARNING":
                    icon = "[W]"
                    style = "bold yellow"
                    suffix = ""
                else:
                    icon = "[i]"
                    style = "cyan"
                    suffix = ""

                lines.append(f"[{style}]{icon} {alert_type}: {message}{suffix}[/] [dim]{time_str}[/]")

            alerts_list.update("\n".join(lines))
        except Exception:
            pass
