"""
Market alerts list widget.

Displays market-wide alerts with persistence tracking:
- Active alerts with severity icons
- Recently cleared alerts (dimmed)

OPT-011: Uses AlertViewModel for incremental updates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult

from ..viewmodels.alert_vm import AlertViewModel


class AlertsList(Widget):
    """Market alerts display with persistence."""

    # Maximum number of persistent alerts to prevent memory leak
    _MAX_PERSISTENT_ALERTS = 100

    # Reactive state
    alerts: reactive[List[Dict[str, Any]]] = reactive(list, init=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._persistent_alerts: Dict[str, Dict] = {}
        self._alert_retention_seconds = 300
        # OPT-011: ViewModel for incremental updates
        self._vm = AlertViewModel()

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

        # Enforce maximum alert limit to prevent memory leak
        if len(self._persistent_alerts) > self._MAX_PERSISTENT_ALERTS:
            # Sort by (is_active, last_seen) - remove oldest inactive first
            sorted_alerts = sorted(
                self._persistent_alerts.items(),
                key=lambda x: (x[1]["is_active"], x[1]["last_seen"])
            )
            excess_count = len(self._persistent_alerts) - self._MAX_PERSISTENT_ALERTS
            for key, _ in sorted_alerts[:excess_count]:
                del self._persistent_alerts[key]

        return display_alerts

    def _render_alerts(self, display_alerts: List[Dict]) -> None:
        """Render alert list with incremental updates."""
        try:
            alerts_list = self.query_one("#alerts-list", Static)

            # OPT-011: Use ViewModel to check if update needed
            needs_update, content = self._vm.needs_update(display_alerts)
            if needs_update:
                alerts_list.update(content)
        except Exception as e:
            self.log.error(f"Failed to render alerts: {e}")
