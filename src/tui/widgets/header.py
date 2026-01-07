"""
Header widget for the Apex Dashboard.

Displays:
- Title: "Live Risk Management System"
- Current time in display timezone
- Environment indicator
- Market status
- Tab navigation hints
"""

from __future__ import annotations

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static, Tabs, Tab
from textual.containers import Horizontal
from textual.app import ComposeResult

from ...utils.market_hours import MarketHours
from ..base import VIEW_TABS


class HeaderWidget(Widget):
    """Header bar with market status and navigation hints."""

    # All styles are in css/dashboard.tcss (P3.7: extracted inline CSS)
    DEFAULT_CSS = ""

    active_tab: reactive[str] = reactive("summary", init=False)

    def __init__(self, env: str = "dev", **kwargs):
        """
        Initialize header widget.

        Args:
            env: Environment name (dev, demo, prod).
        """
        super().__init__(**kwargs)
        self.env = env
        self._display_tz = None

    def compose(self) -> ComposeResult:
        """Compose the header layout."""
        # Note: No Center() wrapper - CSS content-align handles centering
        # Center() inside a 1-row height widget causes layout issues
        with Horizontal(id="header-content"):
            yield Static(self._build_header_text(), id="header-left")
            tabs = [
                Tab(f"[{key}]{label}", id=tab_id)
                for key, label, tab_id, _view in VIEW_TABS
            ]
            yield Tabs(*tabs, id="header-tabs", active=self.active_tab)

    def _build_header_text(self) -> str:
        """Build the header text with all components."""
        from ...utils.timezone import DisplayTimezone
        # Get current time
        if self._display_tz is None:
            self._display_tz = DisplayTimezone("Asia/Hong_Kong")
        current_time = self._display_tz.current_time("%H:%M:%S %Z")

        # Get market status
        market_status = MarketHours.get_market_status()

        sections = [
            "[bold #5fd7ff]Live Risk Management System[/]",
            f"[#c9d1d9]{current_time}[/]",
        ]

        # Environment
        env_upper = self.env.upper()
        if self.env == "prod":
            sections.append(f"[bold #ff6b6b]{env_upper}[/]")
        elif self.env == "demo":
            sections.append(f"[bold #d66efd]{env_upper}[/]")
        else:
            sections.append(f"[bold #f6d365]{env_upper}[/]")

        # Market status
        if market_status == "OPEN":
            market_text = "[bold #7ee787]OPEN[/]"
        elif market_status == "EXTENDED":
            market_text = "[bold #f6d365]EXTENDED HOURS[/]"
        else:
            market_text = "[bold #ff6b6b]CLOSED[/]"
        sections.append(f"Market: {market_text}")

        return " | ".join(sections) + " |"

    def watch_active_tab(self, tab_id: str) -> None:
        """Refresh header when the active tab changes."""
        self._sync_tabs(tab_id)
        self._refresh_header()

    def refresh_time(self) -> None:
        """Refresh the header to update time display."""
        self._refresh_header()

    def _refresh_header(self) -> None:
        """Update header display safely."""
        try:
            content = self.query_one("#header-left", Static)
            content.update(self._build_header_text())
        except Exception as e:
            self.log.error(f"Failed to refresh header: {e}")

    def _sync_tabs(self, tab_id: str) -> None:
        """Sync header tabs with active tab."""
        try:
            tabs = self.query_one("#header-tabs", Tabs)
            if tab_id and tabs.active != tab_id:
                tabs.active = tab_id
        except Exception as e:
            self.log.error(f"Failed to sync tabs: {e}")
