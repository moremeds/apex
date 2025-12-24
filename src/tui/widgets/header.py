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
from textual.widgets import Static
from textual.app import ComposeResult

from ...utils.market_hours import MarketHours


class HeaderWidget(Widget):
    """Header bar with market status and navigation hints."""

    DEFAULT_CSS = """
    HeaderWidget {
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
    }

    HeaderWidget #header-content {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    """

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
        yield Static(self._build_header_text(), id="header-content")

    def _build_header_text(self) -> str:
        """Build the header text with all components."""
        from ...utils.timezone import DisplayTimezone
        from datetime import datetime

        # Get current time
        if self._display_tz is None:
            self._display_tz = DisplayTimezone("Asia/Hong_Kong")
        current_time = self._display_tz.current_time("%H:%M:%S %Z")

        # Get market status
        market_status = MarketHours.get_market_status()

        # Build header parts
        parts = []

        # Title
        parts.append("[bold cyan]Live Risk Management System[/]")
        parts.append("  |  ")

        # Time
        parts.append(f"[white]{current_time}[/]")
        parts.append("  |  ")

        # Environment
        env_upper = self.env.upper()
        if self.env == "prod":
            parts.append(f"[bold red]{env_upper}[/]")
        elif self.env == "demo":
            parts.append(f"[bold magenta]{env_upper}[/]")
        else:
            parts.append(f"[bold yellow]{env_upper}[/]")
        parts.append("  |  ")

        # Market status
        parts.append("Market: ")
        if market_status == "OPEN":
            parts.append("[bold green]OPEN[/]")
        elif market_status == "EXTENDED":
            parts.append("[bold yellow]EXTENDED HOURS[/]")
        else:
            parts.append("[bold red]CLOSED[/]")
        parts.append("  |  ")

        # Tab hints
        tabs = [
            ("1", "Summary"),
            ("2", "Signals"),
            ("3", "IB"),
            ("4", "Futu"),
            ("5", "Lab"),
        ]
        for key, label in tabs:
            parts.append(f"[dim][{key}]{label}[/]  ")

        return "".join(parts)

    def refresh_time(self) -> None:
        """Refresh the header to update time display."""
        try:
            content = self.query_one("#header-content", Static)
            content.update(self._build_header_text())
        except Exception:
            pass
