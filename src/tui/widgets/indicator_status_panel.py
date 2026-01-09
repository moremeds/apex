"""
Indicator status panel for Data view (Tab 7).

Displays indicator update status as a high-level summary with drill-down.
Shows indicator name, symbol count, and last update timestamp.

Layout:
- Summary rows showing indicator name, symbol count, last update
- Expandable to show per-symbol details
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable, Static


def format_ago(dt: Optional[datetime]) -> str:
    """Format datetime as relative time (e.g., '2s ago')."""
    if dt is None:
        return "?"
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt

    if delta.total_seconds() < 0:
        return "future"
    if delta.total_seconds() < 60:
        return f"{int(delta.total_seconds())}s ago"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if delta.total_seconds() < 86400:
        return f"{int(delta.total_seconds() / 3600)}h ago"
    return f"{int(delta.total_seconds() / 86400)}d ago"


def format_time(dt: Optional[datetime]) -> str:
    """Format datetime as HH:MM:SS."""
    if dt is None:
        return "?"
    return dt.strftime("%H:%M:%S")


class IndicatorStatusPanel(Widget):
    """
    Indicator status panel with summary and drill-down.

    Displays:
    - Summary rows: indicator name, symbol count, last update
    - Expanded view: per-symbol details with timestamps
    """

    # Reactive data
    summary: reactive[List[Dict]] = reactive(list, init=False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Track expanded indicators
        self._expanded: Set[str] = set()
        # Details cache: indicator -> list of symbol details
        self._details: Dict[str, List[Dict]] = {}
        # Row keys for navigation
        self._row_keys: List[str] = []
        # Currently selected row index
        self._selected_idx: int = 0
        # Active state for visual feedback
        self._is_active: bool = False

    def compose(self) -> ComposeResult:
        """Compose the indicator panel layout."""
        yield DataTable(id="indicator-table", cursor_type="row")

    def on_mount(self) -> None:
        """Initialize table columns."""
        table = self.query_one("#indicator-table", DataTable)
        table.add_column("", width=2, key="expand")  # Expand indicator
        table.add_column("Indicator", width=12, key="indicator")
        table.add_column("Symbols", width=8, key="symbols")
        table.add_column("Last Update", width=14, key="last_update")

    def watch_summary(self, data: List[Dict]) -> None:
        """Rebuild table when summary data changes."""
        self._rebuild_table()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Set active state for visual feedback."""
        self._is_active = active
        self._rebuild_table()

    def set_details(self, indicator: str, details: List[Dict]) -> None:
        """Set details for an indicator."""
        self._details[indicator] = details
        if indicator in self._expanded:
            self._rebuild_table()

    def move_cursor(self, delta: int) -> None:
        """Move cursor up or down."""
        if not self._row_keys:
            return
        self._selected_idx = max(0, min(len(self._row_keys) - 1, self._selected_idx + delta))
        try:
            table = self.query_one("#indicator-table", DataTable)
            if self._row_keys:
                table.move_cursor(row=self._selected_idx)
        except Exception:
            pass

    def toggle_selected(self) -> Optional[str]:
        """
        Toggle expand/collapse on selected row.

        Returns the indicator name if this is a toggleable row that was expanded,
        or None if collapsed or not a summary row.
        """
        if not self._row_keys or self._selected_idx >= len(self._row_keys):
            return None

        key = self._row_keys[self._selected_idx]
        # Only toggle if it's an indicator row (not a detail row)
        if "/" not in key:  # Indicator rows are just "rsi", detail rows are "rsi/AAPL"
            if key in self._expanded:
                self._expanded.discard(key)
                self._rebuild_table()
                return None  # Collapsed, no need to fetch details
            else:
                self._expanded.add(key)
                self._rebuild_table()
                return key  # Expanded, request details
        return None

    def get_selected_indicator(self) -> Optional[str]:
        """Get currently selected indicator name."""
        if not self._row_keys or self._selected_idx >= len(self._row_keys):
            return None
        key = self._row_keys[self._selected_idx]
        return key.split("/")[0] if "/" in key else key

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        """Rebuild the table with current data and expand state."""
        try:
            table = self.query_one("#indicator-table", DataTable)
        except Exception:
            return

        table.clear()
        self._row_keys = []

        if not self.summary:
            table.add_row("", "[dim]No data[/]", "", "", key="empty")
            self._row_keys.append("empty")
            return

        for record in self.summary:
            indicator = record.get("indicator", "?")
            symbol_count = record.get("symbol_count", 0)
            last_update = record.get("last_update")
            is_expanded = indicator in self._expanded

            # Indicator summary row
            expand_icon = "▼" if is_expanded else "▶"

            # Color based on recency
            ago_str = format_ago(last_update)
            if last_update:
                now = datetime.now(last_update.tzinfo) if last_update.tzinfo else datetime.now()
                delta = now - last_update
                if delta.total_seconds() < 60:
                    ago_color = "[green]"
                elif delta.total_seconds() < 300:
                    ago_color = "[yellow]"
                else:
                    ago_color = "[red]"
                ago_display = f"{ago_color}{ago_str}[/]"
            else:
                ago_display = "[dim]?[/]"

            table.add_row(
                expand_icon,
                f"[bold]{indicator}[/]",
                str(symbol_count),
                ago_display,
                key=indicator,
            )
            self._row_keys.append(indicator)

            # Detail rows if expanded
            if is_expanded:
                details = self._details.get(indicator, [])
                if not details:
                    table.add_row(
                        "",
                        "  [dim]Loading...[/]",
                        "",
                        "",
                        key=f"{indicator}/loading",
                    )
                    self._row_keys.append(f"{indicator}/loading")
                else:
                    for detail in sorted(details, key=lambda d: d.get("symbol", "")):
                        symbol = detail.get("symbol", "?")
                        tf = detail.get("timeframe", "?")
                        detail_update = detail.get("last_update")

                        detail_ago = format_ago(detail_update) if detail_update else "?"
                        detail_key = f"{indicator}/{symbol}"

                        table.add_row(
                            "",
                            f"  [dim]{symbol}/{tf}[/]",
                            "",
                            f"[dim]{detail_ago}[/]",
                            key=detail_key,
                        )
                        self._row_keys.append(detail_key)

        # Restore cursor position
        if self._row_keys and self._selected_idx < len(self._row_keys):
            table.move_cursor(row=self._selected_idx)
