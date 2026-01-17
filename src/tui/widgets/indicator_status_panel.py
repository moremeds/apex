"""
Indicator status panel for Data view (Tab 7).

Displays indicator update status grouped by category with drill-down.
Three-level hierarchy:
1. Category (momentum, trend, volatility, etc.) - collapsible
2. Indicator (rsi, macd, etc.) - collapsible, shows per-symbol details
3. Details - per-symbol info (leaf nodes)

Uses IndicatorStatusViewModel for data transformation.
Widget handles only rendering and UI state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Set

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable

from ..viewmodels.indicator_status_vm import (
    IndicatorRow,
    IndicatorStatusViewModel,
    RowType,
)


class IndicatorStatusPanel(Widget):
    """
    Indicator status panel with category grouping and drill-down.

    Three-level hierarchy:
    - Category (collapsible): momentum, trend, volatility, etc.
    - Indicator (collapsible): rsi, macd, kdj, etc.
    - Details (leaf): per-symbol info
    """

    # Reactive data - list of indicator summaries with category
    summary: reactive[List[Dict]] = reactive(list, init=False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # ViewModel for data transformation
        self._view_model = IndicatorStatusViewModel()
        # Track expanded categories
        self._expanded_categories: Set[str] = set()
        # Track expanded indicators
        self._expanded_indicators: Set[str] = set()
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
        table.add_column("", width=3, key="expand")  # Expand indicator
        table.add_column("Indicator / Description", width=55, key="name")
        table.add_column("Sym", width=4, key="symbols")
        table.add_column("DB Timestamp", width=20, key="stalest")  # Raw timestamp with TZ

    def watch_summary(self, data: List[Dict]) -> None:
        """Rebuild table when summary data changes."""
        self.log.info(f"watch_summary triggered with {len(data) if data else 0} records")
        self._rebuild_table()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Set active state for visual feedback."""
        if self._is_active != active:
            self._is_active = active
            # Don't rebuild - just update visual if needed in future

    def set_details(self, indicator: str, details: List[Dict]) -> None:
        """Set details for an indicator."""
        self._details[indicator] = details
        if indicator in self._expanded_indicators:
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

        Returns:
            - Indicator name if an indicator row was expanded (needs details fetch)
            - None if category toggled or row collapsed
        """
        if not self._row_keys or self._selected_idx >= len(self._row_keys):
            return None

        key = self._row_keys[self._selected_idx]

        # Category row: "cat:momentum"
        if key.startswith("cat:"):
            category = key[4:]
            if category in self._expanded_categories:
                self._expanded_categories.discard(category)
            else:
                self._expanded_categories.add(category)
            self._rebuild_table()
            return None

        # Indicator row: "ind:rsi"
        if key.startswith("ind:"):
            indicator = key[4:]
            if indicator in self._expanded_indicators:
                self._expanded_indicators.discard(indicator)
                self._rebuild_table()
                return None
            else:
                self._expanded_indicators.add(indicator)
                self._rebuild_table()
                return indicator  # Needs details fetch

        # Detail row: "det:rsi/AAPL" - not expandable
        return None

    def get_selected_indicator(self) -> Optional[str]:
        """Get currently selected indicator name."""
        if not self._row_keys or self._selected_idx >= len(self._row_keys):
            return None
        key = self._row_keys[self._selected_idx]
        if key.startswith("ind:"):
            return key[4:]
        if key.startswith("det:"):
            # Extract indicator from "det:rsi/AAPL"
            return key[4:].split("/")[0]
        return None

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        """Rebuild the table using ViewModel for data transformation."""
        try:
            table = self.query_one("#indicator-table", DataTable)
        except Exception:
            return

        table.clear()
        self._row_keys = []

        # Get renderable rows from ViewModel
        rows = self._view_model.build_rows(
            summary=self.summary or [],
            expanded_categories=self._expanded_categories,
            expanded_indicators=self._expanded_indicators,
            details=self._details,
        )

        # Render each row based on type
        for row in rows:
            self._render_row(table, row)
            self._row_keys.append(row.key)

        # Restore cursor position (prevents flickering)
        if self._row_keys:
            self._selected_idx = min(self._selected_idx, len(self._row_keys) - 1)
            try:
                table.move_cursor(row=self._selected_idx)
            except Exception:
                pass

    def _render_row(self, table: DataTable, row: IndicatorRow) -> None:
        """Render a single row based on its type."""
        if row.row_type == RowType.EMPTY:
            table.add_row("", "[dim]No data[/]", "", "", key=row.key)

        elif row.row_type == RowType.CATEGORY:
            icon = "▼" if row.is_expanded else "▶"
            ts_display = self._format_time_colored(row.timestamp) if row.timestamp else ""
            table.add_row(
                icon,
                f"[bold]{row.label}[/] [dim]({row.indicator_count} indicators)[/]",
                "",
                ts_display,
                key=row.key,
            )

        elif row.row_type == RowType.INDICATOR:
            icon = "▼" if row.is_expanded else "▶"
            ts_display = self._format_time_colored(row.timestamp)
            table.add_row(
                f"  {icon}",
                f"  [bold]{row.label}[/] - {row.full_name}",
                str(row.symbol_count),
                ts_display,
                key=row.key,
            )

        elif row.row_type == RowType.DESCRIPTION:
            table.add_row("", f"    [italic cyan]{row.description}[/]", "", "", key=row.key)

        elif row.row_type == RowType.LOADING:
            table.add_row("", "    [dim]Loading symbols...[/]", "", "", key=row.key)

        elif row.row_type == RowType.DETAIL:
            ts_display = self._format_time_colored(row.timestamp)
            table.add_row("", f"    {row.symbol} ({row.timeframe})", "", ts_display, key=row.key)

    def _format_time_colored(self, dt: Optional[datetime], warn_threshold: float = 300) -> str:
        """Format datetime showing RAW database timestamp with timezone for debugging."""
        from datetime import timezone

        if dt is None:
            return "[dim]?[/]"

        # Show RAW timestamp from database with timezone info
        # Format: "HH:MM:SS TZ" or "HH:MM:SS (naive)" if no tzinfo
        if dt.tzinfo is not None:
            # Has timezone - show it
            raw_str = dt.strftime("%m-%d %H:%M %Z")
        else:
            # Naive datetime - flag it
            raw_str = dt.strftime("%m-%d %H:%M") + " NAIVE"

        # Color based on age (assume UTC if naive for comparison)
        dt_utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt_utc
        seconds = delta.total_seconds()

        if seconds < 0:
            return f"[blue]{raw_str}[/]"
        if seconds < 3600:  # < 1 hour - green
            return f"[green]{raw_str}[/]"
        if seconds < 86400:  # < 1 day - yellow
            return f"[yellow]{raw_str}[/]"
        # Very old - red
        return f"[red]{raw_str}[/]"
