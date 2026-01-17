"""
Historical coverage panel for Data view (Tab 7).

Displays historical data coverage grouped by ticker with collapsible rows.
Shows timeframe, date range, and bar count for each symbol.

Layout:
- Collapsible ticker groups
- Expanded view shows per-timeframe details
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable


def format_date(dt: Optional[datetime]) -> str:
    """Format datetime as short date."""
    if dt is None:
        return "?"
    return dt.strftime("%Y-%m-%d")


def format_bars(n: Optional[int]) -> str:
    """Format bar count with K/M suffix."""
    if n is None:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_size(bytes_val: Optional[int]) -> str:
    """Format file size with KB/MB suffix."""
    if bytes_val is None:
        return "?"
    if bytes_val >= 1_000_000:
        return f"{bytes_val / 1_000_000:.1f}MB"
    if bytes_val >= 1_000:
        return f"{bytes_val / 1_000:.1f}KB"
    return f"{bytes_val}B"


class HistoricalCoveragePanel(Widget):
    """
    Historical data coverage panel with collapsible ticker groups.

    Displays:
    - Ticker header with timeframe count (collapsible)
    - Per-timeframe details when expanded
    """

    # Reactive data
    coverage_data: reactive[Dict[str, List[Dict]]] = reactive(dict, init=False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Track expanded symbols
        self._expanded: Set[str] = set()
        # Row keys for navigation
        self._row_keys: List[str] = []
        # Currently selected row index
        self._selected_idx: int = 0
        # Active state for visual feedback
        self._is_active: bool = True

    def compose(self) -> ComposeResult:
        """Compose the coverage panel layout."""
        yield DataTable(id="coverage-table", cursor_type="row")

    def on_mount(self) -> None:
        """Initialize table columns."""
        table = self.query_one("#coverage-table", DataTable)
        table.add_column("", width=2, key="expand")  # Expand indicator
        table.add_column("Symbol / TF", width=14, key="symbol")
        table.add_column("Date Range", width=26, key="range")
        table.add_column("Bars", width=8, key="bars")
        table.add_column("Size", width=8, key="size")

    def watch_coverage_data(self, data: Dict[str, List[Dict]]) -> None:
        """Rebuild table when coverage data changes."""
        self._rebuild_table()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Set active state for visual feedback."""
        if self._is_active != active:
            self._is_active = active
            # Don't rebuild - just update visual if needed in future

    def move_cursor(self, delta: int) -> None:
        """Move cursor up or down."""
        if not self._row_keys:
            return
        self._selected_idx = max(0, min(len(self._row_keys) - 1, self._selected_idx + delta))
        try:
            table = self.query_one("#coverage-table", DataTable)
            if self._row_keys:
                table.move_cursor(row=self._selected_idx)
        except Exception:
            pass

    def toggle_selected(self) -> Optional[str]:
        """Toggle expand/collapse on selected row."""
        if not self._row_keys or self._selected_idx >= len(self._row_keys):
            return None

        key = self._row_keys[self._selected_idx]
        # Only toggle if it's a symbol row (not a detail row)
        if "/" not in key:  # Symbol rows are just "AAPL", detail rows are "AAPL/1d"
            if key in self._expanded:
                self._expanded.discard(key)
            else:
                self._expanded.add(key)
            self._rebuild_table()
            return key
        return None

    def get_selected_symbol(self) -> Optional[str]:
        """Get currently selected symbol."""
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
            table = self.query_one("#coverage-table", DataTable)
        except Exception:
            return

        table.clear()
        self._row_keys = []

        if not self.coverage_data:
            table.add_row("", "[dim]No data[/]", "", "", "", key="empty")
            self._row_keys.append("empty")
            return

        for symbol in sorted(self.coverage_data.keys()):
            records = self.coverage_data[symbol]
            tf_count = len(records)
            is_expanded = symbol in self._expanded

            # Calculate totals for this symbol
            total_bars = sum(r.get("total_bars", 0) or 0 for r in records)
            total_size = sum(r.get("file_size", 0) or 0 for r in records)

            # Symbol header row
            expand_icon = "▼" if is_expanded else "▶"
            symbol_label = f"[bold]{symbol}[/] ({tf_count} tf)"

            # Date range across all timeframes
            all_earliest: List[datetime] = [r["earliest"] for r in records if r.get("earliest") is not None]
            all_latest: List[datetime] = [r["latest"] for r in records if r.get("latest") is not None]
            earliest: Optional[datetime] = min(all_earliest) if all_earliest else None
            latest: Optional[datetime] = max(all_latest) if all_latest else None
            date_range = f"{format_date(earliest)} - {format_date(latest)}"

            table.add_row(
                expand_icon,
                symbol_label,
                date_range,
                format_bars(total_bars),
                format_size(total_size) if total_size else "?",
                key=symbol,
            )
            self._row_keys.append(symbol)

            # Detail rows if expanded
            if is_expanded:
                for record in sorted(records, key=lambda r: r.get("timeframe", "")):
                    tf = record.get("timeframe", "?")
                    earliest_tf = record.get("earliest")
                    latest_tf = record.get("latest")
                    bars = record.get("total_bars", 0)
                    file_size = record.get("file_size")

                    tf_range = f"{format_date(earliest_tf)} - {format_date(latest_tf)}"
                    detail_key = f"{symbol}/{tf}"

                    table.add_row(
                        "",
                        f"  [dim]{tf}[/]",
                        f"  [dim]{tf_range}[/]",
                        f"[dim]{format_bars(bars)}[/]",
                        f"[dim]{format_size(file_size)}[/]",
                        key=detail_key,
                    )
                    self._row_keys.append(detail_key)

        # Restore cursor position
        if self._row_keys and self._selected_idx < len(self._row_keys):
            table.move_cursor(row=self._selected_idx)
