"""
Indicator status panel for Data view (Tab 7).

Displays indicator update status grouped by category with drill-down.
Three-level hierarchy:
1. Category (momentum, trend, volatility, etc.) - collapsible
2. Indicator (rsi, macd, etc.) - collapsible, shows per-symbol details
3. Details - per-symbol info (leaf nodes)

Layout:
- Category headers with indicator count
- Expandable indicators showing symbol count, last update
- Per-symbol details when indicator is expanded
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable


# Category display order and icons
CATEGORY_ORDER = ["momentum", "trend", "volatility", "volume", "moving_avg", "pattern", "other"]
CATEGORY_LABELS = {
    "momentum": "📈 Momentum",
    "trend": "📊 Trend",
    "volatility": "📉 Volatility",
    "volume": "📦 Volume",
    "moving_avg": "〰️ Moving Avg",
    "pattern": "🔷 Patterns",
    "other": "📋 Other",
}


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
        table.add_column("Indicator / Description", width=65, key="name")
        table.add_column("Sym", width=4, key="symbols")
        table.add_column("Stale", width=10, key="stalest")  # Shows oldest/most lagging

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
        """Rebuild the table with category grouping."""
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

        # Group indicators by category
        by_category: Dict[str, List[Dict]] = {}
        for record in self.summary:
            cat = record.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(record)

        # Display in category order
        for category in CATEGORY_ORDER:
            if category not in by_category:
                continue

            indicators = by_category[category]
            is_cat_expanded = category in self._expanded_categories
            cat_icon = "▼" if is_cat_expanded else "▶"
            cat_label = CATEGORY_LABELS.get(category, category.title())

            # Find the oldest (most stale) update across all indicators in this category
            oldest_in_category = None
            for ind in indicators:
                ind_oldest = ind.get("oldest_update")
                if ind_oldest:
                    if oldest_in_category is None or ind_oldest < oldest_in_category:
                        oldest_in_category = ind_oldest

            cat_stale_display = self._format_time_colored(oldest_in_category, warn_threshold=7*86400) if oldest_in_category else ""

            # Category header row
            cat_key = f"cat:{category}"
            table.add_row(
                cat_icon,
                f"[bold]{cat_label}[/] [dim]({len(indicators)} indicators)[/]",
                "",
                cat_stale_display,
                key=cat_key,
            )
            self._row_keys.append(cat_key)

            # Show indicators if category is expanded
            if is_cat_expanded:
                for record in sorted(indicators, key=lambda r: r.get("indicator", "")):
                    self._add_indicator_row(table, record)

        # Restore cursor position once at the end (prevents flickering)
        if self._row_keys:
            # Clamp selected index to valid range
            self._selected_idx = min(self._selected_idx, len(self._row_keys) - 1)
            try:
                table.move_cursor(row=self._selected_idx)
            except Exception:
                pass

    def _add_indicator_row(self, table: DataTable, record: Dict) -> None:
        """Add an indicator row (and details if expanded)."""
        indicator = record.get("indicator", "?")
        full_name = record.get("full_name", indicator)
        description = record.get("description", "")
        symbol_count = record.get("symbol_count", 0)
        # Show OLDEST update (most stale) so lagging data is visible at a glance
        oldest_update = record.get("oldest_update")
        is_expanded = indicator in self._expanded_indicators

        # Indicator row - show short name in bold + full name
        ind_icon = "▼" if is_expanded else "▶"

        # Format oldest update time - this shows the worst case (most stale data)
        update_display = self._format_time_colored(oldest_update, warn_threshold=7*86400)

        # Format: "rsi - Relative Strength Index"
        display_name = f"[bold]{indicator}[/] - {full_name}"

        ind_key = f"ind:{indicator}"
        table.add_row(
            f"  {ind_icon}",
            f"  {display_name}",
            str(symbol_count),
            update_display,
            key=ind_key,
        )
        self._row_keys.append(ind_key)

        # Detail rows if expanded
        if is_expanded:
            # First show description if available
            if description:
                desc_key = f"desc:{indicator}"
                table.add_row(
                    "",
                    f"    [italic cyan]{description}[/]",
                    "",
                    "",
                    key=desc_key,
                )
                self._row_keys.append(desc_key)

            details = self._details.get(indicator, [])
            if not details:
                table.add_row(
                    "",
                    "    [dim]Loading symbols...[/]",
                    "",
                    "",
                    key=f"det:{indicator}/loading",
                )
                self._row_keys.append(f"det:{indicator}/loading")
            else:
                # Sort alphabetically by symbol
                for detail in sorted(details, key=lambda d: d.get("symbol", "")):
                    symbol = detail.get("symbol", "?")
                    tf = detail.get("timeframe", "?")
                    detail_update = detail.get("last_update")

                    detail_display = self._format_time_colored(detail_update, warn_threshold=7*86400)
                    detail_key = f"det:{indicator}/{symbol}/{tf}"

                    table.add_row(
                        "",
                        f"    {symbol} ({tf})",
                        "",
                        detail_display,
                        key=detail_key,
                    )
                    self._row_keys.append(detail_key)

    def _format_time_colored(self, dt: Optional[datetime], warn_threshold: float = 300) -> str:
        """Format datetime with color based on age."""
        if dt is None:
            return "[dim]?[/]"

        ago_str = format_ago(dt)
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt
        seconds = delta.total_seconds()

        if seconds < 0:
            return "[blue]future[/]"
        if seconds < 3600:  # < 1 hour - green
            return f"[green]{ago_str}[/]"
        if seconds < 86400:  # < 1 day - yellow
            return f"[yellow]{ago_str}[/]"
        if seconds < warn_threshold:  # < threshold - yellow
            return f"[yellow]{ago_str}[/]"
        # Very old - red
        return f"[red]{ago_str}[/]"
