"""
Data view for historical coverage and indicator status (Tab 7).

Layout:
- Left (~50%): Historical data coverage grouped by ticker
- Right (~50%): Indicator DB status with summary + drill-down
- Bottom: Stats bar

Keyboard shortcuts:
- r: Manual refresh
- Enter: Expand/collapse selected item
- Up/Down: Navigate lists
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Static

from ..widgets.historical_coverage_panel import HistoricalCoveragePanel
from ..widgets.indicator_status_panel import IndicatorStatusPanel

if TYPE_CHECKING:
    pass


class DataView(Container, can_focus=True):
    """
    Data view showing historical coverage and indicator DB status.

    Provides:
    - Left panel: Historical data coverage grouped by ticker
    - Right panel: Indicator status summary with drill-down
    - Bottom: Stats summary bar
    """

    DEFAULT_CSS = ""

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "toggle_expand", "Expand", show=True),
        Binding("tab", "switch_panel", "Switch Panel", show=True),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active_panel: str = "coverage"  # "coverage" or "indicators"
        self._stats_text: str = "Loading..."

    def on_show(self) -> None:
        """Focus this view when it becomes visible."""
        self.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in either DataTable - auto-switch active panel."""
        table_id = event.data_table.id if event.data_table else ""
        if table_id == "coverage-table":
            if self._active_panel != "coverage":
                self._active_panel = "coverage"
                self._sync_panel_active_state()
        elif table_id == "indicator-table":
            if self._active_panel != "indicators":
                self._active_panel = "indicators"
                self._sync_panel_active_state()

    def _sync_panel_active_state(self) -> None:
        """Sync the active state to both panels."""
        try:
            coverage = self.query_one("#data-coverage", HistoricalCoveragePanel)
            indicators = self.query_one("#data-indicators", IndicatorStatusPanel)
            coverage.set_active(self._active_panel == "coverage")
            indicators.set_active(self._active_panel == "indicators")
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        """Compose the data view layout."""
        with Horizontal(id="data-main"):
            # Left side - Historical coverage (~50%)
            with Vertical(id="data-left"):
                with Vertical(id="data-coverage-panel"):
                    yield Static(
                        "Historical Data Coverage",
                        id="data-coverage-title",
                        classes="panel-title",
                    )
                    yield HistoricalCoveragePanel(id="data-coverage")

            # Right side - Indicator status (~50%)
            with Vertical(id="data-right"):
                with Vertical(id="data-indicator-panel"):
                    yield Static(
                        "Indicator DB Status",
                        id="data-indicator-title",
                        classes="panel-title",
                    )
                    yield IndicatorStatusPanel(id="data-indicators")

        # Bottom - Stats bar
        with Vertical(id="data-stats-bar"):
            yield Static(self._stats_text, id="data-stats")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def update_coverage(self, coverage_data: Dict[str, List[Dict]]) -> None:
        """
        Update historical coverage display.

        Args:
            coverage_data: Dict mapping symbol -> list of coverage records
                Each record: {timeframe, source, earliest, latest, total_bars}
        """
        try:
            panel = self.query_one("#data-coverage", HistoricalCoveragePanel)
            panel.coverage_data = coverage_data
        except Exception:
            self.log.error("Failed to update coverage panel")

    def update_indicator_summary(self, summary: List[Dict]) -> None:
        """
        Update indicator summary display.

        Args:
            summary: List of indicator summaries
                Each: {indicator, symbol_count, last_update, oldest_update}
        """
        try:
            panel = self.query_one("#data-indicators", IndicatorStatusPanel)
            panel.summary = summary
        except Exception:
            self.log.error("Failed to update indicator panel")

    def update_indicator_details(self, indicator: str, details: List[Dict]) -> None:
        """
        Update details for expanded indicator.

        Args:
            indicator: Indicator name
            details: List of per-symbol details
                Each: {symbol, timeframe, last_update, state}
        """
        try:
            panel = self.query_one("#data-indicators", IndicatorStatusPanel)
            panel.set_details(indicator, details)
        except Exception:
            self.log.error("Failed to update indicator details")

    def update_stats(
        self,
        symbol_count: int = 0,
        timeframe_count: int = 0,
        total_bars: int = 0,
        db_connected: bool = False,
    ) -> None:
        """Update bottom stats bar."""
        db_status = "[green]Connected[/]" if db_connected else "[red]Disconnected[/]"
        bars_fmt = f"{total_bars:,}" if total_bars < 1_000_000 else f"{total_bars / 1_000_000:.1f}M"
        self._stats_text = (
            f"{symbol_count} symbols | {timeframe_count} timeframes | "
            f"{bars_fmt} bars | DB: {db_status}"
        )
        try:
            stats = self.query_one("#data-stats", Static)
            stats.update(self._stats_text)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_refresh(self) -> None:
        """Request manual refresh from parent app."""
        self.post_message(DataRefreshRequested())
        self.notify("Refreshing data...", severity="information", timeout=2.0)

    def action_move_up(self) -> None:
        """Move selection up in active panel."""
        self._move_cursor(-1)

    def action_move_down(self) -> None:
        """Move selection down in active panel."""
        self._move_cursor(1)

    def action_toggle_expand(self) -> None:
        """Toggle expand/collapse on selected item."""
        try:
            if self._active_panel == "coverage":
                coverage_panel = self.query_one("#data-coverage", HistoricalCoveragePanel)
                coverage_panel.toggle_selected()
            else:
                indicator_panel = self.query_one("#data-indicators", IndicatorStatusPanel)
                indicator = indicator_panel.toggle_selected()
                if indicator:
                    # Request details from parent app
                    self.post_message(IndicatorDetailsRequested(indicator))
        except Exception:
            self.log.error("Failed to toggle expand")

    def action_switch_panel(self) -> None:
        """Switch focus between coverage and indicator panels."""
        self._active_panel = "indicators" if self._active_panel == "coverage" else "coverage"
        self._sync_panel_active_state()

    def _move_cursor(self, delta: int) -> None:
        """Move cursor in active panel."""
        try:
            if self._active_panel == "coverage":
                coverage_panel = self.query_one("#data-coverage", HistoricalCoveragePanel)
                coverage_panel.move_cursor(delta)
            else:
                indicator_panel = self.query_one("#data-indicators", IndicatorStatusPanel)
                indicator_panel.move_cursor(delta)
        except Exception:
            pass


# -------------------------------------------------------------------------
# Custom Messages
# -------------------------------------------------------------------------


from textual.message import Message


class DataRefreshRequested(Message):
    """Request manual data refresh."""


class IndicatorDetailsRequested(Message):
    """Request details for an indicator."""

    def __init__(self, indicator: str) -> None:
        super().__init__()
        self.indicator = indicator
