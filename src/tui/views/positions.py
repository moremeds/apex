"""
Positions view for broker-specific position details (Tab 3/4).

Layout matching original Rich dashboard:
- Left (~65%): Broker positions table with IV column
- Right top: ATR analysis panel
- Right bottom: Open Orders panel

Keyboard shortcuts:
- w/s: Navigate underlyings
- +/-: Adjust ATR period
- t: Cycle timeframe
- h: Toggle help
- r: Reset ATR period
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical

from ..widgets.atr_panel import ATRPanel
from ..widgets.orders_panel import OrdersPanel
from ..widgets.positions_table import PositionsTable

if TYPE_CHECKING:
    from ...domain.events.domain_events import PositionDeltaEvent
    from ...models.risk_snapshot import RiskSnapshot


class PositionsView(Container, can_focus=True):
    """Broker-specific positions view with ATR analysis."""

    # Styles are defined in css/dashboard.tcss using the Rich-matching palette:
    # - PositionsTable: blue border (#2f6fb3)
    # - ATRPanel: orange border (#f59e0b)
    # - OrdersPanel: gray border (#3a4148)
    DEFAULT_CSS = ""

    BINDINGS = [
        Binding("w", "move_up", "Up", show=True),
        Binding("s", "move_down", "Down", show=True),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("+", "increase_atr_period", "+ATR", show=True),
        Binding("-", "decrease_atr_period", "-ATR", show=True),
        Binding("t", "cycle_timeframe", "Timeframe", show=True),
        Binding("h", "toggle_help", "Help", show=True),
        Binding("r", "reset_atr", "Reset ATR", show=True),
    ]

    def __init__(self, broker: str = "ib", **kwargs: Any) -> None:
        """
        Initialize positions view.

        Args:
            broker: Broker to filter positions ("ib" or "futu").
        """
        super().__init__(**kwargs)
        self.broker = broker
        self.broker_display = broker.upper()

    def on_show(self) -> None:
        """Focus this view when it becomes visible."""
        self.focus()

    def compose(self) -> ComposeResult:
        """Compose the positions view layout."""
        with Horizontal():
            # Left side - Positions table (~65%)
            with Vertical(classes="positions-left"):
                yield PositionsTable(
                    id=f"{self.broker}-positions",
                    broker_filter=self.broker,
                    show_portfolio_row=True,
                    consolidated=False,
                )

            # Right side - ATR + Orders (~35%)
            with Vertical(classes="positions-right"):
                yield ATRPanel(id=f"{self.broker}-atr")
                yield OrdersPanel(broker=self.broker_display, id=f"{self.broker}-orders")

    def on_positions_table_position_selected(self, event: PositionsTable.PositionSelected) -> None:
        """Handle position selection from the table."""
        try:
            atr_panel = self.query_one(f"#{self.broker}-atr", ATRPanel)
            atr_panel.position = event.position
            atr_panel.selected_symbol = event.underlying
        except Exception:
            self.log.error("Failed to update ATR panel on selection")

    def update_data(self, snapshot: Optional["RiskSnapshot"]) -> None:
        """
        Update the view with new position data.

        Args:
            snapshot: RiskSnapshot with position data.
        """
        position_risks = getattr(snapshot, "position_risks", []) if snapshot else []
        try:
            positions_table = self.query_one(f"#{self.broker}-positions", PositionsTable)
            positions_table.positions = position_risks
        except Exception:
            self.log.error("Failed to update positions table")

    def apply_deltas(self, deltas: Dict[str, "PositionDeltaEvent"]) -> None:
        """
        Apply position deltas for streaming updates.

        Fast path that updates specific cells without full table refresh.

        Args:
            deltas: Dict mapping symbol -> PositionDeltaEvent
        """
        try:
            positions_table = self.query_one(f"#{self.broker}-positions", PositionsTable)
            positions_table.apply_deltas(deltas)
        except Exception:
            self.log.error("Failed to apply position deltas")

    def action_move_up(self) -> None:
        """Move selection up in the positions table."""
        try:
            table = self.query_one(f"#{self.broker}-positions", PositionsTable)
            if table.cursor_row is not None and table.cursor_row > 0:
                table.move_cursor(row=table.cursor_row - 1)
                self._on_selection_change()
        except Exception:
            self.log.error("Failed to move up")

    def action_move_down(self) -> None:
        """Move selection down in the positions table."""
        try:
            table = self.query_one(f"#{self.broker}-positions", PositionsTable)
            if table.cursor_row is not None and table.cursor_row < table.row_count - 1:
                table.move_cursor(row=table.cursor_row + 1)
                self._on_selection_change()
        except Exception:
            self.log.error("Failed to move down")

    def action_increase_atr_period(self) -> None:
        """Increase ATR period."""
        try:
            atr_panel = self.query_one(f"#{self.broker}-atr", ATRPanel)
            atr_panel.adjust_period(1)
        except Exception:
            self.log.error("Failed to increase ATR period")

    def action_decrease_atr_period(self) -> None:
        """Decrease ATR period."""
        try:
            atr_panel = self.query_one(f"#{self.broker}-atr", ATRPanel)
            atr_panel.adjust_period(-1)
        except Exception:
            self.log.error("Failed to decrease ATR period")

    def action_cycle_timeframe(self) -> None:
        """Cycle ATR timeframe."""
        try:
            atr_panel = self.query_one(f"#{self.broker}-atr", ATRPanel)
            atr_panel.cycle_timeframe()
        except Exception:
            self.log.error("Failed to cycle timeframe")

    def action_toggle_help(self) -> None:
        """Toggle ATR help mode."""
        try:
            atr_panel = self.query_one(f"#{self.broker}-atr", ATRPanel)
            atr_panel.toggle_help()
        except Exception:
            self.log.error("Failed to toggle help")

    def action_reset_atr(self) -> None:
        """Reset ATR period and timeframe to defaults."""
        try:
            atr_panel = self.query_one(f"#{self.broker}-atr", ATRPanel)
            atr_panel.reset()
        except Exception:
            self.log.error("Failed to reset ATR")

    def _on_selection_change(self) -> None:
        """Handle selection change in the table."""
        try:
            table = self.query_one(f"#{self.broker}-positions", PositionsTable)
            pos = table.get_selected_position()
            symbol = table.get_selected_underlying()
            if symbol:
                atr_panel = self.query_one(f"#{self.broker}-atr", ATRPanel)
                atr_panel.position = pos
                atr_panel.selected_symbol = symbol
        except Exception:
            self.log.error("Failed to update selection")
