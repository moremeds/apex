"""
Summary view for the Account Summary tab (Tab 1).

Layout matching original Rich dashboard:
- Left (60%): Consolidated Positions table with PORTFOLIO row
- Right top: Portfolio Summary panel
- Right bottom: Market Alerts panel
- Bottom: Component Health bar (full width)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static
from textual.app import ComposeResult

from ..widgets.positions_table import PositionsTable
from ..widgets.summary_panel import SummaryPanel
from ..widgets.alerts_list import AlertsList
from ..widgets.health_bar import HealthBar

if TYPE_CHECKING:
    from ...models.risk_snapshot import RiskSnapshot


class SummaryView(Container):
    """Account summary view matching original Rich dashboard layout."""

    # No DEFAULT_CSS - let widgets use auto-sizing

    def compose(self) -> ComposeResult:
        """Compose the summary view layout."""
        with Horizontal(id="summary-main"):
            # Left side - Positions table (60% via 3fr)
            with Vertical(id="summary-left"):
                yield PositionsTable(
                    id="summary-positions",
                    show_portfolio_row=True,
                    consolidated=True,
                )

            # Right side - Summary + Alerts (40% via 2fr)
            with Vertical(id="summary-right"):
                yield SummaryPanel(id="summary-panel")
                yield AlertsList(id="summary-alerts")

        # Bottom - Health bar (full width)
        yield HealthBar(id="summary-health")

    def update_data(
        self,
        snapshot: Optional["RiskSnapshot"],
        alerts: Optional[List[Dict[str, Any]]],
        health: Optional[List[Any]],
    ) -> None:
        """
        Update the view with new data.

        Args:
            snapshot: RiskSnapshot with position data.
            alerts: List of market alerts.
            health: List of component health statuses.
        """
        position_risks = getattr(snapshot, "position_risks", []) if snapshot else []

        # Update positions table
        try:
            positions_table = self.query_one("#summary-positions", PositionsTable)
            positions_table.positions = position_risks
        except Exception:
            pass

        # Update summary panel
        try:
            summary_panel = self.query_one("#summary-panel", SummaryPanel)
            summary_panel.snapshot = snapshot
        except Exception:
            pass

        # Update alerts list
        try:
            alerts_list = self.query_one("#summary-alerts", AlertsList)
            alerts_list.alerts = alerts or []
        except Exception:
            pass

        # Update health bar
        try:
            health_bar = self.query_one("#summary-health", HealthBar)
            health_bar.health = health or []
        except Exception:
            pass
