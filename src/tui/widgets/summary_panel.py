"""
Portfolio summary panel widget.

Displays key portfolio metrics matching the original Rich dashboard layout:
- Account section (IB/Futu/Total NetLiq)
- P&L section
- Exposure section
- Greeks section
- Risk section

Uses SummaryViewModel for incremental field updates.
"""

from __future__ import annotations

from typing import Any, Optional

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult

from ..viewmodels.summary_vm import SummaryViewModel


class SummaryPanel(Widget):
    """Portfolio summary metrics display with incremental updates."""

    # Reactive state
    snapshot: reactive[Optional[Any]] = reactive(None, init=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._view_model = SummaryViewModel()

    def compose(self) -> ComposeResult:
        """Compose the summary panel layout."""
        with Vertical(id="summary-content"):
            # Title
            yield Static("[bold]Portfolio Summary[/]", id="summary-title", classes="section-title")

            # Account section
            yield Static("─── Account ───", classes="section-header")
            yield Static("IB NetLiq:       $0", id="ib-netliq")
            yield Static("Futu NetLiq:     $0", id="futu-netliq")
            yield Static("Total NetLiq:    $0", id="total-netliq")

            # P&L section
            yield Static("─── P&L ───", classes="section-header")
            yield Static("Unrealized P&L:  $0", id="unrealized-pnl")
            yield Static("Daily P&L:       $0", id="daily-pnl")

            # Exposure section
            yield Static("─── Exposure ───", classes="section-header")
            yield Static("Gross Notional: $0", id="gross-notional")
            yield Static("Net Notional:   $0", id="net-notional")

            # Greeks section
            yield Static("─── Greeks ───", classes="section-header")
            yield Static("Portfolio Delta: 0", id="portfolio-delta")
            yield Static("Portfolio Gamma: 0", id="portfolio-gamma")
            yield Static("Portfolio Vega:  0", id="portfolio-vega")
            yield Static("Portfolio Theta: 0", id="portfolio-theta")

            # Risk section
            yield Static("─── Risk ───", classes="section-header")
            yield Static("Max Conc:        0%", id="max-concentration")
            yield Static("Max Symbol:      -", id="max-symbol")
            yield Static("Margin Util:     0%", id="margin-util")

    def watch_snapshot(self, snapshot: Optional[Any]) -> None:
        """Update display when snapshot changes - only update changed fields."""
        if snapshot is None:
            return

        # Get only the changed fields from ViewModel
        updates = self._view_model.compute_field_updates(snapshot)

        # Update only changed fields
        for field_id, value in updates.items():
            self._update(field_id, value)

    def _update(self, widget_id: str, value: str) -> None:
        """Update a Static widget by ID."""
        try:
            self.query_one(f"#{widget_id}", Static).update(value)
        except Exception:
            pass
