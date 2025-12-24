"""
Portfolio summary panel widget.

Displays key portfolio metrics matching the original Rich dashboard layout:
- Account section (IB/Futu/Total NetLiq)
- P&L section
- Exposure section
- Greeks section
- Risk section
"""

from __future__ import annotations

from typing import Any, Optional

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult


class SummaryPanel(Widget):
    """Portfolio summary metrics display matching original layout."""

    # Reactive state
    snapshot: reactive[Optional[Any]] = reactive(None, init=False)

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
        """Update display when snapshot changes."""
        if snapshot is None:
            return

        # Account section
        ib_netliq = getattr(snapshot, "ib_net_liquidation", 0) or 0
        futu_netliq = getattr(snapshot, "futu_net_liquidation", 0) or 0
        total_netliq = getattr(snapshot, "total_net_liquidation", 0) or ib_netliq + futu_netliq

        self._update("ib-netliq", f"IB NetLiq:       ${ib_netliq:,.0f}")
        self._update("futu-netliq", f"Futu NetLiq:     ${futu_netliq:,.0f}")
        self._update("total-netliq", f"[bold]Total NetLiq:    ${total_netliq:,.0f}[/]")

        # P&L section
        unrealized = getattr(snapshot, "total_unrealized_pnl", 0) or 0
        daily = getattr(snapshot, "total_daily_pnl", 0) or 0

        upnl_color = "green" if unrealized >= 0 else "red"
        dpnl_color = "green" if daily >= 0 else "red"

        self._update("unrealized-pnl", f"Unrealized P&L:  [{upnl_color}]${unrealized:+,.2f}[/]")
        self._update("daily-pnl", f"Daily P&L:       [{dpnl_color}]${daily:+,.2f}[/]")

        # Exposure section
        gross = getattr(snapshot, "total_gross_notional", 0) or 0
        net = getattr(snapshot, "total_net_notional", 0) or 0

        self._update("gross-notional", f"Gross Notional: ${gross:,.0f}")
        self._update("net-notional", f"Net Notional:   ${net:,.0f}")

        # Greeks section
        delta = getattr(snapshot, "portfolio_delta", 0) or 0
        gamma = getattr(snapshot, "portfolio_gamma", 0) or 0
        vega = getattr(snapshot, "portfolio_vega", 0) or 0
        theta = getattr(snapshot, "portfolio_theta", 0) or 0

        self._update("portfolio-delta", f"Portfolio Delta: {delta:,.0f}")
        self._update("portfolio-gamma", f"Portfolio Gamma: {gamma:,.2f}")
        self._update("portfolio-vega", f"Portfolio Vega:  {vega:,.0f}")
        self._update("portfolio-theta", f"Portfolio Theta: {theta:,.0f}")

        # Risk section
        concentration = getattr(snapshot, "concentration_pct", 0) or 0
        max_symbol = getattr(snapshot, "max_underlying_symbol", "-") or "-"
        margin_util = getattr(snapshot, "margin_utilization", 0) or 0

        self._update("max-concentration", f"Max Conc:        {concentration:.1%}")
        self._update("max-symbol", f"Max Symbol:      {max_symbol}")
        self._update("margin-util", f"Margin Util:     {margin_util:.1%}")

    def _update(self, widget_id: str, value: str) -> None:
        """Update a Static widget by ID."""
        try:
            self.query_one(f"#{widget_id}", Static).update(value)
        except Exception:
            pass
