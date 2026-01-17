"""
SummaryViewModel - Framework-agnostic portfolio summary transformation.

Extracts field formatting logic from SummaryPanel.
Tracks which fields changed for targeted updates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class SummaryViewModel:
    """
    ViewModel for portfolio summary panel.

    Responsibilities:
    - Extract and format summary metrics from RiskSnapshot
    - Track which fields changed for targeted updates
    """

    FIELD_IDS = [
        "ib-netliq",
        "futu-netliq",
        "total-netliq",
        "unrealized-pnl",
        "daily-pnl",
        "gross-notional",
        "net-notional",
        "portfolio-delta",
        "portfolio-gamma",
        "portfolio-vega",
        "portfolio-theta",
        "max-concentration",
        "max-symbol",
        "margin-util",
    ]

    def __init__(self) -> None:
        self._field_cache: Dict[str, str] = {}

    def compute_display_data(self, snapshot: Optional[Any]) -> Dict[str, str]:
        """Transform snapshot into field values."""
        if snapshot is None:
            return {}

        ib_netliq = getattr(snapshot, "ib_net_liquidation", 0) or 0
        futu_netliq = getattr(snapshot, "futu_net_liquidation", 0) or 0
        total_netliq = getattr(snapshot, "total_net_liquidation", 0) or ib_netliq + futu_netliq

        unrealized = getattr(snapshot, "total_unrealized_pnl", 0) or 0
        daily = getattr(snapshot, "total_daily_pnl", 0) or 0

        gross = getattr(snapshot, "total_gross_notional", 0) or 0
        net = getattr(snapshot, "total_net_notional", 0) or 0

        delta = getattr(snapshot, "portfolio_delta", 0) or 0
        gamma = getattr(snapshot, "portfolio_gamma", 0) or 0
        vega = getattr(snapshot, "portfolio_vega", 0) or 0
        theta = getattr(snapshot, "portfolio_theta", 0) or 0

        concentration = getattr(snapshot, "concentration_pct", 0) or 0
        max_symbol = getattr(snapshot, "max_underlying_symbol", "-") or "-"
        margin_util = getattr(snapshot, "margin_utilization", 0) or 0

        upnl_color = "green" if unrealized >= 0 else "red"
        dpnl_color = "green" if daily >= 0 else "red"

        return {
            "ib-netliq": f"IB NetLiq:       ${ib_netliq:,.0f}",
            "futu-netliq": f"Futu NetLiq:     ${futu_netliq:,.0f}",
            "total-netliq": f"[bold]Total NetLiq:    ${total_netliq:,.0f}[/]",
            "unrealized-pnl": f"Unrealized P&L:  [{upnl_color}]${unrealized:+,.2f}[/]",
            "daily-pnl": f"Daily P&L:       [{dpnl_color}]${daily:+,.2f}[/]",
            "gross-notional": f"Gross Notional: ${gross:,.0f}",
            "net-notional": f"Net Notional:   ${net:,.0f}",
            "portfolio-delta": f"Portfolio Delta: {delta:,.0f}",
            "portfolio-gamma": f"Portfolio Gamma: {gamma:,.2f}",
            "portfolio-vega": f"Portfolio Vega:  {vega:,.0f}",
            "portfolio-theta": f"Portfolio Theta: {theta:,.0f}",
            "max-concentration": f"Max Conc:        {concentration:.1%}",
            "max-symbol": f"Max Symbol:      {max_symbol}",
            "margin-util": f"Margin Util:     {margin_util:.1%}",
        }

    def compute_field_updates(self, snapshot: Optional[Any]) -> Dict[str, str]:
        """Return only fields that changed."""
        new_data = self.compute_display_data(snapshot)
        updates = {}

        for field_id, value in new_data.items():
            if self._field_cache.get(field_id) != value:
                updates[field_id] = value

        self._field_cache = new_data
        return updates

    def invalidate(self) -> None:
        """Clear cache, forcing full refresh on next update."""
        self._field_cache.clear()
