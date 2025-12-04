"""
Portfolio summary panel rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...models.risk_snapshot import RiskSnapshot
from ..formatters import format_pnl


def render_portfolio_summary(snapshot: RiskSnapshot) -> Panel:
    """
    Render portfolio summary panel.

    Args:
        snapshot: Risk snapshot containing portfolio metrics.

    Returns:
        Panel containing the portfolio summary.
    """
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    # Account Info - Per Broker
    table.add_row(Text("--- Account ---", style="bold"), "")
    table.add_row("IB NetLiq", f"${snapshot.ib_net_liquidation:,.0f}")
    table.add_row("Futu NetLiq", f"${snapshot.futu_net_liquidation:,.0f}")
    table.add_row("Total NetLiq", Text(f"${snapshot.total_net_liquidation:,.0f}", style="bold"))

    # P&L
    table.add_row(Text("--- P&L ---", style="bold"), "")
    table.add_row("Unrealized P&L", format_pnl(snapshot.total_unrealized_pnl))
    table.add_row("Daily P&L", format_pnl(snapshot.total_daily_pnl))

    # Notional
    table.add_row(Text("--- Exposure ---", style="bold"), "")
    table.add_row("Gross Notional", f"${snapshot.total_gross_notional:,.0f}")
    table.add_row("Net Notional", f"${snapshot.total_net_notional:,.0f}")

    # Greeks
    table.add_row(Text("--- Greeks ---", style="bold"), "")
    table.add_row("Portfolio Delta", f"{snapshot.portfolio_delta:,.0f}")
    table.add_row("Portfolio Gamma", f"{snapshot.portfolio_gamma:,.2f}")
    table.add_row("Portfolio Vega", f"{snapshot.portfolio_vega:,.0f}")
    table.add_row("Portfolio Theta", f"{snapshot.portfolio_theta:,.0f}")

    # Concentration
    table.add_row(Text("--- Risk ---", style="bold"), "")
    table.add_row("Max Concentration", f"{snapshot.concentration_pct:.1%}")
    table.add_row("Max Underlying", snapshot.max_underlying_symbol)
    table.add_row("Margin Utilization", f"{snapshot.margin_utilization:.1%}")

    return Panel(table, title="Portfolio Summary", border_style="green")
