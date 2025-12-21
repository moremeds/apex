"""
Position panels rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from datetime import date, datetime
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...models.position_risk import PositionRisk
from ...models.risk_snapshot import RiskSnapshot
from ..formatters import format_price, format_quantity, format_number


def render_consolidated_positions(
    position_risks: List[PositionRisk],
    snapshot: Optional[RiskSnapshot] = None,
) -> Panel:
    """
    Render consolidated positions table grouped by underlying only (Tab 1).

    Shows summary row per underlying with aggregated metrics.
    Uses pre-calculated snapshot values when available to avoid re-summing.

    Args:
        position_risks: List of position risk objects.
        snapshot: Optional risk snapshot with pre-calculated totals.

    Returns:
        Panel containing the consolidated positions table.
    """
    if not position_risks:
        return Panel(
            Text("No positions", style="dim"),
            title="Portfolio Positions (Consolidated)",
            border_style="blue",
        )

    # Group position_risks by underlying
    by_underlying: Dict[str, List[PositionRisk]] = {}
    for pos_risk in position_risks:
        if pos_risk.underlying not in by_underlying:
            by_underlying[pos_risk.underlying] = []
        by_underlying[pos_risk.underlying].append(pos_risk)

    # Create table
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Underlying", style="bold", no_wrap=True)
    table.add_column("Positions", justify="right", no_wrap=True)
    table.add_column("Spot", justify="right", no_wrap=True)
    table.add_column("Beta", justify="right", no_wrap=True)
    table.add_column("Mkt Value", justify="right", no_wrap=True)
    table.add_column("P&L", justify="right", no_wrap=True)
    table.add_column("UP&L", justify="right", no_wrap=True)
    table.add_column("Delta $", justify="right", no_wrap=True)
    table.add_column("D(Δ)", justify="right", no_wrap=True)
    table.add_column("G(γ)", justify="right", no_wrap=True)
    table.add_column("V(ν)", justify="right", no_wrap=True)
    table.add_column("Th(Θ)", justify="right", no_wrap=True)

    # Portfolio totals - use pre-calculated snapshot values when available
    if snapshot:
        total_daily_pnl = snapshot.total_daily_pnl
        total_unrealized = snapshot.total_unrealized_pnl
        portfolio_delta = snapshot.portfolio_delta
        portfolio_gamma = snapshot.portfolio_gamma
        portfolio_vega = snapshot.portfolio_vega
        portfolio_theta = snapshot.portfolio_theta
        total_market_value = sum(pr.market_value for pr in position_risks)
        total_delta_dollars = sum(pr.delta_dollars for pr in position_risks)
    else:
        total_market_value = sum(pr.market_value for pr in position_risks)
        total_daily_pnl = sum(pr.daily_pnl for pr in position_risks)
        total_unrealized = sum(pr.unrealized_pnl for pr in position_risks)
        total_delta_dollars = sum(pr.delta_dollars for pr in position_risks)
        portfolio_delta = sum(pr.delta for pr in position_risks)
        portfolio_gamma = sum(pr.gamma for pr in position_risks)
        portfolio_vega = sum(pr.vega for pr in position_risks)
        portfolio_theta = sum(pr.theta for pr in position_risks)

    # Add portfolio total row
    table.add_row(
        ">> PORTFOLIO",
        str(len(position_risks)),
        "",
        "",
        format_number(total_market_value, color=False),
        format_number(total_daily_pnl, color=True),
        format_number(total_unrealized, color=True),
        format_number(total_delta_dollars, color=False),
        format_number(portfolio_delta, color=False),
        format_number(portfolio_gamma, color=False),
        format_number(portfolio_vega, color=False),
        format_number(portfolio_theta, color=False),
        style="bold white on rgb(80,80,80)",
    )

    # Sort underlyings by absolute market value (descending)
    underlying_values = {}
    for underlying, prs in by_underlying.items():
        underlying_values[underlying] = sum(abs(pr.market_value) for pr in prs)
    sorted_underlyings = sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)

    for underlying in sorted_underlyings:
        prs = by_underlying[underlying]

        # Calculate aggregates for this underlying
        underlying_market_value = sum(pr.market_value for pr in prs)
        underlying_daily_pnl = sum(pr.daily_pnl for pr in prs)
        underlying_unrealized = sum(pr.unrealized_pnl for pr in prs)
        underlying_delta_dollars = sum(pr.delta_dollars for pr in prs)
        underlying_delta = sum(pr.delta for pr in prs)
        underlying_gamma = sum(pr.gamma for pr in prs)
        underlying_vega = sum(pr.vega for pr in prs)
        underlying_theta = sum(pr.theta for pr in prs)

        # Get spot price (from stock position if available)
        spot_price = ""
        is_using_close = False
        for pr in prs:
            if not pr.expiry and pr.mark_price:  # Stock position
                spot_price = pr.mark_price
                is_using_close = pr.is_using_close
                break

        # Get beta
        beta_str = ""
        if prs and prs[0].beta is not None:
            beta_str = f"{prs[0].beta:.2f}"

        table.add_row(
            f"  {underlying}",
            str(len(prs)),
            format_price(spot_price, is_using_close, decimals=2) if spot_price else "",
            beta_str,
            format_number(underlying_market_value, color=False),
            format_number(underlying_daily_pnl, color=True),
            format_number(underlying_unrealized, color=True),
            format_number(underlying_delta_dollars, color=False),
            format_number(underlying_delta, color=False),
            format_number(underlying_gamma, color=False),
            format_number(underlying_vega, color=False),
            format_number(underlying_theta, color=False),
            style="white",
        )

    return Panel(table, title="Portfolio Positions (Consolidated)", border_style="blue")


def render_broker_positions(
    position_risks: List[PositionRisk],
    broker: str,
    selected_index: Optional[int] = None,
) -> Panel:
    """
    Render detailed positions for a specific broker (Tab 3 & 4).

    Shows full position details filtered by source (IB or FUTU).
    Supports selection highlighting for ATR level display.

    Args:
        position_risks: List of all position risk objects.
        broker: Broker name ("IB" or "FUTU").
        selected_index: Index of selected underlying (0-based), or None.

    Returns:
        Panel containing the broker positions table.
    """
    from ...models.position import PositionSource
    broker_source = PositionSource.IB if broker == "IB" else PositionSource.FUTU

    filtered_risks = []
    for pr in position_risks:
        pos = pr.position
        if pos.all_sources and broker_source in pos.all_sources:
            filtered_risks.append(pr)
        elif pos.source == broker_source:
            filtered_risks.append(pr)

    if not filtered_risks:
        return Panel(
            Text(f"No {broker} positions", style="dim"),
            title=f"{broker} Positions",
            border_style="blue",
        )

    # Group by underlying
    by_underlying: Dict[str, List[PositionRisk]] = {}
    for pos_risk in filtered_risks:
        if pos_risk.underlying not in by_underlying:
            by_underlying[pos_risk.underlying] = []
        by_underlying[pos_risk.underlying].append(pos_risk)

    # Create table with full details
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Underlying", style="bold", no_wrap=True)
    table.add_column("Positions", justify="right", no_wrap=True)
    table.add_column("Spot", justify="right", no_wrap=True)
    table.add_column("IV", justify="right", no_wrap=True)
    table.add_column("Beta", justify="right", no_wrap=True)
    table.add_column("Mkt Value", justify="right", no_wrap=True)
    table.add_column("P&L", justify="right", no_wrap=True)
    table.add_column("UP&L", justify="right", no_wrap=True)
    table.add_column("Delta $", justify="right", no_wrap=True)
    table.add_column("D(Δ)", justify="right", no_wrap=True)
    table.add_column("G(γ)", justify="right", no_wrap=True)
    table.add_column("V(ν)", justify="right", no_wrap=True)
    table.add_column("Th(Θ)", justify="right", no_wrap=True)

    # Broker totals
    total_market_value = sum(pr.market_value for pr in filtered_risks)
    total_daily_pnl = sum(pr.daily_pnl for pr in filtered_risks)
    total_unrealized = sum(pr.unrealized_pnl for pr in filtered_risks)
    total_delta_dollars = sum(pr.delta_dollars for pr in filtered_risks)
    total_delta = sum(pr.delta for pr in filtered_risks)
    total_gamma = sum(pr.gamma for pr in filtered_risks)
    total_vega = sum(pr.vega for pr in filtered_risks)
    total_theta = sum(pr.theta for pr in filtered_risks)

    # Add broker total row
    table.add_row(
        f">> {broker} Total",
        str(len(filtered_risks)),
        "",
        "",
        "",
        format_number(total_market_value, color=False),
        format_number(total_daily_pnl, color=True),
        format_number(total_unrealized, color=True),
        format_number(total_delta_dollars, color=False),
        format_number(total_delta, color=False),
        format_number(total_gamma, color=False),
        format_number(total_vega, color=False),
        format_number(total_theta, color=False),
        style="bold white on rgb(80,80,80)",
    )

    # Sort underlyings by absolute market value
    underlying_values = {}
    for underlying, prs in by_underlying.items():
        underlying_values[underlying] = sum(abs(pr.market_value) for pr in prs)
    sorted_underlyings = sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)

    # Track current underlying index for selection highlighting
    underlying_index = 0

    for underlying in sorted_underlyings:
        prs = by_underlying[underlying]

        # Add underlying header
        underlying_market_value = sum(pr.market_value for pr in prs)
        underlying_daily_pnl = sum(pr.daily_pnl for pr in prs)
        underlying_unrealized = sum(pr.unrealized_pnl for pr in prs)
        underlying_delta_dollars = sum(pr.delta_dollars for pr in prs)
        underlying_delta = sum(pr.delta for pr in prs)
        underlying_gamma = sum(pr.gamma for pr in prs)
        underlying_vega = sum(pr.vega for pr in prs)
        underlying_theta = sum(pr.theta for pr in prs)

        # Get beta
        beta_str = f"{prs[0].beta:.2f}" if prs and prs[0].beta is not None else ""

        # Selection highlighting: use different style for selected row
        is_selected = selected_index is not None and underlying_index == selected_index
        if is_selected:
            # Selected row: bright background with indicator
            underlying_style = "bold white on rgb(50,100,150)"
            prefix = "> "
        else:
            underlying_style = "bold white"
            prefix = "  "

        table.add_row(
            f"{prefix}{underlying}",
            "",
            "",
            "",
            beta_str,
            format_number(underlying_market_value, color=False),
            format_number(underlying_daily_pnl, color=True),
            format_number(underlying_unrealized, color=True),
            format_number(underlying_delta_dollars, color=False),
            format_number(underlying_delta, color=False),
            format_number(underlying_gamma, color=False),
            format_number(underlying_vega, color=False),
            format_number(underlying_theta, color=False),
            style=underlying_style,
        )

        underlying_index += 1

        # Sort positions: stocks first, then by expiry
        stocks = [pr for pr in prs if not pr.expiry]
        options = sorted([pr for pr in prs if pr.expiry], key=lambda p: p.expiry or "")

        for pr in stocks + options:
            iv_str = f"{pr.iv * 100:.1f}%" if pr.iv is not None else ""
            beta_str = f"{pr.beta:.2f}" if pr.beta is not None else ""

            table.add_row(
                f"  {pr.get_display_name()}",
                format_quantity(pr.quantity),
                format_price(pr.mark_price, pr.is_using_close, decimals=3 if pr.expiry else 2),
                iv_str,
                beta_str,
                format_number(pr.market_value, color=False),
                format_number(pr.daily_pnl, color=True),
                format_number(pr.unrealized_pnl, color=True),
                format_number(pr.delta_dollars, color=False),
                format_number(pr.delta, color=False),
                format_number(pr.gamma, color=False),
                format_number(pr.vega, color=False),
                format_number(pr.theta, color=False),
                style="white",
            )

    return Panel(table, title=f"{broker} Positions ({len(filtered_risks)})", border_style="blue")


def get_broker_underlyings(
    position_risks: List[PositionRisk],
    broker: str,
) -> List[str]:
    """
    Get sorted list of underlyings for a broker.

    Returns underlyings in the same order as they appear in render_broker_positions,
    which is sorted by absolute market value (descending).

    Args:
        position_risks: List of all position risk objects.
        broker: Broker name ("IB" or "FUTU").

    Returns:
        List of underlying symbols in display order.
    """
    from ...models.position import PositionSource
    broker_source = PositionSource.IB if broker == "IB" else PositionSource.FUTU

    # Filter positions for this broker
    filtered_risks = []
    for pr in position_risks:
        pos = pr.position
        if pos.all_sources and broker_source in pos.all_sources:
            filtered_risks.append(pr)
        elif pos.source == broker_source:
            filtered_risks.append(pr)

    if not filtered_risks:
        return []

    # Group by underlying
    by_underlying: Dict[str, List[PositionRisk]] = {}
    for pos_risk in filtered_risks:
        if pos_risk.underlying not in by_underlying:
            by_underlying[pos_risk.underlying] = []
        by_underlying[pos_risk.underlying].append(pos_risk)

    # Sort by absolute market value (same as render function)
    underlying_values = {}
    for underlying, prs in by_underlying.items():
        underlying_values[underlying] = sum(abs(pr.market_value) for pr in prs)

    return sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)


def render_positions_profile(position_risks: List[PositionRisk]) -> Panel:
    """
    Render hierarchical positions table grouped by underlying -> expiry -> position.

    Uses pre-calculated PositionRisk objects from RiskEngine (single source of truth).
    This method does NOT perform any calculations - only displays data.

    Args:
        position_risks: List of position risk objects.

    Returns:
        Panel containing the hierarchical positions table.
    """
    if not position_risks:
        return Panel(
            Text("No positions", style="dim"),
            title="Portfolio Positions",
            border_style="blue",
        )

    # Group position_risks by underlying
    by_underlying = {}
    for pos_risk in position_risks:
        if pos_risk.underlying not in by_underlying:
            by_underlying[pos_risk.underlying] = []
        by_underlying[pos_risk.underlying].append(pos_risk)

    # Create table
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Ticker", style="bold", no_wrap=True)
    table.add_column("Pos", justify="right", no_wrap=True)
    table.add_column("Spot", justify="right", no_wrap=True)
    table.add_column("IV", justify="right", no_wrap=True)
    table.add_column("Beta", justify="right", no_wrap=True)
    table.add_column("Mkt Value", justify="right", no_wrap=True)
    table.add_column("P&L", justify="right", no_wrap=True)
    table.add_column("UP&L", justify="right", no_wrap=True)
    table.add_column("Delta $", justify="right", no_wrap=True)
    table.add_column("VAR", justify="right", no_wrap=True)
    table.add_column("D(delta)", justify="right", no_wrap=True)
    table.add_column("G(gamma)", justify="right", no_wrap=True)
    table.add_column("V(nu)", justify="right", no_wrap=True)
    table.add_column("Th(theta)", justify="right", no_wrap=True)

    # Add portfolio total row
    total_market_value = sum(pr.market_value for pr in position_risks)
    total_daily_pnl = sum(pr.daily_pnl for pr in position_risks)
    total_unrealized = sum(pr.unrealized_pnl for pr in position_risks)
    total_delta_dollars = sum(pr.delta_dollars for pr in position_risks)
    portfolio_delta = sum(pr.delta for pr in position_risks)
    portfolio_gamma = sum(pr.gamma for pr in position_risks)
    portfolio_vega = sum(pr.vega for pr in position_risks)
    portfolio_theta = sum(pr.theta for pr in position_risks)

    table.add_row(
        ">> All Tickers",
        "",
        "",
        "",
        "",
        format_number(total_market_value, color=False),
        format_number(total_daily_pnl, color=True),
        format_number(total_unrealized, color=True),
        format_number(total_delta_dollars, color=False),
        "",
        format_number(portfolio_delta, color=False),
        format_number(portfolio_gamma, color=False),
        format_number(portfolio_vega, color=False),
        format_number(portfolio_theta, color=False),
        style="bold white on rgb(80,80,80)",
    )

    # Sort underlyings by absolute market value (descending)
    underlying_values = {}
    for underlying, underlying_pos_risks in by_underlying.items():
        total_value = sum(abs(pr.market_value) for pr in underlying_pos_risks)
        underlying_values[underlying] = total_value

    sorted_underlyings = sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)

    for underlying in sorted_underlyings:
        underlying_pos_risks = by_underlying[underlying]

        # Group by expiry within underlying
        by_expiry = {}
        stock_pos_risks = []
        for pr in underlying_pos_risks:
            if pr.expiry:
                if pr.expiry not in by_expiry:
                    by_expiry[pr.expiry] = []
                by_expiry[pr.expiry].append(pr)
            else:
                stock_pos_risks.append(pr)

        # Calculate underlying-level totals
        underlying_market_value = sum(pr.market_value for pr in underlying_pos_risks)
        underlying_daily_pnl = sum(pr.daily_pnl for pr in underlying_pos_risks)
        underlying_unrealized = sum(pr.unrealized_pnl for pr in underlying_pos_risks)
        underlying_delta_dollars = sum(pr.delta_dollars for pr in underlying_pos_risks)
        underlying_delta = sum(pr.delta for pr in underlying_pos_risks)
        underlying_gamma = sum(pr.gamma for pr in underlying_pos_risks)
        underlying_vega = sum(pr.vega for pr in underlying_pos_risks)
        underlying_theta = sum(pr.theta for pr in underlying_pos_risks)

        # Get mark price for underlying
        underlying_mark = ""
        if stock_pos_risks:
            pr = stock_pos_risks[0]
            if pr.mark_price:
                underlying_mark = format_price(pr.mark_price, pr.is_using_close, decimals=2)

        # Get beta for underlying
        underlying_beta = ""
        if underlying_pos_risks:
            first_beta = underlying_pos_risks[0].beta
            if first_beta is not None:
                underlying_beta = f"{first_beta:.2f}"

        # Add underlying header row
        table.add_row(
            f">> {underlying} ",
            "",
            underlying_mark,
            "",
            underlying_beta,
            format_number(underlying_market_value, color=False),
            format_number(underlying_daily_pnl, color=True),
            format_number(underlying_unrealized, color=True),
            format_number(underlying_delta_dollars, color=False),
            "",
            format_number(underlying_delta, color=False),
            format_number(underlying_gamma, color=False),
            format_number(underlying_vega, color=False),
            format_number(underlying_theta, color=False),
            style=f"bold white ",
        )

        # Add stock positions
        for pr in stock_pos_risks:
            stock_beta = f"{pr.beta:.2f}" if pr.beta is not None else ""
            table.add_row(
                f" {pr.get_display_name()} ",
                format_quantity(pr.quantity),
                format_price(pr.mark_price, pr.is_using_close, decimals=2),
                "",
                stock_beta,
                format_number(pr.market_value, color=False),
                format_number(pr.daily_pnl, color=True),
                format_number(pr.unrealized_pnl, color=True),
                format_number(pr.delta_dollars, color=False),
                "",
                format_number(pr.delta, color=False),
                "",
                "",
                "",
                style=f"white ",
            )

        # Add expiry groups
        def normalize_expiry(exp):
            if isinstance(exp, date):
                return exp
            elif isinstance(exp, str):
                return datetime.strptime(exp, "%Y%m%d").date()
            return exp

        for expiry in sorted(by_expiry.keys(), key=normalize_expiry):
            expiry_pos_risks = by_expiry[expiry]

            # Calculate expiry-level totals
            expiry_market_value = sum(pr.market_value for pr in expiry_pos_risks)
            expiry_daily_pnl = sum(pr.daily_pnl for pr in expiry_pos_risks)
            expiry_unrealized = sum(pr.unrealized_pnl for pr in expiry_pos_risks)
            expiry_delta_dollars = sum(pr.delta_dollars for pr in expiry_pos_risks)
            expiry_delta = sum(pr.delta for pr in expiry_pos_risks)
            expiry_gamma = sum(pr.gamma for pr in expiry_pos_risks)
            expiry_vega = sum(pr.vega for pr in expiry_pos_risks)
            expiry_theta = sum(pr.theta for pr in expiry_pos_risks)

            # Add expiry header row
            table.add_row(
                f"  >> {expiry}",
                "",
                "",
                "",
                "",
                format_number(expiry_market_value, color=False),
                format_number(expiry_daily_pnl, color=True),
                format_number(expiry_unrealized, color=True),
                format_number(expiry_delta_dollars, color=False),
                "",
                format_number(expiry_delta, color=False),
                format_number(expiry_gamma, color=False),
                format_number(expiry_vega, color=False),
                format_number(expiry_theta, color=False),
                style=f"bold white ",
            )

            # Add individual option positions
            for pr in expiry_pos_risks:
                option_desc = pr.get_display_name()
                iv_display = f"{pr.iv * 100:.1f}%" if pr.iv is not None else ""
                option_beta = f"{pr.beta:.2f}" if pr.beta is not None else ""

                table.add_row(
                    f"    {option_desc}",
                    format_quantity(pr.quantity),
                    format_price(pr.mark_price, pr.is_using_close, decimals=3),
                    iv_display,
                    option_beta,
                    format_number(pr.market_value, color=False),
                    format_number(pr.daily_pnl, color=True),
                    format_number(pr.unrealized_pnl, color=True),
                    format_number(pr.delta_dollars, color=False),
                    "",
                    format_number(pr.delta, color=False),
                    format_number(pr.gamma, color=False),
                    format_number(pr.vega, color=False),
                    format_number(pr.theta, color=False),
                    style=f"white ",
                )

    return Panel(table, title="Portfolio Positions", border_style="blue")
