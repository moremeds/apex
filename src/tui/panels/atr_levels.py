"""
ATR Levels Panel for Terminal Dashboard.

Displays ATR-based stop loss and take profit levels for a selected position.
Horizontal layout with position suggestions and reward/risk calculations.
"""

from __future__ import annotations

from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich import box

from ...domain.indicators.atr import ATRData, ATROptimizationResult
from ...models.position_risk import PositionRisk


def render_atr_levels(
    atr_data: Optional[ATRData],
    position: Optional[PositionRisk] = None,
    optimization: Optional[ATROptimizationResult] = None,
    loading: bool = False,
    current_period: int = 14,
) -> Panel:
    """
    Render ATR levels panel with horizontal layout.

    Layout similar to atr_example.png:
    - Header row with symbol, price, ATR
    - Horizontal price levels bar (Stop Loss | Entry | Take Profit)
    - Position info and projected scenarios
    - Reward/Risk calculation

    Args:
        atr_data: Calculated ATR data with levels.
        position: Selected position for context.
        optimization: Historical optimization result.
        loading: True if ATR data is being fetched.
        current_period: Current ATR period setting.

    Returns:
        Panel with ATR levels display.
    """
    if loading:
        return Panel(
            Text("Loading ATR data...", style="dim italic"),
            title="ATR Analysis",
            border_style="yellow",
        )

    if atr_data is None:
        hint = "Select a position with w/s to view ATR levels"
        if position:
            hint = "ATR data unavailable for this position"
        return Panel(
            Text(hint, style="dim"),
            title="ATR Analysis",
            border_style="dim",
        )

    # Get position details
    qty = position.quantity if position else 0
    avg_price = None
    if position and hasattr(position, 'position'):
        avg_price = getattr(position.position, 'avg_price', None)
    cost_basis = avg_price if avg_price else atr_data.current_price
    total_cost = abs(qty * cost_basis)

    # === Row 1: Header with Symbol, Price, ATR ===
    header_table = Table(box=None, padding=0, expand=True, show_header=False)
    header_table.add_column("ticker", width=8)
    header_table.add_column("last", width=10, justify="right")
    header_table.add_column("atr", width=8, justify="right")
    header_table.add_column("atr_pct", width=7, justify="right")
    header_table.add_column("qty", width=8, justify="right")
    header_table.add_column("cost", width=12, justify="right")

    header_table.add_row(
        Text(atr_data.symbol, style="bold cyan"),
        Text(f"${atr_data.current_price:,.2f}", style="bold white"),
        Text(f"${atr_data.atr_value:.2f}", style="bold yellow"),
        Text(f"{atr_data.atr_percent:.1f}%", style="yellow"),
        Text(f"{qty:,.0f} sh", style="green" if qty > 0 else "red") if qty else Text("-", style="dim"),
        Text(f"${total_cost:,.0f}", style="white") if qty else Text("-", style="dim"),
    )

    # === Row 2: Horizontal Price Levels Bar ===
    levels_table = Table(box=box.SIMPLE_HEAD, padding=0, expand=True)
    levels_table.add_column("", width=6, style="dim")  # Label column
    levels_table.add_column("SL -2x", justify="center", style="red", width=9)
    levels_table.add_column("SL -1.5x", justify="center", style="red", width=9)
    levels_table.add_column("SL -1x", justify="center", style="red", width=9)
    levels_table.add_column("ENTRY", justify="center", style="bold white", width=9)
    levels_table.add_column("TP +7x", justify="center", style="green", width=9)
    levels_table.add_column("TP +8x", justify="center", style="green", width=9)
    levels_table.add_column("TP +9x", justify="center", style="green", width=9)
    levels_table.add_column("TP +10x", justify="center", style="green", width=9)

    # Price row
    levels_table.add_row(
        "Price",
        f"${atr_data.stop_loss_2x:.0f}",
        f"${atr_data.stop_loss_1_5x:.0f}",
        f"${atr_data.stop_loss_1x:.0f}",
        f"${atr_data.current_price:.0f}",
        f"${atr_data.take_profit_7x:.0f}",
        f"${atr_data.take_profit_8x:.0f}",
        f"${atr_data.take_profit_9x:.0f}",
        f"${atr_data.take_profit_10x:.0f}",
    )

    # Percent row
    levels_table.add_row(
        "%",
        f"{atr_data.get_percent_from_entry(atr_data.stop_loss_2x):+.1f}%",
        f"{atr_data.get_percent_from_entry(atr_data.stop_loss_1_5x):+.1f}%",
        f"{atr_data.get_percent_from_entry(atr_data.stop_loss_1x):+.1f}%",
        "0%",
        f"+{atr_data.get_percent_from_entry(atr_data.take_profit_7x):.1f}%",
        f"+{atr_data.get_percent_from_entry(atr_data.take_profit_8x):.1f}%",
        f"+{atr_data.get_percent_from_entry(atr_data.take_profit_9x):.1f}%",
        f"+{atr_data.get_percent_from_entry(atr_data.take_profit_10x):.1f}%",
    )

    # === Row 3: Position Suggestions (if we have quantity) ===
    suggestions_table = Table(box=None, padding=(0, 1), expand=True, show_header=True)
    suggestions_table.add_column("Scenario", style="bold", width=22)
    suggestions_table.add_column("Sell @", justify="right", width=10)
    suggestions_table.add_column("Proceeds", justify="right", width=11)
    suggestions_table.add_column("P&L", justify="right", width=10)
    suggestions_table.add_column("R:R", justify="right", width=6)

    if qty > 0:
        # Calculate scenarios for long position
        scenarios = [
            ("Sell 100% @ +9 ATR", atr_data.take_profit_9x, 1.0, 1.5),
            ("Sell 50% @ +8 ATR", atr_data.take_profit_8x, 0.5, 1.5),
            ("Stop Loss @ -1.5 ATR", atr_data.stop_loss_1_5x, 1.0, 1.5),
        ]

        for name, price, sell_pct, sl_mult in scenarios:
            sell_qty = qty * sell_pct
            proceeds = sell_qty * price
            pnl = (price - cost_basis) * sell_qty
            # R:R = potential profit / risk (using 1.5x ATR as risk)
            risk_per_share = atr_data.atr_value * sl_mult
            reward_per_share = price - cost_basis
            rr = reward_per_share / risk_per_share if risk_per_share > 0 else 0

            pnl_style = "green" if pnl >= 0 else "red"
            suggestions_table.add_row(
                name,
                f"${price:,.0f}",
                f"${proceeds:,.0f}",
                Text(f"${pnl:+,.0f}", style=pnl_style),
                f"{rr:.1f}" if rr > 0 else "-",
            )

        # Add max return scenario
        max_price = atr_data.take_profit_10x
        max_proceeds = qty * max_price
        max_pnl = (max_price - cost_basis) * qty
        max_return_pct = (max_pnl / total_cost * 100) if total_cost > 0 else 0
        suggestions_table.add_row(
            Text("Max Return @ +10 ATR", style="bold cyan"),
            f"${max_price:,.0f}",
            f"${max_proceeds:,.0f}",
            Text(f"${max_pnl:+,.0f}", style="green"),
            Text(f"{max_return_pct:.0f}%", style="bold green"),
        )
    else:
        suggestions_table.add_row(
            Text("No position - showing reference levels", style="dim"),
            "-", "-", "-", "-"
        )

    # === Row 4: Reward/Risk Summary ===
    summary_table = Table(box=None, padding=(0, 2), expand=True, show_header=False)
    summary_table.add_column(width=30)
    summary_table.add_column(width=30)
    summary_table.add_column(width=20)

    # Calculate key metrics
    risk_1_5x = atr_data.atr_value * 1.5
    reward_9x = atr_data.take_profit_9x - atr_data.current_price
    rr_ratio = reward_9x / risk_1_5x if risk_1_5x > 0 else 0

    summary_table.add_row(
        Text.assemble(
            ("Risk (1.5x ATR): ", "dim"),
            (f"${risk_1_5x:.2f}", "red"),
            (f" ({risk_1_5x/atr_data.current_price*100:.1f}%)", "dim red"),
        ),
        Text.assemble(
            ("Reward (9x ATR): ", "dim"),
            (f"${reward_9x:.2f}", "green"),
            (f" ({reward_9x/atr_data.current_price*100:.1f}%)", "dim green"),
        ),
        Text.assemble(
            ("R:R = ", "dim"),
            (f"{rr_ratio:.2f}", "bold cyan"),
        ),
    )

    # === Row 5: Keyboard hints ===
    hints = Text()
    hints.append("[w/s] ", style="cyan")
    hints.append("Select  ", style="dim")
    hints.append("[+/-] ", style="cyan")
    hints.append(f"Period({current_period})  ", style="dim")
    hints.append("[r] ", style="cyan")
    hints.append("Reset", style="dim")

    # Combine all sections
    content = Group(
        header_table,
        Text(""),  # Spacer
        levels_table,
        Text(""),  # Spacer
        suggestions_table,
        Text(""),  # Spacer
        summary_table,
        hints,
    )

    return Panel(
        content,
        title=f"ATR Analysis: {atr_data.symbol} | ATR({current_period})=${atr_data.atr_value:.2f} ({atr_data.atr_percent:.1f}%)",
        border_style="blue",
    )


def render_atr_loading() -> Panel:
    """Render loading state for ATR panel."""
    return Panel(
        Text("Loading ATR data...", style="dim italic"),
        title="ATR Analysis",
        border_style="yellow",
    )


def render_atr_empty(message: str = "Select a position to view ATR levels") -> Panel:
    """Render empty state for ATR panel."""
    return Panel(
        Text(message, style="dim"),
        title="ATR Analysis",
        border_style="dim",
    )


def render_atr_compact(
    atr_data: Optional[ATRData],
    optimization: Optional[ATROptimizationResult] = None,
) -> Panel:
    """
    Render compact ATR summary (for smaller panel spaces).

    Args:
        atr_data: Calculated ATR data.
        optimization: Optional optimization result.

    Returns:
        Compact panel with key ATR info.
    """
    if atr_data is None:
        return Panel(
            Text("No ATR data", style="dim"),
            title="ATR",
            border_style="dim",
        )

    lines = [
        Text.assemble(
            ("ATR(", "dim"),
            (str(atr_data.period), "yellow"),
            ("): ", "dim"),
            (f"${atr_data.atr_value:.2f}", "bold yellow"),
            (f" ({atr_data.atr_percent:.1f}%)", "dim"),
        ),
        Text.assemble(
            ("SL: ", "dim"),
            (f"${atr_data.stop_loss_1_5x:.2f}", "red"),
            ("  TP: ", "dim"),
            (f"${atr_data.take_profit_9x:.2f}", "green"),
        ),
    ]

    if optimization and optimization.is_recommended:
        lines.append(
            Text.assemble(
                ("Rec: ", "dim"),
                (f"{optimization.historical_win_rate:.0f}% win", "cyan"),
            )
        )

    content = Table.grid()
    for line in lines:
        content.add_row(line)

    return Panel(
        content,
        title=f"ATR: {atr_data.symbol}",
        border_style="blue",
    )
