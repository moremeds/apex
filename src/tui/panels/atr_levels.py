"""
ATR Levels Panel for Terminal Dashboard.

Displays ATR-based stop loss and take profit levels for a selected position.
Horizontal layout matching atr_example.png style.
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
    timeframe: str = "Daily",
) -> Panel:
    """
    Render ATR levels panel with horizontal bar layout.
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

    # Calculate values
    atr = atr_data.atr_value
    price = atr_data.current_price

    # Guard against invalid price data
    if price <= 0:
        return Panel(
            Text(f"Invalid price data for {atr_data.symbol} (price=0)", style="yellow"),
            title="ATR Analysis",
            border_style="yellow",
        )

    risk = atr * 1.5  # 1.5x ATR stop

    # SMA21 estimate (slightly below current price for uptrend)
    sma21 = price * 0.97  # Estimate ~3% below current

    # Calculate all levels
    sl_2x = price - (atr * 2)
    sl_1_5x = price - (atr * 1.5)

    # R-multiple levels (1R to 11R)
    r_levels = {i: price + (risk * i) for i in range(1, 12)}

    # Build content
    content = Table.grid(padding=0)
    content.add_column()

    # === Row 1: Header info ===
    header = Table(box=None, padding=(0, 1), show_header=False)
    header.add_column("sym", width=8)
    header.add_column("price", width=10, justify="right")
    header.add_column("atr", width=10, justify="right")
    header.add_column("pct", width=8, justify="right")
    header.add_column("tf", width=8)
    header.add_column("qty", width=12, justify="right")
    header.add_column("cost", width=14, justify="right")

    header.add_row(
        Text(atr_data.symbol, style="bold cyan"),
        Text(f"${price:,.2f}", style="bold white"),
        Text(f"ATR ${atr:.2f}", style="bold yellow"),
        Text(f"({atr_data.atr_percent:.1f}%)", style="yellow"),
        Text(f"[{timeframe}]", style="magenta"),
        Text(f"{qty:,.0f} sh", style="green") if qty else Text("-", style="dim"),
        Text(f"Cost: ${cost_basis:,.0f}", style="white") if qty else Text("", style="dim"),
    )
    content.add_row(header)
    content.add_row(Text(""))

    # === Row 2: Horizontal Price Bar (like atr_example.png) ===
    bar = Table(box=box.SIMPLE_HEAD, padding=0, expand=True, show_header=True)

    # Calculate adaptive column width based on largest price (8R)
    max_price = r_levels[8]
    col_width = len(f"{max_price:.2f}") + 1  # +1 for padding

    # Columns: SL-2x, SL-1.5x, SMA21, Entry, 1R-8R
    bar.add_column("SL-2x", justify="center", style="red", width=col_width)
    bar.add_column("SL-1.5x", justify="center", style="red", width=col_width)
    bar.add_column("SMA21", justify="center", style="blue", width=col_width)
    bar.add_column("Entry", justify="center", style="bold white", width=col_width)
    for i in range(1, 9):
        style = "green" if i <= 4 else "yellow"
        bar.add_column(f"{i}R", justify="center", style=style, width=col_width)

    # Price row - consistent 2 decimal places
    bar.add_row(
        f"{sl_2x:.2f}",
        f"{sl_1_5x:.2f}",
        f"{sma21:.2f}",
        f"{price:.2f}",
        *[f"{r_levels[i]:.2f}" for i in range(1, 9)]
    )

    # Percent row
    def pct(p):
        if price == 0:
            return "—"
        return f"{(p - price) / price * 100:+.0f}%" if p != price else "—"

    bar.add_row(
        pct(sl_2x),
        pct(sl_1_5x),
        pct(sma21),
        "—",
        *[pct(r_levels[i]) for i in range(1, 9)]
    )

    content.add_row(bar)
    content.add_row(Text(""))

    # === Row 3: Two columns - R-Targets | Trailing Stop ===
    two_col = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    two_col.add_column(width=44)
    two_col.add_column(width=36)

    # Left: Key R-Targets
    left = Table(box=None, padding=0, show_header=False)
    left.add_column(width=44)
    left.add_row(Text("R-TARGETS", style="bold green"))
    left.add_row(Text(f"Risk (1R) = ${risk:.0f} (1.5×ATR)", style="dim"))

    targets = Table(box=None, padding=(0, 1), show_header=False)
    targets.add_column(width=8)
    targets.add_column(width=10, justify="right")
    targets.add_column(width=10, justify="right")
    targets.add_column(width=14)

    targets.add_row(Text("Stop", style="red"), f"${sl_1_5x:,.0f}", Text(f"-${risk:.0f}", style="red"), "Exit all")
    targets.add_row(Text("1R", style="green"), f"${r_levels[1]:,.0f}", Text(f"+${risk:.0f}", style="green"), "Trail starts")
    targets.add_row(Text("2R", style="green"), f"${r_levels[2]:,.0f}", Text(f"+${risk*2:.0f}", style="green"), "Sell 33%")
    targets.add_row(Text("3R", style="yellow"), f"${r_levels[3]:,.0f}", Text(f"+${risk*3:.0f}", style="yellow"), "Sell 33%")
    targets.add_row(Text("8R", style="cyan"), f"${r_levels[8]:,.0f}", Text(f"+${risk*8:.0f}", style="cyan"), "Max target")
    left.add_row(targets)

    # Right: Trailing Stop
    right = Table(box=None, padding=0, show_header=False)
    right.add_column(width=36)
    trail = atr * 2

    right.add_row(Text("TRAILING STOP", style="bold yellow"))
    right.add_row(Text(f"Trail = High - 2×ATR (${trail:.0f})", style="dim"))
    right.add_row(Text(""))
    right.add_row(Text(f"Activates at 1R: ${r_levels[1]:,.0f}", style="white"))
    right.add_row(Text(""))
    right.add_row(Text(f"  ${r_levels[2]:,.0f} → stop ${r_levels[2]-trail:,.0f}", style="dim white"))
    right.add_row(Text(f"  ${r_levels[4]:,.0f} → stop ${r_levels[4]-trail:,.0f}", style="dim white"))
    right.add_row(Text(f"  ${r_levels[6]:,.0f} → stop ${r_levels[6]-trail:,.0f}", style="dim white"))

    two_col.add_row(left, right)
    content.add_row(two_col)

    # === Row 4: Exit Plan (if position) ===
    if qty > 0:
        content.add_row(Text("─" * 80, style="dim"))
        plan = Table(box=None, padding=(0, 1), show_header=False)
        plan.add_column(width=80)
        plan.add_row(Text("EXIT PLAN", style="bold magenta"))

        exits = Table(box=None, padding=(0, 2), show_header=False)
        exits.add_column(width=35)
        exits.add_column(width=20, justify="right")
        exits.add_column(width=20, justify="right")

        q3 = qty // 3
        exits.add_row(
            Text(f"├─ Sell {q3:,} @ 2R (${r_levels[2]:,.0f})", style="green"),
            Text(f"+${(r_levels[2]-cost_basis)*q3:,.0f}", style="green"),
            Text(f"{(r_levels[2]-cost_basis)/cost_basis*100:+.1f}%", style="dim green"),
        )
        exits.add_row(
            Text(f"├─ Sell {q3:,} @ 3R (${r_levels[3]:,.0f})", style="yellow"),
            Text(f"+${(r_levels[3]-cost_basis)*q3:,.0f}", style="yellow"),
            Text(f"{(r_levels[3]-cost_basis)/cost_basis*100:+.1f}%", style="dim yellow"),
        )
        exits.add_row(
            Text(f"└─ Trail {qty-2*q3:,} (2×ATR)", style="cyan"),
            Text("→ runners", style="cyan"),
            Text("5R-8R", style="dim cyan"),
        )
        plan.add_row(exits)
        content.add_row(plan)

    # === Row 5: Summary ===
    content.add_row(Text("─" * 80, style="dim"))
    summary = Table(box=None, padding=(0, 2), show_header=False)
    summary.add_column(width=25)
    summary.add_column(width=25)
    summary.add_column(width=25)

    max_return = (r_levels[8] - price) / price * 100
    summary.add_row(
        Text.assemble(("Risk: ", "dim"), (f"${risk:.0f}", "red"), (f" ({risk/price*100:.1f}%)", "dim")),
        Text.assemble(("Max @8R: ", "dim"), (f"${r_levels[8]-price:.0f}", "cyan"), (f" ({max_return:.0f}%)", "dim")),
        Text.assemble(("R:R @8R = ", "dim"), (f"{8:.1f}", "bold cyan")),
    )
    content.add_row(summary)

    # === Keyboard hints ===
    content.add_row(Text(""))
    hints = Text()
    hints.append("[w/s]", style="cyan")
    hints.append(" Sel  ", style="dim")
    hints.append("[+/-]", style="cyan")
    hints.append(f" Per({current_period})  ", style="dim")
    hints.append("[t]", style="cyan")
    hints.append(f" {timeframe}  ", style="magenta")
    hints.append("[h]", style="cyan")
    hints.append(" Help  ", style="dim")
    hints.append("[r]", style="cyan")
    hints.append(" Reset", style="dim")
    content.add_row(hints)

    return Panel(
        content,
        title=f"ATR: {atr_data.symbol} | {timeframe} ATR({current_period})=${atr:.2f} ({atr_data.atr_percent:.1f}%)",
        border_style="blue",
    )


def render_atr_help(
    atr_data: Optional[ATRData] = None,
    position: Optional[PositionRisk] = None,
) -> Panel:
    """Render ATR strategy help - clean two-column layout."""
    entry = atr_data.current_price if atr_data else 483
    atr = atr_data.atr_value if atr_data else 12
    risk = atr * 1.5

    content = Table.grid(padding=(0, 2))
    content.add_column(width=40)
    content.add_column(width=40)

    # Left: R-Multiple
    left = Table(box=None, padding=0, show_header=False)
    left.add_column(width=38)
    left.add_row(Text("R-MULTIPLE TARGETS", style="bold cyan underline"))
    left.add_row(Text(""))
    left.add_row(Text("Fixed exits based on risk:", style="dim"))
    left.add_row(Text(f"  1R = +${risk:.0f}  (get risk back)", style="green"))
    left.add_row(Text(f"  2R = +${risk*2:.0f}  (2× risk)", style="green"))
    left.add_row(Text(f"  3R = +${risk*3:.0f}  (3× risk)", style="yellow"))
    left.add_row(Text(f"  9R = +${risk*9:.0f}  (9× risk)", style="cyan"))
    left.add_row(Text(""))
    left.add_row(Text("Best for: ranging, quick trades", style="dim"))

    # Right: Trailing
    right = Table(box=None, padding=0, show_header=False)
    right.add_column(width=38)
    right.add_row(Text("TRAILING STOP", style="bold yellow underline"))
    right.add_row(Text(""))
    right.add_row(Text("Dynamic exit follows price:", style="dim"))
    right.add_row(Text(f"  Stop = High - 2×ATR", style="yellow"))
    right.add_row(Text(f"  Activates after 1R", style="dim"))
    right.add_row(Text(""))
    right.add_row(Text("Best for: trends, runners", style="dim"))

    content.add_row(left, right)
    content.add_row(Text(""), Text(""))

    # Comparison
    compare = Table(box=box.SIMPLE, padding=(0, 1))
    compare.add_column("", width=12, style="bold")
    compare.add_column("R-Targets", width=22)
    compare.add_column("Trailing", width=22)
    compare.add_row("Exit", "Fixed prices", "Dynamic")
    compare.add_row("Best", "Ranging market", "Trending")
    compare.add_row("Risk", "Exit too early", "Give back gains")

    # Best practice
    best = Table(box=None, padding=0, show_header=False)
    best.add_column(width=80)
    best.add_row(Text(""))
    best.add_row(Text("BEST PRACTICE: Scale Out + Trail", style="bold magenta underline"))
    best.add_row(Text("  • Sell 1/3 at 2R → Lock 2× risk", style="green"))
    best.add_row(Text("  • Sell 1/3 at 3R → Lock 3× risk", style="yellow"))
    best.add_row(Text("  • Trail 1/3 → Catch 5R-11R runners", style="cyan"))

    # Phases
    phases = Table(box=box.SIMPLE, padding=(0, 1))
    phases.add_column("Phase", width=12, style="bold")
    phases.add_column("Stop", width=15)
    phases.add_column("Purpose", width=28)
    phases.add_row("Entry→1R", "Fixed", "Let trade develop")
    phases.add_row("After 1R", "Trail 2×ATR", "Lock profits, ride trend")

    full = Table.grid(padding=0)
    full.add_column()
    full.add_row(content)
    full.add_row(compare)
    full.add_row(best)
    full.add_row(Text(""))
    full.add_row(phases)
    full.add_row(Text(""))
    full.add_row(Text("  [h] to close", style="dim cyan"))

    return Panel(full, title="ATR Strategy Guide", subtitle="[h] close", border_style="cyan")


def render_atr_loading() -> Panel:
    return Panel(Text("Loading...", style="dim"), title="ATR", border_style="yellow")


def render_atr_empty(message: str = "Select position with w/s") -> Panel:
    return Panel(Text(message, style="dim"), title="ATR", border_style="dim")


def render_atr_compact(atr_data: Optional[ATRData], optimization: Optional[ATROptimizationResult] = None) -> Panel:
    if atr_data is None:
        return Panel(Text("No ATR", style="dim"), title="ATR", border_style="dim")
    return Panel(
        Text.assemble(("ATR: ", "dim"), (f"${atr_data.atr_value:.2f}", "yellow")),
        title=atr_data.symbol, border_style="blue"
    )
