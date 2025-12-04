"""
Formatting utilities for the Terminal Dashboard.

Provides consistent number, price, and P&L formatting across all panels.
"""

from __future__ import annotations
from rich.text import Text


def format_price(price: float | None, is_using_close: bool = False, decimals: int = 2) -> str:
    """
    Format price with 'c' indicator if using yesterday's close.

    Args:
        price: The price to format (or None)
        is_using_close: True if price is from yesterday's close (no live data)
        decimals: Number of decimal places (2 for stocks, 3 for options)

    Returns:
        Formatted price string with 'c' suffix if using close
    """
    if price is None:
        return ""
    formatted = f"{price:.{decimals}f}"
    if is_using_close:
        return f"{formatted}c"
    return formatted


def format_quantity(value: float) -> str:
    """Format quantity with decimal places for fractional shares/contracts."""
    if abs(value) < 0.001:
        return ""

    # Show decimals only if needed
    if value == int(value):
        return f"{int(value):,}"
    else:
        return f"{value:,.2f}"


def format_number(value: float, color: bool = False) -> str:
    """Format number with optional color coding for P&L."""
    if abs(value) < 0.01:
        return ""

    formatted = f"{value:,.0f}"

    if not color:
        return formatted

    # Color coding for P&L
    if value > 0:
        return f"[green]{formatted}[/green]"
    elif value < 0:
        return f"[red]{formatted}[/red]"
    return formatted


def format_pnl(value: float) -> Text:
    """Format P&L with color."""
    if value > 0:
        return Text(f"+${value:,.2f}", style="green")
    elif value < 0:
        return Text(f"-${abs(value):,.2f}", style="red")
    else:
        return Text(f"${value:,.2f}", style="dim")


def format_pnl_simple(pnl: float | None) -> Text:
    """Format P&L value as a Text object with color styling."""
    if pnl is None:
        return Text("-", style="dim")
    if pnl > 0:
        return Text(f"+${pnl:,.0f}", style="green")
    elif pnl < 0:
        return Text(f"-${abs(pnl):,.0f}", style="red")
    else:
        return Text("$0", style="dim")
