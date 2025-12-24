"""
Formatting utilities for the Terminal Dashboard.

Provides consistent number, price, and P&L formatting across all panels.
Uses Textual markup syntax (same as Rich markup).
"""

from __future__ import annotations


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

    # Color coding for P&L (green for positive, red for negative)
    if value > 0:
        return f"[green]{formatted}[/]"
    elif value < 0:
        return f"[red]{formatted}[/]"
    return formatted


def format_pnl(value: float) -> str:
    """Format P&L with color markup."""
    if value > 0:
        return f"[green]+${value:,.2f}[/]"
    elif value < 0:
        return f"[red]-${abs(value):,.2f}[/]"
    else:
        return f"[dim]${value:,.2f}[/]"


def format_pnl_simple(pnl: float | None) -> str:
    """Format P&L value with color markup."""
    if pnl is None:
        return "[dim]-[/]"
    if pnl > 0:
        return f"[green]+${pnl:,.0f}[/]"
    elif pnl < 0:
        return f"[red]-${abs(pnl):,.0f}[/]"
    else:
        return "[dim]$0[/]"


def fmt_number(value: float, decimals: int = 0) -> str:
    """Format number with specified decimal places."""
    if abs(value) < 0.01:
        return ""
    return f"{value:,.{decimals}f}"


def fmt_pnl(value: float) -> str:
    """Format P&L with sign and color."""
    if value > 0:
        return f"[green]+${value:,.2f}[/]"
    elif value < 0:
        return f"[red]-${abs(value):,.2f}[/]"
    return "[dim]$0.00[/]"


def fmt_greek(value: float, decimals: int = 0) -> str:
    """Format Greek value."""
    if abs(value) < 0.01:
        return ""
    return f"{value:,.{decimals}f}"
