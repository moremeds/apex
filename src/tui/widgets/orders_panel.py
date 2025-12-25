"""
Open orders panel widget.

Displays pending/open orders for a broker.
Currently shows placeholder content as persistence layer is not configured.
"""

from __future__ import annotations

from typing import Any, List

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult


class OrdersPanel(Widget):
    """Open orders display for a broker."""

    # Reactive state - use factory to avoid mutable default sharing
    orders: reactive[List[Any]] = reactive(list, init=False)

    def __init__(self, broker: str = "IB", **kwargs):
        super().__init__(**kwargs)
        self.broker = broker

    def compose(self) -> ComposeResult:
        """Compose the orders panel layout."""
        with Vertical(id="orders-content"):
            yield Static(f"[bold]Open Orders ({self.broker})[/]", id="orders-title")
            yield Static("[dim]No orders data[/]", id="orders-list")

    def watch_orders(self, orders: List[Any]) -> None:
        """Update display when orders change."""
        self._render_orders(orders)

    def _render_orders(self, orders: List[Any]) -> None:
        """Render order list."""
        try:
            orders_list = self.query_one("#orders-list", Static)

            if not orders:
                orders_list.update("[dim]No orders data[/]")
                return

            lines = []
            for order in orders:
                time_str = getattr(order, "time", "")
                side = getattr(order, "side", "?")
                symbol = getattr(order, "symbol", "?")
                qty = getattr(order, "quantity", 0)
                price = getattr(order, "price", 0)
                status = getattr(order, "status", "PENDING")

                side_style = "green" if side == "BUY" else "red"
                lines.append(
                    f"[dim]{time_str}[/] [{side_style}]{side}[/] [cyan]{symbol}[/] {qty} @ ${price:.2f} [yellow]{status}[/]"
                )

            orders_list.update("\n".join(lines))
        except Exception as e:
            self.log.error(f"Failed to render orders: {e}")
