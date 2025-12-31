"""
Open orders panel widget.

Displays pending/open orders for a broker.
Currently shows placeholder content as persistence layer is not configured.

OPT-011: Uses OrderViewModel for incremental updates.
"""

from __future__ import annotations

from typing import Any, List

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult

from ..viewmodels.order_vm import OrderViewModel


class OrdersPanel(Widget):
    """Open orders display for a broker."""

    # Reactive state - use factory to avoid mutable default sharing
    orders: reactive[List[Any]] = reactive(list, init=False)

    def __init__(self, broker: str = "IB", **kwargs):
        super().__init__(**kwargs)
        self.broker = broker
        # OPT-011: ViewModel for incremental updates
        self._vm = OrderViewModel()

    def compose(self) -> ComposeResult:
        """Compose the orders panel layout."""
        with Vertical(id="orders-content"):
            yield Static(f"[bold]Open Orders ({self.broker})[/]", id="orders-title")
            yield Static("[dim]No orders data[/]", id="orders-list")

    def watch_orders(self, orders: List[Any]) -> None:
        """Update display when orders change."""
        self._render_orders(orders)

    def _render_orders(self, orders: List[Any]) -> None:
        """Render order list with incremental updates."""
        try:
            orders_list = self.query_one("#orders-list", Static)

            # OPT-011: Use ViewModel to check if update needed
            needs_update, content = self._vm.needs_update(orders)
            if needs_update:
                orders_list.update(content)
        except Exception as e:
            self.log.error(f"Failed to render orders: {e}")
