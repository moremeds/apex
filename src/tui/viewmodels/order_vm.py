"""
Order ViewModel for incremental updates.

OPT-011: Provides diff computation to avoid full rebuilds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class OrderDisplayRow:
    """Immutable order row for comparison."""

    order_id: str
    time_str: str
    side: str
    symbol: str
    quantity: float
    price: float
    status: str

    def format_line(self) -> str:
        """Format order for display."""
        side_style = "green" if self.side == "BUY" else "red"
        return (
            f"[dim]{self.time_str}[/] [{side_style}]{self.side}[/] "
            f"[cyan]{self.symbol}[/] {self.quantity} @ ${self.price:.2f} "
            f"[yellow]{self.status}[/]"
        )


class OrderViewModel:
    """
    ViewModel for orders with incremental update detection.

    OPT-011: Only triggers re-render when content actually changes.
    """

    def __init__(self) -> None:
        self._previous_rows: List[OrderDisplayRow] = []
        self._previous_content: Optional[str] = None

    def compute_display(self, orders: List[Any]) -> tuple[List[OrderDisplayRow], str]:
        """
        Compute display rows and formatted content.

        Returns:
            Tuple of (rows, formatted_content)
        """
        if not orders:
            return [], "[dim]No orders data[/]"

        rows = []
        for order in orders:
            order_id = str(getattr(order, "order_id", id(order)))
            row = OrderDisplayRow(
                order_id=order_id,
                time_str=str(getattr(order, "time", "")),
                side=getattr(order, "side", "?"),
                symbol=getattr(order, "symbol", "?"),
                quantity=float(getattr(order, "quantity", 0)),
                price=float(getattr(order, "price", 0)),
                status=getattr(order, "status", "PENDING"),
            )
            rows.append(row)

        content = "\n".join(row.format_line() for row in rows)
        return rows, content

    def needs_update(self, orders: List[Any]) -> tuple[bool, str]:
        """
        Check if update is needed and return new content.

        OPT-011: Avoids re-render if content unchanged.

        Returns:
            Tuple of (needs_update, new_content)
        """
        rows, content = self.compute_display(orders)

        # Compare with previous
        if content == self._previous_content:
            return False, content

        # Update cache
        self._previous_rows = rows
        self._previous_content = content
        return True, content

    def invalidate(self) -> None:
        """Clear cache, forcing update on next call."""
        self._previous_rows = []
        self._previous_content = None
