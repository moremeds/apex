"""
Position history panels rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import datetime
import logging
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..formatters import format_pnl_simple

if TYPE_CHECKING:
    from ...infrastructure.persistence.persistence_manager import PersistenceManager

logger = logging.getLogger(__name__)


def render_position_history_today(
    broker: str,
    persistence_manager: "PersistenceManager | None",
) -> Panel:
    """
    Render today's executed trades for a broker.

    Shows actual trade executions from the trades table in database.

    Args:
        broker: Broker name ("IB" or "FUTU").
        persistence_manager: Persistence manager for database queries.

    Returns:
        Panel containing today's trades.
    """
    if not persistence_manager:
        return Panel(
            Text("Persistence not enabled", style="dim"),
            title="Today's Trades",
            border_style="dim",
        )

    try:
        from src.models.order import OrderSource

        source = OrderSource.IB if broker == "IB" else OrderSource.FUTU
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = persistence_manager.orders.get_trades(
            source=source,
            start_time=today_start,
            limit=20,
        )

        if not today_trades:
            return Panel(
                Text("No trades today", style="dim"),
                title=f"Today's Trades ({broker})",
                border_style="dim",
            )

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Time", style="dim", no_wrap=True, width=8)
        table.add_column("Side", style="bold", no_wrap=True, width=4)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("Price", justify="right", width=10)

        for trade in today_trades[:8]:
            trade_time = trade.get("trade_time")
            time_str = trade_time.strftime("%H:%M:%S") if trade_time else ""

            side = trade.get("side", "")
            if side == "BUY":
                side_style = "green"
                side_str = "BUY"
            else:
                side_style = "red"
                side_str = "SELL"

            qty = trade.get("quantity", 0)
            price = trade.get("price", 0)

            table.add_row(
                time_str,
                Text(side_str, style=side_style),
                trade.get("symbol", "")[:18],
                f"{qty:,.0f}",
                f"${price:,.2f}" if price else "-",
            )

        return Panel(
            table,
            title=f"Today's Trades ({len(today_trades)})",
            border_style="cyan",
        )

    except Exception as e:
        logger.warning(f"Failed to load today's trades: {e}")
        return Panel(
            Text(f"Error: {e}", style="red"),
            title="Today's Trades",
            border_style="red",
        )


def render_open_orders(
    broker: str,
    persistence_manager: "PersistenceManager | None",
) -> Panel:
    """
    Render open/pending orders from database for a broker.

    Shows orders that are not yet filled (PENDING, SUBMITTED, PARTIALLY_FILLED).

    Args:
        broker: Broker name ("IB" or "FUTU").
        persistence_manager: Persistence manager for database queries.

    Returns:
        Panel containing open orders.
    """
    if not persistence_manager:
        return Panel(
            Text("Persistence not enabled", style="dim"),
            title="Open Orders",
            border_style="dim",
        )

    try:
        from src.models.order import OrderSource

        source = OrderSource.IB if broker == "IB" else OrderSource.FUTU
        open_orders = persistence_manager.orders.get_open_orders(source=source)

        if not open_orders:
            return Panel(
                Text("No open orders", style="dim"),
                title=f"Open Orders ({broker})",
                border_style="dim",
            )

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Time", style="dim", no_wrap=True, width=8)
        table.add_column("Side", style="bold", no_wrap=True, width=4)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Qty", justify="right", width=6)
        table.add_column("Price", justify="right", width=8)
        table.add_column("Status", style="yellow", width=8)

        for order in open_orders[:8]:
            created_time = order.get("created_time")
            time_str = created_time.strftime("%H:%M:%S") if created_time else ""

            side = order.get("side", "")
            side_style = "green" if side == "BUY" else "red"

            qty = order.get("quantity", 0)
            filled = order.get("filled_quantity", 0)
            limit_price = order.get("limit_price")

            if filled > 0:
                qty_str = f"{filled:.0f}/{qty:.0f}"
            else:
                qty_str = f"{qty:.0f}"

            price_str = f"${limit_price:,.2f}" if limit_price else "MKT"

            status = order.get("status", "PENDING")
            if status == "PARTIALLY_FILLED":
                status_str = "PARTIAL"
            elif status == "SUBMITTED":
                status_str = "SENT"
            else:
                status_str = status[:7]

            table.add_row(
                time_str,
                Text(side[:1], style=side_style),
                order.get("symbol", "")[:15],
                qty_str,
                price_str,
                status_str,
            )

        return Panel(
            table,
            title=f"Open Orders ({len(open_orders)})",
            border_style="yellow",
        )

    except Exception as e:
        logger.warning(f"Failed to load open orders: {e}")
        return Panel(
            Text(f"Error: {e}", style="red"),
            title="Open Orders",
            border_style="red",
        )


def render_position_history_recent(
    broker: str,
    persistence_manager: "PersistenceManager | None",
) -> Panel:
    """
    Render all stored positions from database for a broker.

    Shows positions currently tracked in the database to verify persistence is working.

    Args:
        broker: Broker name ("IB" or "FUTU").
        persistence_manager: Persistence manager for database queries.

    Returns:
        Panel containing stored positions.
    """
    if not persistence_manager:
        return Panel(
            Text("Persistence not enabled", style="dim"),
            title="Stored Positions (DB)",
            border_style="dim",
        )

    try:
        all_positions = persistence_manager.get_all_position_snapshots(limit=100)
        broker_positions = [
            p for p in all_positions
            if p.get("source") == broker
        ]

        if not broker_positions:
            stats = persistence_manager.get_stats()
            info_text = Text()
            info_text.append("No positions stored for this broker\n\n", style="dim")
            info_text.append(f"DB Stats:\n", style="cyan")
            info_text.append(f"  Snapshots saved: {stats.get('snapshots_saved', 0)}\n", style="white")
            info_text.append(f"  Changes detected: {stats.get('changes_detected', 0)}\n", style="white")
            info_text.append(f"  Tracked positions: {stats.get('tracked_positions', 0)}\n", style="white")
            return Panel(
                info_text,
                title=f"Stored Positions ({broker})",
                border_style="dim",
            )

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("AvgPx", justify="right", width=10)
        table.add_column("Mark", justify="right", width=10)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Updated", style="dim", width=10)

        broker_positions.sort(key=lambda p: (p.get("underlying", ""), p.get("symbol", "")))

        for pos in broker_positions[:20]:
            symbol = pos.get("symbol", "")[:18]
            qty = pos.get("quantity")
            avg_price = pos.get("avg_price")
            mark_price = pos.get("mark_price")
            pnl = pos.get("unrealized_pnl")
            snapshot_time = pos.get("snapshot_time")

            qty_str = f"{qty:,.0f}" if qty is not None else "-"
            avg_str = f"${avg_price:,.2f}" if avg_price else "-"
            mark_str = f"${mark_price:,.2f}" if mark_price else "-"
            pnl_text = format_pnl_simple(pnl)
            time_str = snapshot_time.strftime("%H:%M:%S") if snapshot_time else "-"

            table.add_row(
                symbol,
                qty_str,
                avg_str,
                mark_str,
                pnl_text,
                time_str,
            )

        stats = persistence_manager.get_stats()
        title = f"Stored Positions ({len(broker_positions)} in DB, {stats.get('changes_detected', 0)} changes)"

        return Panel(
            table,
            title=title,
            border_style="blue",
        )

    except Exception as e:
        logger.warning(f"Failed to load stored positions: {e}")
        return Panel(
            Text(f"Error: {e}", style="red"),
            title="Stored Positions",
            border_style="red",
        )
