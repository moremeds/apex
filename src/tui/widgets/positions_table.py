"""
Positions table widget for displaying position data.

Two modes:
- Consolidated: Groups positions by underlying (for Summary view)
- Detailed: Shows individual positions under underlying headers (for IB/Futu views)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual.widgets import DataTable
from textual.reactive import reactive
from textual.message import Message

from ..formatters import format_price, format_quantity, format_number


class PositionsTable(DataTable):
    """
    DataTable for displaying positions with selection support.

    Supports two display modes:
    - consolidated=True: Groups by underlying, shows aggregated metrics (Summary view)
    - consolidated=False: Shows individual positions under underlying headers (IB/Futu views)
    """

    class PositionSelected(Message):
        """Posted when a position row is selected."""

        def __init__(self, symbol: str, underlying: str) -> None:
            self.symbol = symbol
            self.underlying = underlying
            super().__init__()

    # Column definitions for consolidated view (Summary)
    COLUMNS_CONSOLIDATED = [
        ("Ticker", 12),
        ("Qty", 5),
        ("Spot", 10),
        ("Beta", 5),
        ("Mkt Value", 11),
        ("P&L", 9),
        ("UP&L", 9),
        ("Delta $", 9),
        ("D(Δ)", 6),
        ("G(γ)", 6),
        ("V(ν)", 6),
        ("Th(Θ)", 6),
    ]

    # Column definitions for detailed view (IB/Futu) with IV
    COLUMNS_DETAILED = [
        ("Ticker", 22),
        ("Qty", 8),
        ("Spot", 7),
        ("IV", 6),
        ("Beta", 5),
        ("Mkt Value", 11),
        ("P&L", 9),
        ("UP&L", 9),
        ("Delta $", 9),
        ("D(Δ)", 6),
        ("G(γ)", 6),
        ("V(ν)", 6),
        ("Th(Θ)", 6),
    ]

    # Reactive properties
    positions: reactive[List[Any]] = reactive([], init=False)

    def __init__(
        self,
        broker_filter: Optional[str] = None,
        show_portfolio_row: bool = True,
        consolidated: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize positions table.

        Args:
            broker_filter: Filter positions by broker ("ib", "futu", or None for all).
            show_portfolio_row: Show portfolio totals row at top.
            consolidated: True for grouped by underlying, False for individual positions.
        """
        super().__init__(cursor_type="row", zebra_stripes=True, **kwargs)
        self.broker_filter = broker_filter
        self.show_portfolio_row = show_portfolio_row
        self.consolidated = consolidated
        self._position_map: Dict[str, Any] = {}
        self._underlying_list: List[str] = []

    def on_mount(self) -> None:
        """Set up columns when widget is mounted."""
        columns = self.COLUMNS_CONSOLIDATED if self.consolidated else self.COLUMNS_DETAILED
        for name, width in columns:
            self.add_column(name, width=width)

    def _get_position_source(self, pos: Any) -> str:
        """Get the source/broker for a position."""
        # PositionRisk wraps Position, source is an enum
        if hasattr(pos, "position") and hasattr(pos.position, "source"):
            source = pos.position.source
            return source.value.lower() if hasattr(source, "value") else str(source).lower()
        if hasattr(pos, "source"):
            source = pos.source
            return source.value.lower() if hasattr(source, "value") else str(source).lower()
        return ""

    def watch_positions(self, positions: List[Any]) -> None:
        """React to position changes by rebuilding the table."""
        self.clear()
        self._position_map.clear()
        self._underlying_list.clear()

        if not positions:
            return

        # Filter by broker if specified
        filtered = positions
        if self.broker_filter:
            filtered = [
                p for p in positions
                if self._get_position_source(p) == self.broker_filter.lower()
            ]

        if not filtered:
            return

        if self.consolidated:
            self._render_consolidated(filtered)
        else:
            self._render_detailed(filtered)

    def _render_consolidated(self, positions: List[Any]) -> None:
        """Render consolidated view grouped by underlying."""
        # Group by underlying
        by_underlying: Dict[str, List[Any]] = {}
        for pos in positions:
            underlying = getattr(pos, "underlying", None) or getattr(pos, "symbol", "?")
            if underlying not in by_underlying:
                by_underlying[underlying] = []
            by_underlying[underlying].append(pos)

        # Calculate portfolio totals
        total_mkt_value = sum(getattr(p, "market_value", 0) or 0 for p in positions)
        total_pnl = sum(getattr(p, "daily_pnl", 0) or 0 for p in positions)
        total_upnl = sum(getattr(p, "unrealized_pnl", 0) or 0 for p in positions)
        total_delta_dollars = sum(getattr(p, "delta_dollars", 0) or 0 for p in positions)
        total_delta = sum(getattr(p, "delta", 0) or 0 for p in positions)
        total_gamma = sum(getattr(p, "gamma", 0) or 0 for p in positions)
        total_vega = sum(getattr(p, "vega", 0) or 0 for p in positions)
        total_theta = sum(getattr(p, "theta", 0) or 0 for p in positions)

        # Add portfolio row
        if self.show_portfolio_row:
            self.add_row(
                "[bold]>> PORTFOLIO[/]",
                str(len(positions)),
                "",
                "",
                format_number(total_mkt_value),
                format_number(total_pnl, color=True),
                format_number(total_upnl, color=True),
                format_number(total_delta_dollars),
                format_number(total_delta),
                format_number(total_gamma),
                format_number(total_vega),
                format_number(total_theta),
                key="__portfolio__",
            )

        # Sort underlyings by absolute market value
        sorted_underlyings = sorted(
            by_underlying.keys(),
            key=lambda u: sum(abs(getattr(p, "market_value", 0) or 0) for p in by_underlying[u]),
            reverse=True,
        )
        self._underlying_list = sorted_underlyings

        for underlying in sorted_underlyings:
            prs = by_underlying[underlying]

            # Calculate underlying totals
            mkt_value = sum(getattr(p, "market_value", 0) or 0 for p in prs)
            pnl = sum(getattr(p, "daily_pnl", 0) or 0 for p in prs)
            upnl = sum(getattr(p, "unrealized_pnl", 0) or 0 for p in prs)
            delta_dollars = sum(getattr(p, "delta_dollars", 0) or 0 for p in prs)
            delta = sum(getattr(p, "delta", 0) or 0 for p in prs)
            gamma = sum(getattr(p, "gamma", 0) or 0 for p in prs)
            vega = sum(getattr(p, "vega", 0) or 0 for p in prs)
            theta = sum(getattr(p, "theta", 0) or 0 for p in prs)

            # Get spot price from stock position
            spot = ""
            for p in prs:
                if not getattr(p, "expiry", None):
                    mark = getattr(p, "mark_price", None)
                    if mark:
                        is_close = getattr(p, "is_using_close", False)
                        spot = format_price(mark, is_close, decimals=2)
                        break

            # Get beta
            beta = ""
            if prs and getattr(prs[0], "beta", None):
                beta = f"{prs[0].beta:.2f}"

            row_key = f"underlying-{underlying}"
            self._position_map[row_key] = prs[0]

            self.add_row(
                f"  {underlying}",
                str(len(prs)),
                spot,
                beta,
                format_number(mkt_value),
                format_number(pnl, color=True),
                format_number(upnl, color=True),
                format_number(delta_dollars),
                format_number(delta),
                format_number(gamma),
                format_number(vega),
                format_number(theta),
                key=row_key,
            )

    def _render_detailed(self, positions: List[Any]) -> None:
        """Render detailed view with individual positions under underlying headers."""
        # Group by underlying
        by_underlying: Dict[str, List[Any]] = {}
        for pos in positions:
            underlying = getattr(pos, "underlying", None) or getattr(pos, "symbol", "?")
            if underlying not in by_underlying:
                by_underlying[underlying] = []
            by_underlying[underlying].append(pos)

        # Calculate broker totals
        total_mkt_value = sum(getattr(p, "market_value", 0) or 0 for p in positions)
        total_pnl = sum(getattr(p, "daily_pnl", 0) or 0 for p in positions)
        total_upnl = sum(getattr(p, "unrealized_pnl", 0) or 0 for p in positions)
        total_delta_dollars = sum(getattr(p, "delta_dollars", 0) or 0 for p in positions)
        total_delta = sum(getattr(p, "delta", 0) or 0 for p in positions)
        total_gamma = sum(getattr(p, "gamma", 0) or 0 for p in positions)
        total_vega = sum(getattr(p, "vega", 0) or 0 for p in positions)
        total_theta = sum(getattr(p, "theta", 0) or 0 for p in positions)

        broker_name = (self.broker_filter or "ALL").upper()

        # Add broker total row
        if self.show_portfolio_row:
            self.add_row(
                f"[bold]>> {broker_name} Total[/]",
                str(len(positions)),
                "",
                "",
                "",
                format_number(total_mkt_value),
                format_number(total_pnl, color=True),
                format_number(total_upnl, color=True),
                format_number(total_delta_dollars),
                format_number(total_delta),
                format_number(total_gamma),
                format_number(total_vega),
                format_number(total_theta),
                key="__total__",
            )

        # Sort underlyings by absolute market value
        sorted_underlyings = sorted(
            by_underlying.keys(),
            key=lambda u: sum(abs(getattr(p, "market_value", 0) or 0) for p in by_underlying[u]),
            reverse=True,
        )
        self._underlying_list = sorted_underlyings

        row_idx = 0
        for underlying in sorted_underlyings:
            prs = by_underlying[underlying]

            # Calculate underlying totals
            mkt_value = sum(getattr(p, "market_value", 0) or 0 for p in prs)
            pnl = sum(getattr(p, "daily_pnl", 0) or 0 for p in prs)
            upnl = sum(getattr(p, "unrealized_pnl", 0) or 0 for p in prs)
            delta_dollars = sum(getattr(p, "delta_dollars", 0) or 0 for p in prs)
            delta = sum(getattr(p, "delta", 0) or 0 for p in prs)
            gamma = sum(getattr(p, "gamma", 0) or 0 for p in prs)
            vega = sum(getattr(p, "vega", 0) or 0 for p in prs)
            theta = sum(getattr(p, "theta", 0) or 0 for p in prs)

            # Get beta
            beta = ""
            if prs and getattr(prs[0], "beta", None):
                beta = f"{prs[0].beta:.2f}"

            # Add underlying header row
            header_key = f"header-{underlying}"
            self._position_map[header_key] = prs[0]

            self.add_row(
                f"[bold]>> {underlying}[/]",
                "",
                "",
                "",
                beta,
                format_number(mkt_value),
                format_number(pnl, color=True),
                format_number(upnl, color=True),
                format_number(delta_dollars),
                format_number(delta),
                format_number(gamma),
                format_number(vega),
                format_number(theta),
                key=header_key,
            )

            # Sort: stocks first, then options by expiry
            stocks = [p for p in prs if not getattr(p, "expiry", None)]
            options = sorted(
                [p for p in prs if getattr(p, "expiry", None)],
                key=lambda p: getattr(p, "expiry", "") or "",
            )

            for pos in stocks + options:
                row_idx += 1
                display_name = self._get_display_name(pos)
                qty = getattr(pos, "quantity", 0)
                mark = getattr(pos, "mark_price", None)
                is_close = getattr(pos, "is_using_close", False)
                iv = getattr(pos, "iv", None)
                pos_beta = getattr(pos, "beta", None)

                # Determine price decimals (3 for options, 2 for stocks)
                decimals = 3 if getattr(pos, "expiry", None) else 2

                row_key = f"pos-{row_idx}-{display_name}"
                self._position_map[row_key] = pos

                self.add_row(
                    f"  {display_name}",
                    format_quantity(qty),
                    format_price(mark, is_close, decimals=decimals),
                    f"{iv * 100:.1f}%" if iv else "",
                    f"{pos_beta:.2f}" if pos_beta else "",
                    format_number(getattr(pos, "market_value", 0) or 0),
                    format_number(getattr(pos, "daily_pnl", 0) or 0, color=True),
                    format_number(getattr(pos, "unrealized_pnl", 0) or 0, color=True),
                    format_number(getattr(pos, "delta_dollars", 0) or 0),
                    format_number(getattr(pos, "delta", 0) or 0),
                    format_number(getattr(pos, "gamma", 0) or 0),
                    format_number(getattr(pos, "vega", 0) or 0),
                    format_number(getattr(pos, "theta", 0) or 0),
                    key=row_key,
                )

    def _get_display_name(self, pos: Any) -> str:
        """Get display name for a position (e.g., 'QQQ 20251226P615.0')."""
        # Try get_display_name method first
        if hasattr(pos, "get_display_name"):
            return pos.get_display_name()

        # Build from position attributes
        symbol = getattr(pos, "symbol", "?")
        expiry = getattr(pos, "expiry", None)

        if not expiry:
            # Stock - just return symbol
            return symbol

        # Option - format as "SYM YYYYMMDD[C/P]STRIKE"
        strike = getattr(pos, "strike", None)
        right = getattr(pos, "right", None)

        # Parse expiry to YYYYMMDD
        if isinstance(expiry, str):
            if len(expiry) == 8:  # YYYYMMDD
                expiry_str = expiry
            elif "-" in expiry:  # YYYY-MM-DD
                expiry_str = expiry.replace("-", "")
            else:
                expiry_str = expiry
        else:
            expiry_str = str(expiry)

        underlying = getattr(pos, "underlying", symbol.split()[0] if " " in symbol else symbol)

        if strike and right:
            return f"{underlying} {expiry_str}{right}{strike:.1f}"
        return symbol

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key is not None:
            key = str(event.row_key.value)
            if key in ("__portfolio__", "__total__"):
                return

            if key in self._position_map:
                pos = self._position_map[key]
                underlying = getattr(pos, "underlying", None) or getattr(pos, "symbol", "?")
                symbol = getattr(pos, "symbol", underlying)
                self.post_message(self.PositionSelected(symbol=symbol, underlying=underlying))

    def get_selected_underlying(self) -> Optional[str]:
        """Get the underlying of the currently selected row."""
        if self.cursor_row is None:
            return None

        try:
            row_key = self.get_row_at(self.cursor_row)
            if row_key:
                key = str(self.get_row_key(self.cursor_row))
                if key in self._position_map:
                    pos = self._position_map[key]
                    return getattr(pos, "underlying", None) or getattr(pos, "symbol", None)
        except Exception:
            pass
        return None

    def get_underlyings(self) -> List[str]:
        """Get list of underlyings in display order."""
        return self._underlying_list.copy()
