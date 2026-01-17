"""
PositionViewModel - Framework-agnostic position data transformation.

Extracts business logic from PositionsTable:
- Position filtering by broker
- Grouping by underlying
- Aggregation of totals (market_value, P&L, Greeks)
- Sorting by absolute market value
- Row key generation for stable updates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .base import BaseViewModel, CellUpdate

if TYPE_CHECKING:
    from ...domain.events.domain_events import PositionDeltaEvent


class PositionViewModel(BaseViewModel[List[Any]]):
    """
    ViewModel for position tables.

    Responsibilities:
    - Group positions by underlying
    - Aggregate totals (market_value, pnl, Greeks)
    - Sort by absolute market value
    - Track row ordering for cursor preservation
    - Compute cell-level diffs for incremental updates
    - Apply streaming deltas for O(1) cell updates
    """

    def __init__(
        self,
        broker_filter: Optional[str] = None,
        consolidated: bool = True,
        show_portfolio_row: bool = True,
    ) -> None:
        super().__init__()
        self.broker_filter = broker_filter
        self.consolidated = consolidated
        self.show_portfolio_row = show_portfolio_row
        self._underlying_order: List[str] = []
        self._position_map: Dict[str, Any] = {}  # row_key -> position
        # Delta support: track numeric values for O(1) delta application
        self._symbol_to_row_keys: Dict[str, List[str]] = {}  # symbol -> [row_keys]
        self._values_cache: Dict[str, Dict[str, float]] = {}  # row_key -> {metric -> value}

    def compute_display_data(self, positions: List[Any]) -> Dict[str, List[str]]:
        """Transform position list into display rows."""
        if not positions:
            return {}

        # Filter by broker
        filtered = self._filter_positions(positions)
        if not filtered:
            return {}

        # Group by underlying
        by_underlying = self._group_by_underlying(filtered)

        # Sort underlyings by absolute market value
        sorted_underlyings = sorted(
            by_underlying.keys(),
            key=lambda u: sum(abs(getattr(p, "market_value", 0) or 0) for p in by_underlying[u]),
            reverse=True,
        )
        self._underlying_order = sorted_underlyings

        if self.consolidated:
            return self._build_consolidated_rows(filtered, by_underlying, sorted_underlyings)
        else:
            return self._build_detailed_rows(filtered, by_underlying, sorted_underlyings)

    def get_row_order(self, positions: List[Any]) -> List[str]:
        """Return ordered list of row keys."""
        display_data = self.compute_display_data(positions)
        return self._order_row_keys(list(display_data.keys()))

    def _filter_positions(self, positions: List[Any]) -> List[Any]:
        """Filter positions by broker."""
        if not self.broker_filter:
            return positions
        return [p for p in positions if self._get_source(p) == self.broker_filter.lower()]

    def _get_source(self, pos: Any) -> str:
        """Extract broker source from position."""
        if hasattr(pos, "position") and hasattr(pos.position, "source"):
            source = pos.position.source
            return source.value.lower() if hasattr(source, "value") else str(source).lower()
        if hasattr(pos, "source"):
            source = pos.source
            return source.value.lower() if hasattr(source, "value") else str(source).lower()
        return ""

    def _group_by_underlying(self, positions: List[Any]) -> Dict[str, List[Any]]:
        """Group positions by underlying symbol."""
        result: Dict[str, List[Any]] = {}
        for pos in positions:
            underlying = getattr(pos, "underlying", None) or getattr(pos, "symbol", "?")
            if underlying not in result:
                result[underlying] = []
            result[underlying].append(pos)
        return result

    def _build_consolidated_rows(
        self,
        positions: List[Any],
        by_underlying: Dict[str, List[Any]],
        sorted_underlyings: List[str],
    ) -> Dict[str, List[str]]:
        """Build consolidated view rows (grouped by underlying)."""
        result: Dict[str, List[str]] = {}
        self._position_map.clear()
        self._symbol_to_row_keys.clear()
        self._values_cache.clear()

        # Portfolio totals row
        if self.show_portfolio_row:
            totals = self._compute_totals(positions)
            self._values_cache["__portfolio__"] = totals.copy()
            result["__portfolio__"] = [
                "[bold]>> PORTFOLIO[/]",
                str(len(positions)),
                "",
                "",
                self._fmt_number(totals["mkt_value"]),
                self._fmt_number(totals["pnl"], color=True),
                self._fmt_number(totals["upnl"], color=True),
                self._fmt_number(totals["delta_$"]),
                self._fmt_number(totals["delta"]),
                self._fmt_number(totals["gamma"]),
                self._fmt_number(totals["vega"]),
                self._fmt_number(totals["theta"]),
            ]
            # All symbols contribute to portfolio row
            for pos in positions:
                symbol = getattr(pos, "symbol", "")
                if symbol not in self._symbol_to_row_keys:
                    self._symbol_to_row_keys[symbol] = []
                self._symbol_to_row_keys[symbol].append("__portfolio__")

        for underlying in sorted_underlyings:
            prs = by_underlying[underlying]
            totals = self._compute_totals(prs)
            spot, beta = self._get_spot_and_beta(prs)

            row_key = f"underlying-{underlying}"
            self._position_map[row_key] = prs[0]
            self._values_cache[row_key] = totals.copy()

            # Track symbol -> row_key mappings
            for pos in prs:
                symbol = getattr(pos, "symbol", "")
                if symbol not in self._symbol_to_row_keys:
                    self._symbol_to_row_keys[symbol] = []
                self._symbol_to_row_keys[symbol].append(row_key)

            result[row_key] = [
                f"  {underlying}",
                str(len(prs)),
                spot,
                beta,
                self._fmt_number(totals["mkt_value"]),
                self._fmt_number(totals["pnl"], color=True),
                self._fmt_number(totals["upnl"], color=True),
                self._fmt_number(totals["delta_$"]),
                self._fmt_number(totals["delta"]),
                self._fmt_number(totals["gamma"]),
                self._fmt_number(totals["vega"]),
                self._fmt_number(totals["theta"]),
            ]

        return result

    def _build_detailed_rows(
        self,
        positions: List[Any],
        by_underlying: Dict[str, List[Any]],
        sorted_underlyings: List[str],
    ) -> Dict[str, List[str]]:
        """Build detailed view rows with individual positions."""
        result: Dict[str, List[str]] = {}
        self._position_map.clear()
        self._symbol_to_row_keys.clear()
        self._values_cache.clear()

        # Broker total row
        if self.show_portfolio_row:
            totals = self._compute_totals(positions)
            self._values_cache["__total__"] = totals.copy()
            broker_name = (self.broker_filter or "ALL").upper()
            result["__total__"] = [
                f"[bold]>> {broker_name} Total[/]",
                str(len(positions)),
                "",
                "",
                "",
                self._fmt_number(totals["mkt_value"]),
                self._fmt_number(totals["pnl"], color=True),
                self._fmt_number(totals["upnl"], color=True),
                self._fmt_number(totals["delta_$"]),
                self._fmt_number(totals["delta"]),
                self._fmt_number(totals["gamma"]),
                self._fmt_number(totals["vega"]),
                self._fmt_number(totals["theta"]),
            ]
            # All symbols contribute to total row
            for pos in positions:
                symbol = getattr(pos, "symbol", "")
                if symbol not in self._symbol_to_row_keys:
                    self._symbol_to_row_keys[symbol] = []
                self._symbol_to_row_keys[symbol].append("__total__")

        for underlying in sorted_underlyings:
            prs = by_underlying[underlying]
            totals = self._compute_totals(prs)
            _, beta = self._get_spot_and_beta(prs)

            # Header row
            header_key = f"header-{underlying}"
            self._position_map[header_key] = prs[0]
            self._values_cache[header_key] = totals.copy()

            # Track symbol -> header row mapping
            for pos in prs:
                symbol = getattr(pos, "symbol", "")
                if symbol not in self._symbol_to_row_keys:
                    self._symbol_to_row_keys[symbol] = []
                self._symbol_to_row_keys[symbol].append(header_key)

            result[header_key] = [
                f"[bold]>> {underlying}[/]",
                "",
                "",
                "",
                beta,
                self._fmt_number(totals["mkt_value"]),
                self._fmt_number(totals["pnl"], color=True),
                self._fmt_number(totals["upnl"], color=True),
                self._fmt_number(totals["delta_$"]),
                self._fmt_number(totals["delta"]),
                self._fmt_number(totals["gamma"]),
                self._fmt_number(totals["vega"]),
                self._fmt_number(totals["theta"]),
            ]

            # Position rows (stocks first, then options by expiry)
            stocks = [p for p in prs if not getattr(p, "expiry", None)]
            options = sorted(
                [p for p in prs if getattr(p, "expiry", None)],
                key=lambda p: getattr(p, "expiry", "") or "",
            )

            for pos in stocks + options:
                display_name = self._get_display_name(pos)
                expiry = getattr(pos, "expiry", None)
                symbol = getattr(pos, "symbol", "?")

                # Stable row key based on position identity
                expiry_key = str(expiry) if expiry else "stock"
                row_key = f"pos-{underlying}-{symbol}-{expiry_key}"
                self._position_map[row_key] = pos

                # Cache position values for delta application
                self._values_cache[row_key] = self._extract_position_values(pos)
                if symbol not in self._symbol_to_row_keys:
                    self._symbol_to_row_keys[symbol] = []
                self._symbol_to_row_keys[symbol].append(row_key)

                result[row_key] = self._build_position_row(pos, display_name)

        return result

    def _build_position_row(self, pos: Any, display_name: str) -> List[str]:
        """Build a single position row (detailed view with IV column)."""
        qty = getattr(pos, "quantity", 0)
        mark = getattr(pos, "mark_price", None)
        is_close = getattr(pos, "is_using_close", False)
        iv = getattr(pos, "iv", None)
        pos_beta = getattr(pos, "beta", None)
        decimals = 3 if getattr(pos, "expiry", None) else 2

        return [
            f"  {display_name}",
            self._fmt_quantity(qty),
            self._fmt_price(mark, is_close, decimals),
            f"{iv * 100:.1f}%" if iv else "",
            f"{pos_beta:.2f}" if pos_beta else "",
            self._fmt_number(getattr(pos, "market_value", 0) or 0),
            self._fmt_number(getattr(pos, "daily_pnl", 0) or 0, color=True),
            self._fmt_number(getattr(pos, "unrealized_pnl", 0) or 0, color=True),
            self._fmt_number(getattr(pos, "delta_dollars", 0) or 0),
            self._fmt_number(getattr(pos, "delta", 0) or 0),
            self._fmt_number(getattr(pos, "gamma", 0) or 0),
            self._fmt_number(getattr(pos, "vega", 0) or 0),
            self._fmt_number(getattr(pos, "theta", 0) or 0),
        ]

    def _compute_totals(self, positions: List[Any]) -> Dict[str, float]:
        """Compute aggregated totals for a list of positions."""
        return {
            "mkt_value": sum(getattr(p, "market_value", 0) or 0 for p in positions),
            "pnl": sum(getattr(p, "daily_pnl", 0) or 0 for p in positions),
            "upnl": sum(getattr(p, "unrealized_pnl", 0) or 0 for p in positions),
            "delta_$": sum(getattr(p, "delta_dollars", 0) or 0 for p in positions),
            "delta": sum(getattr(p, "delta", 0) or 0 for p in positions),
            "gamma": sum(getattr(p, "gamma", 0) or 0 for p in positions),
            "vega": sum(getattr(p, "vega", 0) or 0 for p in positions),
            "theta": sum(getattr(p, "theta", 0) or 0 for p in positions),
        }

    def _get_spot_and_beta(self, positions: List[Any]) -> Tuple[str, str]:
        """Get spot price and beta for an underlying group."""
        spot = ""
        beta = ""
        for p in positions:
            if not getattr(p, "expiry", None):
                mark = getattr(p, "mark_price", None)
                if mark:
                    is_close = getattr(p, "is_using_close", False)
                    spot = self._fmt_price(mark, is_close, 2)
                    break
        if positions and getattr(positions[0], "beta", None):
            beta = f"{positions[0].beta:.2f}"
        return spot, beta

    def _get_display_name(self, pos: Any) -> str:
        """Get display name for a position."""
        if hasattr(pos, "get_display_name"):
            return pos.get_display_name()

        symbol = getattr(pos, "symbol", "?")
        expiry = getattr(pos, "expiry", None)

        if not expiry:
            return symbol

        strike = getattr(pos, "strike", None)
        right = getattr(pos, "right", None)

        if isinstance(expiry, str):
            if len(expiry) == 8:
                expiry_str = expiry
            elif "-" in expiry:
                expiry_str = expiry.replace("-", "")
            else:
                expiry_str = expiry
        else:
            expiry_str = str(expiry)

        underlying = getattr(pos, "underlying", symbol.split()[0] if " " in symbol else symbol)

        if strike and right:
            return f"{underlying} {expiry_str}{right}{strike:.1f}"
        return symbol

    def _order_row_keys(self, keys: List[str]) -> List[str]:
        """Order row keys for display."""
        keys = list(keys)
        result = []

        # Portfolio/total row first
        for k in ["__portfolio__", "__total__"]:
            if k in keys:
                result.append(k)
                keys.remove(k)

        # For detailed view: headers and their positions together
        if not self.consolidated:
            for underlying in self._underlying_order:
                header_key = f"header-{underlying}"
                if header_key in keys:
                    result.append(header_key)
                    keys.remove(header_key)

                # Add all positions for this underlying
                pos_keys = [k for k in keys if k.startswith(f"pos-{underlying}-")]
                for pk in pos_keys:
                    result.append(pk)
                    keys.remove(pk)
        else:
            # Consolidated: just underlying rows in order
            for underlying in self._underlying_order:
                key = f"underlying-{underlying}"
                if key in keys:
                    result.append(key)
                    keys.remove(key)

        # Any remaining keys
        result.extend(keys)

        return result

    # Formatting helpers (with Rich markup for Textual)
    def _fmt_number(self, value: float, color: bool = False) -> str:
        """Format a number with optional color coding."""
        if abs(value) < 0.01:
            return ""
        formatted = f"{value:,.0f}"
        if not color:
            return formatted
        if value > 0:
            return f"[green]{formatted}[/]"
        elif value < 0:
            return f"[red]{formatted}[/]"
        return formatted

    def _fmt_price(self, price: Optional[float], is_close: bool, decimals: int) -> str:
        """Format a price with optional close indicator."""
        if price is None:
            return ""
        formatted = f"{price:.{decimals}f}"
        return f"{formatted}c" if is_close else formatted

    def _fmt_quantity(self, value: float) -> str:
        """Format a quantity."""
        if abs(value) < 0.001:
            return ""
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.2f}"

    def get_underlying_order(self) -> List[str]:
        """Get ordered list of underlyings."""
        return self._underlying_order.copy()

    def get_position_for_key(self, row_key: str) -> Optional[Any]:
        """Get the position object for a row key."""
        return self._position_map.get(row_key)

    def _extract_position_values(self, pos: Any) -> Dict[str, float]:
        """Extract numeric values from a position for delta tracking."""
        return {
            "mkt_value": getattr(pos, "market_value", 0) or 0,
            "pnl": getattr(pos, "daily_pnl", 0) or 0,
            "upnl": getattr(pos, "unrealized_pnl", 0) or 0,
            "delta_$": getattr(pos, "delta_dollars", 0) or 0,
            "delta": getattr(pos, "delta", 0) or 0,
            "gamma": getattr(pos, "gamma", 0) or 0,
            "vega": getattr(pos, "vega", 0) or 0,
            "theta": getattr(pos, "theta", 0) or 0,
            "mark_price": getattr(pos, "mark_price", 0) or 0,
        }

    def apply_deltas(self, deltas: Dict[str, "PositionDeltaEvent"]) -> List[CellUpdate]:
        """
        Apply position deltas and return cell updates.

        Fast path for streaming updates - O(n) where n is number of deltas.
        Skips symbols not in the current view.

        Args:
            deltas: Dict mapping symbol -> PositionDeltaEvent

        Returns:
            List of CellUpdate objects for changed cells.
        """
        cell_updates: List[CellUpdate] = []

        for symbol, delta in deltas.items():
            row_keys = self._symbol_to_row_keys.get(symbol, [])
            if not row_keys:
                continue  # Symbol not in current view

            for row_key in row_keys:
                if row_key not in self._values_cache:
                    continue

                values = self._values_cache[row_key]

                # Apply delta changes to cached values
                values["pnl"] = values.get("pnl", 0) + delta.daily_pnl_change
                values["upnl"] = values.get("upnl", 0) + delta.pnl_change
                values["delta"] = values.get("delta", 0) + delta.delta_change
                values["gamma"] = values.get("gamma", 0) + delta.gamma_change
                values["vega"] = values.get("vega", 0) + delta.vega_change
                values["theta"] = values.get("theta", 0) + delta.theta_change
                values["mkt_value"] = values.get("mkt_value", 0) + delta.notional_change
                # Update mark_price and delta_$ from delta event
                values["mark_price"] = delta.new_mark_price
                values["delta_$"] = values.get("delta_$", 0) + delta.delta_dollars_change

                # Compute cell updates based on row type
                updates = self._compute_delta_cell_updates(row_key, values, delta)
                cell_updates.extend(updates)

        return cell_updates

    def _compute_delta_cell_updates(
        self,
        row_key: str,
        values: Dict[str, float],
        delta: "PositionDeltaEvent",
    ) -> List[CellUpdate]:
        """
        Compute cell updates for a single row after delta application.

        Column indices differ between consolidated and detailed views.
        """
        updates: List[CellUpdate] = []

        if self.consolidated:
            # Consolidated view: Ticker, Qty, Spot, Beta, Mkt Value, P&L, UP&L, Delta $, D, G, V, Th
            # Indices:              0      1     2     3       4        5      6       7    8  9 10 11

            # Update Spot column (index 2) with underlying price
            # Skip aggregate rows (__portfolio__, __total__) as Spot is meaningless for them
            if not row_key.startswith("__"):
                updates.append(
                    CellUpdate(
                        row_key,
                        2,
                        self._fmt_price(delta.underlying_price, is_close=False, decimals=2),
                    )
                )

            updates.extend(
                [
                    CellUpdate(row_key, 4, self._fmt_number(values["mkt_value"])),
                    CellUpdate(row_key, 5, self._fmt_number(values["pnl"], color=True)),
                    CellUpdate(row_key, 6, self._fmt_number(values["upnl"], color=True)),
                    CellUpdate(row_key, 7, self._fmt_number(values["delta_$"])),  # Delta $ column
                    CellUpdate(row_key, 8, self._fmt_number(values["delta"])),
                    CellUpdate(row_key, 9, self._fmt_number(values["gamma"])),
                    CellUpdate(row_key, 10, self._fmt_number(values["vega"])),
                    CellUpdate(row_key, 11, self._fmt_number(values["theta"])),
                ]
            )
        else:
            # Detailed view: Ticker, Qty, Spot, IV, Beta, Mkt Value, P&L, UP&L, Delta $, D, G, V, Th
            # Indices:           0      1     2    3    4       5        6      7       8    9 10 11 12

            # Position rows get mark price update too
            if row_key.startswith("pos-"):
                # Update mark price in Spot column (index 2)
                is_close = False  # Delta events are live, not close prices
                decimals = 3 if "-stock" not in row_key else 2
                updates.append(
                    CellUpdate(
                        row_key, 2, self._fmt_price(delta.new_mark_price, is_close, decimals)
                    )
                )

            updates.extend(
                [
                    CellUpdate(row_key, 5, self._fmt_number(values["mkt_value"])),
                    CellUpdate(row_key, 6, self._fmt_number(values["pnl"], color=True)),
                    CellUpdate(row_key, 7, self._fmt_number(values["upnl"], color=True)),
                    CellUpdate(row_key, 8, self._fmt_number(values["delta_$"])),  # Delta $ column
                    CellUpdate(row_key, 9, self._fmt_number(values["delta"])),
                    CellUpdate(row_key, 10, self._fmt_number(values["gamma"])),
                    CellUpdate(row_key, 11, self._fmt_number(values["vega"])),
                    CellUpdate(row_key, 12, self._fmt_number(values["theta"])),
                ]
            )

        return updates
