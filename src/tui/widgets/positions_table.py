"""
Positions table widget for displaying position data.

Two modes:
- Consolidated: Groups positions by underlying (for Summary view)
- Detailed: Shows individual positions under underlying headers (for IB/Futu views)

Uses PositionViewModel for business logic and incremental updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable

from ..viewmodels.position_vm import PositionViewModel

if TYPE_CHECKING:
    from ...domain.events.domain_events import PositionDeltaEvent


class PositionsTable(DataTable):
    """
    DataTable for displaying positions with selection support.

    Supports two display modes:
    - consolidated=True: Groups by underlying, shows aggregated metrics (Summary view)
    - consolidated=False: Shows individual positions under underlying headers (IB/Futu views)

    Uses PositionViewModel for data transformation and incremental cell updates.
    """

    class PositionSelected(Message):
        """Posted when a position row is selected."""

        def __init__(self, symbol: str, underlying: str, position: Any) -> None:
            self.symbol = symbol
            self.underlying = underlying
            self.position = position
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

    # Reactive properties - use factory to avoid mutable default shared across instances
    positions: reactive[List[Any]] = reactive(list, init=False)

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

        # ViewModel for business logic and diff computation
        self._view_model = PositionViewModel(
            broker_filter=broker_filter,
            consolidated=consolidated,
            show_portfolio_row=show_portfolio_row,
        )

        # Column key mapping for update_cell
        self._column_keys: List[str] = []

        # Row tracking for cursor management
        self._row_keys: List[str] = []
        self._selected_row_key: Optional[str] = None
        self._selected_role: Optional[str] = None
        self._selected_display_name: Optional[str] = None
        self._selected_underlying: Optional[str] = None

    def on_mount(self) -> None:
        """Set up columns when widget is mounted."""
        columns = self.COLUMNS_CONSOLIDATED if self.consolidated else self.COLUMNS_DETAILED
        self._column_keys.clear()
        for idx, (name, width) in enumerate(columns):
            col_key = f"col-{idx}"
            self.add_column(name, width=width, key=col_key)
            self._column_keys.append(col_key)

    def watch_positions(self, positions: List[Any]) -> None:
        """React to position changes with incremental updates when possible."""
        if not positions:
            self._full_rebuild([])
            return

        if self._view_model.full_refresh_needed(positions):
            self._full_rebuild(positions)
        else:
            self._incremental_update(positions)

    def _full_rebuild(self, positions: List[Any]) -> None:
        """Full table rebuild (first load or structural change)."""
        selected_key = self._selected_row_key or self._get_cursor_row_key()
        self.clear()
        self._row_keys.clear()

        if not positions:
            self._clear_selection()
            self._view_model.invalidate()
            return

        # Get display data from ViewModel
        display_data = self._view_model.compute_display_data(positions)
        row_order = self._view_model.get_row_order(positions)

        # Add rows in order
        for row_key in row_order:
            if row_key in display_data:
                values = display_data[row_key]
                self.add_row(*values, key=row_key)
                self._row_keys.append(row_key)

        self._restore_cursor(selected_key)

    def _incremental_update(self, positions: List[Any]) -> None:
        """Incremental cell-level updates (efficient path)."""
        row_ops, cell_updates, new_order = self._view_model.compute_updates(positions)

        # Handle row removals first
        for row_op in row_ops:
            if row_op.action == "remove":
                try:
                    self.remove_row(row_op.row_key)
                    if row_op.row_key in self._row_keys:
                        self._row_keys.remove(row_op.row_key)
                except Exception as e:
                    self.log.error(f"Failed to remove row {row_op.row_key}: {e}")

        # Handle row additions
        for row_op in row_ops:
            if row_op.action == "add" and row_op.values:
                try:
                    self.add_row(*row_op.values, key=row_op.row_key)
                    self._row_keys.append(row_op.row_key)
                except Exception as e:
                    self.log.error(f"Failed to add row {row_op.row_key}: {e}")

        # Handle cell updates (the efficient path)
        for cell in cell_updates:
            try:
                if cell.column_index < len(self._column_keys):
                    col_key = self._column_keys[cell.column_index]
                    self.update_cell(cell.row_key, col_key, cell.value)
            except Exception as e:
                self.log.error(f"Failed to update cell {cell.row_key}: {e}")

        # Update row order tracking
        # NOTE: Textual DataTable doesn't support row reordering natively.
        # This updates our internal tracking but displayed rows stay in insertion order.
        # Full rebuild would be needed for visual reordering (causes flicker).
        self._row_keys = [k for k in new_order if k in self._row_keys]

    def apply_deltas(self, deltas: Dict[str, "PositionDeltaEvent"]) -> None:
        """
        Apply position deltas for streaming updates (O(1) per delta).

        Fast path that updates specific cells without full table refresh.
        Skips deltas for symbols not currently in the table.

        Args:
            deltas: Dict mapping symbol -> PositionDeltaEvent
        """
        if not deltas:
            return

        # Get cell updates from ViewModel
        cell_updates = self._view_model.apply_deltas(deltas)

        # Apply cell updates directly
        for cell in cell_updates:
            try:
                if cell.column_index < len(self._column_keys):
                    col_key = self._column_keys[cell.column_index]
                    self.update_cell(cell.row_key, col_key, cell.value)
            except Exception as e:
                self.log.error(f"Failed to apply delta cell update {cell.row_key}: {e}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key is not None:
            self._emit_position_selected(event.row_key)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlight to keep selection in sync with cursor."""
        if event.row_key is not None:
            self._emit_position_selected(event.row_key)

    def _emit_position_selected(self, row_key) -> None:
        """Post selection message for a row key."""
        key = str(row_key.value)
        if key in ("__portfolio__", "__total__"):
            return

        pos = self._view_model.get_position_for_key(key)
        if pos:
            underlying = getattr(pos, "underlying", None) or getattr(pos, "symbol", "?")
            symbol = getattr(pos, "symbol", underlying)
            self._selected_row_key = key
            self._selected_role = self._row_role_from_key(key)
            self._selected_display_name = self._position_display_name(pos)
            self._selected_underlying = underlying
            self.post_message(
                self.PositionSelected(symbol=symbol, underlying=underlying, position=pos)
            )

    def get_selected_underlying(self) -> Optional[str]:
        """Get the underlying of the currently selected row."""
        if self.cursor_row is None:
            return None

        try:
            if 0 <= self.cursor_row < len(self._row_keys):
                key = self._row_keys[self.cursor_row]
                pos = self._view_model.get_position_for_key(key)
                if pos:
                    return getattr(pos, "underlying", None) or getattr(pos, "symbol", None)
        except Exception as e:
            self.log.error(f"Failed to get selected underlying: {e}")
        return None

    def get_selected_position(self) -> Optional[Any]:
        """Get the position object for the currently selected row."""
        if self.cursor_row is None:
            return None
        try:
            if 0 <= self.cursor_row < len(self._row_keys):
                key = self._row_keys[self.cursor_row]
                return self._view_model.get_position_for_key(key)
        except Exception as e:
            self.log.error(f"Failed to get selected position: {e}")
        return None

    def get_underlyings(self) -> List[str]:
        """Get list of underlyings in display order."""
        return self._view_model.get_underlying_order()

    def _get_cursor_row_key(self) -> Optional[str]:
        """Get row key from current cursor position."""
        if self.cursor_row is None:
            return None
        if 0 <= self.cursor_row < len(self._row_keys):
            return self._row_keys[self.cursor_row]
        return None

    def _restore_cursor(self, row_key: Optional[str]) -> None:
        """Restore cursor to a previously selected row key if present."""
        candidate_key = row_key if row_key in self._row_keys else None
        if candidate_key is None:
            candidate_key = self._find_row_key_for_selection()
        if candidate_key and candidate_key in self._row_keys:
            row_index = self._row_keys.index(candidate_key)
            self.move_cursor(row=row_index, column=0, scroll=False)
            self._selected_row_key = candidate_key

    def _find_row_key_for_selection(self) -> Optional[str]:
        """Find a matching row key based on the last selection details."""
        if self._selected_role == "pos" and self._selected_display_name:
            for key in self._row_keys:
                if not key.startswith("pos-"):
                    continue
                pos = self._view_model.get_position_for_key(key)
                if pos and self._position_display_name(pos) == self._selected_display_name:
                    return key

        if self._selected_underlying:
            header_key = f"header-{self._selected_underlying}"
            if header_key in self._row_keys:
                return header_key
            underlying_key = f"underlying-{self._selected_underlying}"
            if underlying_key in self._row_keys:
                return underlying_key

        return None

    def _position_display_name(self, pos: Any) -> str:
        """Return a stable display name for a position."""
        if hasattr(pos, "get_display_name"):
            return pos.get_display_name()
        return getattr(pos, "symbol", "?")

    def _row_role_from_key(self, row_key: str) -> Optional[str]:
        """Classify row key for selection restore."""
        if row_key.startswith("pos-"):
            return "pos"
        if row_key.startswith("header-"):
            return "header"
        if row_key.startswith("underlying-"):
            return "underlying"
        return None

    def _clear_selection(self) -> None:
        """Clear selection tracking when no positions are available."""
        self._selected_row_key = None
        self._selected_role = None
        self._selected_display_name = None
        self._selected_underlying = None
