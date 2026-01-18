"""
Risk signals table widget for full-screen display.

Shows all risk signals with detailed information:
- Status, Severity, Symbol, Layer, Rule, Current, Limit, Breach %, Action, Times

Uses SignalViewModel for persistence tracking and incremental updates.
"""

from __future__ import annotations

from typing import Any, List, Optional

from textual.reactive import reactive
from textual.widgets import DataTable

from ..viewmodels.signal_vm import SignalViewModel


class SignalsTable(DataTable):
    """Full-screen risk signals display with incremental updates."""

    COLUMNS = [
        ("Status", 10),
        ("Severity", 10),
        ("Symbol", 12),
        ("Layer", 8),
        ("Trigger Rule", 30),
        ("Current", 12),
        ("Limit", 12),
        ("Breach %", 10),
        ("Action", 15),
        ("First Seen", 10),
        ("Last Seen", 10),
    ]

    # Reactive state - use factory to avoid mutable default shared across instances
    signals: reactive[List[Any]] = reactive(list, init=False)
    snapshot: reactive[Optional[Any]] = reactive(lambda: None, init=False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(cursor_type="row", **kwargs)

        # ViewModel for persistence tracking and diff computation
        self._view_model = SignalViewModel(retention_seconds=300)

        # Column key mapping for update_cell
        self._column_keys: List[str] = []

        # Row tracking
        self._row_keys: List[str] = []

    def on_mount(self) -> None:
        """Set up columns when mounted."""
        self._column_keys.clear()
        for idx, (name, width) in enumerate(self.COLUMNS):
            col_key = f"col-{idx}"
            self.add_column(name, width=width, key=col_key)
            self._column_keys.append(col_key)

    def watch_signals(self, signals: List[Any]) -> None:
        """Update display when signals change."""
        if self._view_model.full_refresh_needed(signals):
            self._full_rebuild(signals)
        else:
            self._incremental_update(signals)

    def _full_rebuild(self, signals: List[Any]) -> None:
        """Full table rebuild."""
        self.clear()
        self._row_keys.clear()

        # Get display data from ViewModel
        display_data = self._view_model.compute_display_data(signals)
        row_order = self._view_model.get_row_order(signals)

        # Add rows in order
        for row_key in row_order:
            if row_key in display_data:
                values = display_data[row_key]
                self.add_row(*values, key=row_key)
                self._row_keys.append(row_key)

    def _incremental_update(self, signals: List[Any]) -> None:
        """Incremental cell-level updates."""
        row_ops, cell_updates, new_order = self._view_model.compute_updates(signals)

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

        # Handle cell updates
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
