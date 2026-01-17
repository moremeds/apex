"""
Trading signals table widget with color-coded direction display.

Displays trading signals from the signal engine with:
- Color coding: green (BUY), red (SELL), yellow (ALERT)
- Strength indicator bar
- Timeframe and indicator extraction from signal_id
- FIFO eviction when max_signals exceeded
- Incremental updates via TradingSignalViewModel (OPT-PERF)
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from textual.reactive import reactive
from textual.widgets import DataTable

from ..viewmodels.trading_signal_vm import TradingSignalViewModel


def extract_signal_metadata(signal: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract timeframe and indicator from a signal.

    First checks explicit attributes, then parses from signal_id.
    Signal ID format: "{category}:{indicator}:{symbol}:{timeframe}"

    Args:
        signal: TradingSignal or TradingSignalEvent

    Returns:
        Tuple of (timeframe, indicator)
    """
    # Try explicit attributes first
    timeframe = getattr(signal, "timeframe", None)
    indicator = getattr(signal, "indicator", None)

    # Fall back to parsing signal_id
    signal_id = getattr(signal, "signal_id", None)
    if isinstance(signal_id, str):
        parts = signal_id.split(":")
        if len(parts) >= 4:
            if not indicator:
                indicator = parts[1]
            if not timeframe:
                timeframe = parts[3]

    return (
        str(timeframe) if timeframe else None,
        str(indicator) if indicator else None,
    )


class TradingSignalsTable(DataTable):
    """
    Signal feed table with direction color coding.

    Displays trading signals with columns for time, symbol, direction,
    strength, timeframe, indicator, rule, and message.
    """

    COLUMNS = [
        ("Time", 9),
        ("Symbol", 8),
        ("Dir", 5),
        ("Str", 4),
        ("TF", 4),
        ("Indicator", 14),
        ("Rule", 24),
        ("Message", 50),
    ]

    # Reactive signal list
    signals: reactive[List[Any]] = reactive(list, init=False)

    def __init__(self, max_signals: int = 100, **kwargs: Any) -> None:
        """
        Initialize the trading signals table.

        Args:
            max_signals: Maximum signals to display (FIFO eviction)
        """
        super().__init__(cursor_type="row", zebra_stripes=True, **kwargs)
        self._max_signals = max_signals
        self._column_keys: List[str] = []
        self._row_keys: List[str] = []

        # ViewModel for incremental updates (OPT-PERF)
        # Will be initialized in on_mount when app's display_tz is available
        self._view_model: Optional[TradingSignalViewModel] = None

    def on_mount(self) -> None:
        """Initialize table columns."""
        # Initialize viewmodel with app's display timezone
        display_tz = getattr(self.app, "display_tz", "Asia/Hong_Kong")
        self._view_model = TradingSignalViewModel(
            max_signals=self._max_signals,
            display_tz=display_tz,
        )

        self._column_keys.clear()
        for idx, (name, width) in enumerate(self.COLUMNS):
            col_key = f"col-{idx}"
            self.add_column(name, width=width, key=col_key)
            self._column_keys.append(col_key)

    def watch_signals(self, signals: List[Any]) -> None:
        """Update table when signals change using incremental updates (OPT-PERF)."""
        if not signals:
            self._full_rebuild([])
            return

        # Guard: viewmodel may not be initialized yet (before on_mount)
        if self._view_model is None:
            return

        # Use ViewModel to determine if full refresh is needed
        if self._view_model.full_refresh_needed(signals):
            self._full_rebuild(signals)
        else:
            self._incremental_update(signals)

    def add_signal(self, signal: Any) -> None:
        """
        Add a single signal to the table.

        Maintains max_signals limit with FIFO eviction.
        """
        current = list(self.signals)
        current.insert(0, signal)  # Newest first
        if len(current) > self._max_signals:
            current = current[: self._max_signals]
        self.signals = current

    def clear_signals(self) -> None:
        """Clear all signals from the table."""
        if self._view_model is not None:
            self._view_model.invalidate()
        self.signals = []

    def _full_rebuild(self, signals: List[Any]) -> None:
        """Full table rebuild (first load or structural change)."""
        self.clear()
        self._row_keys.clear()

        if not signals:
            self._add_placeholder_row()
            if self._view_model is not None:
                self._view_model.invalidate()
            return

        if self._view_model is None:
            return

        # Get display data from ViewModel
        display_data = self._view_model.compute_display_data(signals)
        row_order = self._view_model.get_row_order(signals)

        # Add rows in order
        for row_key in row_order:
            if row_key in display_data:
                self.add_row(*display_data[row_key], key=row_key)
                self._row_keys.append(row_key)

    def _incremental_update(self, signals: List[Any]) -> None:
        """Incremental cell-level updates (efficient path - OPT-PERF)."""
        if self._view_model is None:
            return
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

        # Handle cell updates (most efficient path)
        for cell in cell_updates:
            try:
                if cell.column_index < len(self._column_keys):
                    col_key = self._column_keys[cell.column_index]
                    self.update_cell(cell.row_key, col_key, cell.value)
            except Exception as e:
                self.log.error(f"Failed to update cell {cell.row_key}: {e}")

        # Update row order tracking
        self._row_keys = [
            k
            for k in new_order
            if k in self._row_keys or any(op.action == "add" and op.row_key == k for op in row_ops)
        ]

    def _add_placeholder_row(self) -> None:
        """Add a placeholder row when no signals exist."""
        placeholders = [
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]No signals[/]",
            "[dim]Waiting for events...[/]",
            "[dim]-[/]",
        ]
        self.add_row(*placeholders, key="_placeholder_empty")
        self._row_keys.append("_placeholder_empty")
