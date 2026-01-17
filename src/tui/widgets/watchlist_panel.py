"""
Watchlist panel for trading signals view.

Displays a symbol list from the trading signal universe with
timeframe selector toggle. Emits selection events for filtering
signals in the right panel.

Layout:
- Timeframe toggle bar (1m, 5m, 15m, 1h, 4h, 1d)
- Symbol DataTable with cursor navigation
- Keyboard hints
"""

from __future__ import annotations

from typing import Dict, List, Optional

from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable, Static


class WatchlistPanel(Widget):
    """
    Universe watchlist with timeframe toggle and symbol selection.

    Emits:
        SymbolSelected: When a symbol row is highlighted/selected
        TimeframeChanged: When timeframe toggle changes
    """

    class SymbolSelected(Message):
        """Posted when a symbol is selected or highlighted."""

        def __init__(self, symbol: str, timeframe: str) -> None:
            self.symbol = symbol
            self.timeframe = timeframe
            super().__init__()

    class TimeframeChanged(Message):
        """Posted when the timeframe toggle changes."""

        def __init__(self, timeframe: str) -> None:
            self.timeframe = timeframe
            super().__init__()

    # Standard timeframes in display order (short to long)
    TIMEFRAMES: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # Preferred timeframes for default selection (1d first for trading signals)
    PREFERRED_TIMEFRAMES: List[str] = ["1d", "4h", "1h", "15m", "5m", "1m"]

    # Reactive state
    symbols: reactive[List[str]] = reactive(list, init=False)
    selected_symbol: reactive[Optional[str]] = reactive(None, init=False)
    selected_timeframe: reactive[str] = reactive("1d", init=False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Universe map: timeframe -> symbols
        self._symbols_by_timeframe: Dict[str, List[str]] = {}
        # Track row keys for cursor positioning
        self._row_keys: List[str] = []

    def compose(self) -> ComposeResult:
        """Compose the watchlist layout."""
        yield Static("Watchlist", id="watchlist-title", classes="panel-title")
        yield Static("", id="watchlist-timeframes")
        yield DataTable(id="watchlist-table", cursor_type="row")
        yield Static(
            "w/s: select   t: timeframe",
            id="watchlist-hints",
            classes="panel-hints",
        )

    def on_mount(self) -> None:
        """Initialize table columns and render timeframe toggle."""
        table = self.query_one("#watchlist-table", DataTable)
        table.add_column("Symbol", width=12, key="symbol")
        self._render_timeframe_toggle()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_symbols_by_timeframe(self, symbols_by_tf: Dict[str, List[str]]) -> None:
        """
        Set the full universe map of timeframe -> symbols.

        Args:
            symbols_by_tf: Mapping from timeframe (e.g., "1h") to symbol list
        """
        self._symbols_by_timeframe = {tf: sorted(set(syms)) for tf, syms in symbols_by_tf.items()}
        # If current timeframe has no symbols, switch to first available
        if self.selected_timeframe not in self._symbols_by_timeframe:
            self.selected_timeframe = self._first_available_timeframe()
        else:
            # Trigger symbol list update
            self.symbols = list(self._symbols_by_timeframe.get(self.selected_timeframe, []))

    def set_symbols(self, symbols: List[str]) -> None:
        """
        Set symbols for the current timeframe only.

        Args:
            symbols: List of symbols to display
        """
        self._symbols_by_timeframe[self.selected_timeframe] = list(symbols)
        self.symbols = list(symbols)

    def select_timeframe(self, timeframe: str) -> None:
        """
        Select a specific timeframe if valid.

        Args:
            timeframe: Timeframe string (e.g., "1h")
        """
        if timeframe in self.TIMEFRAMES:
            self.selected_timeframe = timeframe

    def cycle_timeframe(self, delta: int = 1) -> None:
        """
        Cycle to next/previous timeframe.

        Args:
            delta: Direction (+1 forward, -1 backward)
        """
        try:
            idx = self.TIMEFRAMES.index(self.selected_timeframe)
        except ValueError:
            idx = 0
        next_idx = (idx + delta) % len(self.TIMEFRAMES)
        self.selected_timeframe = self.TIMEFRAMES[next_idx]

    def move_cursor(self, delta: int) -> None:
        """
        Move selection cursor by delta rows.

        Args:
            delta: Number of rows to move (+1 down, -1 up)
        """
        table = self.query_one("#watchlist-table", DataTable)
        if table.row_count == 0:
            return
        current = table.cursor_row or 0
        next_row = max(0, min(current + delta, table.row_count - 1))
        table.move_cursor(row=next_row)

    def get_selected_symbol(self) -> Optional[str]:
        """Get the currently selected symbol."""
        return self.selected_symbol

    def get_selected_timeframe(self) -> str:
        """Get the currently selected timeframe."""
        return self.selected_timeframe

    # -------------------------------------------------------------------------
    # Reactive watchers
    # -------------------------------------------------------------------------

    def watch_selected_timeframe(self, timeframe: str) -> None:
        """Update display and symbols when timeframe changes."""
        self._render_timeframe_toggle()
        symbols = self._symbols_by_timeframe.get(timeframe, [])
        self.symbols = list(symbols)
        self.post_message(self.TimeframeChanged(timeframe))

    def watch_symbols(self, symbols: List[str]) -> None:
        """Rebuild the symbol table when data changes."""
        self._rebuild_table(symbols)

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle explicit row selection (Enter key)."""
        self._handle_row_event(event.row_key)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlight changes (cursor movement)."""
        self._handle_row_event(event.row_key)

    def _handle_row_event(self, row_key) -> None:
        """Process row selection/highlight event."""
        if row_key is None:
            return
        key_str = str(row_key.value) if hasattr(row_key, "value") else str(row_key)
        # Skip placeholder rows
        if key_str.startswith("_placeholder"):
            return
        self.selected_symbol = key_str
        self.post_message(self.SymbolSelected(key_str, self.selected_timeframe))

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _rebuild_table(self, symbols: List[str]) -> None:
        """Rebuild the symbols table with new data."""
        table = self.query_one("#watchlist-table", DataTable)
        prev_selected = self.selected_symbol
        table.clear()
        self._row_keys.clear()

        if not symbols:
            table.add_row("[dim]No symbols[/]", key="_placeholder_empty")
            return

        for symbol in symbols:
            table.add_row(symbol, key=symbol)
            self._row_keys.append(symbol)

        # Restore selection if symbol still exists
        if prev_selected and prev_selected in self._row_keys:
            idx = self._row_keys.index(prev_selected)
            table.move_cursor(row=idx)
        elif self._row_keys:
            # Select first symbol
            table.move_cursor(row=0)
            self.selected_symbol = self._row_keys[0]
            self.post_message(self.SymbolSelected(self._row_keys[0], self.selected_timeframe))

    def _render_timeframe_toggle(self) -> None:
        """Render the timeframe toggle bar with active highlight."""
        tokens = []
        for tf in self.TIMEFRAMES:
            if tf == self.selected_timeframe:
                tokens.append(f"[bold #5fd7ff]{tf}[/]")
            else:
                tokens.append(f"[#8b949e]{tf}[/]")
        text = "TF: " + "  ".join(tokens)
        try:
            self.query_one("#watchlist-timeframes", Static).update(text)
        except Exception:
            pass  # Widget not mounted yet

    def _first_available_timeframe(self) -> str:
        """Get first available timeframe, preferring longer ones (1d first)."""
        for tf in self.PREFERRED_TIMEFRAMES:
            if tf in self._symbols_by_timeframe and self._symbols_by_timeframe[tf]:
                return tf
        return "1d"  # Default to daily
