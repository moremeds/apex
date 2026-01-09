"""
Trading Signals view for Tab 6.

Layout matching the Lab view pattern:
- Left (~33%): Watchlist panel with timeframe toggle
- Right top (~67%): Trading signals table
- Right bottom: Confluence score panel

Keyboard shortcuts:
- w/s or Up/Down: Navigate watchlist
- t: Cycle timeframe
- c: Clear signals
- h: Toggle history view (Current State vs History)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static

from ..widgets.watchlist_panel import WatchlistPanel
from ..widgets.trading_signals_table import TradingSignalsTable, extract_signal_metadata
from ..widgets.confluence_panel import ConfluencePanel


class TradingSignalsView(Container, can_focus=True):
    """
    Trading signals view with watchlist and signal feed.

    Provides:
    - Left panel: Symbol watchlist with timeframe selector
    - Right top: Filtered trading signals table
    - Right bottom: Confluence score for selected symbol/timeframe
    """

    DEFAULT_CSS = ""

    BINDINGS = [
        Binding("w", "move_up", "Up", show=True),
        Binding("s", "move_down", "Down", show=True),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("t", "cycle_timeframe", "TF", show=True),
        Binding("c", "clear_signals", "Clear", show=True),
        Binding("h", "toggle_history", "History", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Signal cache (all signals, unfiltered)
        self._signals: List[Any] = []
        # Confluence cache: (symbol, timeframe) -> ConfluenceScore
        self._confluence_scores: Dict[Tuple[str, str], Any] = {}
        # MTF alignment cache: symbol -> MTFAlignment
        self._alignments: Dict[str, Any] = {}
        # Max signals to retain
        self._max_signals = 500
        # History mode: True = full history, False = current state only
        self._history_mode: bool = True

    def on_show(self) -> None:
        """Focus this view when it becomes visible."""
        self.focus()

    def compose(self) -> ComposeResult:
        """Compose the trading signals layout."""
        with Horizontal(id="trading-main"):
            # Left side - Watchlist (~33%)
            with Vertical(id="trading-left"):
                with Vertical(id="watchlist-panel"):
                    yield WatchlistPanel(id="trading-watchlist")

            # Right side - Signals + Confluence (~67%)
            with Vertical(id="trading-right"):
                with Vertical(id="trading-signals-panel"):
                    yield Static(
                        "Trading Signals",
                        id="trading-signals-title",
                        classes="panel-title",
                    )
                    yield TradingSignalsTable(id="trading-signals-table")
                with Vertical(id="trading-confluence-panel"):
                    yield Static(
                        "Confluence",
                        id="trading-confluence-title",
                        classes="panel-title",
                    )
                    yield ConfluencePanel(id="trading-confluence")

    # -------------------------------------------------------------------------
    # Public API for Orchestrator integration
    # -------------------------------------------------------------------------

    def refresh_view(self) -> None:
        """Refresh signals and confluence based on current selection."""
        self._refresh_signals()
        self._refresh_confluence()

    def update_signals(self, signals: Optional[List[Any]]) -> None:
        """
        Update signal cache and refresh table.

        Args:
            signals: List of TradingSignal or TradingSignalEvent objects
        """
        if signals is None:
            return
        self._signals = list(signals)[-self._max_signals :]
        self._refresh_signals()

    def add_signal(self, signal: Any) -> None:
        """
        Add a single signal to the cache.

        Args:
            signal: TradingSignal or TradingSignalEvent
        """
        self._signals.insert(0, signal)
        if len(self._signals) > self._max_signals:
            self._signals = self._signals[: self._max_signals]
        self._refresh_signals()

    def update_confluence_score(self, score: Any) -> None:
        """
        Update a single confluence score.

        Args:
            score: ConfluenceScore object
        """
        symbol = getattr(score, "symbol", None)
        timeframe = getattr(score, "timeframe", None)
        if symbol and timeframe:
            self._confluence_scores[(symbol, timeframe)] = score
            self._refresh_confluence()

    def update_confluence_scores(self, scores: Optional[List[Any]]) -> None:
        """
        Update confluence score cache (batch).

        Args:
            scores: List of ConfluenceScore objects
        """
        if scores is None:
            return
        self._confluence_scores = {
            (getattr(s, "symbol", ""), getattr(s, "timeframe", "")): s
            for s in scores
        }
        self._refresh_confluence()

    def update_alignment(self, alignment: Any) -> None:
        """
        Update a single MTF alignment.

        Args:
            alignment: MTFAlignment object
        """
        symbol = getattr(alignment, "symbol", None)
        if symbol:
            self._alignments[symbol] = alignment
            self._refresh_confluence()

    def update_alignments(self, alignments: Optional[List[Any]]) -> None:
        """
        Update MTF alignment cache (batch).

        Args:
            alignments: List of MTFAlignment objects
        """
        if alignments is None:
            return
        self._alignments = {getattr(a, "symbol", ""): a for a in alignments}
        self._refresh_confluence()

    def set_universe(self, symbols_by_timeframe: Dict[str, List[str]]) -> None:
        """
        Load universe symbols into the watchlist.

        Args:
            symbols_by_timeframe: Mapping of timeframe -> symbol list
        """
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            watchlist.set_symbols_by_timeframe(symbols_by_timeframe)
        except Exception:
            self.log.exception("Failed to load watchlist symbols")

    def get_selected_symbol(self) -> Optional[str]:
        """Get currently selected symbol."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            return watchlist.get_selected_symbol()
        except Exception:
            return None

    def get_selected_timeframe(self) -> str:
        """Get currently selected timeframe."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            return watchlist.get_selected_timeframe()
        except Exception:
            return "1h"

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def on_watchlist_panel_symbol_selected(
        self, event: WatchlistPanel.SymbolSelected
    ) -> None:
        """Handle symbol selection changes."""
        self._refresh_signals()
        self._refresh_confluence()

    def on_watchlist_panel_timeframe_changed(
        self, event: WatchlistPanel.TimeframeChanged
    ) -> None:
        """Handle timeframe changes."""
        self._refresh_signals()
        self._refresh_confluence()

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_move_up(self) -> None:
        """Move selection up in watchlist."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            watchlist.move_cursor(-1)
        except Exception:
            self.log.exception("Failed to move up in watchlist")

    def action_move_down(self) -> None:
        """Move selection down in watchlist."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            watchlist.move_cursor(1)
        except Exception:
            self.log.exception("Failed to move down in watchlist")

    def action_cycle_timeframe(self) -> None:
        """Cycle watchlist timeframe."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            watchlist.cycle_timeframe()
        except Exception:
            self.log.exception("Failed to cycle timeframe")

    def action_clear_signals(self) -> None:
        """Clear all signals."""
        self._signals.clear()
        self._refresh_signals()
        self.notify("Signals cleared", severity="information", timeout=2.0)

    def action_toggle_history(self) -> None:
        """Toggle between history and current state view."""
        self._history_mode = not self._history_mode
        self._update_title()
        self._refresh_signals()

        mode_name = "History" if self._history_mode else "Current State"
        self.notify(f"View: {mode_name}", severity="information", timeout=2.0)

    def _update_title(self) -> None:
        """Update panel title to reflect current mode."""
        try:
            title = self.query_one("#trading-signals-title", Static)
            mode_indicator = "[dim](History)[/]" if self._history_mode else "[green](Current)[/]"
            title.update(f"Trading Signals {mode_indicator}")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _refresh_signals(self) -> None:
        """Filter and display signals based on watchlist selection and mode."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            table = self.query_one("#trading-signals-table", TradingSignalsTable)
        except Exception:
            return

        selected_symbol = watchlist.get_selected_symbol()
        selected_tf = watchlist.get_selected_timeframe()

        # Filter signals by symbol and timeframe
        filtered = []
        for signal in self._signals:
            symbol = getattr(signal, "symbol", None)
            if selected_symbol and symbol != selected_symbol:
                continue

            timeframe, _ = extract_signal_metadata(signal)
            if timeframe and selected_tf and timeframe != selected_tf:
                continue

            filtered.append(signal)

        # In Current State mode, keep only latest per (symbol, indicator)
        if not self._history_mode:
            filtered = self._dedupe_to_current_state(filtered)

        table.signals = filtered

    def _dedupe_to_current_state(self, signals: List[Any]) -> List[Any]:
        """
        Deduplicate signals to show only latest per (symbol, indicator).

        In Current State mode, we want to see the latest signal state
        for each unique symbol/indicator combination.
        """
        seen: Dict[Tuple[str, str], Any] = {}  # (symbol, indicator) -> signal

        for signal in signals:
            symbol = getattr(signal, "symbol", "")
            indicator = getattr(signal, "indicator", "")
            key = (symbol, indicator)

            # Keep the first occurrence (signals are already sorted newest-first)
            if key not in seen:
                seen[key] = signal

        return list(seen.values())

    def _refresh_confluence(self) -> None:
        """Refresh confluence panel based on current selection."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            panel = self.query_one("#trading-confluence", ConfluencePanel)
        except Exception:
            return

        symbol = watchlist.get_selected_symbol()
        timeframe = watchlist.get_selected_timeframe()

        # Get confluence score for selected symbol/timeframe
        score = None
        if symbol and timeframe:
            score = self._confluence_scores.get((symbol, timeframe))

        # Fallback: if no exact match, find any score for this symbol
        # This handles timeframe mismatch (e.g., computed for "1d" but watchlist shows "1h")
        if score is None and symbol:
            for (s, tf), sc in self._confluence_scores.items():
                if s == symbol:
                    score = sc
                    break

        # Get MTF alignment for symbol
        alignment = self._alignments.get(symbol) if symbol else None

        panel.score = score
        panel.alignment = alignment
