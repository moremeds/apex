"""
Unified Signals view for Tab 2.

Combines:
- Risk signals table (breach alerts, Greeks warnings)
- Trading signals view (watchlist + indicator signals + confluence)

Key bindings:
- 1/r: Switch to Risk panel
- 2: Switch to Trading panel
- t: Cycle timeframe (Trading panel only)
- w/s: Move watchlist cursor
- h: Toggle history mode
- c: Clear signals
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import ContentSwitcher, Static

from ..widgets.confluence_panel import ConfluencePanel
from ..widgets.signals_table import SignalsTable
from ..widgets.trading_signals_table import TradingSignalsTable, extract_signal_metadata
from ..widgets.watchlist_panel import WatchlistPanel

if TYPE_CHECKING:
    from ...models.risk_snapshot import RiskSnapshot
    from ...models.risk_signal import RiskSignal


class UnifiedSignalsView(Container, can_focus=True):
    """
    Unified signals view with Risk/Trading panel switcher.

    Provides:
    - Risk panel: Full-screen risk signals table
    - Trading panel: Watchlist + trading signals + confluence
    """

    DEFAULT_CSS = """
    UnifiedSignalsView {
        height: 1fr;
        width: 1fr;
        layout: vertical;
    }

    #signals-header {
        height: 1;
        width: 1fr;
        background: $surface;
        padding: 0 1;
    }

    #signals-header-title {
        width: 1fr;
    }

    #signals-switcher {
        height: 1fr;
        width: 1fr;
    }

    /* Risk panel - full container */
    #risk-panel {
        height: 1fr;
        width: 1fr;
    }

    /* Trading panel layout */
    #trading-panel {
        height: 1fr;
        width: 1fr;
    }

    #trading-main {
        height: 1fr;
        width: 1fr;
    }

    #trading-left {
        width: 1fr;
        min-width: 20;
    }

    #trading-right {
        width: 2fr;
    }

    #trading-signals-panel {
        height: 2fr;
    }

    #trading-confluence-panel {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("1", "show_risk", "Risk", show=True),
        Binding("r", "show_risk", "Risk", show=False),
        Binding("2", "show_trading", "Trading", show=True),
        Binding("t", "cycle_timeframe", "TF", show=True),
        Binding("w", "move_up", "Up", show=True),
        Binding("s", "move_down", "Down", show=True),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("c", "clear_signals", "Clear", show=True),
        Binding("h", "toggle_history", "History", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Trading signal cache (unfiltered)
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

    def on_mount(self) -> None:
        """Default to risk panel on mount."""
        self._set_active_panel("risk-panel")

    def compose(self) -> ComposeResult:
        """Compose the unified signals layout."""
        with Horizontal(id="signals-header"):
            yield Static("", id="signals-header-title")
        with ContentSwitcher(id="signals-switcher", initial="risk-panel"):
            # Risk signals panel - Vertical to ensure proper layout fill
            with Vertical(id="risk-panel"):
                yield SignalsTable(id="signals-table")
            # Trading signals panel
            with Vertical(id="trading-panel"):
                with Horizontal(id="trading-main"):
                    # Left side - Watchlist (~33%)
                    with Vertical(id="trading-left"):
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
    # Public API for Risk Signals (from orchestrator snapshot)
    # -------------------------------------------------------------------------

    def update_data(
        self,
        signals: Optional[List["RiskSignal"]],
        snapshot: Optional["RiskSnapshot"],
    ) -> None:
        """
        Update the view with new risk signal data.

        Args:
            signals: List of risk signals.
            snapshot: Current risk snapshot for context.
        """
        try:
            signals_table = self.query_one("#signals-table", SignalsTable)
            signals_table.signals = signals or []
            signals_table.snapshot = snapshot
        except Exception:
            self.log.exception("Failed to update risk signals table")

    # -------------------------------------------------------------------------
    # Public API for Trading Signals (from event bus)
    # -------------------------------------------------------------------------

    def add_trading_signal(self, signal: Any) -> None:
        """
        Add a single trading signal to the cache.

        Args:
            signal: TradingSignal or TradingSignalEvent
        """
        self._signals.insert(0, signal)
        if len(self._signals) > self._max_signals:
            self._signals = self._signals[: self._max_signals]
        self.log.info(f"Signal received: {getattr(signal, 'symbol', '?')} - Total cached: {len(self._signals)}")
        self._refresh_trading_signals()

    def add_signal(self, signal: Any) -> None:
        """Backwards-compatible alias for add_trading_signal."""
        self.add_trading_signal(signal)

    def update_signals(self, signals: Optional[List[Any]]) -> None:
        """
        Batch update trading signal cache.

        Args:
            signals: List of TradingSignal or TradingSignalEvent objects
        """
        if signals is None:
            return
        self._signals = list(signals)[-self._max_signals :]
        self._refresh_trading_signals()

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
        Batch update confluence score cache.

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
        Batch update MTF alignment cache.

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

    def refresh_view(self) -> None:
        """Refresh trading signals and confluence based on current selection."""
        self._refresh_trading_signals()
        self._refresh_confluence()

    def get_selected_symbol(self) -> Optional[str]:
        """Get currently selected symbol from watchlist."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            return watchlist.get_selected_symbol()
        except Exception:
            return None

    def get_selected_timeframe(self) -> str:
        """Get currently selected timeframe from watchlist."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            return watchlist.get_selected_timeframe()
        except Exception:
            return "1h"

    # -------------------------------------------------------------------------
    # Watchlist event handlers
    # -------------------------------------------------------------------------

    def on_watchlist_panel_symbol_selected(
        self, event: WatchlistPanel.SymbolSelected
    ) -> None:
        """Handle symbol selection changes."""
        self._refresh_trading_signals()
        self._refresh_confluence()

    def on_watchlist_panel_timeframe_changed(
        self, event: WatchlistPanel.TimeframeChanged
    ) -> None:
        """Handle timeframe changes."""
        self._refresh_trading_signals()
        self._refresh_confluence()

    # -------------------------------------------------------------------------
    # Actions - Panel switching
    # -------------------------------------------------------------------------

    def action_show_risk(self) -> None:
        """Show the risk signals panel."""
        self._set_active_panel("risk-panel")

    def action_show_trading(self) -> None:
        """Show the trading signals panel."""
        self._set_active_panel("trading-panel")

    # -------------------------------------------------------------------------
    # Actions - Trading panel (only active when trading panel shown)
    # -------------------------------------------------------------------------

    def action_move_up(self) -> None:
        """Move selection up in watchlist."""
        if not self._is_trading_active():
            return
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            watchlist.move_cursor(-1)
        except Exception:
            self.log.exception("Failed to move up in watchlist")

    def action_move_down(self) -> None:
        """Move selection down in watchlist."""
        if not self._is_trading_active():
            return
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            watchlist.move_cursor(1)
        except Exception:
            self.log.exception("Failed to move down in watchlist")

    def action_cycle_timeframe(self) -> None:
        """Cycle watchlist timeframe."""
        if not self._is_trading_active():
            return
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            watchlist.cycle_timeframe()
        except Exception:
            self.log.exception("Failed to cycle timeframe")

    def action_clear_signals(self) -> None:
        """Clear all trading signals."""
        if not self._is_trading_active():
            return
        self._signals.clear()
        self._refresh_trading_signals()
        self.notify("Signals cleared", severity="information", timeout=2.0)

    def action_toggle_history(self) -> None:
        """Toggle between history and current state view."""
        if not self._is_trading_active():
            return
        self._history_mode = not self._history_mode
        self._update_trading_title()
        self._refresh_trading_signals()

        mode_name = "History" if self._history_mode else "Current State"
        self.notify(f"View: {mode_name}", severity="information", timeout=2.0)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _set_active_panel(self, panel_id: str) -> None:
        """Switch the ContentSwitcher to the requested panel."""
        try:
            switcher = self.query_one("#signals-switcher", ContentSwitcher)
            switcher.current = panel_id
            self._update_header()
            if panel_id == "trading-panel":
                self.refresh_view()
        except Exception:
            self.log.exception("Failed to switch signals panel")

    def _is_trading_active(self) -> bool:
        """Check if trading panel is currently active."""
        try:
            switcher = self.query_one("#signals-switcher", ContentSwitcher)
            return switcher.current == "trading-panel"
        except Exception:
            return False

    def _update_header(self) -> None:
        """Update header bar to reflect active panel."""
        try:
            title = self.query_one("#signals-header-title", Static)
            is_trading = self._is_trading_active()
            risk_label = "[bold #ff6b6b]Risk[/]" if not is_trading else "Risk"
            trading_label = "[bold #7ee787]Trading[/]" if is_trading else "Trading"
            title.update(f"[1] {risk_label}   [2] {trading_label}")
        except Exception:
            pass

    def _update_trading_title(self) -> None:
        """Update trading panel title to reflect current mode."""
        try:
            title = self.query_one("#trading-signals-title", Static)
            mode_indicator = (
                "[dim](History)[/]" if self._history_mode else "[green](Current)[/]"
            )
            title.update(f"Trading Signals {mode_indicator}")
        except Exception:
            pass

    def _refresh_trading_signals(self) -> None:
        """Filter and display signals based on watchlist selection and mode."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
            table = self.query_one("#trading-signals-table", TradingSignalsTable)
        except Exception:
            return

        self._ensure_timeframe_alignment()
        selected_symbol = watchlist.get_selected_symbol()
        selected_tf = watchlist.get_selected_timeframe()

        # Filter signals: when no symbol selected, show ALL signals
        filtered = []
        for signal in self._signals:
            symbol = getattr(signal, "symbol", None)

            # Skip mismatched symbols; show symbol-agnostic signals (symbol=None/"") always
            if selected_symbol and symbol and symbol != selected_symbol:
                continue

            # Filter by timeframe (only if both signal timeframe and selection exist)
            timeframe, _ = extract_signal_metadata(signal)
            if selected_tf and timeframe and timeframe != selected_tf:
                continue

            filtered.append(signal)

        # In Current State mode, keep only latest per (symbol, indicator)
        if not self._history_mode:
            filtered = self._dedupe_to_current_state(filtered)

        table.signals = filtered

    def _ensure_timeframe_alignment(self) -> None:
        """Align watchlist timeframe with available confluence/signal data."""
        try:
            watchlist = self.query_one("#trading-watchlist", WatchlistPanel)
        except Exception:
            return

        symbol = watchlist.get_selected_symbol()
        if not symbol:
            return

        selected_tf = watchlist.get_selected_timeframe()
        if self._has_data_for(symbol, selected_tf):
            return

        best_tf = self._best_timeframe_for_symbol(symbol)
        if best_tf and best_tf != selected_tf:
            watchlist.select_timeframe(best_tf)

    def _has_data_for(self, symbol: str, timeframe: str) -> bool:
        """Check whether signals or confluence exist for symbol/timeframe."""
        if (symbol, timeframe) in self._confluence_scores:
            return True

        for signal in self._signals:
            if getattr(signal, "symbol", None) != symbol:
                continue
            signal_tf, _ = extract_signal_metadata(signal)
            if signal_tf == timeframe:
                return True

        return False

    def _best_timeframe_for_symbol(self, symbol: str) -> Optional[str]:
        """Pick a timeframe with data, preferring the latest confluence score."""
        best_tf: Optional[str] = None
        best_ts = None

        for (sym, tf), score in self._confluence_scores.items():
            if sym != symbol:
                continue
            ts = getattr(score, "timestamp", None)
            if ts and (best_ts is None or ts > best_ts):
                best_ts = ts
                best_tf = tf
            elif best_tf is None:
                best_tf = tf

        if best_tf:
            return best_tf

        for signal in self._signals:
            if getattr(signal, "symbol", None) != symbol:
                continue
            signal_tf, _ = extract_signal_metadata(signal)
            if signal_tf:
                return signal_tf

        return None

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

        self._ensure_timeframe_alignment()
        symbol = watchlist.get_selected_symbol()
        timeframe = watchlist.get_selected_timeframe()

        # Get confluence score for selected symbol/timeframe
        score = None
        if symbol and timeframe:
            score = self._confluence_scores.get((symbol, timeframe))

        # Fallback: if no exact match, find any score for this symbol
        if score is None and symbol:
            for (s, _tf), sc in self._confluence_scores.items():
                if s == symbol:
                    score = sc
                    break

        # Get MTF alignment for symbol
        alignment = self._alignments.get(symbol) if symbol else None

        panel.score = score
        panel.alignment = alignment


# Backwards-compatible alias
SignalsView = UnifiedSignalsView
