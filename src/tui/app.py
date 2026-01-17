"""
Apex Dashboard - Pure Textual Implementation.

Real-time terminal UI for risk monitoring with tabbed views:
- Tab 1: Account Summary (consolidated positions, summary, alerts, health)
- Tab 2: Signals (unified: risk signals + trading signals with panel switcher)
- Tab 3: IB Positions (detailed IB positions with ATR levels)
- Tab 4: Futu Positions (detailed Futu positions with ATR levels)
- Tab 5: Lab (backtest strategies with parameters and performance results)
- Tab 6: Data (historical coverage + indicator DB status)
- Tab 7: Signal Introspection (indicator states, rule evaluations, cooldowns)

Signal Persistence Integration:
- Loads historical signals from database on startup (non-blocking)
- Connects to PostgreSQL NOTIFY via SignalListener for real-time updates
- Ensures no signals are lost between startup and event bus subscription
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, TabbedContent, TabPane, Tabs

from .event_bus import TUIEventBus
from .views.data import DataRefreshRequested, DataView, IndicatorDetailsRequested
from .views.lab import LabView
from .views.positions import PositionsView
from .views.signal_introspection import SignalIntrospectionView
from .views.signals import SignalsView
from .views.summary import SummaryView
from .widgets.header import HeaderWidget

if TYPE_CHECKING:
    from ..domain.interfaces.signal_persistence import SignalPersistencePort
    from ..infrastructure.monitoring import ComponentHealth
    from ..infrastructure.persistence.signal_listener import SignalListener
    from ..models.risk_signal import RiskSignal
    from ..models.risk_snapshot import RiskSnapshot


class ApexApp(App):
    """
    Apex Risk Dashboard - Pure Textual Implementation.

    Provides real-time display of:
    - Portfolio metrics and positions
    - Risk signals and alerts
    - ATR analysis for position management
    - Strategy backtesting
    """

    CSS_PATH = "css/dashboard.tcss"

    BINDINGS = [
        Binding("1", "switch_tab('summary')", "Summary", show=True),
        Binding("2", "switch_tab('signals')", "Signals", show=True),
        Binding("3", "switch_tab('ib')", "IB", show=True),
        Binding("4", "switch_tab('futu')", "Futu", show=True),
        Binding("5", "switch_tab('lab')", "Lab", show=True),
        Binding("6", "switch_tab('data')", "Data", show=True),
        Binding("7", "switch_tab('introspection')", "Intro", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    # Reactive state - changes auto-propagate to widgets
    # Using callable factories to avoid mutable default sharing
    snapshot: reactive[Optional[Any]] = reactive(None)
    risk_signals: reactive[List[Any]] = reactive(list)
    health: reactive[List[Any]] = reactive(list)
    market_alerts: reactive[List[Dict[str, Any]]] = reactive(list)

    def __init__(self, env: str = "dev", display_tz: str = "Asia/Hong_Kong", **kwargs):
        """
        Initialize the Apex Dashboard.

        Args:
            env: Environment name (dev, demo, prod).
            display_tz: IANA timezone name for display (e.g., "Asia/Hong_Kong").
        """
        super().__init__(**kwargs)
        self.env = env
        self.display_tz = display_tz
        self.ta_service = None
        self.historical_service = None
        self._event_loop = None
        self._event_bus = None

        # Consolidated event bus for all TUI updates (replaces 4 separate queues)
        self._tui_events = TUIEventBus()

        self._poll_timer = None
        # Pending universe to apply after mount (query_one fails before compose)
        self._pending_universe: List[str] = []
        # Track mount state for deferred universe application
        self._is_mounted = False

        # Signal persistence integration
        self._signal_persistence: Optional["SignalPersistencePort"] = None
        self._signal_listener: Optional["SignalListener"] = None
        self._history_loaded = False

        # Tab 7 (Data) services
        self._coverage_store = None  # DuckDBCoverageStore
        self._data_poll_timer = None  # Slower polling for Tab 7

    def on_mount(self) -> None:
        """Set up update polling when app is mounted."""
        # OPT-001: 4Hz target (0.25s) instead of 10Hz (0.1s) for consistent updates
        self._poll_timer = self.set_interval(0.25, self._poll_updates)
        self._sync_header_tab()

        # Slower polling for Tab 7 data (coverage + indicators)
        self._data_poll_timer = self.set_interval(10.0, self._poll_data_view)

        # Log service status for debugging
        self.log.info(
            f"Services: coverage_store={self._coverage_store is not None}, "
            f"signal_persistence={self._signal_persistence is not None}, "
            f"event_loop={self._event_loop is not None}"
        )

        # Mark as mounted so deferred set_trading_universe() calls apply immediately
        self._is_mounted = True

        # Apply pending universe now that widgets are ready
        if self._pending_universe:
            self._apply_trading_universe(self._pending_universe)

        # Load historical signals from database (non-blocking)
        if self._signal_persistence and not self._history_loaded:
            self._load_signal_history()

        # Connect SignalListener for real-time NOTIFY updates
        if self._signal_listener:
            self._connect_signal_listener()

    def on_unmount(self) -> None:
        """Clean up timers on unmount."""
        if self._poll_timer:
            self._poll_timer.stop()
        if self._data_poll_timer:
            self._data_poll_timer.stop()

    def _poll_updates(self) -> None:
        """Poll all event queues and dispatch to appropriate handlers."""
        result = self._tui_events.poll()

        # Dispatch trading signals (no conflation - each signal matters)
        for signal in result.signals:
            self._dispatch_trading_signal(signal)

        # Dispatch confluence (conflated to latest)
        if result.confluence is not None:
            self._dispatch_confluence(result.confluence)

        # Dispatch alignment (conflated to latest)
        if result.alignment is not None:
            self._dispatch_alignment(result.alignment)

        # Note: Deltas are dispatched via call_from_thread in _flush_deltas()
        # to achieve sub-100ms latency. Polling path disabled to prevent double-apply.

        # Dispatch snapshot (conflated to latest)
        if result.snapshot is not None:
            self.update_data(
                result.snapshot.snapshot,
                result.snapshot.signals,
                result.snapshot.health,
                result.snapshot.alerts,
            )

    def queue_update(self, snapshot, signals, health, alerts) -> None:
        """Queue a data update (thread-safe)."""
        self._tui_events.push_snapshot(
            snapshot=snapshot,
            signals=signals,
            health=health,
            alerts=alerts,
        )

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield HeaderWidget(env=self.env, display_tz=self.display_tz, id="header")
        with TabbedContent(initial="summary", id="main-tabs"):
            with TabPane("Summary", id="summary"):
                yield SummaryView(id="summary-view")
            with TabPane("Signals", id="signals"):
                yield SignalsView(id="signals-view")
            with TabPane("IB", id="ib"):
                yield PositionsView(broker="ib", id="ib-view")
            with TabPane("Futu", id="futu"):
                yield PositionsView(broker="futu", id="futu-view")
            with TabPane("Lab", id="lab"):
                yield LabView(id="lab-view")
            with TabPane("Data", id="data"):
                yield DataView(id="data-view")
            with TabPane("Intro", id="introspection"):
                yield SignalIntrospectionView(id="introspection-view")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id
        self._set_header_tab(tab_id)
        # OPT-002: Immediately update newly visible view with current data
        self._update_views()
        # Trigger immediate data refresh when switching to Tab 7
        if tab_id == "data":
            self._refresh_data_view()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Sync header when tab activation changes."""
        pane_id = event.pane.id if event.pane else ""
        if pane_id:
            self._set_header_tab(pane_id)
            # Trigger immediate data refresh when switching to Tab 7
            if pane_id == "data":
                self._refresh_data_view()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Handle header tab clicks."""
        tab_id = event.tab.id or ""
        if tab_id in {"summary", "signals", "ib", "futu", "lab", "data", "introspection"}:
            tabs = self.query_one("#main-tabs", TabbedContent)
            if tabs.active != tab_id:
                tabs.active = tab_id
            self._set_header_tab(tab_id)

    def inject_services(
        self,
        ta_service,
        event_loop,
        historical_service=None,
        event_bus=None,
        signal_persistence: Optional["SignalPersistencePort"] = None,
        signal_listener: Optional["SignalListener"] = None,
        coverage_store=None,
    ) -> None:
        """
        Inject services for ATR calculation, backtesting, and signal events.

        Args:
            ta_service: TAService instance for ATR calculation.
            event_loop: Main event loop for scheduling async operations.
            historical_service: HistoricalDataService for backtest data.
            event_bus: PriorityEventBus for subscribing to TRADING_SIGNAL events.
            signal_persistence: SignalPersistencePort for loading historical signals.
            signal_listener: SignalListener for PostgreSQL NOTIFY updates.
            coverage_store: DuckDBCoverageStore for Tab 7 historical coverage.
        """
        self.ta_service = ta_service
        self._event_loop = event_loop
        self.historical_service = historical_service
        self._event_bus = event_bus
        self._signal_persistence = signal_persistence
        self._signal_listener = signal_listener
        self._coverage_store = coverage_store

        # Subscribe to trading signal events if event bus provided
        if event_bus is not None:
            self._subscribe_trading_signals(event_bus)
        else:
            self.log.warning("No event_bus provided - trading signals will not stream in real-time")

    def set_signal_introspection(self, introspection) -> None:
        """
        Set the signal introspection adapter for Tab 7 Intro view.

        Args:
            introspection: SignalIntrospectionPort implementation (adapter).
        """
        try:
            intro_view = self.query_one("#introspection-view", SignalIntrospectionView)
            intro_view.set_introspection(introspection)
            self.log.info("Signal introspection adapter wired to TUI")
        except Exception as e:
            self.log.error(f"Failed to wire signal introspection: {e}")

    def _subscribe_trading_signals(self, event_bus) -> None:
        """Subscribe to trading signals and position delta events."""
        from ..domain.events.event_types import EventType

        event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)
        event_bus.subscribe(EventType.CONFLUENCE_UPDATE, self._on_confluence_update)
        event_bus.subscribe(EventType.ALIGNMENT_UPDATE, self._on_alignment_update)
        event_bus.subscribe(EventType.POSITION_DELTA, self._on_position_delta)
        self.log.info(
            "Subscribed to TRADING_SIGNAL, CONFLUENCE_UPDATE, ALIGNMENT_UPDATE, POSITION_DELTA events"
        )

    def _on_trading_signal(self, payload) -> None:
        """
        Event bus callback for trading signals (called from event bus thread).

        Queues the signal for processing in the TUI thread.
        """
        self._tui_events.push_signal(payload)

    def _dispatch_trading_signal(self, signal) -> None:
        """Dispatch a trading signal to the unified Signals view (runs in TUI thread)."""
        try:
            signals_view = self.query_one("#signals-view", SignalsView)
            signals_view.add_trading_signal(signal)
        except Exception as e:
            self.log.error(f"Failed to dispatch trading signal: {e}")

    def _on_confluence_update(self, score) -> None:
        """
        Callback for confluence score updates (called from event bus).

        Queues the score for processing in the TUI thread.

        Args:
            score: ConfluenceUpdateEvent from event bus
        """
        self._tui_events.push_confluence(score)

    def _on_alignment_update(self, alignment) -> None:
        """
        Callback for MTF alignment updates (called from coordinator thread).

        Queues the alignment for processing in the TUI thread.

        Args:
            alignment: MTFAlignment object from MTFDivergenceAnalyzer
        """
        self._tui_events.push_alignment(alignment)

    def _on_position_delta(self, delta) -> None:
        """
        Event bus callback for position deltas (called from DeltaPublisher thread).

        Uses coalescing buffer + scheduled flush for sub-100ms latency while
        preventing UI thread overload under high tick rates.

        Args:
            delta: PositionDeltaEvent with incremental P&L/Greeks changes.
        """
        # Push to coalescing buffer (latest delta per symbol wins)
        self._tui_events.push_delta(delta)

        # Schedule flush via call_from_thread (debounced - only one pending at a time)
        if not getattr(self, "_delta_flush_pending", False):
            self._delta_flush_pending = True
            self.call_from_thread(self._flush_deltas)

    def _dispatch_confluence(self, score) -> None:
        """Dispatch confluence score to unified Signals view (runs in TUI thread)."""
        try:
            signals_view = self.query_one("#signals-view", SignalsView)
            signals_view.update_confluence_score(score)
        except Exception as e:
            self.log.error(f"Failed to dispatch confluence score: {e}")

    def _dispatch_alignment(self, alignment) -> None:
        """Dispatch MTF alignment to unified Signals view (runs in TUI thread)."""
        try:
            signals_view = self.query_one("#signals-view", SignalsView)
            signals_view.update_alignment(alignment)
        except Exception as e:
            self.log.error(f"Failed to dispatch MTF alignment: {e}")

    def _flush_deltas(self) -> None:
        """
        Flush coalesced deltas to all views (called via call_from_thread).

        Runs in TUI thread. Clears the pending flag to allow next flush.
        """
        # Clear pending flag to allow next flush
        self._delta_flush_pending = False

        # Poll deltas from buffer (clears buffer)
        deltas = self._tui_events._poll_deltas()
        if not deltas:
            return

        # Dispatch to ALL position views
        self._dispatch_deltas(deltas)

    def _dispatch_deltas(self, deltas: Dict[str, Any]) -> None:
        """
        Dispatch position deltas to ALL position views (runs in TUI thread).

        Updates all views to keep them in sync, preventing stale data on tab switch.
        Deltas are applied directly to table cells without waiting for snapshots.

        Args:
            deltas: Dict mapping symbol -> PositionDeltaEvent
        """
        if not deltas:
            return

        # Apply to ALL position views (not just active)
        try:
            summary_view = self.query_one("#summary-view", SummaryView)
            summary_view.apply_deltas(deltas)
        except Exception as e:
            self.log.error(f"Failed to dispatch deltas to Summary view: {e}")

        try:
            ib_view = self.query_one("#ib-view", PositionsView)
            ib_view.apply_deltas(deltas)
        except Exception as e:
            self.log.error(f"Failed to dispatch deltas to IB view: {e}")

        try:
            futu_view = self.query_one("#futu-view", PositionsView)
            futu_view.apply_deltas(deltas)
        except Exception as e:
            self.log.error(f"Failed to dispatch deltas to Futu view: {e}")

    def set_trading_universe(self, symbols: List[str]) -> None:
        """
        Set the trading universe symbols for the watchlist.

        Stores symbols for deferred application if app not yet mounted.

        Args:
            symbols: List of symbol strings (e.g., ["AAPL", "TSLA"]).
        """
        if not symbols:
            return

        # Always store for potential re-application
        self._pending_universe = list(symbols)

        # Try to apply immediately if already mounted
        if getattr(self, "_is_mounted", False):
            self._apply_trading_universe(symbols)

    def _apply_trading_universe(self, symbols: List[str]) -> None:
        """Apply trading universe to the unified Signals view (called after mount)."""
        # Map symbols to all standard timeframes
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        symbols_by_tf = {tf: list(symbols) for tf in timeframes}

        try:
            signals_view = self.query_one("#signals-view", SignalsView)
            signals_view.set_universe(symbols_by_tf)
            self.log.info(f"Trading universe applied: {len(symbols)} symbols")
        except Exception as e:
            self.log.error(f"Failed to apply trading universe: {e}")

    def update_data(
        self,
        snapshot: "RiskSnapshot",
        signals: List["RiskSignal"],
        health: List["ComponentHealth"],
        alerts: List[Dict[str, Any]],
    ) -> None:
        """
        Update dashboard with latest data.

        Called from Orchestrator thread via call_from_thread.

        Args:
            snapshot: Latest risk snapshot with position data.
            signals: List of active risk signals.
            health: List of component health statuses.
            alerts: List of market-wide alerts.
        """
        self.snapshot = snapshot
        self.risk_signals = signals
        self.health = health
        self.market_alerts = alerts

        # Propagate to views
        self._update_views()

    def _update_views(self) -> None:
        """Propagate data updates to active view only (OPT-002)."""
        # Get active tab to only update visible view
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
            active_tab = tabs.active
        except Exception as e:
            self.log.error(f"Failed to get active tab: {e}")
            active_tab = "summary"  # Default fallback

        # OPT-002: Only update the active view to reduce overhead
        try:
            if active_tab == "summary":
                summary_view = self.query_one("#summary-view", SummaryView)
                summary_view.update_data(
                    self.snapshot,
                    self.market_alerts,
                    self.health,
                )
            elif active_tab == "signals":
                signals_view = self.query_one("#signals-view", SignalsView)
                # Update risk signals (unified view handles both risk + trading)
                signals_view.update_data(self.risk_signals, self.snapshot)
            elif active_tab == "ib":
                ib_view = self.query_one("#ib-view", PositionsView)
                ib_view.update_data(self.snapshot)
            elif active_tab == "futu":
                futu_view = self.query_one("#futu-view", PositionsView)
                futu_view.update_data(self.snapshot)
            elif active_tab == "introspection":
                intro_view = self.query_one("#introspection-view", SignalIntrospectionView)
                intro_view.refresh_view()
            # Lab and Data views update on-demand, not on polling cycle
        except Exception as e:
            self.log.error(f"Failed to update {active_tab} view: {e}")

        # Always update header (lightweight operation)
        try:
            header = self.query_one("#header", HeaderWidget)
            header.refresh_time()
        except Exception as e:
            self.log.error(f"Failed to update header: {e}")

    def _sync_header_tab(self) -> None:
        """Sync header with current active tab."""
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
            if tabs.active:
                self._set_header_tab(tabs.active)
        except Exception as e:
            self.log.error(f"Failed to sync header tab: {e}")

    def _set_header_tab(self, tab_id: str) -> None:
        """Set active tab in header."""
        try:
            header = self.query_one("#header", HeaderWidget)
            header.active_tab = tab_id
        except Exception as e:
            self.log.error(f"Failed to set header tab: {e}")

    # -------------------------------------------------------------------------
    # Tab 7 (Data) Polling and Event Handlers
    # -------------------------------------------------------------------------

    def _poll_data_view(self) -> None:
        """
        Poll data for Tab 7 (coverage and indicators).

        Runs on slower 10s interval. Only fetches when data tab is active.
        """
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
            if tabs.active != "data":
                return  # Only poll when data tab is active
        except Exception:
            return

        self._refresh_data_view()

    def _refresh_data_view(self) -> None:
        """Refresh data view with coverage and indicator data."""
        # Update coverage from DuckDB
        if self._coverage_store:
            try:
                coverage_data = self._get_coverage_data()
                data_view = self.query_one("#data-view", DataView)
                data_view.update_coverage(coverage_data)

                # Update stats
                symbol_count = len(coverage_data)
                timeframe_set = set()
                total_bars = 0
                for records in coverage_data.values():
                    for r in records:
                        timeframe_set.add(r.get("timeframe", ""))
                        total_bars += r.get("total_bars", 0) or 0

                data_view.update_stats(
                    symbol_count=symbol_count,
                    timeframe_count=len(timeframe_set),
                    total_bars=total_bars,
                    db_connected=self._signal_persistence is not None,
                )
            except Exception as e:
                self.log.error(f"Failed to update coverage: {e}")

        # Update indicator summary from PostgreSQL (async via worker)
        if self._signal_persistence:
            self._fetch_indicator_summary()
        else:
            self.log.warning("Indicator query skipped: signal_persistence not set")

    def _fetch_indicator_summary(self) -> None:
        """Fetch indicator summary using Textual worker to avoid blocking."""

        async def _fetch():
            try:
                summary = await self._signal_persistence.get_indicator_summary()
                self.log.info(f"Indicator summary fetched: {len(summary)} indicators")
                # Workers run in same event loop when thread=False
                data_view = self.query_one("#data-view", DataView)
                data_view.update_indicator_summary(summary)
            except Exception as e:
                self.log.error(f"Failed to fetch indicator summary: {e}")

        # CRITICAL: thread=False ensures we run in the main event loop
        # asyncpg connections are NOT thread-safe and are tied to the event loop
        # they were created in. Using thread=True would create a new event loop
        # where the database connection wouldn't work.
        self.run_worker(_fetch, exclusive=False, name="indicator_summary", thread=False)

    def _get_coverage_data(self) -> Dict[str, List[Dict]]:
        """Get coverage data from DuckDB grouped by symbol, including file sizes."""
        if not self._coverage_store:
            return {}

        try:
            # Get coverage from DuckDB
            coverage = self._coverage_store.get_all_coverage()

            # Add file sizes from the parquet files
            from pathlib import Path

            base_dir = Path("data/historical")

            if base_dir.exists():
                for symbol, records in coverage.items():
                    symbol_dir = base_dir / symbol.upper()
                    if symbol_dir.exists():
                        for record in records:
                            tf = record.get("timeframe", "")
                            parquet_file = symbol_dir / f"{tf}.parquet"
                            if parquet_file.exists():
                                try:
                                    record["file_size"] = parquet_file.stat().st_size
                                except OSError:
                                    record["file_size"] = None
                            else:
                                record["file_size"] = None

            return coverage
        except Exception as e:
            self.log.error(f"Failed to get coverage data: {e}")
            return {}

    def on_data_refresh_requested(self, event: DataRefreshRequested) -> None:
        """Handle manual refresh request from Data view."""
        self._refresh_data_view()

    def on_indicator_details_requested(self, event: IndicatorDetailsRequested) -> None:
        """Handle indicator drill-down request from Data view."""
        if not self._signal_persistence:
            return

        indicator = event.indicator

        async def fetch_details():
            try:
                details = await self._signal_persistence.get_indicator_details(indicator)
                # Apply directly - we're in the same event loop with thread=False
                data_view = self.query_one("#data-view", DataView)
                data_view.update_indicator_details(indicator, details)
            except Exception as e:
                self.log.error(f"Failed to fetch indicator details: {e}")

        # thread=False for asyncpg compatibility (see _fetch_indicator_summary)
        self.run_worker(fetch_details, exclusive=True, thread=False)

    # NOTE: watch_snapshot removed to fix double rendering bug.
    # The update_data() method calls _update_views() directly, so having
    # watch_snapshot also call _update_views() caused double rendering
    # on every update cycle.

    # -------------------------------------------------------------------------
    # Signal History Loading (Background Worker)
    # -------------------------------------------------------------------------

    def _load_signal_history(self) -> None:
        """
        Load historical signals from database in background worker.

        Uses Textual's run_worker with thread=False to run in the main event loop,
        which is required for asyncpg database connections.

        Loads the last 100 signals and applies them to the unified Signals view.
        """
        if not self._signal_persistence:
            return

        async def _fetch_signals():
            """Async worker to fetch signals from database."""
            try:
                signals = await self._signal_persistence.get_recent_signals(limit=100)
                if signals:
                    # Apply directly - we're in the same event loop with thread=False
                    signals_view = self.query_one("#signals-view", SignalsView)
                    for signal in reversed(signals):
                        signals_view.add_signal(signal)
                    self.log.info(f"Loaded {len(signals)} historical signals from database")
                else:
                    self.log.info("No historical signals found in database")
                self._history_loaded = True
            except Exception as e:
                self.log.error(f"Failed to load signal history: {e}")

        # thread=False for asyncpg compatibility (connections are event-loop bound)
        self.run_worker(_fetch_signals, exclusive=True, thread=False)

    # -------------------------------------------------------------------------
    # SignalListener Integration (PostgreSQL NOTIFY)
    # -------------------------------------------------------------------------

    def _connect_signal_listener(self) -> None:
        """
        Connect SignalListener callbacks for real-time database notifications.

        Routes PostgreSQL NOTIFY payloads to the same queues as EventBus,
        unifying the data path for both live and database-sourced signals.
        """
        if not self._signal_listener:
            return

        # Route signal notifications to the existing queue
        self._signal_listener.on_signal(self._on_db_signal_notify)
        self._signal_listener.on_confluence(self._on_db_confluence_notify)

        self.log.info("Connected SignalListener for PostgreSQL NOTIFY updates")

    def _on_db_signal_notify(self, payload: Dict[str, Any]) -> None:
        """
        Handle signal notification from PostgreSQL NOTIFY.

        Routes database payloads through the same TUIEventBus as EventBus signals.

        Args:
            payload: JSON payload from PostgreSQL NOTIFY trigger.
        """
        self._tui_events.push_signal(payload)

    def _on_db_confluence_notify(self, payload: Dict[str, Any]) -> None:
        """
        Handle confluence notification from PostgreSQL NOTIFY.

        Routes database payloads through the same TUIEventBus as EventBus confluence.

        Args:
            payload: JSON payload from PostgreSQL NOTIFY trigger.
        """
        self._tui_events.push_confluence(payload)
