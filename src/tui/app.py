"""
Apex Dashboard - Pure Textual Implementation.

Real-time terminal UI for risk monitoring with tabbed views:
- Tab 1: Account Summary (consolidated positions, summary, alerts, health)
- Tab 2: Risk Signals (full screen risk signals)
- Tab 3: IB Positions (detailed IB positions with ATR levels)
- Tab 4: Futu Positions (detailed Futu positions with ATR levels)
- Tab 5: Lab (backtest strategies with parameters and performance results)
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, TabbedContent, TabPane, Tabs
from textual.reactive import reactive

from .views.summary import SummaryView
from .views.signals import SignalsView
from .views.positions import PositionsView
from .views.lab import LabView
from .widgets.header import HeaderWidget

if TYPE_CHECKING:
    from ..models.risk_snapshot import RiskSnapshot
    from ..models.risk_signal import RiskSignal
    from ..infrastructure.monitoring import ComponentHealth


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
        Binding("q", "quit", "Quit", show=True),
    ]

    # Reactive state - changes auto-propagate to widgets
    # Using callable factories to avoid mutable default sharing
    snapshot: reactive[Optional[Any]] = reactive(None)
    risk_signals: reactive[List[Any]] = reactive(list)
    health: reactive[List[Any]] = reactive(list)
    market_alerts: reactive[List[Dict[str, Any]]] = reactive(list)

    def __init__(self, env: str = "dev", **kwargs):
        """
        Initialize the Apex Dashboard.

        Args:
            env: Environment name (dev, demo, prod).
        """
        super().__init__(**kwargs)
        self.env = env
        self.ta_service = None
        self.historical_service = None
        self._event_loop = None
        self._update_queue: queue.Queue = queue.Queue()
        self._poll_timer = None

    def on_mount(self) -> None:
        """Set up update polling when app is mounted."""
        # OPT-001: 4Hz target (0.25s) instead of 10Hz (0.1s) for consistent updates
        self._poll_timer = self.set_interval(0.25, self._poll_updates)
        self._sync_header_tab()

    def on_unmount(self) -> None:
        """Clean up timer on unmount."""
        if self._poll_timer:
            self._poll_timer.stop()

    def _poll_updates(self) -> None:
        """Poll the update queue with conflation - only process latest update."""
        latest_update = None
        try:
            # Drain queue but only keep the latest update (conflation)
            while True:
                latest_update = self._update_queue.get_nowait()
        except queue.Empty:
            pass

        # Only process the latest update
        if latest_update is not None:
            self.update_data(
                latest_update.snapshot,
                latest_update.signals,
                latest_update.health,
                latest_update.alerts,
            )

    def queue_update(self, snapshot, signals, health, alerts) -> None:
        """Queue a data update (thread-safe)."""
        from .dashboard import DashboardUpdate
        update = DashboardUpdate(
            snapshot=snapshot,
            signals=signals,
            health=health,
            alerts=alerts,
        )
        try:
            self._update_queue.put_nowait(update)
        except queue.Full:
            # Drop old update if full
            try:
                self._update_queue.get_nowait()
                self._update_queue.put_nowait(update)
            except Exception:
                pass

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield HeaderWidget(env=self.env, id="header")
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
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id
        self._set_header_tab(tab_id)
        # OPT-002: Immediately update newly visible view with current data
        self._update_views()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Sync header when tab activation changes."""
        pane_id = event.pane.id if event.pane else ""
        if pane_id:
            self._set_header_tab(pane_id)

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Handle header tab clicks."""
        tab_id = event.tab.id or ""
        if tab_id in {"summary", "signals", "ib", "futu", "lab"}:
            tabs = self.query_one("#main-tabs", TabbedContent)
            if tabs.active != tab_id:
                tabs.active = tab_id
            self._set_header_tab(tab_id)

    def inject_services(
        self,
        ta_service,
        event_loop,
        historical_service=None,
    ) -> None:
        """
        Inject services for ATR calculation and backtesting.

        Args:
            ta_service: TAService instance for ATR calculation.
            event_loop: Main event loop for scheduling async operations.
            historical_service: HistoricalDataService for backtest data.
        """
        self.ta_service = ta_service
        self._event_loop = event_loop
        self.historical_service = historical_service

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
                signals_view.update_data(self.risk_signals, self.snapshot)
            elif active_tab == "ib":
                ib_view = self.query_one("#ib-view", PositionsView)
                ib_view.update_data(self.snapshot)
            elif active_tab == "futu":
                futu_view = self.query_one("#futu-view", PositionsView)
                futu_view.update_data(self.snapshot)
            # Lab view updates on-demand, not on polling cycle
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

    # NOTE: watch_snapshot removed to fix double rendering bug.
    # The update_data() method calls _update_views() directly, so having
    # watch_snapshot also call _update_views() caused double rendering
    # on every update cycle.
