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
from textual.widgets import Footer, TabbedContent, TabPane
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
    snapshot: reactive[Optional[Any]] = reactive(None)
    risk_signals: reactive[List[Any]] = reactive([])
    health: reactive[List[Any]] = reactive([])
    market_alerts: reactive[List[Dict[str, Any]]] = reactive([])

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

    def on_mount(self) -> None:
        """Set up update polling when app is mounted."""
        self.set_interval(0.1, self._poll_updates)

    def _poll_updates(self) -> None:
        """Poll the update queue and apply updates."""
        try:
            while True:
                update = self._update_queue.get_nowait()
                self.update_data(
                    update.snapshot,
                    update.signals,
                    update.health,
                    update.alerts,
                )
        except queue.Empty:
            pass

    def queue_update(self, snapshot, signals, health, alerts) -> None:
        """Queue a data update (thread-safe)."""
        from .textual_dashboard import DashboardUpdate
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
        """Propagate data updates to all views."""
        # Update summary view
        try:
            summary_view = self.query_one("#summary-view", SummaryView)
            summary_view.update_data(
                self.snapshot,
                self.market_alerts,
                self.health,
            )
        except Exception:
            pass

        # Update signals view
        try:
            signals_view = self.query_one("#signals-view", SignalsView)
            signals_view.update_data(self.risk_signals, self.snapshot)
        except Exception:
            pass

        # Update IB positions view
        try:
            ib_view = self.query_one("#ib-view", PositionsView)
            ib_view.update_data(self.snapshot)
        except Exception:
            pass

        # Update Futu positions view
        try:
            futu_view = self.query_one("#futu-view", PositionsView)
            futu_view.update_data(self.snapshot)
        except Exception:
            pass

        # Update header
        try:
            header = self.query_one("#header", HeaderWidget)
            header.refresh_time()
        except Exception:
            pass

    def watch_snapshot(self, snapshot: Optional[Any]) -> None:
        """React to snapshot changes."""
        if snapshot is not None:
            self._update_views()
