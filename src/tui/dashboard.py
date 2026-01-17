"""
Textual Dashboard Wrapper.

for easy integration with orchestrator.py.

IMPORTANT: Textual must run in the main thread. This wrapper uses a queue-based
approach where updates are posted to a queue and processed by the Textual app.

Methods:
- start(): Does nothing (compatibility shim)
- stop(): Stop the dashboard
- update(): Queue data for display
- set_ta_service(): Inject services for ATR calculation
- set_signal_persistence(): Inject signal persistence for history loading
- run_async(): Actually runs the app (blocking, must be called from main)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .app import ApexApp

if TYPE_CHECKING:
    from ..domain.interfaces.signal_persistence import SignalPersistencePort
    from ..infrastructure.monitoring import ComponentHealth
    from ..infrastructure.persistence.signal_listener import SignalListener
    from ..models.risk_signal import RiskSignal
    from ..models.risk_snapshot import RiskSnapshot


@dataclass
class DashboardUpdate:
    """Data update for the dashboard."""

    snapshot: Any
    signals: List[Any]
    health: List[Any]
    alerts: List[Dict[str, Any]]


class TextualDashboard:
    """
    Wrapper around ApexApp to provide the same interface as TerminalDashboard.

    Uses a queue-based approach for thread-safe updates. The app must be run
    in the main thread using run_async().
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        env: str = "dev",
    ) -> None:
        """
        Initialize the Textual dashboard.

        Args:
            config: Dashboard configuration dict. May contain 'display_tz' key
                    for timezone display (e.g., "Asia/Hong_Kong").
            env: Environment name (dev, demo, prod).
        """
        self.env = env
        self.config = config or {}
        # Extract display timezone from config (default: Asia/Hong_Kong)
        self.display_tz = self.config.get("display_tz", "Asia/Hong_Kong")
        self._app: Optional[ApexApp] = None
        # Set _running = True immediately so update_loop doesn't exit early
        # It will be set to False when the app actually exits
        self._running = True
        self._quit_requested = False
        self._ta_service = None
        self._event_loop = None
        self._historical_service = None
        self._event_bus = None
        self._pending_universe: List[str] = []  # Symbols to apply when app starts

        # Signal persistence integration
        self._signal_persistence: Optional["SignalPersistencePort"] = None
        self._signal_listener: Optional["SignalListener"] = None

        # Tab 7 (Data) coverage store
        self._coverage_store = None

        # Tab 7 (Intro) signal introspection
        self._signal_introspection = None

    def start(self) -> None:
        """
        Compatibility shim - does nothing.

        The actual start happens when run_async() is called.
        """
        pass

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._app:
            try:
                self._app.exit()
            except Exception:
                pass

    @property
    def quit_requested(self) -> bool:
        """Check if user requested quit."""
        return self._quit_requested

    def update(
        self,
        snapshot: "RiskSnapshot",
        signals: List["RiskSignal"],
        health: List["ComponentHealth"],
        alerts: List[Dict[str, Any]],
    ) -> None:
        """
        Queue a data update for the dashboard.

        Thread-safe: can be called from any thread.

        Args:
            snapshot: Latest risk snapshot.
            signals: List of risk signals.
            health: List of component health statuses.
            alerts: List of market alerts.
        """
        if not self._running:
            return

        if self._app:
            # Queue update to the app (thread-safe)
            self._app.queue_update(snapshot, signals, health, alerts)

    def set_ta_service(
        self,
        ta_service,
        event_loop: asyncio.AbstractEventLoop,
        historical_service=None,
        event_bus=None,
    ) -> None:
        """
        Inject services for ATR calculation, backtesting, and signal events.

        Args:
            ta_service: TAService instance.
            event_loop: Main event loop for async operations.
            historical_service: HistoricalDataService for backtest data.
            event_bus: PriorityEventBus for subscribing to TRADING_SIGNAL events.
        """
        self._ta_service = ta_service
        self._event_loop = event_loop
        self._historical_service = historical_service
        self._event_bus = event_bus

        if self._app:
            self._app.inject_services(
                ta_service,
                event_loop,
                historical_service,
                event_bus,
                self._signal_persistence,
                self._signal_listener,
                self._coverage_store,
            )

    def set_trading_universe(self, symbols: List[str]) -> None:
        """
        Set the trading universe symbols for Tab 2 Signals (trading panel watchlist).

        Args:
            symbols: List of symbol strings (e.g., ["AAPL", "TSLA"]).
        """
        self._pending_universe = list(symbols)
        if self._app:
            self._app.set_trading_universe(symbols)

    def set_signal_introspection(self, introspection) -> None:
        """
        Set the signal introspection adapter for Tab 7 Intro view.

        Args:
            introspection: SignalIntrospectionPort implementation (adapter).
        """
        self._signal_introspection = introspection
        if self._app:
            self._app.set_signal_introspection(introspection)

    def set_event_bus(
        self,
        event_bus,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Set the event bus for signal event subscriptions.

        CRITICAL: This must be called BEFORE run_async() to ensure the TUI
        subscribes to TRADING_SIGNAL, CONFLUENCE_UPDATE, and ALIGNMENT_UPDATE
        events before any signals are generated.

        Args:
            event_bus: PriorityEventBus instance for event subscriptions.
            event_loop: Main event loop for async operations.
        """
        self._event_bus = event_bus
        self._event_loop = event_loop

        # If app already exists (rare), inject immediately
        if self._app:
            self._app.inject_services(
                self._ta_service,
                self._event_loop,
                self._historical_service,
                self._event_bus,
                self._signal_persistence,
                self._signal_listener,
                self._coverage_store,
            )

    def set_signal_persistence(
        self,
        persistence: Optional["SignalPersistencePort"] = None,
        listener: Optional["SignalListener"] = None,
    ) -> None:
        """
        Set signal persistence for history loading and real-time NOTIFY.

        This enables:
        1. Loading historical signals from database on TUI startup
        2. Receiving real-time signal updates via PostgreSQL NOTIFY

        Args:
            persistence: SignalPersistencePort for database queries.
            listener: SignalListener for PostgreSQL NOTIFY updates.
        """
        self._signal_persistence = persistence
        self._signal_listener = listener

        # If app already exists, inject immediately
        if self._app:
            self._app.inject_services(
                self._ta_service,
                self._event_loop,
                self._historical_service,
                self._event_bus,
                self._signal_persistence,
                self._signal_listener,
                self._coverage_store,
            )

    def set_coverage_store(self, coverage_store) -> None:
        """
        Set coverage store for Tab 7 historical data display.

        Args:
            coverage_store: DuckDBCoverageStore instance.
        """
        self._coverage_store = coverage_store

        # If app already exists, inject immediately
        if self._app:
            self._app.inject_services(
                self._ta_service,
                self._event_loop,
                self._historical_service,
                self._event_bus,
                self._signal_persistence,
                self._signal_listener,
                self._coverage_store,
            )

    async def run_async(
        self,
        orchestrator_callback: Optional[Callable] = None,
    ) -> None:
        """
        Run the dashboard asynchronously.

        This is the main entry point that runs in the main thread.
        The orchestrator callback is called periodically to fetch data.

        Args:
            orchestrator_callback: Async function that returns (snapshot, signals, health, alerts).
        """
        self._app = ApexApp(env=self.env, display_tz=self.display_tz)

        # Inject services if already set (event_bus is critical for signal subscriptions)
        if self._event_bus or self._ta_service or self._signal_persistence or self._coverage_store:
            self._app.inject_services(
                self._ta_service,
                self._event_loop,
                self._historical_service,
                self._event_bus,
                self._signal_persistence,
                self._signal_listener,
                self._coverage_store,
            )

        # Apply pending trading universe if set
        if self._pending_universe:
            self._app.set_trading_universe(self._pending_universe)

        # Apply signal introspection if set
        if self._signal_introspection:
            self._app.set_signal_introspection(self._signal_introspection)

        # NOTE: Confluence/alignment updates now use event bus (CONFLUENCE_UPDATE,
        # ALIGNMENT_UPDATE events) instead of direct callbacks. The TUI subscribes
        # to these events in inject_services() via _subscribe_trading_signals().

        # Run the app (blocks until user quits)
        # The app's on_mount sets up the update polling timer
        await self._app.run_async()

        self._running = False
        self._quit_requested = True

    @property
    def running(self):
        return self._running


# Alias for backward compatibility
TerminalDashboard = TextualDashboard
