"""
Terminal Dashboard using rich library.

Real-time terminal UI for risk monitoring with tabbed views:
- Tab 1: Account Summary (consolidated positions by underlying, summary, alerts, health)
- Tab 2: Risk Signals (full screen risk signals)
- Tab 3: IB Positions (detailed Interactive Brokers positions)
- Tab 4: Futu Positions (detailed Futu positions)

Keyboard shortcuts:
- 1: Account Summary view (default)
- 2: Risk Signals view
- 3: IB Positions view
- 4: Futu Positions view
- q: Quit
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging
import threading
import sys
import select

from rich.console import Console
from rich.live import Live

from .base import DashboardView
from .layouts import (
    create_layout_account_summary,
    create_layout_risk_signals,
    create_layout_broker_positions,
)
from .panels import (
    render_header,
    render_portfolio_summary,
    render_market_alerts,
    update_persistent_alerts,
    render_breaches,
    render_risk_signals_fullscreen,
    render_consolidated_positions,
    render_broker_positions,
    render_health,
    render_position_history_today,
    render_open_orders,
    render_position_history_recent,
)

from ..models.risk_snapshot import RiskSnapshot
from ..models.risk_signal import RiskSignal
from src.domain.services.risk.rule_engine import LimitBreach
from ..infrastructure.monitoring import ComponentHealth

logger = logging.getLogger(__name__)


class TerminalDashboard:
    """
    Terminal dashboard using rich library.

    Provides real-time display of:
    - Portfolio metrics
    - Limit breaches
    - Health status
    - Position details (if enabled)
    """

    def __init__(self, config: dict, env: str = "dev"):
        """
        Initialize dashboard.

        Args:
            config: Dashboard configuration dict.
            env: Environment name (dev, demo, prod).
        """
        self.config = config
        self.env = env
        self.show_positions = config.get("show_positions", True)
        self.console = Console()
        self.live: Optional[Live] = None

        # View state
        self._current_view = DashboardView.ACCOUNT_SUMMARY
        self._layout_account_summary = create_layout_account_summary()
        self._layout_risk_signals = create_layout_risk_signals()
        self._layout_broker_positions = create_layout_broker_positions()
        self.layout = self._layout_account_summary

        # Keyboard input handling
        self._input_thread: Optional[threading.Thread] = None
        self._stop_input = threading.Event()
        self._quit_requested = False

        # Persistent alert tracking
        self._persistent_alerts: Dict[str, Dict] = {}
        self._persistent_risk_signals: Dict[str, Dict] = {}
        self._alert_retention_seconds = config.get("alert_retention_seconds", 300)

        # Store latest data for view switching
        self._last_snapshot: Optional[RiskSnapshot] = None
        self._last_breaches: List = []
        self._last_health: List[ComponentHealth] = []
        self._last_market_alerts: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Start live dashboard (blocking)."""
        self.live = Live(self.layout, console=self.console, refresh_per_second=2)
        self.live.start()
        self._start_keyboard_listener()
        logger.info("Terminal dashboard started")

    def stop(self) -> None:
        """Stop live dashboard."""
        self._stop_keyboard_listener()
        if self.live:
            self.live.stop()
            logger.info("Terminal dashboard stopped")

    def _start_keyboard_listener(self) -> None:
        """Start background thread for keyboard input."""
        self._stop_input.clear()
        self._input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._input_thread.start()
        logger.debug("Keyboard listener started")

    def _stop_keyboard_listener(self) -> None:
        """Stop keyboard input thread."""
        self._stop_input.set()
        if self._input_thread and self._input_thread.is_alive():
            self._input_thread.join(timeout=1.0)
        logger.debug("Keyboard listener stopped")

    def _keyboard_listener(self) -> None:
        """Background thread that listens for keyboard input."""
        import termios
        import tty

        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except termios.error:
            logger.warning("Cannot get terminal settings, keyboard shortcuts disabled")
            return

        try:
            tty.setcbreak(sys.stdin.fileno())

            while not self._stop_input.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    self._handle_keypress(char)
        except Exception as e:
            logger.error(f"Keyboard listener error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _handle_keypress(self, char: str) -> None:
        """Handle a single keypress."""
        if char == '1':
            self._switch_view(DashboardView.ACCOUNT_SUMMARY)
        elif char == '2':
            self._switch_view(DashboardView.RISK_SIGNALS)
        elif char == '3':
            self._switch_view(DashboardView.IB_POSITIONS)
        elif char == '4':
            self._switch_view(DashboardView.FUTU_POSITIONS)
        elif char in ('q', 'Q', '\x03'):
            self._quit_requested = True
            logger.info("Quit requested via keyboard")

    def _switch_view(self, new_view: DashboardView) -> None:
        """Switch to a different dashboard view."""
        if new_view == self._current_view:
            return

        self._current_view = new_view
        logger.info(f"Switched to {new_view.value} view")

        if new_view == DashboardView.ACCOUNT_SUMMARY:
            self.layout = self._layout_account_summary
        elif new_view == DashboardView.RISK_SIGNALS:
            self.layout = self._layout_risk_signals
        elif new_view in (DashboardView.IB_POSITIONS, DashboardView.FUTU_POSITIONS):
            self.layout = create_layout_broker_positions()

        if self.live:
            self.live.update(self.layout)

        if self._last_snapshot:
            self.update(
                self._last_snapshot,
                self._last_breaches,
                self._last_health,
                self._last_market_alerts,
            )

    @property
    def quit_requested(self) -> bool:
        """Check if user requested quit via keyboard."""
        return self._quit_requested

    def update(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
        health: List[ComponentHealth],
        market_alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Update dashboard with latest data.

        Args:
            snapshot: Latest risk snapshot (contains pre-calculated position_risks).
            breaches: List of limit breaches (legacy) or risk signals (new).
            health: List of component health statuses.
            market_alerts: List of market-wide alerts (VIX spikes, etc).
        """
        # Cache data for view switching
        self._last_snapshot = snapshot
        self._last_breaches = breaches
        self._last_health = health
        self._last_market_alerts = market_alerts or []

        if self._current_view == DashboardView.ACCOUNT_SUMMARY:
            self._update_account_summary_view(snapshot, breaches, health, market_alerts or [])
        elif self._current_view == DashboardView.RISK_SIGNALS:
            self._update_risk_signals_view(snapshot, breaches)
        elif self._current_view == DashboardView.IB_POSITIONS:
            self._update_broker_positions_view(snapshot, "IB")
        elif self._current_view == DashboardView.FUTU_POSITIONS:
            self._update_broker_positions_view(snapshot, "FUTU")

    def _update_account_summary_view(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
        health: List[ComponentHealth],
        market_alerts: List[Dict[str, Any]],
    ) -> None:
        """Update the account summary view layout (Tab 1)."""
        layout = self._layout_account_summary

        # Update persistent alerts and get display list
        display_alerts = update_persistent_alerts(
            market_alerts,
            self._persistent_alerts,
            self._alert_retention_seconds,
        )

        layout["header"].update(render_header(self.env, self._current_view))
        layout["body"]["positions"].update(
            render_consolidated_positions(snapshot.position_risks, snapshot)
        )
        layout["body"]["right"]["summary"].update(render_portfolio_summary(snapshot))
        layout["body"]["right"]["alerts"].update(render_market_alerts(display_alerts))
        layout["footer"].update(render_health(health))

    def _update_risk_signals_view(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
    ) -> None:
        """Update the risk signals view layout (Tab 2)."""
        layout = self._layout_risk_signals
        layout["header"].update(render_header(self.env, self._current_view))
        layout["signals"].update(
            render_risk_signals_fullscreen(
                breaches,
                snapshot,
                self._persistent_risk_signals,
                self._alert_retention_seconds,
            )
        )

    def _update_broker_positions_view(
        self,
        snapshot: RiskSnapshot,
        broker: str,
    ) -> None:
        """Update the broker positions view layout (Tab 3 & 4)."""
        layout = self.layout
        layout["header"].update(render_header(self.env, self._current_view))
        layout["body"]["positions"].update(
            render_broker_positions(snapshot.position_risks, broker)
        )
        layout["body"]["history_panel"]["history_today"].update(
            render_position_history_today(broker)
        )
        layout["body"]["history_panel"]["open_orders"].update(
            render_open_orders(broker)
        )
        layout["body"]["history_panel"]["history_recent"].update(
            render_position_history_recent(broker)
        )
