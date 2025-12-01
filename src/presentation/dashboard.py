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
from datetime import date, datetime
from enum import Enum
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import logging
import threading
import sys
import select

from ..models.risk_snapshot import RiskSnapshot
from ..models.position_risk import PositionRisk
from ..models.risk_signal import RiskSignal, SignalSeverity
from src.domain.services.risk.rule_engine import LimitBreach, BreachSeverity
from ..infrastructure.monitoring import ComponentHealth, HealthStatus
from ..infrastructure.persistence import PersistenceManager
from ..utils.market_hours import MarketHours

# Type checking imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..infrastructure.persistence.persistence_manager import PersistenceManager

logger = logging.getLogger(__name__)


class DashboardView(Enum):
    """Available dashboard views."""
    ACCOUNT_SUMMARY = "account_summary"  # Tab 1: Consolidated view
    RISK_SIGNALS = "risk_signals"        # Tab 2: Risk signals only
    IB_POSITIONS = "ib_positions"        # Tab 3: IB detailed positions
    FUTU_POSITIONS = "futu_positions"    # Tab 4: Futu detailed positions


class TerminalDashboard:
    """
    Terminal dashboard using rich library.

    Provides real-time display of:
    - Portfolio metrics
    - Limit breaches
    - Health status
    - Position details (if enabled)
    """

    def __init__(self, config: dict, env: str = "dev", persistence_manager: Optional[PersistenceManager] = None):
        """
        Initialize dashboard.

        Args:
            config: Dashboard configuration dict.
            env: Environment name (dev, demo, prod).
            persistence_manager: Optional persistence manager for history queries.
        """
        self.config = config
        self.env = env
        self.show_positions = config.get("show_positions", True)
        self.console = Console()
        self.live: Optional[Live] = None
        self.persistence_manager = persistence_manager

        # View state
        self._current_view = DashboardView.ACCOUNT_SUMMARY
        self._layout_account_summary = self._create_layout_account_summary()
        self._layout_risk_signals = self._create_layout_risk_signals()
        self._layout_broker_positions = self._create_layout_broker_positions()
        self.layout = self._layout_account_summary  # Default to account summary view

        # Keyboard input handling
        self._input_thread: Optional[threading.Thread] = None
        self._stop_input = threading.Event()
        self._quit_requested = False

        # Persistent alert tracking: {alert_key: {alert_data, first_seen, last_seen, is_active}}
        self._persistent_alerts: Dict[str, Dict] = {}
        self._persistent_risk_signals: Dict[str, Dict] = {}
        # How long to keep cleared alerts visible (seconds)
        self._alert_retention_seconds = config.get("alert_retention_seconds", 300)  # 5 minutes default

        # Store latest data for view switching
        self._last_snapshot: Optional[RiskSnapshot] = None
        self._last_breaches: List = []
        self._last_health: List[ComponentHealth] = []
        self._last_market_alerts: List[Dict[str, Any]] = []

    def _create_layout_account_summary(self) -> Layout:
        """Create account summary view layout (Tab 1)."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),  # Main body - expands to fill available space
            Layout(name="footer", size=5),  # Health status at bottom
        )
        # Split body into left (consolidated positions) and right (summary + alerts)
        layout["body"].split_row(
            Layout(name="positions", ratio=3),  # Left: Consolidated positions (60%)
            Layout(name="right", ratio=2),  # Right: Summary + Alerts (40%)
        )
        # Split right side into summary and alerts
        layout["right"].split_column(
            Layout(name="summary", size=18),  # Upper: Portfolio summary
            Layout(name="alerts"),  # Lower: Market alerts
        )
        return layout

    def _create_layout_risk_signals(self) -> Layout:
        """Create risk signals view layout (Tab 2) - full screen signals only."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="signals"),  # Full screen for risk signals
        )
        return layout

    def _create_layout_broker_positions(self) -> Layout:
        """Create broker positions view layout (Tab 3 & 4) - positions with history on right."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        # Split body into positions (left) and history (right)
        layout["body"].split_row(
            Layout(name="positions", ratio=3),  # Current positions (left 60%)
            Layout(name="history_panel", ratio=2),  # Position history panel (right 40%)
        )
        # Split history panel into today and recent (5 days)
        layout["body"]["history_panel"].split_column(
            Layout(name="history_today"),       # Today's changes (top)
            Layout(name="history_recent"),      # Recent 5 days changes (bottom)
        )
        return layout

    def _create_layout(self) -> Layout:
        """Create default layout (for backwards compatibility)."""
        return self._create_layout_account_summary()

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

        # Save terminal settings
        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except termios.error:
            logger.warning("Cannot get terminal settings, keyboard shortcuts disabled")
            return

        try:
            # Use cbreak mode instead of raw mode
            # cbreak allows single-char input without echo, but preserves terminal signals
            # and doesn't interfere with rich's output like raw mode does
            tty.setcbreak(sys.stdin.fileno())

            while not self._stop_input.is_set():
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    self._handle_keypress(char)
        except Exception as e:
            logger.error(f"Keyboard listener error: {e}")
        finally:
            # Restore terminal settings
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
        elif char in ('q', 'Q', '\x03'):  # q or Ctrl+C
            self._quit_requested = True
            logger.info("Quit requested via keyboard")

    def _switch_view(self, new_view: DashboardView) -> None:
        """Switch to a different dashboard view."""
        if new_view == self._current_view:
            return

        self._current_view = new_view
        logger.info(f"Switched to {new_view.value} view")

        # Update layout reference based on view
        if new_view == DashboardView.ACCOUNT_SUMMARY:
            self.layout = self._layout_account_summary
        elif new_view == DashboardView.RISK_SIGNALS:
            self.layout = self._layout_risk_signals
        elif new_view in (DashboardView.IB_POSITIONS, DashboardView.FUTU_POSITIONS):
            # Recreate layout for broker positions (different title)
            self.layout = self._create_layout_broker_positions()

        # Update live display with new layout
        if self.live:
            self.live.update(self.layout)

        # Re-render with cached data
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
                Each alert is a dict with keys: 'type', 'message', 'severity'

        Note:
            The snapshot.position_risks field contains all pre-calculated metrics.
            This dashboard is a "dumb" presentation layer - it does NOT perform
            any calculations, only displays the data from the RiskEngine.
        """
        # Cache data for view switching
        self._last_snapshot = snapshot
        self._last_breaches = breaches
        self._last_health = health
        self._last_market_alerts = market_alerts or []

        # Render based on current view
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
        layout["header"].update(self._render_header())
        layout["body"]["positions"].update(
            self._render_consolidated_positions(snapshot.position_risks, snapshot)
        )
        layout["body"]["right"]["summary"].update(self._render_portfolio_summary(snapshot))
        layout["body"]["right"]["alerts"].update(self._render_market_alerts(market_alerts))
        layout["footer"].update(self._render_health(health))

    def _update_risk_signals_view(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
    ) -> None:
        """Update the risk signals view layout (Tab 2)."""
        layout = self._layout_risk_signals
        layout["header"].update(self._render_header())
        layout["signals"].update(
            self._render_risk_signals_fullscreen(breaches, snapshot)
        )

    def _update_broker_positions_view(
        self,
        snapshot: RiskSnapshot,
        broker: str,
    ) -> None:
        """Update the broker positions view layout (Tab 3 & 4)."""
        layout = self.layout
        layout["header"].update(self._render_header())
        layout["body"]["positions"].update(
            self._render_broker_positions(snapshot.position_risks, broker)
        )
        layout["body"]["history_panel"]["history_today"].update(
            self._render_position_history_today(broker)
        )
        layout["body"]["history_panel"]["history_recent"].update(
            self._render_position_history_recent(broker)
        )

    def _render_header(self) -> Panel:
        """Render header panel with market status, environment, and view tabs."""
        # Get market status
        market_status = MarketHours.get_market_status()

        # Create header text
        header = Text("Live Risk Management System", style="bold cyan")

        # Add environment indicator
        header.append("  |  ", style="dim")
        env_upper = self.env.upper()
        if self.env == "prod":
            header.append(env_upper, style="bold red")
        elif self.env == "demo":
            header.append(env_upper, style="bold magenta")
        else:  # dev
            header.append(env_upper, style="bold yellow")

        # Add market status indicator
        header.append("  |  ", style="dim")
        if market_status == "OPEN":
            header.append("Market: ", style="dim")
            header.append("OPEN", style="bold green")
        elif market_status == "EXTENDED":
            header.append("Market: ", style="dim")
            header.append("EXTENDED HOURS", style="bold yellow")
        else:
            header.append("Market: ", style="dim")
            header.append("CLOSED", style="bold red")

        # Add view tabs
        header.append("  |  ", style="dim")
        tabs = [
            ("1", "Summary", DashboardView.ACCOUNT_SUMMARY),
            ("2", "Signals", DashboardView.RISK_SIGNALS),
            ("3", "IB", DashboardView.IB_POSITIONS),
            ("4", "Futu", DashboardView.FUTU_POSITIONS),
        ]
        for i, (key, label, view) in enumerate(tabs):
            if i > 0:
                header.append(" ", style="dim")
            if self._current_view == view:
                header.append(f"[{key}]{label}", style="bold white on blue")
            else:
                header.append(f"[{key}]{label}", style="dim")

        header.justify = "center"
        return Panel(header, style="bold")

    def _render_portfolio_summary(self, snapshot: RiskSnapshot) -> Panel:
        """Render portfolio summary panel."""
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        # Account Info - Per Broker
        table.add_row(Text("─── Account ───", style="bold"), "")
        table.add_row("IB NetLiq", f"${snapshot.ib_net_liquidation:,.0f}")
        table.add_row("Futu NetLiq", f"${snapshot.futu_net_liquidation:,.0f}")
        table.add_row("Total NetLiq", Text(f"${snapshot.total_net_liquidation:,.0f}", style="bold"))

        # P&L
        table.add_row(Text("─── P&L ───", style="bold"), "")
        table.add_row("Unrealized P&L", self._format_pnl(snapshot.total_unrealized_pnl))
        table.add_row("Daily P&L", self._format_pnl(snapshot.total_daily_pnl))

        # Notional
        table.add_row(Text("─── Exposure ───", style="bold"), "")
        table.add_row("Gross Notional", f"${snapshot.total_gross_notional:,.0f}")
        table.add_row("Net Notional", f"${snapshot.total_net_notional:,.0f}")

        # Greeks
        table.add_row(Text("─── Greeks ───", style="bold"), "")
        table.add_row("Portfolio Delta", f"{snapshot.portfolio_delta:,.0f}")
        table.add_row("Portfolio Gamma", f"{snapshot.portfolio_gamma:,.2f}")
        table.add_row("Portfolio Vega", f"{snapshot.portfolio_vega:,.0f}")
        table.add_row("Portfolio Theta", f"{snapshot.portfolio_theta:,.0f}")

        # Concentration
        table.add_row(Text("─── Risk ───", style="bold"), "")
        table.add_row("Max Concentration", f"{snapshot.concentration_pct:.1%}")
        table.add_row("Max Underlying", snapshot.max_underlying_symbol)
        table.add_row("Margin Utilization", f"{snapshot.margin_utilization:.1%}")

        return Panel(table, title="Portfolio Summary", border_style="green")

    def _update_persistent_alerts(self, current_alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update persistent alert tracking and return alerts to display.

        Active alerts are updated with last_seen timestamp.
        Cleared alerts are kept for alert_retention_seconds with is_active=False.

        Args:
            current_alerts: List of currently active alerts

        Returns:
            List of alerts to display (active + recently cleared)
        """
        now = datetime.now()

        # Build set of current alert keys
        current_keys = set()
        for alert in current_alerts:
            alert_key = f"{alert.get('type', 'UNKNOWN')}_{alert.get('severity', 'INFO')}"
            current_keys.add(alert_key)

            if alert_key in self._persistent_alerts:
                # Update existing alert
                self._persistent_alerts[alert_key]["alert_data"] = alert
                self._persistent_alerts[alert_key]["last_seen"] = now
                self._persistent_alerts[alert_key]["is_active"] = True
            else:
                # New alert
                self._persistent_alerts[alert_key] = {
                    "alert_data": alert,
                    "first_seen": now,
                    "last_seen": now,
                    "is_active": True,
                }

        # Mark alerts not in current set as inactive
        for alert_key in self._persistent_alerts:
            if alert_key not in current_keys:
                self._persistent_alerts[alert_key]["is_active"] = False

        # Build display list and cleanup expired alerts
        display_alerts = []
        expired_keys = []

        for alert_key, alert_info in self._persistent_alerts.items():
            age_seconds = (now - alert_info["last_seen"]).total_seconds()

            if alert_info["is_active"]:
                # Active alert - always display
                display_alerts.append({
                    **alert_info["alert_data"],
                    "first_seen": alert_info["first_seen"],
                    "last_seen": alert_info["last_seen"],
                    "is_active": True,
                })
            elif age_seconds <= self._alert_retention_seconds:
                # Recently cleared - display with dimmed style
                display_alerts.append({
                    **alert_info["alert_data"],
                    "first_seen": alert_info["first_seen"],
                    "last_seen": alert_info["last_seen"],
                    "is_active": False,
                })
            else:
                # Expired - mark for cleanup
                expired_keys.append(alert_key)

        # Cleanup expired alerts
        for key in expired_keys:
            del self._persistent_alerts[key]

        return display_alerts

    def _render_market_alerts(self, alerts: List[Dict[str, Any]]) -> Panel:
        """
        Render market-wide alerts panel with persistent alert tracking.

        Args:
            alerts: List of market alerts. Each alert is a dict with:
                - type: str (e.g., "VIX_SPIKE", "MARKET_DROP", "VOLATILITY")
                - message: str (e.g., "VIX jumped 15% to 28.5")
                - severity: str ("INFO", "WARNING", "CRITICAL")
        """
        # Update persistent alerts and get display list
        display_alerts = self._update_persistent_alerts(alerts)

        if not display_alerts:
            text = Text("✓ No market alerts", style="dim")
            return Panel(text, title="Market Alerts", border_style="dim")

        table = Table(show_header=False, box=None)
        table.add_column("Alert", style="bold")
        table.add_column("Details", justify="left")
        table.add_column("Time", justify="right", style="dim")

        for alert in display_alerts:
            alert_type = alert.get("type", "UNKNOWN")
            message = alert.get("message", "")
            severity = alert.get("severity", "INFO")
            is_active = alert.get("is_active", True)
            last_seen = alert.get("last_seen")

            # Format time display
            time_str = ""
            if last_seen:
                time_str = last_seen.strftime("%H:%M:%S")

            # Set style based on severity and active status
            if not is_active:
                # Cleared alert - dimmed style
                style = "dim"
                icon = "○"
                status_suffix = " [cleared]"
            elif severity == "CRITICAL":
                style = "bold red"
                icon = "🔴"
                status_suffix = ""
            elif severity == "WARNING":
                style = "bold yellow"
                icon = "⚠️"
                status_suffix = ""
            else:
                style = "cyan"
                icon = "ℹ️"
                status_suffix = ""

            table.add_row(
                Text(f"{icon} {alert_type}", style=style),
                Text(f"{message}{status_suffix}", style=style),
                Text(time_str, style="dim")
            )

        # Set border color based on highest severity of ACTIVE alerts
        active_alerts = [a for a in display_alerts if a.get("is_active", True)]
        has_critical = any(a.get("severity") == "CRITICAL" for a in active_alerts)
        has_warning = any(a.get("severity") == "WARNING" for a in active_alerts)

        active_count = len(active_alerts)
        cleared_count = len(display_alerts) - active_count

        if has_critical:
            border_style = "red"
            title = f"🔴 Market Alerts ({active_count} active"
        elif has_warning:
            border_style = "yellow"
            title = f"⚠️  Market Alerts ({active_count} active"
        elif active_count > 0:
            border_style = "cyan"
            title = f"Market Alerts ({active_count} active"
        else:
            border_style = "dim"
            title = f"Market Alerts (0 active"

        if cleared_count > 0:
            title += f", {cleared_count} cleared)"
        else:
            title += ")"

        return Panel(table, title=title, border_style=border_style)

    def _update_persistent_risk_signals(self, current_signals: List[RiskSignal]) -> List[Dict]:
        """
        Update persistent risk signal tracking and return signals to display.

        Active signals are updated with last_seen timestamp.
        Cleared signals are kept for alert_retention_seconds with is_active=False.

        Args:
            current_signals: List of currently active risk signals

        Returns:
            List of signal info dicts to display (active + recently cleared)
        """
        now = datetime.now()

        # Build set of current signal keys
        current_keys = set()
        for signal in current_signals:
            signal_key = f"{signal.symbol or 'PORTFOLIO'}_{signal.trigger_rule}_{signal.severity.value}"
            current_keys.add(signal_key)

            if signal_key in self._persistent_risk_signals:
                # Update existing signal
                self._persistent_risk_signals[signal_key]["signal"] = signal
                self._persistent_risk_signals[signal_key]["last_seen"] = now
                self._persistent_risk_signals[signal_key]["is_active"] = True
            else:
                # New signal
                self._persistent_risk_signals[signal_key] = {
                    "signal": signal,
                    "first_seen": now,
                    "last_seen": now,
                    "is_active": True,
                }

        # Mark signals not in current set as inactive
        for signal_key in self._persistent_risk_signals:
            if signal_key not in current_keys:
                self._persistent_risk_signals[signal_key]["is_active"] = False

        # Build display list and cleanup expired signals
        display_signals = []
        expired_keys = []

        for signal_key, signal_info in self._persistent_risk_signals.items():
            age_seconds = (now - signal_info["last_seen"]).total_seconds()

            if signal_info["is_active"]:
                # Active signal - always display
                display_signals.append({
                    "signal": signal_info["signal"],
                    "first_seen": signal_info["first_seen"],
                    "last_seen": signal_info["last_seen"],
                    "is_active": True,
                })
            elif age_seconds <= self._alert_retention_seconds:
                # Recently cleared - display with dimmed style
                display_signals.append({
                    "signal": signal_info["signal"],
                    "first_seen": signal_info["first_seen"],
                    "last_seen": signal_info["last_seen"],
                    "is_active": False,
                })
            else:
                # Expired - mark for cleanup
                expired_keys.append(signal_key)

        # Cleanup expired signals
        for key in expired_keys:
            del self._persistent_risk_signals[key]

        return display_signals

    def _render_breaches(self, breaches: List[LimitBreach] | List[RiskSignal]) -> Panel:
        """Render portfolio risk alerts panel (supports both LimitBreach and RiskSignal)."""
        # Check if we're using RiskSignals or legacy LimitBreaches
        is_risk_signals = breaches and isinstance(breaches[0], RiskSignal)

        if is_risk_signals:
            return self._render_risk_signals(breaches)
        else:
            # For legacy breaches, check if empty
            if not breaches:
                # Also check persistent risk signals for cleared items
                display_signals = self._update_persistent_risk_signals([])
                if display_signals:
                    return self._render_risk_signals_from_persistent(display_signals)
                text = Text("✓ All risk limits OK", style="green")
                return Panel(text, title="Portfolio Risk Alert", border_style="green")
            return self._render_legacy_breaches(breaches)

    def _render_legacy_breaches(self, breaches: List[LimitBreach]) -> Panel:
        """Render legacy LimitBreach objects."""
        table = Table(show_header=True, box=None)
        table.add_column("Severity", style="bold")
        table.add_column("Risk Metric", style="cyan")
        table.add_column("Status", justify="right")

        for breach in breaches:
            severity_style = "red" if breach.severity == BreachSeverity.HARD else "yellow"
            severity_text = "HARD" if breach.severity == BreachSeverity.HARD else "SOFT"

            table.add_row(
                Text(severity_text, style=severity_style),
                breach.limit_name,
                f"{breach.breach_pct():.1f}%",
            )

        border_style = "red" if any(b.severity == BreachSeverity.HARD for b in breaches) else "yellow"
        return Panel(table, title=f"⚠ Portfolio Risk Alert ({len(breaches)})", border_style=border_style)

    def _render_risk_signals(self, signals: List[RiskSignal]) -> Panel:
        """Render RiskSignal objects with persistent tracking and enhanced display."""
        # Update persistent tracking and get display list
        display_signals = self._update_persistent_risk_signals(signals)
        return self._render_risk_signals_from_persistent(display_signals)

    def _render_risk_signals_from_persistent(self, display_signals: List[Dict]) -> Panel:
        """Render risk signals from persistent tracking data."""
        if not display_signals:
            text = Text("✓ All risk limits OK", style="green")
            return Panel(text, title="Portfolio Risk Alert", border_style="green")

        table = Table(show_header=True, box=None)
        table.add_column("Severity", style="bold", no_wrap=True)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Rule", style="white")
        table.add_column("Action", style="yellow", justify="right")
        table.add_column("Time", style="dim", justify="right")

        # Sort by active status first (active first), then by severity
        sorted_signals = sorted(
            display_signals,
            key=lambda s: (
                0 if s["is_active"] else 1,
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[s["signal"].severity.value]
            )
        )

        for signal_info in sorted_signals:
            signal = signal_info["signal"]
            is_active = signal_info["is_active"]
            last_seen = signal_info["last_seen"]

            # Format time display
            time_str = last_seen.strftime("%H:%M:%S") if last_seen else ""

            if not is_active:
                # Cleared signal - dimmed style
                severity_style = "dim"
                icon = "○"
                status_suffix = " [cleared]"
            else:
                severity_style = {
                    "CRITICAL": "bold red",
                    "WARNING": "bold yellow",
                    "INFO": "cyan"
                }[signal.severity.value]

                icon = {
                    "CRITICAL": "🔴",
                    "WARNING": "⚠️",
                    "INFO": "ℹ️"
                }[signal.severity.value]
                status_suffix = ""

            # Format action
            action_text = signal.suggested_action.value
            if signal.breach_pct:
                action_text += f" ({signal.breach_pct:.0f}%)"

            rule_text = signal.trigger_rule + status_suffix

            table.add_row(
                Text(f"{icon} {signal.severity.value}", style=severity_style),
                signal.symbol or "PORTFOLIO",
                Text(rule_text, style=severity_style if not is_active else "white"),
                Text(action_text, style=severity_style if not is_active else "yellow"),
                Text(time_str, style="dim")
            )

        # Set border color based on highest severity of ACTIVE signals
        active_signals = [s for s in display_signals if s["is_active"]]
        has_critical = any(s["signal"].severity == SignalSeverity.CRITICAL for s in active_signals)
        has_warning = any(s["signal"].severity == SignalSeverity.WARNING for s in active_signals)

        active_count = len(active_signals)
        cleared_count = len(display_signals) - active_count

        if has_critical:
            border_style = "red"
            title = f"🔴 Portfolio Risk Alert ({active_count} active"
        elif has_warning:
            border_style = "yellow"
            title = f"⚠️  Portfolio Risk Alert ({active_count} active"
        elif active_count > 0:
            border_style = "cyan"
            title = f"Portfolio Risk Alert ({active_count} active"
        else:
            border_style = "dim"
            title = f"Portfolio Risk Alert (0 active"

        if cleared_count > 0:
            title += f", {cleared_count} cleared)"
        else:
            title += ")"

        return Panel(table, title=title, border_style=border_style)

    def _render_risk_signals_fullscreen(
        self,
        breaches: List[LimitBreach] | List[RiskSignal],
        snapshot: RiskSnapshot,
    ) -> Panel:
        """
        Render full-screen risk signals view with detailed information.

        Shows:
        - All active signals with full details
        - Signal history (recently cleared)
        - Grouping by severity (CRITICAL → WARNING → INFO)
        - More context per signal (current value, limit, breach %)
        """
        # Get display signals (active + recently cleared)
        if breaches and isinstance(breaches[0], RiskSignal):
            display_signals = self._update_persistent_risk_signals(breaches)
        else:
            # Legacy breaches - convert to display format
            display_signals = self._update_persistent_risk_signals([])

        # Create main table with expanded columns
        table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
        table.add_column("Status", style="bold", no_wrap=True, width=10)
        table.add_column("Severity", style="bold", no_wrap=True, width=10)
        table.add_column("Symbol", style="cyan", no_wrap=True, width=12)
        table.add_column("Layer", style="dim", no_wrap=True, width=8)
        table.add_column("Trigger Rule", style="white", width=30)
        table.add_column("Current", justify="right", width=12)
        table.add_column("Limit", justify="right", width=12)
        table.add_column("Breach %", justify="right", width=10)
        table.add_column("Action", style="yellow", width=15)
        table.add_column("First Seen", style="dim", justify="right", width=10)
        table.add_column("Last Seen", style="dim", justify="right", width=10)

        if not display_signals:
            # Show "all clear" message with summary stats
            table.add_row(
                Text("✓", style="green"),
                "",
                "PORTFOLIO",
                "",
                Text("All risk limits within acceptable range", style="green"),
                "",
                "",
                "",
                "",
                "",
                "",
            )
        else:
            # Sort by: active first, then by severity, then by symbol
            sorted_signals = sorted(
                display_signals,
                key=lambda s: (
                    0 if s["is_active"] else 1,
                    {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[s["signal"].severity.value],
                    s["signal"].symbol or "ZZZZZ",  # Portfolio signals last within severity
                )
            )

            for signal_info in sorted_signals:
                signal = signal_info["signal"]
                is_active = signal_info["is_active"]
                first_seen = signal_info["first_seen"]
                last_seen = signal_info["last_seen"]

                # Format time displays
                first_seen_str = first_seen.strftime("%H:%M:%S") if first_seen else ""
                last_seen_str = last_seen.strftime("%H:%M:%S") if last_seen else ""

                # Status indicator
                if is_active:
                    status_text = Text("● ACTIVE", style="bold green")
                else:
                    status_text = Text("○ CLEARED", style="dim")

                # Severity styling
                if not is_active:
                    severity_style = "dim"
                    icon = "○"
                else:
                    severity_style = {
                        "CRITICAL": "bold red",
                        "WARNING": "bold yellow",
                        "INFO": "cyan"
                    }[signal.severity.value]
                    icon = {
                        "CRITICAL": "🔴",
                        "WARNING": "⚠️",
                        "INFO": "ℹ️"
                    }[signal.severity.value]

                # Format current value and limit (threshold)
                current_str = f"{signal.current_value:,.2f}" if signal.current_value is not None else "-"
                limit_str = f"{signal.threshold:,.2f}" if signal.threshold is not None else "-"
                breach_str = f"{signal.breach_pct:.1f}%" if signal.breach_pct is not None else "-"

                # Layer info
                layer_str = f"L{signal.layer}" if hasattr(signal, 'layer') and signal.layer else "-"

                # Action
                action_str = signal.suggested_action.value if signal.suggested_action else "-"

                table.add_row(
                    status_text,
                    Text(f"{icon} {signal.severity.value}", style=severity_style),
                    signal.symbol or "PORTFOLIO",
                    layer_str,
                    Text(signal.trigger_rule, style=severity_style if not is_active else "white"),
                    Text(current_str, style=severity_style if not is_active else "white"),
                    Text(limit_str, style="dim"),
                    Text(breach_str, style=severity_style if not is_active else "yellow"),
                    Text(action_str, style=severity_style if not is_active else "yellow"),
                    Text(first_seen_str, style="dim"),
                    Text(last_seen_str, style="dim"),
                )

        # Calculate summary stats
        active_signals = [s for s in display_signals if s["is_active"]]
        cleared_signals = [s for s in display_signals if not s["is_active"]]

        critical_count = sum(1 for s in active_signals if s["signal"].severity == SignalSeverity.CRITICAL)
        warning_count = sum(1 for s in active_signals if s["signal"].severity == SignalSeverity.WARNING)
        info_count = sum(1 for s in active_signals if s["signal"].severity == SignalSeverity.INFO)

        # Build title with summary
        title_parts = ["Risk Signals"]
        if active_signals:
            title_parts.append(f"({len(active_signals)} active")
            if critical_count:
                title_parts.append(f"🔴{critical_count}")
            if warning_count:
                title_parts.append(f"⚠️{warning_count}")
            if info_count:
                title_parts.append(f"ℹ️{info_count}")
            if cleared_signals:
                title_parts.append(f", {len(cleared_signals)} cleared)")
            else:
                title_parts.append(")")
        elif cleared_signals:
            title_parts.append(f"(0 active, {len(cleared_signals)} cleared)")
        else:
            title_parts.append("(✓ All Clear)")

        title = " ".join(title_parts)

        # Border color based on highest active severity
        if critical_count > 0:
            border_style = "red"
        elif warning_count > 0:
            border_style = "yellow"
        elif info_count > 0:
            border_style = "cyan"
        else:
            border_style = "green"

        return Panel(table, title=title, border_style=border_style)

    def _render_health(self, health: List[ComponentHealth]) -> Panel:
        """Render health status panel horizontally."""
        if not health:
            return Panel(Text("No health data", style="dim"), title="Health", border_style="dim")

        # Create table with horizontal layout - each component is a column
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)

        # Add columns for each health component
        for h in health:
            table.add_column(justify="center", no_wrap=True)

        # First row: Status icons
        icons = []
        styles = []
        for h in health:
            if h.status == HealthStatus.HEALTHY:
                style = "green"
                icon = "✓"
            elif h.status == HealthStatus.DEGRADED:
                style = "yellow"
                icon = "⚠"
            elif h.status == HealthStatus.UNHEALTHY:
                style = "red"
                icon = "✗"
            else:  # UNKNOWN
                style = "dim"
                icon = "○"
            icons.append(Text(icon, style=style))
            styles.append(style)

        table.add_row(*icons)

        # Second row: Component names
        names = []
        for h in health:
            names.append(Text(h.component_name, style="cyan"))
        table.add_row(*names)

        # Third row: Details
        details_list = []
        for h in health:
            details = h.message if h.message else ""

            # Add metadata info for market data coverage (show for all statuses)
            if h.component_name == "market_data_coverage" and h.metadata:
                if "missing_count" in h.metadata and "total" in h.metadata:
                    missing = h.metadata['missing_count']
                    total = h.metadata['total']
                    if missing > 0:
                        details = f"{missing}/{total} missing MD"
                    else:
                        details = f"All {total} OK" if total > 0 else "No positions"
            # Add metadata info for other degraded/unhealthy components
            elif h.status != HealthStatus.HEALTHY and h.metadata and details == "":
                # Convert metadata to string, handling various types
                if isinstance(h.metadata, dict):
                    details = str(h.metadata)
                else:
                    details = str(h.metadata) if h.metadata else ""

            # Ensure details is always a string
            details = str(details) if details is not None else ""
            details_list.append(Text(details, style="dim"))

        table.add_row(*details_list)

        # Set border color based on worst status
        has_unhealthy = any(h.status == HealthStatus.UNHEALTHY for h in health)
        has_degraded = any(h.status == HealthStatus.DEGRADED for h in health)

        if has_unhealthy:
            border_style = "red"
        elif has_degraded:
            border_style = "yellow"
        else:
            border_style = "green"

        return Panel(table, title="Component Health", border_style=border_style)

    def _render_consolidated_positions(
        self, position_risks: List[PositionRisk], snapshot: RiskSnapshot = None
    ) -> Panel:
        """
        Render consolidated positions table grouped by underlying only (Tab 1).

        Shows summary row per underlying with aggregated metrics.
        Uses pre-calculated snapshot values when available to avoid re-summing.
        """
        if not position_risks:
            return Panel(
                Text("No positions", style="dim"),
                title="Portfolio Positions (Consolidated)",
                border_style="blue",
            )

        # Group position_risks by underlying
        by_underlying: Dict[str, List[PositionRisk]] = {}
        for pos_risk in position_risks:
            if pos_risk.underlying not in by_underlying:
                by_underlying[pos_risk.underlying] = []
            by_underlying[pos_risk.underlying].append(pos_risk)

        # Create table
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Underlying", style="bold", no_wrap=True)
        table.add_column("Positions", justify="right", no_wrap=True)
        table.add_column("Spot", justify="right", no_wrap=True)
        table.add_column("Beta", justify="right", no_wrap=True)
        table.add_column("Mkt Value", justify="right", no_wrap=True)
        table.add_column("P&L", justify="right", no_wrap=True)
        table.add_column("UP&L", justify="right", no_wrap=True)
        table.add_column("Delta $", justify="right", no_wrap=True)
        table.add_column("D(Δ)", justify="right", no_wrap=True)
        table.add_column("G(γ)", justify="right", no_wrap=True)
        table.add_column("V(ν)", justify="right", no_wrap=True)
        table.add_column("Th(Θ)", justify="right", no_wrap=True)

        # Portfolio totals - use pre-calculated snapshot values when available
        if snapshot:
            total_daily_pnl = snapshot.total_daily_pnl
            total_unrealized = snapshot.total_unrealized_pnl
            portfolio_delta = snapshot.portfolio_delta
            portfolio_gamma = snapshot.portfolio_gamma
            portfolio_vega = snapshot.portfolio_vega
            portfolio_theta = snapshot.portfolio_theta
            # These are not in snapshot, calculate once
            total_market_value = sum(pr.market_value for pr in position_risks)
            total_delta_dollars = sum(pr.delta_dollars for pr in position_risks)
        else:
            # Fallback to calculating from position_risks
            total_market_value = sum(pr.market_value for pr in position_risks)
            total_daily_pnl = sum(pr.daily_pnl for pr in position_risks)
            total_unrealized = sum(pr.unrealized_pnl for pr in position_risks)
            total_delta_dollars = sum(pr.delta_dollars for pr in position_risks)
            portfolio_delta = sum(pr.delta for pr in position_risks)
            portfolio_gamma = sum(pr.gamma for pr in position_risks)
            portfolio_vega = sum(pr.vega for pr in position_risks)
            portfolio_theta = sum(pr.theta for pr in position_risks)

        # Add portfolio total row
        table.add_row(
            "▼ PORTFOLIO",
            str(len(position_risks)),
            "",
            "",
            self._format_number(total_market_value, color=False),
            self._format_number(total_daily_pnl, color=True),
            self._format_number(total_unrealized, color=True),
            self._format_number(total_delta_dollars, color=False),
            self._format_number(portfolio_delta, color=False),
            self._format_number(portfolio_gamma, color=False),
            self._format_number(portfolio_vega, color=False),
            self._format_number(portfolio_theta, color=False),
            style="bold white on rgb(80,80,80)",
        )

        # Sort underlyings by absolute market value (descending)
        underlying_values = {}
        for underlying, prs in by_underlying.items():
            underlying_values[underlying] = sum(abs(pr.market_value) for pr in prs)
        sorted_underlyings = sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)

        for underlying in sorted_underlyings:
            prs = by_underlying[underlying]

            # Calculate aggregates for this underlying
            underlying_market_value = sum(pr.market_value for pr in prs)
            underlying_daily_pnl = sum(pr.daily_pnl for pr in prs)
            underlying_unrealized = sum(pr.unrealized_pnl for pr in prs)
            underlying_delta_dollars = sum(pr.delta_dollars for pr in prs)
            underlying_delta = sum(pr.delta for pr in prs)
            underlying_gamma = sum(pr.gamma for pr in prs)
            underlying_vega = sum(pr.vega for pr in prs)
            underlying_theta = sum(pr.theta for pr in prs)

            # Get spot price (from stock position if available)
            # Note: stocks may have expiry=None or expiry="" depending on broker
            spot_price = ""
            is_using_close = False
            for pr in prs:
                if not pr.expiry and pr.mark_price:  # Stock position
                    spot_price = pr.mark_price
                    is_using_close = pr.is_using_close
                    break

            # Get beta
            beta_str = ""
            if prs and prs[0].beta is not None:
                beta_str = f"{prs[0].beta:.2f}"

            table.add_row(
                f"  {underlying}",
                str(len(prs)),
                self._format_price(spot_price, is_using_close, decimals=2) if spot_price else "",
                beta_str,
                self._format_number(underlying_market_value, color=False),
                self._format_number(underlying_daily_pnl, color=True),
                self._format_number(underlying_unrealized, color=True),
                self._format_number(underlying_delta_dollars, color=False),
                self._format_number(underlying_delta, color=False),
                self._format_number(underlying_gamma, color=False),
                self._format_number(underlying_vega, color=False),
                self._format_number(underlying_theta, color=False),
                style="white",
            )

        return Panel(table, title="Portfolio Positions (Consolidated)", border_style="blue")

    def _render_broker_positions(
        self, position_risks: List[PositionRisk], broker: str
    ) -> Panel:
        """
        Render detailed positions for a specific broker (Tab 3 & 4).

        Shows full position details filtered by source (IB or FUTU).
        """
        # Filter positions by broker source (using all_sources to show positions in multiple brokers)
        from ..models.position import PositionSource
        broker_source = PositionSource.IB if broker == "IB" else PositionSource.FUTU

        filtered_risks = []
        for pr in position_risks:
            pos = pr.position
            # Check all_sources if populated, otherwise fall back to source
            if pos.all_sources and broker_source in pos.all_sources:
                filtered_risks.append(pr)
            elif pos.source == broker_source:
                filtered_risks.append(pr)

        if not filtered_risks:
            return Panel(
                Text(f"No {broker} positions", style="dim"),
                title=f"{broker} Positions",
                border_style="blue",
            )

        # Group by underlying
        by_underlying: Dict[str, List[PositionRisk]] = {}
        for pos_risk in filtered_risks:
            if pos_risk.underlying not in by_underlying:
                by_underlying[pos_risk.underlying] = []
            by_underlying[pos_risk.underlying].append(pos_risk)

        # Create table with full details
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Symbol", style="bold", no_wrap=True)
        table.add_column("Pos", justify="right", no_wrap=True)
        table.add_column("Spot", justify="right", no_wrap=True)
        table.add_column("IV", justify="right", no_wrap=True)
        table.add_column("Beta", justify="right", no_wrap=True)
        table.add_column("Mkt Value", justify="right", no_wrap=True)
        table.add_column("P&L", justify="right", no_wrap=True)
        table.add_column("UP&L", justify="right", no_wrap=True)
        table.add_column("Delta $", justify="right", no_wrap=True)
        table.add_column("D(Δ)", justify="right", no_wrap=True)
        table.add_column("G(γ)", justify="right", no_wrap=True)
        table.add_column("V(ν)", justify="right", no_wrap=True)
        table.add_column("Th(Θ)", justify="right", no_wrap=True)

        # Broker totals
        total_market_value = sum(pr.market_value for pr in filtered_risks)
        total_daily_pnl = sum(pr.daily_pnl for pr in filtered_risks)
        total_unrealized = sum(pr.unrealized_pnl for pr in filtered_risks)
        total_delta_dollars = sum(pr.delta_dollars for pr in filtered_risks)
        total_delta = sum(pr.delta for pr in filtered_risks)
        total_gamma = sum(pr.gamma for pr in filtered_risks)
        total_vega = sum(pr.vega for pr in filtered_risks)
        total_theta = sum(pr.theta for pr in filtered_risks)

        # Add broker total row
        table.add_row(
            f"▼ {broker} Total",
            str(len(filtered_risks)),
            "",
            "",
            "",
            self._format_number(total_market_value, color=False),
            self._format_number(total_daily_pnl, color=True),
            self._format_number(total_unrealized, color=True),
            self._format_number(total_delta_dollars, color=False),
            self._format_number(total_delta, color=False),
            self._format_number(total_gamma, color=False),
            self._format_number(total_vega, color=False),
            self._format_number(total_theta, color=False),
            style="bold white on rgb(80,80,80)",
        )

        # Sort underlyings by absolute market value
        underlying_values = {}
        for underlying, prs in by_underlying.items():
            underlying_values[underlying] = sum(abs(pr.market_value) for pr in prs)
        sorted_underlyings = sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)

        for underlying in sorted_underlyings:
            prs = by_underlying[underlying]

            # Add underlying header
            underlying_market_value = sum(pr.market_value for pr in prs)
            underlying_daily_pnl = sum(pr.daily_pnl for pr in prs)
            underlying_unrealized = sum(pr.unrealized_pnl for pr in prs)
            underlying_delta_dollars = sum(pr.delta_dollars for pr in prs)
            underlying_delta = sum(pr.delta for pr in prs)
            underlying_gamma = sum(pr.gamma for pr in prs)
            underlying_vega = sum(pr.vega for pr in prs)
            underlying_theta = sum(pr.theta for pr in prs)

            # Get beta
            beta_str = f"{prs[0].beta:.2f}" if prs and prs[0].beta is not None else ""

            table.add_row(
                f"▼ {underlying}",
                "",
                "",
                "",
                beta_str,
                self._format_number(underlying_market_value, color=False),
                self._format_number(underlying_daily_pnl, color=True),
                self._format_number(underlying_unrealized, color=True),
                self._format_number(underlying_delta_dollars, color=False),
                self._format_number(underlying_delta, color=False),
                self._format_number(underlying_gamma, color=False),
                self._format_number(underlying_vega, color=False),
                self._format_number(underlying_theta, color=False),
                style="bold white",
            )

            # Sort positions: stocks first, then by expiry
            # Note: stocks may have expiry=None or expiry="" (empty string) depending on broker
            stocks = [pr for pr in prs if not pr.expiry]
            options = sorted([pr for pr in prs if pr.expiry], key=lambda p: p.expiry or "")

            for pr in stocks + options:
                iv_str = f"{pr.iv * 100:.1f}%" if pr.iv is not None else ""
                beta_str = f"{pr.beta:.2f}" if pr.beta is not None else ""

                table.add_row(
                    f"  {pr.get_display_name()}",
                    self._format_quantity(pr.quantity),
                    self._format_price(pr.mark_price, pr.is_using_close, decimals=3 if pr.expiry else 2),
                    iv_str,
                    beta_str,
                    self._format_number(pr.market_value, color=False),
                    self._format_number(pr.daily_pnl, color=True),
                    self._format_number(pr.unrealized_pnl, color=True),
                    self._format_number(pr.delta_dollars, color=False),
                    self._format_number(pr.delta, color=False),
                    self._format_number(pr.gamma, color=False),
                    self._format_number(pr.vega, color=False),
                    self._format_number(pr.theta, color=False),
                    style="white",
                )

        return Panel(table, title=f"{broker} Positions ({len(filtered_risks)})", border_style="blue")

    def _render_position_history_today(self, broker: str) -> Panel:
        """
        Render today's position change history for a broker.

        Shows OPEN/CLOSE/MODIFY events from today.
        """
        if not self.persistence_manager:
            return Panel(
                Text("Persistence not enabled", style="dim"),
                title=f"Today's Changes",
                border_style="dim",
            )

        try:
            # Get today's position changes filtered by broker
            changes = self.persistence_manager.positions.get_changes_today()

            # Filter by broker source
            broker_changes = [
                c for c in changes
                if c.get("source") == broker or (c.get("source") is None and broker == "IB")
            ]

            if not broker_changes:
                return Panel(
                    Text("No position changes today", style="dim"),
                    title=f"Today's Changes ({broker})",
                    border_style="dim",
                )

            # Create table
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("Time", style="dim", no_wrap=True, width=8)
            table.add_column("Action", style="bold", no_wrap=True, width=7)
            table.add_column("Symbol", style="cyan", no_wrap=True)
            table.add_column("Qty", justify="right", width=10)
            table.add_column("Price", justify="right", width=10)

            # Show most recent first, limit to 8 rows
            for change in broker_changes[:8]:
                change_time = change.get("change_time")
                time_str = change_time.strftime("%H:%M:%S") if change_time else ""

                change_type = change.get("change_type", "UNKNOWN")

                # Style based on change type
                if change_type == "OPEN":
                    type_style = "green"
                    icon = "+"
                elif change_type == "CLOSE":
                    type_style = "red"
                    icon = "−"
                else:  # MODIFY
                    type_style = "yellow"
                    icon = "~"

                qty_after = change.get("quantity_after")
                avg_price = change.get("avg_price_after") or change.get("avg_price_before")

                # Show qty change for MODIFY, otherwise just qty_after
                qty_str = f"{qty_after:,.0f}" if qty_after is not None else "-"

                table.add_row(
                    time_str,
                    Text(f"{icon}{change_type[:3]}", style=type_style),
                    change.get("symbol", "")[:20],  # Truncate long symbols
                    qty_str,
                    f"${avg_price:,.2f}" if avg_price else "-",
                )

            return Panel(
                table,
                title=f"Today's Changes ({len(broker_changes)})",
                border_style="cyan",
            )

        except Exception as e:
            logger.warning(f"Failed to load today's position history: {e}")
            return Panel(
                Text(f"Error: {e}", style="red"),
                title="Today's Changes",
                border_style="red",
            )

    def _render_position_history_recent(self, broker: str) -> Panel:
        """
        Render all stored positions from database for a broker.

        Shows positions currently tracked in the database to verify persistence is working.
        """
        if not self.persistence_manager:
            return Panel(
                Text("Persistence not enabled", style="dim"),
                title="Stored Positions (DB)",
                border_style="dim",
            )

        try:
            # Get all stored positions from database
            all_positions = self.persistence_manager.get_all_position_snapshots(limit=100)

            # Filter by broker source
            broker_positions = [
                p for p in all_positions
                if p.get("source") == broker
            ]

            if not broker_positions:
                # Show persistence stats to help debug
                stats = self.persistence_manager.get_stats()
                info_text = Text()
                info_text.append("No positions stored for this broker\n\n", style="dim")
                info_text.append(f"DB Stats:\n", style="cyan")
                info_text.append(f"  Snapshots saved: {stats.get('snapshots_saved', 0)}\n", style="white")
                info_text.append(f"  Changes detected: {stats.get('changes_detected', 0)}\n", style="white")
                info_text.append(f"  Tracked positions: {stats.get('tracked_positions', 0)}\n", style="white")
                return Panel(
                    info_text,
                    title=f"Stored Positions ({broker})",
                    border_style="dim",
                )

            # Create table
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("Symbol", style="cyan", no_wrap=True)
            table.add_column("Qty", justify="right", width=8)
            table.add_column("AvgPx", justify="right", width=10)
            table.add_column("Mark", justify="right", width=10)
            table.add_column("P&L", justify="right", width=10)
            table.add_column("Updated", style="dim", width=10)

            # Sort by underlying then symbol
            broker_positions.sort(key=lambda p: (p.get("underlying", ""), p.get("symbol", "")))

            for pos in broker_positions[:20]:  # Limit display
                symbol = pos.get("symbol", "")[:18]
                qty = pos.get("quantity")
                avg_price = pos.get("avg_price")
                mark_price = pos.get("mark_price")
                pnl = pos.get("unrealized_pnl")
                snapshot_time = pos.get("snapshot_time")

                # Format values
                qty_str = f"{qty:,.0f}" if qty is not None else "-"
                avg_str = f"${avg_price:,.2f}" if avg_price else "-"
                mark_str = f"${mark_price:,.2f}" if mark_price else "-"

                # P&L with color
                if pnl is not None:
                    if pnl > 0:
                        pnl_str = Text(f"+${pnl:,.0f}", style="green")
                    elif pnl < 0:
                        pnl_str = Text(f"-${abs(pnl):,.0f}", style="red")
                    else:
                        pnl_str = Text("$0", style="dim")
                else:
                    pnl_str = Text("-", style="dim")

                # Time
                time_str = snapshot_time.strftime("%H:%M:%S") if snapshot_time else "-"

                table.add_row(
                    symbol,
                    qty_str,
                    avg_str,
                    mark_str,
                    pnl_str,
                    time_str,
                )

            # Show count and persistence stats
            stats = self.persistence_manager.get_stats()
            title = f"Stored Positions ({len(broker_positions)} in DB, {stats.get('changes_detected', 0)} changes)"

            return Panel(
                table,
                title=title,
                border_style="blue",
            )

        except Exception as e:
            logger.warning(f"Failed to load stored positions: {e}")
            return Panel(
                Text(f"Error: {e}", style="red"),
                title="Stored Positions",
                border_style="red",
            )

    def _render_positions_profile(
        self, position_risks: List[PositionRisk]
    ) -> Panel:
        """
        Render hierarchical positions table grouped by underlying -> expiry -> position.

        Uses pre-calculated PositionRisk objects from RiskEngine (single source of truth).
        This method does NOT perform any calculations - only displays data.
        """
        if not position_risks:
            return Panel(
                Text("No positions", style="dim"),
                title="Portfolio Positions",
                border_style="blue",
            )

        # Group position_risks by underlying
        by_underlying = {}
        for pos_risk in position_risks:
            if pos_risk.underlying not in by_underlying:
                by_underlying[pos_risk.underlying] = []
            by_underlying[pos_risk.underlying].append(pos_risk)

        # Create table
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Ticker", style="bold", no_wrap=True)
        table.add_column("Pos", justify="right", no_wrap=True)
        table.add_column("Spot", justify="right", no_wrap=True)
        table.add_column("IV", justify="right", no_wrap=True)
        table.add_column("Beta", justify="right", no_wrap=True)
        table.add_column("Mkt Value", justify="right", no_wrap=True)
        table.add_column("P&L", justify="right", no_wrap=True)
        table.add_column("UP&L", justify="right", no_wrap=True)
        table.add_column("Delta $", justify="right", no_wrap=True)
        table.add_column("VAR", justify="right", no_wrap=True)
        table.add_column("D(Δ)", justify="right", no_wrap=True)
        table.add_column("G(γ)", justify="right", no_wrap=True)
        table.add_column("V(ν)", justify="right", no_wrap=True)
        table.add_column("Th(Θ)", justify="right", no_wrap=True)

        # Add portfolio total row (use pre-calculated values from PositionRisk)
        total_market_value = sum(pr.market_value for pr in position_risks)
        total_daily_pnl = sum(pr.daily_pnl for pr in position_risks)
        total_unrealized = sum(pr.unrealized_pnl for pr in position_risks)
        total_delta_dollars = sum(pr.delta_dollars for pr in position_risks)

        # Get portfolio-level Greek aggregations (pre-calculated)
        portfolio_delta = sum(pr.delta for pr in position_risks)
        portfolio_gamma = sum(pr.gamma for pr in position_risks)
        portfolio_vega = sum(pr.vega for pr in position_risks)
        portfolio_theta = sum(pr.theta for pr in position_risks)

        table.add_row(
            "▼ All Tickers",
            "",
            "",
            "",
            "",  # Beta - empty for portfolio total
            self._format_number(total_market_value, color=False),
            self._format_number(total_daily_pnl, color=True),
            self._format_number(total_unrealized, color=True),
            self._format_number(total_delta_dollars, color=False),
            "",
            self._format_number(portfolio_delta, color=False),
            self._format_number(portfolio_gamma, color=False),
            self._format_number(portfolio_vega, color=False),
            self._format_number(portfolio_theta, color=False),
            style="bold white on rgb(80,80,80)",
        )

        # Calculate absolute market value for each underlying for sorting (use pre-calculated values)
        underlying_values = {}
        for underlying, underlying_pos_risks in by_underlying.items():
            total_value = sum(abs(pr.market_value) for pr in underlying_pos_risks)
            underlying_values[underlying] = total_value

        # Sort underlyings by absolute market value (descending)
        sorted_underlyings = sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)

        for underlying in sorted_underlyings:
            underlying_pos_risks = by_underlying[underlying]

            # Group by expiry within underlying
            by_expiry = {}
            stock_pos_risks = []
            for pr in underlying_pos_risks:
                if pr.expiry:
                    if pr.expiry not in by_expiry:
                        by_expiry[pr.expiry] = []
                    by_expiry[pr.expiry].append(pr)
                else:
                    stock_pos_risks.append(pr)

            # Calculate underlying-level totals (use pre-calculated values)
            underlying_market_value = sum(pr.market_value for pr in underlying_pos_risks)
            underlying_daily_pnl = sum(pr.daily_pnl for pr in underlying_pos_risks)
            underlying_unrealized = sum(pr.unrealized_pnl for pr in underlying_pos_risks)
            underlying_delta_dollars = sum(pr.delta_dollars for pr in underlying_pos_risks)

            underlying_delta = sum(pr.delta for pr in underlying_pos_risks)
            underlying_gamma = sum(pr.gamma for pr in underlying_pos_risks)
            underlying_vega = sum(pr.vega for pr in underlying_pos_risks)
            underlying_theta = sum(pr.theta for pr in underlying_pos_risks)

            # Get mark price for underlying (with 'c' indicator if using close)
            underlying_mark = ""
            if stock_pos_risks:
                pr = stock_pos_risks[0]
                if pr.mark_price:
                    underlying_mark = self._format_price(pr.mark_price, pr.is_using_close, decimals=2)

            # Get beta for underlying (use first position's beta)
            underlying_beta = ""
            if underlying_pos_risks:
                first_beta = underlying_pos_risks[0].beta
                if first_beta is not None:
                    underlying_beta = f"{first_beta:.2f}"

            # Add underlying header row
            table.add_row(
                f"▼ {underlying} ",
                "",
                underlying_mark,
                "",
                underlying_beta,
                self._format_number(underlying_market_value, color=False),
                self._format_number(underlying_daily_pnl, color=True),
                self._format_number(underlying_unrealized, color=True),
                self._format_number(underlying_delta_dollars, color=False),
                "",
                self._format_number(underlying_delta, color=False),
                self._format_number(underlying_gamma, color=False),
                self._format_number(underlying_vega, color=False),
                self._format_number(underlying_theta, color=False),
                style=f"bold white ",
            )

            # Add stock positions (if any) - use pre-calculated values
            for pr in stock_pos_risks:
                # Format beta for stock position
                stock_beta = f"{pr.beta:.2f}" if pr.beta is not None else ""
                table.add_row(
                    f" {pr.get_display_name()} ",
                    self._format_quantity(pr.quantity),
                    self._format_price(pr.mark_price, pr.is_using_close, decimals=2),
                    "",  # IV - not applicable for stocks
                    stock_beta,
                    self._format_number(pr.market_value, color=False),
                    self._format_number(pr.daily_pnl, color=True),
                    self._format_number(pr.unrealized_pnl, color=True),
                    self._format_number(pr.delta_dollars, color=False),
                    "",
                    self._format_number(pr.delta, color=False),
                    "",
                    "",
                    "",
                    style=f"white ",
                )

            # Add expiry groups
            # Normalize expiry keys to date objects for sorting
            # All expiry values should be in YYYYMMDD format
            def normalize_expiry(exp):
                if isinstance(exp, date):
                    return exp
                elif isinstance(exp, str):
                    # Standard format is YYYYMMDD
                    return datetime.strptime(exp, "%Y%m%d").date()
                return exp

            for expiry in sorted(by_expiry.keys(), key=normalize_expiry):
                expiry_pos_risks = by_expiry[expiry]

                # Calculate expiry-level totals (use pre-calculated values)
                expiry_market_value = sum(pr.market_value for pr in expiry_pos_risks)
                expiry_daily_pnl = sum(pr.daily_pnl for pr in expiry_pos_risks)
                expiry_unrealized = sum(pr.unrealized_pnl for pr in expiry_pos_risks)
                expiry_delta_dollars = sum(pr.delta_dollars for pr in expiry_pos_risks)

                expiry_delta = sum(pr.delta for pr in expiry_pos_risks)
                expiry_gamma = sum(pr.gamma for pr in expiry_pos_risks)
                expiry_vega = sum(pr.vega for pr in expiry_pos_risks)
                expiry_theta = sum(pr.theta for pr in expiry_pos_risks)

                # Add expiry header row
                table.add_row(
                    f"  ▼ {expiry}",
                    "",
                    "",
                    "",
                    "",  # Beta - empty for expiry header
                    self._format_number(expiry_market_value, color=False),
                    self._format_number(expiry_daily_pnl, color=True),
                    self._format_number(expiry_unrealized, color=True),
                    self._format_number(expiry_delta_dollars, color=False),
                    "",
                    self._format_number(expiry_delta, color=False),
                    self._format_number(expiry_gamma, color=False),
                    self._format_number(expiry_vega, color=False),
                    self._format_number(expiry_theta, color=False),
                    style=f"bold white ",
                )

                # Add individual option positions (use pre-calculated values)
                for pr in expiry_pos_risks:
                    option_desc = pr.get_display_name()

                    # Format IV as percentage (e.g., 0.25 -> 25%)
                    iv_display = ""
                    if pr.iv is not None:
                        iv_display = f"{pr.iv * 100:.1f}%"

                    # Format beta for option position
                    option_beta = f"{pr.beta:.2f}" if pr.beta is not None else ""

                    table.add_row(
                        f"    {option_desc}",
                        self._format_quantity(pr.quantity),
                        self._format_price(pr.mark_price, pr.is_using_close, decimals=3),
                        iv_display,
                        option_beta,
                        self._format_number(pr.market_value, color=False),
                        self._format_number(pr.daily_pnl, color=True),
                        self._format_number(pr.unrealized_pnl, color=True),
                        self._format_number(pr.delta_dollars, color=False),
                        "",
                        self._format_number(pr.delta, color=False),
                        self._format_number(pr.gamma, color=False),
                        self._format_number(pr.vega, color=False),
                        self._format_number(pr.theta, color=False),
                        style=f"white ",
                    )

        return Panel(table, title="Portfolio Positions", border_style="blue")

    def _format_price(self, price: float | None, is_using_close: bool = False, decimals: int = 2) -> str:
        """
        Format price with 'c' indicator if using yesterday's close.

        Args:
            price: The price to format (or None)
            is_using_close: True if price is from yesterday's close (no live data)
            decimals: Number of decimal places (2 for stocks, 3 for options)

        Returns:
            Formatted price string with 'c' suffix if using close
        """
        if price is None:
            return ""
        formatted = f"{price:.{decimals}f}"
        if is_using_close:
            return f"{formatted}c"
        return formatted

    def _format_quantity(self, value: float) -> str:
        """Format quantity with decimal places for fractional shares/contracts."""
        if abs(value) < 0.001:
            return ""

        # Show decimals only if needed
        if value == int(value):
            return f"{int(value):,}"
        else:
            return f"{value:,.2f}"

    def _format_number(self, value: float, color: bool = False) -> str:
        """Format number with optional color coding for P&L."""
        if abs(value) < 0.01:
            return ""

        formatted = f"{value:,.0f}"

        if not color:
            return formatted

        # Color coding for P&L
        if value > 0:
            return f"[green]{formatted}[/green]"
        elif value < 0:
            return f"[red]{formatted}[/red]"
        return formatted

    def _format_pnl(self, value: float) -> Text:
        """Format P&L with color."""
        if value > 0:
            return Text(f"+${value:,.2f}", style="green")
        elif value < 0:
            return Text(f"-${abs(value):,.2f}", style="red")
        else:
            return Text(f"${value:,.2f}", style="dim")
