"""
Terminal Dashboard using rich library.

Real-time terminal UI for risk monitoring with:
- Portfolio summary (P&L, Greeks, notional)
- Limit breaches (SOFT/HARD)
- Top contributors
- Health status
- Position table (optional)
"""

from __future__ import annotations
from typing import List, Optional, Dict
from datetime import date, datetime
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import logging

from ..models.risk_snapshot import RiskSnapshot
from ..models.position_risk import PositionRisk
from ..models.risk_signal import RiskSignal, SignalSeverity
from ..domain.services.rule_engine import LimitBreach, BreachSeverity
from ..infrastructure.monitoring import ComponentHealth, HealthStatus
from ..utils.market_hours import MarketHours


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

    def __init__(self, config: dict):
        """
        Initialize dashboard.

        Args:
            config: Dashboard configuration dict.
        """
        self.config = config
        self.show_positions = config.get("show_positions", True)
        self.console = Console()
        self.layout = self._create_layout()
        self.live: Optional[Live] = None

        # Persistent alert tracking: {alert_key: {alert_data, first_seen, last_seen, is_active}}
        self._persistent_alerts: Dict[str, Dict] = {}
        self._persistent_risk_signals: Dict[str, Dict] = {}
        # How long to keep cleared alerts visible (seconds)
        self._alert_retention_seconds = config.get("alert_retention_seconds", 300)  # 5 minutes default

    def _create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),  # Main body - expands to fill available space
            Layout(name="footer", size=5),  # Health status at bottom (horizontal layout)
        )
        # Split body into left (positions) and right (summary + breaches)
        layout["body"].split_row(
            Layout(name="profile", ratio=3),  # Left: Portfolio positions (60%)
            Layout(name="right", ratio=2),  # Right: Summary + Breaches (40%)
        )
        # Split right side into three sections: summary, market alerts, and breaches
        layout["right"].split_column(
            Layout(name="upper", size=15),  # Upper: Portfolio summary
            Layout(name="alerts", size=8),  # Middle: Market alerts
            Layout(name="lower"),  # Lower: Limit breaches
        )
        return layout

    def start(self) -> None:
        """Start live dashboard (blocking)."""
        self.live = Live(self.layout, console=self.console, refresh_per_second=2)
        self.live.start()
        logger.info("Terminal dashboard started")

    def stop(self) -> None:
        """Stop live dashboard."""
        if self.live:
            self.live.stop()
            logger.info("Terminal dashboard stopped")

    def update(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
        health: List[ComponentHealth],
        market_alerts: Optional[List[Dict[str, any]]] = None,
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
        self.layout["header"].update(self._render_header())
        self.layout["profile"].update(
            self._render_positions_profile(snapshot.position_risks)
        )
        self.layout["right"]["upper"].update(self._render_portfolio_summary(snapshot))
        self.layout["right"]["alerts"].update(self._render_market_alerts(market_alerts or []))
        self.layout["right"]["lower"].update(self._render_breaches(breaches))
        self.layout["footer"].update(self._render_health(health))

    def _render_header(self) -> Panel:
        """Render header panel with market status."""
        # Get market status
        market_status = MarketHours.get_market_status()

        # Create header text
        header = Text("Live Risk Management System", style="bold cyan")

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

    def _update_persistent_alerts(self, current_alerts: List[Dict[str, any]]) -> List[Dict[str, any]]:
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

    def _render_market_alerts(self, alerts: List[Dict[str, any]]) -> Panel:
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

            # Add underlying header row
            table.add_row(
                f"▼ {underlying} ",
                "",
                underlying_mark,
                "",
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
                table.add_row(
                    f" {pr.get_display_name()} ",
                    self._format_quantity(pr.quantity),
                    self._format_price(pr.mark_price, pr.is_using_close, decimals=2),
                    "",  # IV - not applicable for stocks
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

                    table.add_row(
                        f"    {option_desc}",
                        self._format_quantity(pr.quantity),
                        self._format_price(pr.mark_price, pr.is_using_close, decimals=3),
                        iv_display,
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
