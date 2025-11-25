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
from typing import List, Optional
from datetime import date, datetime
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import logging

from ..models.risk_snapshot import RiskSnapshot, PositionRisk
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
        # Split right side into upper (summary) and lower (breaches)
        layout["right"].split_column(
            Layout(name="upper", size=15),  # Upper: Portfolio summary
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
        breaches: List[LimitBreach],
        health: List[ComponentHealth],
        positions: Optional[List] = None,
        market_data: Optional[Dict] = None,
    ) -> None:
        """
        Update dashboard with latest data.

        Args:
            snapshot: Latest risk snapshot.
            breaches: List of limit breaches.
            health: List of component health statuses.
            positions: List of positions for detailed display.
            market_data: Market data by symbol.
        """
        self.layout["header"].update(self._render_header())
        self.layout["profile"].update(
            self._render_positions_profile(snapshot)
        )
        self.layout["right"]["upper"].update(self._render_portfolio_summary(snapshot))
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

        # P&L
        table.add_row("Unrealized P&L", self._format_pnl(snapshot.total_unrealized_pnl))
        table.add_row("Daily P&L", self._format_pnl(snapshot.total_daily_pnl))

        # Notional
        table.add_row("Gross Notional", f"${snapshot.total_gross_notional:,.0f}")
        table.add_row("Net Notional", f"${snapshot.total_net_notional:,.0f}")

        # Greeks
        table.add_row("Portfolio Delta", f"{snapshot.portfolio_delta:,.0f}")
        table.add_row("Portfolio Gamma", f"{snapshot.portfolio_gamma:,.2f}")
        table.add_row("Portfolio Vega", f"{snapshot.portfolio_vega:,.0f}")
        table.add_row("Portfolio Theta", f"{snapshot.portfolio_theta:,.0f}")

        # Concentration
        table.add_row("Max Concentration", f"{snapshot.concentration_pct:.1%}")
        table.add_row("Max Underlying", snapshot.max_underlying_symbol)

        # Margin
        table.add_row("Margin Utilization", f"{snapshot.margin_utilization:.1%}")

        return Panel(table, title="Portfolio Summary", border_style="green")

    def _render_breaches(self, breaches: List[LimitBreach]) -> Panel:
        """Render limit breaches panel."""
        if not breaches:
            text = Text("✓ All limits within range", style="green")
            return Panel(text, title="Limit Status", border_style="green")

        table = Table(show_header=True, box=None)
        table.add_column("Severity", style="bold")
        table.add_column("Limit", style="cyan")
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
        return Panel(table, title=f"⚠ Limit Breaches ({len(breaches)})", border_style=border_style)

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
                details = str(h.metadata)

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
        self, snapshot: RiskSnapshot
    ) -> Panel:
        """Render hierarchical positions table grouped by underlying -> expiry -> position."""
        metrics: List[PositionRisk] = getattr(snapshot, "positions", [])
        if not metrics:
            return Panel(
                Text("No positions", style="dim"),
                title="Portfolio Positions",
                border_style="blue",
            )

        # Group positions by underlying
        by_underlying = {}
        for m in metrics:
            if m.underlying not in by_underlying:
                by_underlying[m.underlying] = []
            by_underlying[m.underlying].append(m)

        # Create table
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Ticker", style="bold", no_wrap=True)
        table.add_column("Pos", justify="right", no_wrap=True)
        table.add_column("Spot", justify="right", no_wrap=True)
        table.add_column("Mkt Value", justify="right", no_wrap=True)
        table.add_column("Daily P&L", justify="right", no_wrap=True)
        table.add_column("Unrealised P&L", justify="right", no_wrap=True)
        table.add_column("Delta $", justify="right", no_wrap=True)
        table.add_column("VAR", justify="right", no_wrap=True)
        table.add_column("D(Δ)", justify="right", no_wrap=True)
        table.add_column("G(γ)", justify="right", no_wrap=True)
        table.add_column("V(ν)", justify="right", no_wrap=True)
        table.add_column("Th(Θ)", justify="right", no_wrap=True)

        # Add portfolio total row
        total_market_value = sum(m.market_value for m in metrics)
        total_daily_pnl = sum(m.daily_pnl for m in metrics)
        total_unrealized = sum(m.unrealized_pnl for m in metrics)
        total_delta_dollars = sum(m.delta_dollars for m in metrics)

        # Get portfolio-level aggregations
        portfolio_delta = sum(m.delta_contribution for m in metrics)
        portfolio_gamma = sum(m.gamma_contribution for m in metrics)
        portfolio_vega = sum(m.vega_contribution for m in metrics)
        portfolio_theta = sum(m.theta_contribution for m in metrics)

        table.add_row(
            "▼ All Tickers",
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

        # Calculate absolute market value for each underlying for sorting
        underlying_values = {}
        for underlying, underlying_positions in by_underlying.items():
            total_value = sum(abs(p.market_value) for p in underlying_positions)
            underlying_values[underlying] = total_value

        # Sort underlyings by absolute market value (descending)
        sorted_underlyings = sorted(by_underlying.keys(), key=lambda u: underlying_values[u], reverse=True)

        for underlying in sorted_underlyings:
            underlying_positions = by_underlying[underlying]

            # Group by expiry within underlying
            by_expiry = {}
            stock_positions = []
            for pos in underlying_positions:
                if pos.expiry:
                    if pos.expiry not in by_expiry:
                        by_expiry[pos.expiry] = []
                    by_expiry[pos.expiry].append(pos)
                else:
                    stock_positions.append(pos)

            # Calculate underlying-level totals
            underlying_market_value = sum(p.market_value for p in underlying_positions)
            underlying_daily_pnl = sum(p.daily_pnl for p in underlying_positions)
            underlying_unrealized = sum(p.unrealized_pnl for p in underlying_positions)
            underlying_delta_dollars = sum(p.delta_dollars for p in underlying_positions)

            underlying_delta = sum(p.delta_contribution for p in underlying_positions)
            underlying_gamma = sum(p.gamma_contribution for p in underlying_positions)
            underlying_vega = sum(p.vega_contribution for p in underlying_positions)
            underlying_theta = sum(p.theta_contribution for p in underlying_positions)

            # Get mark price for underlying
            underlying_mark = ""
            if stock_positions and stock_positions[0].mark is not None:
                underlying_mark = f"{stock_positions[0].mark:.2f}"

            # Add underlying header row
            table.add_row(
                f"▼ {underlying} ",
                "",
                underlying_mark,
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

            # Add stock positions (if any)
            for pos in stock_positions:
                mark = pos.mark

                table.add_row(
                    f" {pos.get_display_name()} ",
                    self._format_quantity(pos.quantity),
                    f"{mark:.2f}" if mark else "",
                    self._format_number(pos.market_value, color=False),
                    self._format_number(pos.daily_pnl, color=True),
                    self._format_number(pos.unrealized_pnl, color=True),
                    self._format_number(pos.delta_dollars, color=False),
                    "",
                    self._format_number(pos.delta_contribution, color=False),
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
                expiry_positions = by_expiry[expiry]

                # Calculate expiry-level totals
                expiry_market_value = sum(p.market_value for p in expiry_positions)
                expiry_daily_pnl = sum(p.daily_pnl for p in expiry_positions)
                expiry_unrealized = sum(p.unrealized_pnl for p in expiry_positions)
                expiry_delta_dollars = sum(p.delta_dollars for p in expiry_positions)

                expiry_delta = sum(p.delta_contribution for p in expiry_positions)
                expiry_gamma = sum(p.gamma_contribution for p in expiry_positions)
                expiry_vega = sum(p.vega_contribution for p in expiry_positions)
                expiry_theta = sum(p.theta_contribution for p in expiry_positions)

                # Add expiry header row
                table.add_row(
                    f"  ▼ {expiry}",
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

                # Add individual option positions
                for pos in expiry_positions:
                    mark = pos.mark

                    option_desc = f"{pos.symbol}"
                    if pos.strike and pos.right:
                        option_desc = f"{pos.get_display_name()}"

                    table.add_row(
                        f"    {option_desc}",
                        self._format_quantity(pos.quantity),
                        f"{mark:.3f}" if mark else "",
                        self._format_number(pos.market_value, color=False),
                        self._format_number(pos.daily_pnl, color=True),
                        self._format_number(pos.unrealized_pnl, color=True),
                        self._format_number(pos.delta_dollars, color=False),
                        "",
                        self._format_number(pos.delta_contribution, color=False),
                        self._format_number(pos.gamma_contribution, color=False),
                        self._format_number(pos.vega_contribution, color=False),
                        self._format_number(pos.theta_contribution, color=False),
                        style=f"white ",
                    )

        return Panel(table, title="Portfolio Positions", border_style="blue")

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
