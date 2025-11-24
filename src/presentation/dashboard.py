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
from datetime import date
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import logging

from ..models.risk_snapshot import RiskSnapshot
from ..models.position import Position
from ..models.market_data import MarketData
from ..domain.services.rule_engine import LimitBreach, BreachSeverity
from ..infrastructure.monitoring import ComponentHealth, HealthStatus


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
            Layout(name="profile"),  # Portfolio positions section
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right"),
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
        positions: Optional[List[Position]] = None,
        market_data: Optional[Dict[str, MarketData]] = None,
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
            self._render_positions_profile(positions or [], market_data or {})
        )
        self.layout["left"].update(self._render_portfolio_summary(snapshot))
        self.layout["right"].update(self._render_breaches(breaches))
        self.layout["footer"].update(self._render_health(health))

    def _render_header(self) -> Panel:
        """Render header panel."""
        text = Text("Live Risk Management System", style="bold cyan", justify="center")
        return Panel(text, style="bold")

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
        """Render health status panel."""
        if not health:
            return Panel(Text("No health data", style="dim"), title="Health", border_style="dim")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Status", style="bold", no_wrap=True)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", no_wrap=False)

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

            # Show message if available
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

            table.add_row(
                Text(icon, style=style),
                h.component_name,
                details,
            )

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
        self, positions: List[Position], market_data: Dict[str, MarketData]
    ) -> Panel:
        """Render hierarchical positions table grouped by underlying -> expiry -> position."""
        if not positions:
            return Panel(
                Text("No positions", style="dim"),
                title="Portfolio Positions",
                border_style="blue",
            )

        # Group positions by underlying
        by_underlying = {}
        for pos in positions:
            if pos.underlying not in by_underlying:
                by_underlying[pos.underlying] = []
            by_underlying[pos.underlying].append(pos)

        # Create table
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Underlying", style="bold", no_wrap=True)
        table.add_column("Position", justify="right", no_wrap=True)
        table.add_column("Mark", justify="right", no_wrap=True)
        table.add_column("Unrealized\nP&L", justify="right", no_wrap=True)
        table.add_column("Delta\nDollars", justify="right", no_wrap=True)
        table.add_column("VAR", justify="right", no_wrap=True)
        table.add_column("Delta (Δ)", justify="right", no_wrap=True)
        table.add_column("Gamma...", justify="right", no_wrap=True)
        table.add_column("Vega (ν)", justify="right", no_wrap=True)
        table.add_column("Theta (Θ)", justify="right", no_wrap=True)
        table.add_column("Hedge", justify="right", no_wrap=True)

        # Add portfolio total row
        total_unrealized = sum(
            self._calculate_unrealized_pnl(p, market_data.get(p.symbol)) for p in positions
        )
        total_delta_dollars = sum(
            self._calculate_delta_dollars(p, market_data.get(p.symbol)) for p in positions
        )

        # Get portfolio-level aggregations
        portfolio_delta = sum(self._get_position_delta(p, market_data.get(p.symbol)) for p in positions)
        portfolio_gamma = sum(self._get_greek(p.symbol, market_data, "gamma", 0) * p.quantity * p.multiplier for p in positions)
        portfolio_vega = sum(self._get_greek(p.symbol, market_data, "vega", 0) * p.quantity * p.multiplier for p in positions)
        portfolio_theta = sum(self._get_greek(p.symbol, market_data, "theta", 0) * p.quantity * p.multiplier for p in positions)

        table.add_row(
            "▼ All Underlyings",
            "",
            "",
            self._format_number(total_unrealized, color=True),
            self._format_number(total_delta_dollars, color=False),
            "",
            self._format_number(portfolio_delta, color=False),
            self._format_number(portfolio_gamma, color=False),
            self._format_number(portfolio_vega, color=False),
            self._format_number(portfolio_theta, color=False),
            "",
            style="bold white on rgb(80,80,80)",
        )

        # Add each underlying group (alternating colors)
        color_idx = 0
        colors = ["on rgb(70,130,180)", "on rgb(60,179,113)"]  # Blue, Green

        for underlying in sorted(by_underlying.keys()):
            underlying_positions = by_underlying[underlying]
            bg_color = colors[color_idx % 2]
            color_idx += 1

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
            underlying_unrealized = sum(
                self._calculate_unrealized_pnl(p, market_data.get(p.symbol))
                for p in underlying_positions
            )
            underlying_delta_dollars = sum(
                self._calculate_delta_dollars(p, market_data.get(p.symbol))
                for p in underlying_positions
            )

            underlying_delta = sum(self._get_position_delta(p, market_data.get(p.symbol)) for p in underlying_positions)
            underlying_gamma = sum(self._get_greek(p.symbol, market_data, "gamma", 0) * p.quantity * p.multiplier for p in underlying_positions)
            underlying_vega = sum(self._get_greek(p.symbol, market_data, "vega", 0) * p.quantity * p.multiplier for p in underlying_positions)
            underlying_theta = sum(self._get_greek(p.symbol, market_data, "theta", 0) * p.quantity * p.multiplier for p in underlying_positions)

            # Get mark price for underlying
            underlying_mark = ""
            if stock_positions:
                md = market_data.get(stock_positions[0].symbol)
                if md and md.effective_mid():
                    underlying_mark = f"{md.effective_mid():.2f}"

            # Add underlying header row
            table.add_row(
                f"▼ {underlying} <ARCA>",
                "",
                underlying_mark,
                self._format_number(underlying_unrealized, color=True),
                self._format_number(underlying_delta_dollars, color=False),
                "",
                self._format_number(underlying_delta, color=False),
                self._format_number(underlying_gamma, color=False),
                self._format_number(underlying_vega, color=False),
                self._format_number(underlying_theta, color=False),
                "",
                style=f"bold white {bg_color}",
            )

            # Add stock positions (if any)
            for pos in stock_positions:
                md = market_data.get(pos.symbol)
                mark = md.effective_mid() if md else None

                table.add_row(
                    f"  {pos.symbol} Stock ({pos.source.value})",
                    self._format_number(pos.quantity, color=False),
                    f"{mark:.2f}" if mark else "",
                    self._format_number(self._calculate_unrealized_pnl(pos, md), color=True),
                    self._format_number(self._calculate_delta_dollars(pos, md), color=False),
                    "",
                    self._format_number(self._get_position_delta(pos, md), color=False),
                    "",
                    "",
                    "",
                    "",
                    style=f"white {bg_color}",
                )

            # Add expiry groups
            for expiry in sorted(by_expiry.keys()):
                expiry_positions = by_expiry[expiry]

                # Calculate expiry-level totals
                expiry_unrealized = sum(
                    self._calculate_unrealized_pnl(p, market_data.get(p.symbol))
                    for p in expiry_positions
                )
                expiry_delta_dollars = sum(
                    self._calculate_delta_dollars(p, market_data.get(p.symbol))
                    for p in expiry_positions
                )

                expiry_delta = sum(self._get_position_delta(p, market_data.get(p.symbol)) for p in expiry_positions)
                expiry_gamma = sum(self._get_greek(p.symbol, market_data, "gamma", 0) * p.quantity * p.multiplier for p in expiry_positions)
                expiry_vega = sum(self._get_greek(p.symbol, market_data, "vega", 0) * p.quantity * p.multiplier for p in expiry_positions)
                expiry_theta = sum(self._get_greek(p.symbol, market_data, "theta", 0) * p.quantity * p.multiplier for p in expiry_positions)

                # Add expiry header row
                table.add_row(
                    f"  ▼ {expiry}",
                    "",
                    "",
                    self._format_number(expiry_unrealized, color=True),
                    self._format_number(expiry_delta_dollars, color=False),
                    "",
                    self._format_number(expiry_delta, color=False),
                    self._format_number(expiry_gamma, color=False),
                    self._format_number(expiry_vega, color=False),
                    self._format_number(expiry_theta, color=False),
                    "",
                    style=f"bold white {bg_color}",
                )

                # Add individual option positions
                for pos in expiry_positions:
                    md = market_data.get(pos.symbol)
                    mark = md.effective_mid() if md else None

                    option_desc = f"{pos.symbol}"
                    if pos.strike and pos.right:
                        option_desc = f"{underlying} {expiry.strftime('%b %d \'%y')} {pos.strike} {pos.right}ut"

                    table.add_row(
                        f"    {option_desc}",
                        self._format_number(pos.quantity, color=False),
                        f"{mark:.3f}" if mark else "",
                        self._format_number(self._calculate_unrealized_pnl(pos, md), color=True),
                        self._format_number(self._calculate_delta_dollars(pos, md), color=False),
                        "",
                        self._format_number(self._get_position_delta(pos, md), color=False),
                        self._format_number(self._get_greek(pos.symbol, market_data, "gamma", 0) * pos.quantity * pos.multiplier, color=False),
                        self._format_number(self._get_greek(pos.symbol, market_data, "vega", 0) * pos.quantity * pos.multiplier, color=False),
                        self._format_number(self._get_greek(pos.symbol, market_data, "theta", 0) * pos.quantity * pos.multiplier, color=False),
                        "",
                        style=f"white {bg_color}",
                    )

        return Panel(table, title="Portfolio Positions", border_style="blue")

    def _calculate_unrealized_pnl(
        self, pos: Position, md: Optional[MarketData]
    ) -> float:
        """Calculate unrealized P&L for a position."""
        if not md or not md.effective_mid():
            return 0.0
        mark = md.effective_mid()
        return (mark - pos.avg_price) * pos.quantity * pos.multiplier

    def _calculate_delta_dollars(
        self, pos: Position, md: Optional[MarketData]
    ) -> float:
        """Calculate delta dollars (delta * mark * quantity * multiplier)."""
        if not md:
            return 0.0
        mark = md.effective_mid() or 0
        delta = md.delta if md.delta is not None else (1.0 if pos.asset_type.value == "STOCK" else 0.0)
        return delta * mark * pos.quantity * pos.multiplier

    def _get_position_delta(
        self, pos: Position, md: Optional[MarketData]
    ) -> float:
        """Get position delta (delta * quantity * multiplier)."""
        if not md:
            delta = 1.0 if pos.asset_type.value == "STOCK" else 0.0
            return delta * pos.quantity * pos.multiplier

        delta = md.delta if md.delta is not None else (1.0 if pos.asset_type.value == "STOCK" else 0.0)
        return delta * pos.quantity * pos.multiplier

    def _get_greek(
        self, symbol: str, market_data: Dict[str, MarketData], greek: str, default: float
    ) -> float:
        """Get Greek value from market data."""
        md = market_data.get(symbol)
        if not md:
            return default
        return getattr(md, greek, default) or default

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
            return Text(f"+${value:,.0f}", style="green")
        elif value < 0:
            return Text(f"-${abs(value):,.0f}", style="red")
        else:
            return Text(f"${value:,.0f}", style="dim")
