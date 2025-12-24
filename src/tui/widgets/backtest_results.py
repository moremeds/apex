"""
Backtest results panel widget.

Displays last backtest performance:
- Strategy name and period
- Performance metrics (Return, CAGR)
- Risk metrics (Sharpe, Sortino, Max DD)
- Trade metrics (Total, Win Rate, PF)
"""

from __future__ import annotations

from typing import Any, Optional

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult


class BacktestResultsPanel(Widget):
    """Backtest results display."""

    # Reactive state
    strategy_name: reactive[Optional[str]] = reactive(None, init=False)
    result: reactive[Optional[Any]] = reactive(None, init=False)
    error_message: reactive[Optional[str]] = reactive(None, init=False)

    def compose(self) -> ComposeResult:
        """Compose the results panel layout."""
        with Vertical(id="results-content"):
            yield Static("[bold]Last Backtest[/]", id="results-title")
            yield Static(self._render_empty(), id="results-body")

    def watch_result(self, result: Optional[Any]) -> None:
        """Update display when result changes."""
        self._update_display()

    def watch_error_message(self, error: Optional[str]) -> None:
        """Update display when error changes."""
        self._update_display()

    def watch_strategy_name(self, name: Optional[str]) -> None:
        """Update display when strategy changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the panel display."""
        try:
            body = self.query_one("#results-body", Static)

            if self.error_message:
                body.update(self._render_error())
            elif self.result:
                body.update(self._render_result())
            else:
                body.update(self._render_empty())
        except Exception:
            pass

    def _render_empty(self) -> str:
        """Render empty state."""
        lines = [
            "[dim]No backtest results available[/]",
            "",
            "[dim]Run a backtest to see performance:[/]",
            "[dim]  Press Enter on a strategy[/]",
            "[dim]  or use CLI:[/]",
            "[dim]  python -m src.runners.backtest_runner <spec.yaml>[/]",
        ]
        return "\n".join(lines)

    def _render_error(self) -> str:
        """Render error state."""
        lines = [
            "[bold red]Backtest failed[/]",
            "",
            f"[red]{self.error_message}[/]",
            "",
            "[dim]Tip: check config/base.yaml (brokers.ibkr.host/port)[/]",
            "[dim]and ensure TWS/IB Gateway is running.[/]",
        ]
        return "\n".join(lines)

    def _render_result(self) -> str:
        """Render backtest result."""
        if not self.result:
            return self._render_empty()

        lines = []

        # Header
        strategy_name = getattr(self.result, "strategy_name", self.strategy_name or "Unknown")
        start_date = getattr(self.result, "start_date", "?")
        end_date = getattr(self.result, "end_date", "?")

        lines.append(f"[bold]{strategy_name}[/]")
        lines.append(f"[dim]{start_date} to {end_date}[/]")
        lines.append("")

        # Performance
        perf = getattr(self.result, "performance", None)
        if perf:
            total_return = getattr(perf, "total_return_pct", 0)
            cagr = getattr(perf, "cagr", 0)

            return_style = "green" if total_return >= 0 else "red"
            lines.append("[bold]Performance:[/]")
            lines.append(f"  Return: [{return_style}]{total_return:+.2f}%[/]")
            lines.append(f"  CAGR:   {cagr:.2f}%")
            lines.append("")

        # Risk
        risk = getattr(self.result, "risk", None)
        if risk:
            sharpe = getattr(risk, "sharpe_ratio", 0)
            sortino = getattr(risk, "sortino_ratio", 0)
            max_dd = getattr(risk, "max_drawdown", 0)

            lines.append("[bold]Risk:[/]")
            lines.append(f"  Sharpe:   {sharpe:.2f}")
            lines.append(f"  Sortino:  {sortino:.2f}")
            lines.append(f"  Max DD:   [red]{max_dd:.2f}%[/]")
            lines.append("")

        # Trades
        trades = getattr(self.result, "trades", None)
        if trades:
            total_trades = getattr(trades, "total_trades", 0)
            win_rate = getattr(trades, "win_rate", 0)
            profit_factor = getattr(trades, "profit_factor", 0)

            lines.append("[bold]Trades:[/]")
            lines.append(f"  Total:    {total_trades}")
            lines.append(f"  Win Rate: {win_rate:.1f}%")
            lines.append(f"  PF:       {profit_factor:.2f}")

        return "\n".join(lines)
