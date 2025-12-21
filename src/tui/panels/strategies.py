"""
Strategy Lab panels for the Terminal Dashboard.

Displays backtest strategies:
- Registered trading strategies with parameters
- Last backtest performance results
- Strategy metadata (author, version, description)

Interactive features:
- j/k: Select strategy
- Enter: Run backtest
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import inspect

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import render_header, render_health
from ...domain.strategy import StrategyRegistry

from ...domain.strategy.registry import StrategyRegistry, get_strategy_info
from ...domain.strategy.base import Strategy
from ...domain.backtest.backtest_result import BacktestResult

if TYPE_CHECKING:
    from ..dashboard import BacktestStatus, BacktestConfig


def render_strategy_list() -> Panel:
    """
    Render panel showing all registered backtest strategies.

    Returns:
        Panel with table of registered strategies and key info.
    """
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        expand=True,
    )

    table.add_column("Name", style="bold white", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Version", style="dim", justify="center")
    table.add_column("Author", style="dim")

    strategies = StrategyRegistry.list_strategies()

    if not strategies:
        table.add_row(
            "[dim]No strategies registered[/dim]",
            "[dim]Import strategy modules to register[/dim]",
            "",
            "",
        )
    else:
        for name in sorted(strategies):
            info = get_strategy_info(name)
            if info:
                table.add_row(
                    name,
                    info.get("description", "-")[:50],
                    info.get("version", "1.0"),
                    info.get("author", "-"),
                )

    return Panel(
        table,
        title="[bold]Backtest Strategies[/bold]",
        border_style="blue",
    )


def render_strategy_params(
    strategy_name: Optional[str] = None,
    config: Optional["BacktestConfig"] = None,
) -> Panel:
    """
    Render panel showing strategy parameters and backtest config.

    Args:
        strategy_name: Name of strategy to show params for. If None, shows first available.
        config: Backtest configuration to display.

    Returns:
        Panel with strategy parameters and backtest config.
    """
    strategies = StrategyRegistry.list_strategies()

    if not strategies:
        return Panel(
            "[dim]No strategy selected[/dim]",
            title="[bold]Strategy Parameters[/bold]",
            border_style="cyan",
        )

    # Use first strategy if none specified
    if strategy_name is None:
        strategy_name = sorted(strategies)[0] if strategies else None

    if strategy_name is None:
        return Panel(
            "[dim]No strategy available[/dim]",
            title="[bold]Strategy Parameters[/bold]",
            border_style="cyan",
        )

    strategy_class = StrategyRegistry.get(strategy_name)
    info = get_strategy_info(strategy_name)

    if not strategy_class:
        return Panel(
            f"[dim]Strategy '{strategy_name}' not found[/dim]",
            title="[bold]Strategy Parameters[/bold]",
            border_style="cyan",
        )

    # Extract parameters from __init__ signature
    params = _extract_strategy_params(strategy_class)

    lines = []
    lines.append(f"[bold cyan]{strategy_name}[/bold cyan]")
    lines.append(f"[dim]{info.get('description', '')}[/dim]")
    lines.append("")

    # Strategy parameters
    lines.append("[bold]Strategy Params:[/bold]")
    if params:
        for param_name, param_info in params.items():
            default = param_info.get("default", "required")
            if default == inspect.Parameter.empty:
                default = "[red]required[/red]"
            else:
                default = f"[green]{default}[/green]"
            lines.append(f"  {param_name}: {default}")
    else:
        lines.append("  [dim]No configurable parameters[/dim]")

    # Backtest configuration
    if config:
        lines.append("")
        lines.append("[bold]Backtest Config:[/bold]")
        lines.append(f"  Symbols: [cyan]{', '.join(config.symbols)}[/cyan]")
        lines.append(f"  Period:  [cyan]{config.start_date} to {config.end_date}[/cyan]")
        lines.append(f"  Capital: [cyan]${config.initial_capital:,.0f}[/cyan]")
        lines.append(f"  Data:    [cyan]{config.data_source}[/cyan]")

    content = Text.from_markup("\n".join(lines))

    return Panel(
        content,
        title=f"[bold]{strategy_name} Config[/bold]",
        border_style="cyan",
    )


def render_backtest_performance(
    result: Optional[BacktestResult] = None,
    error_message: Optional[str] = None,
) -> Panel:
    """
    Render panel showing last backtest performance.

    Args:
        result: BacktestResult to display. If None, shows placeholder.

    Returns:
        Panel with performance metrics.
    """
    if error_message:
        content = Text()
        content.append("Backtest failed\n", style="bold red")
        content.append("\n")
        content.append(error_message, style="red")
        content.append("\n\n", style="red")
        content.append(
            "Tip: check `config/base.yaml` (brokers.ibkr.host/port) and ensure TWS/IB Gateway is running.",
            style="dim",
        )
        return Panel(
            content,
            title="[bold]Last Backtest[/bold]",
            border_style="red",
        )

    if result is None:
        lines = [
            "[dim]No backtest results available[/dim]",
            "",
            "[dim]Run a backtest to see performance:[/dim]",
            "[dim]  python -m src.runners.backtest_runner <spec.yaml>[/dim]",
        ]
        content = Text.from_markup("\n".join(lines))
        return Panel(
            content,
            title="[bold]Last Backtest[/bold]",
            border_style="green",
        )

    # Build performance summary
    lines = [f"[bold]{result.strategy_name}[/bold]", f"[dim]{result.start_date} to {result.end_date}[/dim]", ""]

    # Performance metrics
    if result.performance:
        perf = result.performance
        return_style = "green" if perf.total_return_pct >= 0 else "red"
        lines.append("[bold]Performance:[/bold]")
        lines.append(f"  Return: [{return_style}]{perf.total_return_pct:+.2f}%[/{return_style}]")
        lines.append(f"  CAGR:   {perf.cagr:.2f}%")
        lines.append("")

    # Risk metrics
    if result.risk:
        risk = result.risk
        lines.append("[bold]Risk:[/bold]")
        lines.append(f"  Sharpe:   {risk.sharpe_ratio:.2f}")
        lines.append(f"  Sortino:  {risk.sortino_ratio:.2f}")
        lines.append(f"  Max DD:   [red]{risk.max_drawdown:.2f}%[/red]")
        lines.append("")

    # Trade metrics
    if result.trades:
        trades = result.trades
        lines.append("[bold]Trades:[/bold]")
        lines.append(f"  Total:    {trades.total_trades}")
        lines.append(f"  Win Rate: {trades.win_rate:.1f}%")
        lines.append(f"  PF:       {trades.profit_factor:.2f}")

    content = Text.from_markup("\n".join(lines))

    return Panel(
        content,
        title="[bold]Last Backtest[/bold]",
        border_style="green",
    )


def render_lab_main(
    backtest_results: Optional[Dict[str, BacktestResult]] = None,
    backtest_failures: Optional[Dict[str, str]] = None,
    selected_index: int = 0,
    backtest_status: Optional["BacktestStatus"] = None,
    running_strategy: Optional[str] = None,
) -> Panel:
    """
    Render main lab panel (left side of view).

    Shows all strategies with their last backtest metrics.

    Args:
        backtest_results: Dict mapping strategy name to last BacktestResult.
        selected_index: Index of currently selected strategy.
        backtest_status: Current backtest status.
        running_strategy: Name of strategy currently being backtested.

    Returns:
        Panel with strategies and their performance.
    """
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        expand=True,
    )

    table.add_column("", width=2)  # Selection indicator
    table.add_column("Strategy", style="bold white", no_wrap=True)
    table.add_column("Description", style="white", max_width=25)
    table.add_column("Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("Status", justify="center", width=10)

    strategies = sorted(StrategyRegistry.list_strategies())
    backtest_results = backtest_results or {}
    backtest_failures = backtest_failures or {}

    if not strategies:
        table.add_row(
            "", "[dim]No strategies[/dim]",
            "[dim]Import strategy modules[/dim]",
            "-", "-", "-", "-", "-", "",
        )
    else:
        for idx, name in enumerate(strategies):
            info = get_strategy_info(name)
            result = backtest_results.get(name)

            # Selection indicator
            is_selected = idx == selected_index
            selector = "[bold cyan]>[/bold cyan]" if is_selected else ""

            # Strategy name with highlight if selected
            name_style = "[bold cyan]" if is_selected else ""
            name_end = "[/bold cyan]" if is_selected else ""
            name_display = f"{name_style}{name}{name_end}"

            desc = info.get("description", "-")[:25] if info else "-"

            # Performance metrics
            if result and result.performance and result.risk and result.trades:
                ret = result.performance.total_return_pct
                ret_style = "green" if ret >= 0 else "red"
                ret_str = f"[{ret_style}]{ret:+.1f}%[/{ret_style}]"

                sharpe = f"{result.risk.sharpe_ratio:.2f}"
                max_dd = f"[red]{result.risk.max_drawdown:.1f}%[/red]"
                trades = str(result.trades.total_trades)
                win_rate = f"{result.trades.win_rate:.0f}%"
            else:
                ret_str = "[dim]-[/dim]"
                sharpe = "[dim]-[/dim]"
                max_dd = "[dim]-[/dim]"
                trades = "[dim]-[/dim]"
                win_rate = "[dim]-[/dim]"

            # Status indicator
            if running_strategy == name:
                status = "[yellow]Running...[/yellow]"
            elif name in backtest_failures:
                status = "[red]Failed[/red]"
            elif name in backtest_results:
                status = "[green]Done[/green]"
            else:
                status = "[dim]-[/dim]"

            table.add_row(
                selector, name_display, desc, ret_str, sharpe, max_dd, trades, win_rate, status
            )

    # Build title with instructions
    title = "[bold]Strategy Lab[/bold]"
    subtitle = "[dim]Up/Down/jk:select  Enter:run backtest[/dim]"

    return Panel(
        table,
        title=title,
        subtitle=subtitle,
        border_style="blue",
    )


def _extract_strategy_params(strategy_class: type) -> Dict[str, Dict[str, Any]]:
    """
    Extract configurable parameters from strategy __init__ signature.

    Excludes standard params like strategy_id, symbols, context.
    """
    excluded = {"strategy_id", "symbols", "context", "self"}
    params = {}

    try:
        sig = inspect.signature(strategy_class.__init__)
        for name, param in sig.parameters.items():
            if name in excluded:
                continue
            params[name] = {
                "default": param.default,
                "annotation": param.annotation if param.annotation != inspect.Parameter.empty else None,
            }
    except (ValueError, TypeError):
        pass

    return params


def _update_lab_view(_layout_lab, _backtest_state, env, _current_view, _last_health) -> None:
    """Update the strategy lab view layout (Tab 5)."""
    layout = _layout_lab
    strategies = sorted(StrategyRegistry.list_strategies())
    if _backtest_state.selected_index >= len(strategies) and strategies:
        _backtest_state.selected_index = len(strategies) - 1
    selected_strategy = strategies[_backtest_state.selected_index] if strategies else None
    selected_error = (
        _backtest_state.failures.get(selected_strategy) if selected_strategy else None
    )

    layout["header"].update(render_header(env, _current_view))
    layout["body"]["strategies"].update(
        render_lab_main(
            backtest_results=_backtest_state.results,
            backtest_failures=_backtest_state.failures,
            selected_index=_backtest_state.selected_index,
            backtest_status=_backtest_state.status,
            running_strategy=_backtest_state.running_strategy,
        )
    )
    layout["body"]["details"]["params"].update(
        render_strategy_params(
            strategy_name=selected_strategy,
            config=_backtest_state.config,
        )
    )
    layout["body"]["details"]["performance"].update(
        render_backtest_performance(
            result=_backtest_state.results.get(selected_strategy) if selected_strategy else None,
            error_message=selected_error,
        )
    )
    layout["footer"].update(render_health(_last_health))