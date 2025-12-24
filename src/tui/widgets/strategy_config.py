"""
Strategy configuration panel widget.

Displays strategy parameters and backtest configuration:
- Strategy name and description
- Strategy parameters
- Backtest settings (symbols, period, capital, data source)
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult


class StrategyConfigPanel(Widget):
    """Strategy configuration display."""

    # Reactive state
    strategy_name: reactive[Optional[str]] = reactive(None, init=False)
    strategy_info: reactive[Dict[str, Any]] = reactive({}, init=False)

    def compose(self) -> ComposeResult:
        """Compose the config panel layout."""
        with Vertical(id="config-content"):
            yield Static("[bold]Strategy Config[/]", id="config-title")
            yield Static("[dim]Select a strategy[/]", id="config-body")

    def watch_strategy_name(self, name: Optional[str]) -> None:
        """Update display when strategy changes."""
        self._update_display()

    def watch_strategy_info(self, info: Dict[str, Any]) -> None:
        """Update display when strategy info changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the panel display."""
        try:
            title = self.query_one("#config-title", Static)
            body = self.query_one("#config-body", Static)

            if not self.strategy_name:
                title.update("[bold]Strategy Config[/]")
                body.update("[dim]Select a strategy[/]")
                return

            title.update(f"[bold cyan]{self.strategy_name} Config[/]")
            body.update(self._render_config())
        except Exception:
            pass

    def _render_config(self) -> str:
        """Render strategy configuration."""
        if not self.strategy_name:
            return "[dim]No strategy selected[/]"

        lines = []

        # Strategy name and description
        lines.append(f"[bold cyan]{self.strategy_name}[/]")
        desc = self.strategy_info.get("description", "")
        if desc:
            lines.append(f"[dim]{desc}[/]")
        lines.append("")

        # Strategy parameters
        lines.append("[bold]Strategy Params:[/]")
        params = self._extract_strategy_params()
        if params:
            for param_name, param_info in params.items():
                default = param_info.get("default")
                if default is None or default == inspect.Parameter.empty:
                    default_str = "[red]required[/]"
                else:
                    default_str = f"[green]{default}[/]"
                lines.append(f"  {param_name}: {default_str}")
        else:
            lines.append("  [dim]No configurable parameters[/]")
        lines.append("")

        # Backtest configuration (defaults)
        lines.append("[bold]Backtest Config:[/]")
        lines.append("  Symbols: [cyan]AAPL, MSFT[/]")
        lines.append("  Period:  [cyan]2024-01-01 to 2024-06-30[/]")
        lines.append("  Capital: [cyan]$100,000[/]")
        lines.append("  Data:    [cyan]IB Historical[/]")

        return "\n".join(lines)

    def _extract_strategy_params(self) -> Dict[str, Dict[str, Any]]:
        """Extract configurable parameters from strategy class."""
        if not self.strategy_name:
            return {}

        try:
            from ...domain.strategy.registry import StrategyRegistry

            strategy_class = StrategyRegistry.get(self.strategy_name)
            if not strategy_class:
                return {}

            excluded = {"strategy_id", "symbols", "context", "self"}
            params = {}

            sig = inspect.signature(strategy_class.__init__)
            for name, param in sig.parameters.items():
                if name in excluded:
                    continue
                params[name] = {
                    "default": param.default,
                    "annotation": param.annotation if param.annotation != inspect.Parameter.empty else None,
                }

            return params
        except Exception:
            return {}
