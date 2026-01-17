"""
Strategy configuration panel widget.

Displays strategy parameters and backtest configuration:
- Strategy name and description
- Strategy parameters
- Backtest settings (symbols, period, capital, data source)

Uses StrategyConfigViewModel for introspection logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from ..viewmodels.strategy_config_vm import StrategyConfigViewModel


class StrategyConfigPanel(Widget):
    """Strategy configuration display."""

    # Reactive state - use factory to avoid mutable default sharing
    strategy_name: reactive[Optional[str]] = reactive(None, init=False)
    strategy_info: reactive[Dict[str, Any]] = reactive(dict, init=False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._view_model = StrategyConfigViewModel()

    def compose(self) -> ComposeResult:
        """Compose the config panel layout."""
        with Vertical(id="config-content"):
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
            body = self.query_one("#config-body", Static)

            if not self.strategy_name:
                body.update("[dim]Select a strategy[/]")
                return

            body.update(self._render_config())
        except Exception as e:
            self.log.error(f"Failed to update config display: {e}")

    def _render_config(self) -> str:
        """Render strategy configuration."""
        if not self.strategy_name:
            return "[dim]No strategy selected[/]"

        # Get config data from ViewModel
        config = self._view_model.get_config(self.strategy_name, self.strategy_info)
        if not config:
            return "[dim]Strategy not found[/]"

        lines = []

        # Strategy name and description
        lines.append(f"[bold cyan]{config.name}[/]")
        if config.description:
            lines.append(f"[dim]{config.description}[/]")
        lines.append("")

        # Strategy parameters
        lines.append("[bold]Strategy Params:[/]")
        if config.params:
            for param in config.params.values():
                if param.required:
                    default_str = "[red]required[/]"
                else:
                    default_str = f"[green]{param.default}[/]"
                lines.append(f"  {param.name}: {default_str}")
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
