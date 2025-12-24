"""
Strategy list widget for the Lab view.

Displays available strategies from the StrategyRegistry with backtest results.
Matches original Rich layout with columns:
- Strategy, Description, Return, Sharpe, Max DD, Trades, Win%, Status
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual.widgets import DataTable
from textual.reactive import reactive
from textual.message import Message


class StrategyList(DataTable):
    """
    Strategy list display with selection support.

    Shows registered strategies with their last backtest results.
    """

    class StrategySelected(Message):
        """Message sent when a strategy is selected."""

        def __init__(self, strategy_name: str, strategy_info: dict) -> None:
            self.strategy_name = strategy_name
            self.strategy_info = strategy_info
            super().__init__()

    class StrategyActivated(Message):
        """Message sent when a strategy is activated (run requested)."""

        def __init__(self, strategy_name: str, strategy_info: dict) -> None:
            self.strategy_name = strategy_name
            self.strategy_info = strategy_info
            super().__init__()

    # Column definitions matching original Rich layout
    COLUMNS = [
        ("", 2),  # Selection indicator
        ("Strategy", 18),
        ("Description", 25),
        ("Return", 8),
        ("Sharpe", 7),
        ("Max DD", 8),
        ("Trades", 7),
        ("Win%", 6),
        ("Status", 10),
    ]

    # Reactive state
    backtest_results: reactive[Dict[str, Any]] = reactive({}, init=False)
    backtest_failures: reactive[Dict[str, str]] = reactive({}, init=False)
    running_strategy: reactive[Optional[str]] = reactive(None, init=False)

    def __init__(self, **kwargs) -> None:
        super().__init__(cursor_type="row", zebra_stripes=True, **kwargs)
        self._strategy_map: Dict[str, dict] = {}
        self._strategy_list: List[str] = []
        self._selected_strategy: Optional[str] = None

    def on_mount(self) -> None:
        """Set up columns when widget is mounted."""
        for name, width in self.COLUMNS:
            self.add_column(name, width=width)

        # Load strategies from registry
        self._load_strategies()

    def _load_strategies(self) -> None:
        """Load strategies from the StrategyRegistry."""
        try:
            # Import example strategies to ensure they're registered
            from ...domain.strategy import examples  # noqa: F401
            from ...domain.strategy.registry import StrategyRegistry, get_strategy_info

            self._strategy_list = sorted(StrategyRegistry.list_strategies())
            self._strategy_map.clear()

            for name in self._strategy_list:
                info = get_strategy_info(name)
                if info:
                    self._strategy_map[name] = {
                        "name": name,
                        "description": info.get("description", ""),
                        "version": info.get("version", "1.0"),
                        "author": info.get("author", ""),
                    }
                else:
                    self._strategy_map[name] = {
                        "name": name,
                        "description": "",
                        "version": "1.0",
                        "author": "",
                    }

            self._rebuild_table()
        except Exception:
            # Registry not available - show empty state
            self.clear()
            self.add_row(
                "",
                "[dim]No strategies[/]",
                "[dim]Import strategy modules[/]",
                "-", "-", "-", "-", "-", "",
                key="__empty__",
            )

    def _rebuild_table(self) -> None:
        """Rebuild the table with current data."""
        selected_name = self._selected_strategy
        if not selected_name and self.cursor_row is not None and self._strategy_list:
            if 0 <= self.cursor_row < len(self._strategy_list):
                selected_name = self._strategy_list[self.cursor_row]

        self.clear()

        if not self._strategy_list:
            self.add_row(
                "",
                "[dim]No strategies[/]",
                "[dim]Import strategy modules[/]",
                "-", "-", "-", "-", "-", "",
                key="__empty__",
            )
            return

        results = self.backtest_results or {}
        failures = self.backtest_failures or {}
        selected_idx = (
            self._strategy_list.index(selected_name)
            if selected_name in self._strategy_list
            else 0
        )

        for idx, name in enumerate(self._strategy_list):
            info = self._strategy_map.get(name, {})
            result = results.get(name)

            # Selection indicator
            is_selected = idx == selected_idx
            selector = "[bold cyan]>[/]" if is_selected else ""

            # Strategy name with highlight
            name_display = f"[bold cyan]{name}[/]" if is_selected else name

            # Description (truncated)
            desc = info.get("description", "-")[:25]

            # Performance metrics from backtest result
            if result:
                perf = getattr(result, "performance", None)
                risk = getattr(result, "risk", None)
                trades = getattr(result, "trades", None)

                if perf and hasattr(perf, "total_return_pct"):
                    ret = perf.total_return_pct
                    ret_str = f"[green]{ret:+.1f}%[/]" if ret >= 0 else f"[red]{ret:+.1f}%[/]"
                else:
                    ret_str = "[dim]-[/]"

                sharpe = f"{risk.sharpe_ratio:.2f}" if risk and hasattr(risk, "sharpe_ratio") else "[dim]-[/]"
                max_dd = f"[red]{risk.max_drawdown:.1f}%[/]" if risk and hasattr(risk, "max_drawdown") else "[dim]-[/]"
                trade_count = str(trades.total_trades) if trades and hasattr(trades, "total_trades") else "[dim]-[/]"
                win_rate = f"{trades.win_rate:.0f}%" if trades and hasattr(trades, "win_rate") else "[dim]-[/]"
            else:
                ret_str = "[dim]-[/]"
                sharpe = "[dim]-[/]"
                max_dd = "[dim]-[/]"
                trade_count = "[dim]-[/]"
                win_rate = "[dim]-[/]"

            # Status
            if self.running_strategy == name:
                status = "[yellow]Running...[/]"
            elif name in failures:
                status = "[red]Failed[/]"
            elif name in results:
                status = "[green]Done[/]"
            else:
                status = "[dim]-[/]"

            self.add_row(
                selector,
                name_display,
                desc,
                ret_str,
                sharpe,
                max_dd,
                trade_count,
                win_rate,
                status,
                key=name,
            )

        if self.row_count > 0:
            selected_idx = min(selected_idx, self.row_count - 1)
            self.move_cursor(row=selected_idx, column=0, scroll=False)

    def watch_backtest_results(self, results: Dict[str, Any]) -> None:
        """Update display when backtest results change."""
        self._rebuild_table()

    def watch_backtest_failures(self, failures: Dict[str, str]) -> None:
        """Update display when backtest failures change."""
        self._rebuild_table()

    def watch_running_strategy(self, strategy: Optional[str]) -> None:
        """Update display when running strategy changes."""
        self._rebuild_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key is not None:
            key = str(event.row_key.value)
            if key != "__empty__" and key in self._strategy_map:
                self._selected_strategy = key
                self.post_message(
                    self.StrategyActivated(key, self._strategy_map[key])
                )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlight to update selection state."""
        if event.row_key is not None:
            key = str(event.row_key.value)
            if key != "__empty__" and key in self._strategy_map:
                self._selected_strategy = key
                self.post_message(
                    self.StrategySelected(key, self._strategy_map[key])
                )

    def get_selected_strategy(self) -> Optional[str]:
        """Get the currently selected strategy name."""
        if self.cursor_row is not None and self._strategy_list:
            try:
                if self.cursor_row < len(self._strategy_list):
                    return self._strategy_list[self.cursor_row]
            except Exception:
                pass
        return None

    def refresh_strategies(self) -> None:
        """Reload strategies from registry."""
        self._load_strategies()

    def set_backtest_result(self, strategy_name: str, result: Any) -> None:
        """Set backtest result for a strategy."""
        results = dict(self.backtest_results) if self.backtest_results else {}
        results[strategy_name] = result
        self.backtest_results = results

    def set_backtest_failure(self, strategy_name: str, error: str) -> None:
        """Set backtest failure for a strategy."""
        failures = dict(self.backtest_failures) if self.backtest_failures else {}
        failures[strategy_name] = error
        self.backtest_failures = failures

    def set_running_strategy(self, strategy_name: Optional[str]) -> None:
        """Set currently running strategy."""
        self.running_strategy = strategy_name
