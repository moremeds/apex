"""
StrategyViewModel - Framework-agnostic strategy list data transformation.

Extracts business logic from StrategyList:
- Strategy info management
- Backtest result formatting
- Status tracking
- Row key generation for stable updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseViewModel


@dataclass
class StrategyDisplayState:
    """Combined state for strategy display."""

    results: Dict[str, Any] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    running: Optional[str] = None


class StrategyViewModel(BaseViewModel[List[str]]):
    """
    ViewModel for strategy list.

    Responsibilities:
    - Format backtest metrics (return, Sharpe, drawdown)
    - Track running/completed/failed status
    - Compute cell-level diffs for incremental updates
    """

    COLUMN_COUNT = 9  # selector, name, desc, return, sharpe, max_dd, trades, win_rate, status

    def __init__(self) -> None:
        super().__init__()
        self._strategy_map: Dict[str, dict] = {}
        self._state = StrategyDisplayState()
        self._selected_strategy: Optional[str] = None

    def set_strategies(self, strategy_map: Dict[str, dict]) -> None:
        """Set the strategy info map."""
        self._strategy_map = strategy_map
        # Invalidate cache when strategies change
        self.invalidate()

    def set_state(self, state: StrategyDisplayState) -> None:
        """Set the combined display state."""
        self._state = state

    def set_selected(self, strategy_name: Optional[str]) -> None:
        """Set the currently selected strategy."""
        self._selected_strategy = strategy_name

    def compute_display_data(self, strategies: List[str]) -> Dict[str, List[str]]:
        """Transform strategy list into display rows."""
        result: Dict[str, List[str]] = {}

        if not strategies:
            result["__empty__"] = [
                "",
                "[dim]No strategies[/]",
                "[dim]Import strategy modules[/]",
                "-",
                "-",
                "-",
                "-",
                "-",
                "",
            ]
            return result

        for name in strategies:
            is_selected = name == self._selected_strategy
            info = self._strategy_map.get(name, {})
            backtest_result = self._state.results.get(name)

            result[name] = [
                "[bold cyan]>[/]" if is_selected else "",
                f"[bold cyan]{name}[/]" if is_selected else name,
                info.get("description", "-")[:25],
                self._format_return(backtest_result),
                self._format_sharpe(backtest_result),
                self._format_max_dd(backtest_result),
                self._format_trades(backtest_result),
                self._format_win_rate(backtest_result),
                self._format_status(name),
            ]

        return result

    def get_row_order(self, strategies: List[str]) -> List[str]:
        """Return ordered list of row keys."""
        if not strategies:
            return ["__empty__"]
        return list(strategies)

    # Formatting helpers
    def _format_return(self, result: Any) -> str:
        if not result:
            return "[dim]-[/]"
        perf = getattr(result, "performance", None)
        if not perf or not hasattr(perf, "total_return_pct"):
            return "[dim]-[/]"
        ret = perf.total_return_pct
        return f"[green]{ret:+.1f}%[/]" if ret >= 0 else f"[red]{ret:+.1f}%[/]"

    def _format_sharpe(self, result: Any) -> str:
        if not result:
            return "[dim]-[/]"
        risk = getattr(result, "risk", None)
        if not risk or not hasattr(risk, "sharpe_ratio"):
            return "[dim]-[/]"
        return f"{risk.sharpe_ratio:.2f}"

    def _format_max_dd(self, result: Any) -> str:
        if not result:
            return "[dim]-[/]"
        risk = getattr(result, "risk", None)
        if not risk or not hasattr(risk, "max_drawdown"):
            return "[dim]-[/]"
        return f"[red]{risk.max_drawdown:.1f}%[/]"

    def _format_trades(self, result: Any) -> str:
        if not result:
            return "[dim]-[/]"
        trades = getattr(result, "trades", None)
        if not trades or not hasattr(trades, "total_trades"):
            return "[dim]-[/]"
        return str(trades.total_trades)

    def _format_win_rate(self, result: Any) -> str:
        if not result:
            return "[dim]-[/]"
        trades = getattr(result, "trades", None)
        if not trades or not hasattr(trades, "win_rate"):
            return "[dim]-[/]"
        return f"{trades.win_rate:.0f}%"

    def _format_status(self, name: str) -> str:
        if self._state.running == name:
            return "[yellow]Running...[/]"
        if name in self._state.failures:
            return "[red]Failed[/]"
        if name in self._state.results:
            return "[green]Done[/]"
        return "[dim]-[/]"
