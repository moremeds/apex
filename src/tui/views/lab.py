"""
Lab view for strategy backtesting (Tab 5).

Layout matching original Rich dashboard:
- Left (35%): Strategy list with columns
- Right top (65%): Strategy config panel
- Right bottom: Last backtest results panel
- Bottom: Component Health bar

Keyboard shortcuts:
- w/s: Navigate strategies
- Enter: Run backtest for selected strategy
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static
from textual.app import ComposeResult
from textual.binding import Binding

from ..widgets.strategy_list import StrategyList
from ..widgets.strategy_config import StrategyConfigPanel
from ..widgets.backtest_results import BacktestResultsPanel
from ..widgets.health_bar import HealthBar

if TYPE_CHECKING:
    from ...domain.backtest.backtest_result import BacktestResult


class LabView(Container):
    """Strategy lab view for backtesting."""

    DEFAULT_CSS = """
    LabView {
        height: 1fr;
        width: 1fr;
    }

    LabView > Horizontal {
        height: 1fr;
        width: 1fr;
    }

    LabView > Horizontal > Vertical {
        height: 1fr;
    }

    LabView #lab-left {
        width: 1fr;
    }

    LabView #lab-right {
        width: 2fr;
    }

    LabView #lab-health {
        height: 5;
        width: 1fr;
    }

    LabView StrategyList {
        height: 1fr;
        border: solid blue;
    }

    LabView StrategyConfigPanel {
        height: 1fr;
        border: solid green;
    }

    LabView BacktestResultsPanel {
        height: 1fr;
        border: solid yellow;
    }
    """

    BINDINGS = [
        Binding("w", "move_up", "Up", show=True),
        Binding("s", "move_down", "Down", show=True),
        Binding("enter", "run_backtest", "Run Backtest", show=True),
    ]

    def compose(self) -> ComposeResult:
        """Compose the lab view layout."""
        with Horizontal(id="lab-main"):
            # Left side - Strategy list (35%)
            with Vertical(id="lab-left"):
                yield Static("Strategy Lab", classes="panel-title")
                yield StrategyList(id="lab-strategies")

            # Right side - Config + Results (65%)
            with Vertical(id="lab-right"):
                yield StrategyConfigPanel(id="lab-config")
                yield BacktestResultsPanel(id="lab-results")

        # Bottom - Health bar (full width)
        yield HealthBar(id="lab-health")

    def on_strategy_list_strategy_selected(
        self, event: StrategyList.StrategySelected
    ) -> None:
        """Handle strategy selection."""
        try:
            config_panel = self.query_one("#lab-config", StrategyConfigPanel)
            config_panel.strategy_name = event.strategy_name
            config_panel.strategy_info = event.strategy_info
        except Exception:
            pass

        try:
            results_panel = self.query_one("#lab-results", BacktestResultsPanel)
            results_panel.strategy_name = event.strategy_name
        except Exception:
            pass

    def action_move_up(self) -> None:
        """Move selection up in the strategy list."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            if strategy_list.cursor_row is not None and strategy_list.cursor_row > 0:
                strategy_list.cursor_coordinate = (strategy_list.cursor_row - 1, 0)
        except Exception:
            pass

    def action_move_down(self) -> None:
        """Move selection down in the strategy list."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            if strategy_list.cursor_row is not None and strategy_list.cursor_row < strategy_list.row_count - 1:
                strategy_list.cursor_coordinate = (strategy_list.cursor_row + 1, 0)
        except Exception:
            pass

    def action_run_backtest(self) -> None:
        """Run backtest for the selected strategy."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            strategy_name = strategy_list.get_selected_strategy()
            if strategy_name:
                self._execute_backtest(strategy_name)
        except Exception:
            pass

    def _execute_backtest(self, strategy_name: str) -> None:
        """Execute backtest in background."""
        # TODO: Implement backtest execution using @work decorator
        pass

    def update_health(self, health: List[Any]) -> None:
        """Update health bar."""
        try:
            health_bar = self.query_one("#lab-health", HealthBar)
            health_bar.health = health
        except Exception:
            pass

    def set_backtest_result(self, strategy_name: str, result: "BacktestResult") -> None:
        """Set backtest result for a strategy."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            strategy_list.set_backtest_result(strategy_name, result)
        except Exception:
            pass

        try:
            results_panel = self.query_one("#lab-results", BacktestResultsPanel)
            if results_panel.strategy_name == strategy_name:
                results_panel.result = result
        except Exception:
            pass
