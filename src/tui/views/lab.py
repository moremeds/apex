"""
Lab view for strategy backtesting (Tab 5).

Layout matching original Rich dashboard:
- Left (65%): Strategy list with columns
- Right top (35%): Strategy config panel
- Right bottom: Last backtest results panel
- Bottom: Component Health bar

Keyboard shortcuts:
- w/s: Navigate strategies
- Enter: Run backtest for selected strategy
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static
from textual.app import ComposeResult
from textual.binding import Binding
from textual import work

from ..widgets.strategy_list import StrategyList
from ..widgets.strategy_config import StrategyConfigPanel
from ..widgets.backtest_results import BacktestResultsPanel
from ..widgets.health_bar import HealthBar

if TYPE_CHECKING:
    from ...domain.backtest.backtest_result import BacktestResult


class LabView(Container, can_focus=True):
    """Strategy lab view for backtesting."""

    # Styles are in css/dashboard.tcss using Rich-matching palette
    DEFAULT_CSS = ""

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "run_backtest", "Run Backtest", show=True),
    ]

    def on_show(self) -> None:
        """Focus this view when it becomes visible."""
        self.focus()

    def on_unmount(self) -> None:
        """Cancel any running workers on unmount to prevent orphaned tasks."""
        # Textual @work(exclusive=True) workers are tracked and can be cancelled
        self.workers.cancel_all()

    def compose(self) -> ComposeResult:
        """Compose the lab view layout."""
        with Horizontal(id="lab-main"):
            # Left side - Strategy list (~65%)
            with Vertical(id="lab-left"):
                with Vertical(id="lab-left-panel"):
                    yield Static("Strategy Lab", id="lab-left-title", classes="panel-title")
                    yield StrategyList(id="lab-strategies")
                    yield Static(
                        "Up/Down: select   Enter: run backtest",
                        id="lab-hints",
                        classes="panel-hints",
                    )

            # Right side - Config + Results (~35%)
            with Vertical(id="lab-right"):
                with Vertical(id="lab-config-panel"):
                    yield Static("Strategy Config", id="lab-config-title", classes="panel-title")
                    yield StrategyConfigPanel(id="lab-config")
                with Vertical(id="lab-results-panel"):
                    yield Static("Last Backtest", id="lab-results-title", classes="panel-title")
                    yield BacktestResultsPanel(id="lab-results")

        # Bottom - Health bar (full width)
        with Vertical(id="lab-health-panel"):
            yield Static("Component Health", id="lab-health-title", classes="panel-title")
            yield HealthBar(id="lab-health", classes="compact-health")

    def on_strategy_list_strategy_selected(
        self, event: StrategyList.StrategySelected
    ) -> None:
        """Handle strategy selection."""
        try:
            config_title = self.query_one("#lab-config-title", Static)
            config_title.update(f"[bold #7ee787]{event.strategy_name} Config[/]")
            config_panel = self.query_one("#lab-config", StrategyConfigPanel)
            config_panel.strategy_name = event.strategy_name
            config_panel.strategy_info = event.strategy_info
        except Exception:
            self.log.exception("Failed to update config panel")

        try:
            results_panel = self.query_one("#lab-results", BacktestResultsPanel)
            results_panel.strategy_name = event.strategy_name
        except Exception:
            self.log.exception("Failed to update results panel")

    def on_strategy_list_strategy_activated(
        self, event: StrategyList.StrategyActivated
    ) -> None:
        """Handle strategy activation (run backtest)."""
        self._execute_backtest(event.strategy_name)

    def action_move_up(self) -> None:
        """Move selection up in the strategy list."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            if strategy_list.cursor_row is not None and strategy_list.cursor_row > 0:
                strategy_list.move_cursor(row=strategy_list.cursor_row - 1)
        except Exception:
            self.log.exception("Failed to move up in strategy list")

    def action_move_down(self) -> None:
        """Move selection down in the strategy list."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            if strategy_list.cursor_row is not None and strategy_list.cursor_row < strategy_list.row_count - 1:
                strategy_list.move_cursor(row=strategy_list.cursor_row + 1)
        except Exception:
            self.log.exception("Failed to move down in strategy list")

    def action_run_backtest(self) -> None:
        """Run backtest for the selected strategy."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            strategy_name = strategy_list.get_selected_strategy()
            if strategy_name:
                self._execute_backtest(strategy_name)
        except Exception:
            self.log.exception("Failed to run backtest")

    def _execute_backtest(self, strategy_name: str) -> None:
        """Execute backtest in background."""
        # Mark strategy as running
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            strategy_list.set_running_strategy(strategy_name)
        except Exception:
            self.log.exception("Failed to mark strategy as running")

        # Run backtest in background worker
        self._run_backtest_worker(strategy_name)

    @work(exclusive=True)
    async def _run_backtest_worker(self, strategy_name: str) -> None:
        """Run backtest as async task on main event loop."""
        try:
            import yaml
            from pathlib import Path
            from ...infrastructure.backtest.backtest_engine import BacktestEngine, BacktestConfig
            from ...infrastructure.backtest.data_feeds import BarCacheDataFeed

            # Load config from base.yaml
            config_path = Path("config/base.yaml")
            ib_host = "127.0.0.1"
            ib_port = 4001
            ib_client_id = 10  # Use last client from historical_pool

            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                    ibkr = cfg.get("brokers", {}).get("ibkr", {})
                    ib_host = ibkr.get("host", ib_host)
                    ib_port = ibkr.get("port", ib_port)
                    historical_pool = ibkr.get("client_ids", {}).get("historical_pool", [10])
                    if historical_pool:
                        ib_client_id = historical_pool[-1]  # Use last client ID

            # Default backtest configuration
            config = BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 6, 30),
                symbols=["AAPL", "MSFT"],
                initial_capital=100000.0,
                strategy_name=strategy_name,
            )

            # Create engine
            engine = BacktestEngine(config)
            engine.set_strategy(strategy_name=strategy_name)

            # Use IB historical data with config from base.yaml
            feed = BarCacheDataFeed(
                symbols=config.symbols,
                start_date=config.start_date,
                end_date=config.end_date,
                host=ib_host,
                port=ib_port,
                client_id=ib_client_id,
            )

            engine.set_data_feed(feed)

            # Run backtest (async on main loop)
            result = await engine.run()
            self._on_backtest_complete(strategy_name, result)

        except Exception:
            self._on_backtest_error(strategy_name, str(e))

    def _on_backtest_complete(self, strategy_name: str, result: "BacktestResult") -> None:
        """Handle backtest completion."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            strategy_list.set_running_strategy(None)
            strategy_list.set_backtest_result(strategy_name, result)
        except Exception:
            self.log.exception("Failed to update strategy list on complete")

        try:
            results_panel = self.query_one("#lab-results", BacktestResultsPanel)
            results_panel.result = result
        except Exception:
            self.log.exception("Failed to update results panel on complete")

    def _on_backtest_error(self, strategy_name: str, error: str) -> None:
        """Handle backtest error with user notification."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            strategy_list.set_running_strategy(None)
            strategy_list.set_backtest_failure(strategy_name, error)
        except Exception:
            self.log.exception("Failed to update strategy list on error")

        try:
            results_panel = self.query_one("#lab-results", BacktestResultsPanel)
            results_panel.error_message = error
        except Exception:
            self.log.exception("Failed to update results panel on error")

        # User notification for backtest errors
        self.notify(
            f"Backtest failed for {strategy_name}: {error[:100]}",
            severity="error",
            timeout=10.0,
        )

    def update_health(self, health: List[Any]) -> None:
        """Update health bar."""
        try:
            health_bar = self.query_one("#lab-health", HealthBar)
            health_bar.health = health
        except Exception:
            self.log.exception("Failed to update health bar")

    def set_backtest_result(self, strategy_name: str, result: "BacktestResult") -> None:
        """Set backtest result for a strategy."""
        try:
            strategy_list = self.query_one("#lab-strategies", StrategyList)
            strategy_list.set_backtest_result(strategy_name, result)
        except Exception:
            self.log.exception("Failed to set backtest result in strategy list")

        try:
            results_panel = self.query_one("#lab-results", BacktestResultsPanel)
            if results_panel.strategy_name == strategy_name:
                results_panel.result = result
        except Exception:
            self.log.exception("Failed to set backtest result in results panel")
