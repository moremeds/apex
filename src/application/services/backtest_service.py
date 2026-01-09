"""
BacktestService - Application layer service for running backtests.

Encapsulates backtest orchestration logic:
- Engine instantiation
- Data feed configuration
- Config loading from YAML

This service is framework-agnostic and can be called from TUI, CLI, or API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import yaml

if TYPE_CHECKING:
    from src.backtest.execution.engines.backtest_engine import BacktestResult


@dataclass
class BacktestRequest:
    """Request parameters for running a backtest."""

    strategy_name: str
    symbols: List[str]
    start_date: date
    end_date: date
    initial_capital: float = 100000.0


class BacktestService:
    """
    Service for running strategy backtests.

    Responsibilities:
    - Load broker configuration from YAML
    - Create and configure BacktestEngine
    - Create and configure data feed
    - Execute backtest asynchronously
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        ib_host: str = "127.0.0.1",
        ib_port: int = 4001,
        ib_client_id: int = 10,
    ) -> None:
        """
        Initialize the backtest service.

        Args:
            config_path: Path to base.yaml config (optional, will auto-load).
            ib_host: IB Gateway host (fallback if config not found).
            ib_port: IB Gateway port (fallback if config not found).
            ib_client_id: IB client ID (fallback if config not found).
        """
        self._config_path = config_path or Path("config/base.yaml")
        self._ib_host = ib_host
        self._ib_port = ib_port
        self._ib_client_id = ib_client_id

        # Load config on init
        self._load_config()

    def _load_config(self) -> None:
        """Load broker configuration from YAML."""
        if not self._config_path.exists():
            return

        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)

        ibkr = cfg.get("brokers", {}).get("ibkr", {})
        self._ib_host = ibkr.get("host", self._ib_host)
        self._ib_port = ibkr.get("port", self._ib_port)

        # Use last client ID from historical pool
        historical_pool = ibkr.get("client_ids", {}).get("historical_pool", [])
        if historical_pool:
            self._ib_client_id = historical_pool[-1]

    async def run(self, request: BacktestRequest) -> "BacktestResult":
        """
        Run a backtest for the given strategy.

        Args:
            request: BacktestRequest with strategy name, symbols, dates.

        Returns:
            BacktestResult with performance metrics.

        Raises:
            Exception: If backtest fails (engine error, data error, etc.)
        """
        from src.backtest.execution.engines.backtest_engine import (
            BacktestConfig,
            BacktestEngine,
        )
        from src.backtest.data.feeds import BarCacheDataFeed

        # Create engine config
        config = BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            symbols=request.symbols,
            initial_capital=request.initial_capital,
            strategy_name=request.strategy_name,
        )

        # Create engine and set strategy
        engine = BacktestEngine(config)
        engine.set_strategy(strategy_name=request.strategy_name)

        # Create data feed
        feed = BarCacheDataFeed(
            symbols=config.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            host=self._ib_host,
            port=self._ib_port,
            client_id=self._ib_client_id,
        )
        engine.set_data_feed(feed)

        # Run backtest
        return await engine.run()

    async def run_strategy(
        self,
        strategy_name: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        initial_capital: float = 100000.0,
    ) -> "BacktestResult":
        """
        Convenience method to run a backtest with defaults.

        Args:
            strategy_name: Name of the registered strategy.
            symbols: List of symbols (defaults to ["AAPL", "MSFT"]).
            start_date: Start date (defaults to 2024-01-01).
            end_date: End date (defaults to 2024-06-30).
            initial_capital: Initial capital (defaults to 100,000).

        Returns:
            BacktestResult with performance metrics.
        """
        request = BacktestRequest(
            strategy_name=strategy_name,
            symbols=symbols or ["AAPL", "MSFT"],
            start_date=start_date or date(2024, 1, 1),
            end_date=end_date or date(2024, 6, 30),
            initial_capital=initial_capital,
        )
        return await self.run(request)
