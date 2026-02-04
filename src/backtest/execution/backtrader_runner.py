"""
Backtrader Runner - Alternative backtest engine using Backtrader framework.

Requires: pip install backtrader
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.backtest.backtest_result import BacktestResult
from ...domain.backtest.backtest_spec import BacktestSpec
from ...domain.strategy.registry import get_strategy_class, list_strategies

# Check if backtrader is available
try:
    import backtrader as bt

    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None

logger = logging.getLogger(__name__)


class BacktraderRunner:
    """Runner using Backtrader engine. Requires: pip install backtrader"""

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100000.0,
        data_source: str = "csv",
        data_dir: str = "./data",
        bar_size: str = "1d",
        strategy_params: Optional[Dict[str, Any]] = None,
        commission: float = 0.001,
    ):
        if not BACKTRADER_AVAILABLE:
            raise ImportError("backtrader not installed. Run: pip install backtrader")

        self.strategy_name = strategy_name
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data_source = data_source
        self.data_dir = data_dir
        self.bar_size = bar_size
        self.strategy_params = strategy_params or {}
        self.commission = commission
        self._spec: Optional[BacktestSpec] = None

    async def run(self) -> BacktestResult:
        """Run the backtest using Backtrader engine."""
        import time

        from ...domain.reality import RealityModelPack, get_preset_pack
        from .backtrader_adapter import run_backtest_with_backtrader

        self._print_config()

        reality_pack = None
        if self._spec:
            if self._spec.reality_model:
                try:
                    reality_pack = RealityModelPack.from_config(self._spec.reality_model)
                except Exception as e:
                    logger.error(f"Failed to load reality_model from spec: {e}")

            if (
                reality_pack is None
                and hasattr(self._spec.execution, "reality_pack")
                and self._spec.execution.reality_pack
            ):
                try:
                    reality_pack = get_preset_pack(self._spec.execution.reality_pack)
                except Exception as e:
                    logger.error(f"Failed to load reality_pack preset: {e}")

        start_time = time.time()

        strategy_class = get_strategy_class(self.strategy_name)
        if strategy_class is None:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. Available: {list_strategies()}"
            )

        data_feeds = self._create_data_feeds()

        results = run_backtest_with_backtrader(
            apex_strategy_class=strategy_class,
            data_feeds=data_feeds,
            initial_cash=self.initial_capital,
            commission=self.commission,
            strategy_params=self.strategy_params,
            reality_pack=reality_pack,
        )

        run_duration = time.time() - start_time
        result = self._convert_result(results, run_duration)
        result.print_summary()

        return result

    def _create_data_feeds(self) -> List[Any]:
        """Create Backtrader data feeds."""
        feeds = []

        for symbol in self.symbols:
            if self.data_source == "csv":
                csv_path = Path(self.data_dir) / f"{symbol}.csv"
                if not csv_path.exists():
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")

                feed = bt.feeds.GenericCSVData(
                    dataname=str(csv_path),
                    dtformat="%Y-%m-%d",
                    fromdate=datetime.combine(self.start_date, datetime.min.time()),
                    todate=datetime.combine(self.end_date, datetime.max.time()),
                    datetime=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=5,
                    openinterest=-1,
                )
                feed._name = symbol
                feeds.append(feed)

            elif self.data_source == "yahoo":
                feed = bt.feeds.YahooFinanceData(
                    dataname=symbol,
                    fromdate=datetime.combine(self.start_date, datetime.min.time()),
                    todate=datetime.combine(self.end_date, datetime.max.time()),
                )
                feed._name = symbol
                feeds.append(feed)

            else:
                raise ValueError(f"Unsupported data source for Backtrader: {self.data_source}")

        return feeds

    def _convert_result(self, bt_results: Dict[str, Any], run_duration: float) -> BacktestResult:
        """Convert Backtrader results to BacktestResult."""
        from ...domain.backtest.backtest_result import (
            CostMetrics,
            PerformanceMetrics,
            RiskMetrics,
            TradeMetrics,
        )

        initial = self.initial_capital
        final = bt_results.get("final_value", initial)
        total_return = (final - initial) / initial if initial > 0 else 0

        trading_days = (self.end_date - self.start_date).days * 252 // 365

        performance = PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return * 100,
            cagr=self._calculate_cagr(initial, final, trading_days),
            annualized_return=total_return * 252 / max(trading_days, 1),
        )

        sharpe = bt_results.get("sharpe_ratio") or 0.0
        max_dd = bt_results.get("max_drawdown") or 0.0
        risk = RiskMetrics(max_drawdown=max_dd, max_drawdown_duration_days=0, sharpe_ratio=sharpe)

        total_trades = bt_results.get("total_trades", 0)
        trades = TradeMetrics(total_trades=total_trades)

        estimated_commission = total_trades * 2 * 100 * self.commission
        costs = CostMetrics(
            total_commission=estimated_commission,
            cost_pct_of_capital=(estimated_commission / initial * 100) if initial > 0 else 0,
        )

        return BacktestResult(
            strategy_name=self.strategy_name,
            strategy_id=f"backtrader-{self.strategy_name}",
            start_date=self.start_date,
            end_date=self.end_date,
            trading_days=trading_days,
            initial_capital=initial,
            final_capital=final,
            symbols=self.symbols,
            performance=performance,
            risk=risk,
            trades=trades,
            costs=costs,
            equity_curve=[],
            run_duration_seconds=run_duration,
            engine="backtrader",
        )

    def _calculate_cagr(self, initial: float, final: float, days: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        if initial <= 0 or days <= 0:
            return 0.0
        years = days / 252
        if years <= 0:
            return 0.0
        return float(((final / initial) ** (1 / years) - 1) * 100)

    def _print_config(self) -> None:
        """Log backtest configuration using structured logging."""
        config_info = {
            "engine": "backtrader",
            "strategy": self.strategy_name,
            "symbols": self.symbols,
            "start": str(self.start_date),
            "end": str(self.end_date),
            "capital": self.initial_capital,
            "data_source": self.data_source,
            "bar_size": self.bar_size,
            "commission": self.commission,
        }

        if self.strategy_params:
            config_info["params"] = self.strategy_params

        logger.info(
            f"Backtrader config: {self.strategy_name} on {','.join(self.symbols)} "
            f"({self.start_date} to {self.end_date})"
        )
        logger.debug(f"Full config: {config_info}")
