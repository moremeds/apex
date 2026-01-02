"""
ApexEngine - Event-driven backtest engine adapter.

Wraps the legacy BacktestEngine (bar-by-bar event-driven) to implement
the BaseEngine interface used by the systematic runner.

Key characteristics:
- Full-featured: Supports all strategy features (scheduled actions, fills, etc.)
- Event-driven: Processes bars one at a time with strategy callbacks
- Slower: ~100-1000x slower than VectorBT due to Python overhead
- Use case: Final validation of top candidates from VectorBT screening

Usage:
    engine = ApexEngine(config)
    result = engine.run(spec, data)  # RunSpec -> RunResult
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ...core import RunSpec, RunResult, RunMetrics, RunStatus
from .interface import BaseEngine, EngineConfig, EngineType

logger = logging.getLogger(__name__)


@dataclass
class ApexEngineConfig(EngineConfig):
    """Configuration for the Apex event-driven engine."""

    engine_type: EngineType = field(default=EngineType.APEX)

    # Data settings
    data_source: str = "historical"  # historical (parquet), ib, csv
    bar_size: str = "1d"

    # Execution settings
    fill_model: str = "immediate"  # immediate, next_bar, slippage
    reality_pack_name: Optional[str] = None  # e.g., "ib", "futu_us"

    # Streaming mode for large datasets
    streaming: bool = True


class ApexEngine(BaseEngine):
    """
    Event-driven backtest engine implementing BaseEngine interface.

    Wraps the legacy BacktestEngine to provide:
    - run(spec, data) -> RunResult (standard interface)
    - Full strategy feature support (scheduled actions, on_fill, etc.)
    - Reality pack integration (slippage, fees, fill models)

    Trade-offs vs VectorBT:
    - Slower: Python event loop overhead
    - More accurate: Full strategy lifecycle, realistic execution
    - Full-featured: All Strategy methods work (on_bar, on_fill, scheduled)
    """

    def __init__(self, config: Optional[ApexEngineConfig] = None):
        """
        Initialize ApexEngine.

        Args:
            config: Engine configuration (defaults to ApexEngineConfig)
        """
        super().__init__(config or ApexEngineConfig())

        # Cast to specific config type
        self._apex_config = config if isinstance(config, ApexEngineConfig) else ApexEngineConfig()

    @property
    def engine_type(self) -> EngineType:
        """Return APEX engine type."""
        return EngineType.APEX

    @property
    def supports_vectorization(self) -> bool:
        """ApexEngine does not support vectorization."""
        return False

    def run(self, spec: RunSpec, data: Optional[pd.DataFrame] = None) -> RunResult:
        """
        Execute a single backtest run.

        Converts RunSpec to BacktestConfig, runs the event-driven engine,
        and converts BacktestResult to RunResult.

        Args:
            spec: Run specification (symbol, window, params)
            data: Optional pre-loaded OHLCV data (not used - ApexEngine uses DataFeed)

        Returns:
            RunResult with metrics and optional equity curve
        """
        started_at = datetime.now()

        try:
            # Run async engine in sync context
            backtest_result = self._run_async_safely(spec)

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            # Convert BacktestResult to RunResult
            return self._convert_result(spec, backtest_result, started_at, completed_at, duration)

        except Exception as e:
            logger.error(f"ApexEngine run failed for {spec.symbol}: {e}")
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            return RunResult(
                run_id=spec.run_id or f"apex-{spec.symbol}",
                trial_id=spec.trial_id,
                experiment_id=spec.experiment_id or "",
                symbol=spec.symbol,
                window_id=spec.window.window_id,
                profile_version=spec.profile_version,
                data_version=spec.data_version,
                status=RunStatus.FAIL_EXECUTION,
                error=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                is_train=spec.window.is_train,
                is_oos=spec.window.is_oos,
                params=spec.params,
            )

    def _run_async_safely(self, spec: RunSpec):
        """
        Run the async BacktestEngine safely.

        Handles event loop creation/reuse for both main and worker processes.
        """
        # Import here to avoid circular dependency
        from ..simulated import SimulatedExecution, FillModel as SimFillModel
        from ...data.feeds import HistoricalStoreDataFeed, CsvDataFeed, create_data_feed

        # Build BacktestConfig from RunSpec
        from .backtest_engine import BacktestEngine, BacktestConfig

        backtest_dict = spec.to_backtest_config()

        # Map fill model
        fill_model_map = {
            "immediate": SimFillModel.IMMEDIATE,
            "next_bar": SimFillModel.NEXT_BAR,
            "slippage": SimFillModel.SLIPPAGE,
        }

        config = BacktestConfig(
            start_date=backtest_dict["start_date"],
            end_date=backtest_dict["end_date"],
            symbols=backtest_dict["symbols"],
            initial_capital=backtest_dict["initial_capital"],
            bar_size=self._apex_config.bar_size,
            fill_model=fill_model_map.get(self._apex_config.fill_model, SimFillModel.IMMEDIATE),
            slippage_bps=backtest_dict.get("slippage_bps", 5.0),
            commission_per_share=backtest_dict.get("commission_per_share", 0.005),
            reality_pack_name=self._apex_config.reality_pack_name,
            strategy_params=backtest_dict.get("strategy_params", {}),
        )

        # Get strategy name from params
        strategy_name = spec.params.get("strategy_name") or spec.params.get("strategy")

        async def run_backtest():
            engine = BacktestEngine(config)

            # Set strategy
            if strategy_name:
                engine.set_strategy(strategy_name=strategy_name, params=spec.params)

            # Create data feed based on config
            feed = create_data_feed(
                source=self._apex_config.data_source,
                symbols=config.symbols,
                start_date=config.start_date,
                end_date=config.end_date,
                streaming=self._apex_config.streaming,
                bar_size=self._apex_config.bar_size,
            )
            engine.set_data_feed(feed)

            return await engine.run()

        # Run in event loop
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_backtest())
                return future.result()
        except RuntimeError:
            # No running loop - create one
            return asyncio.run(run_backtest())

    def _convert_result(
        self,
        spec: RunSpec,
        backtest_result,
        started_at: datetime,
        completed_at: datetime,
        duration: float,
    ) -> RunResult:
        """Convert BacktestResult to RunResult."""
        # Use the existing conversion helper
        metrics = RunMetrics.from_backtest_result(backtest_result)

        # Add additional metrics from trade tracker
        if backtest_result.trades:
            metrics.max_consecutive_wins = backtest_result.trades.max_consecutive_wins
            metrics.max_consecutive_losses = backtest_result.trades.max_consecutive_losses
            metrics.avg_trade_duration_days = (
                backtest_result.trades.avg_trade_duration_seconds / 86400
                if backtest_result.trades.avg_trade_duration_seconds
                else 0
            )

        return RunResult(
            run_id=spec.run_id or backtest_result.strategy_id,
            trial_id=spec.trial_id,
            experiment_id=spec.experiment_id or "",
            symbol=spec.symbol,
            window_id=spec.window.window_id,
            profile_version=spec.profile_version,
            data_version=spec.data_version,
            status=RunStatus.SUCCESS,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            metrics=metrics,
            is_train=spec.window.is_train,
            is_oos=spec.window.is_oos,
            equity_curve=backtest_result.equity_curve,
            trade_log=[
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                }
                for t in backtest_result.trade_log
            ] if backtest_result.trade_log else None,
            params=spec.params,
        )

    def run_batch(
        self,
        specs: List[RunSpec],
        data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[RunResult]:
        """
        Execute multiple runs sequentially.

        ApexEngine does not support vectorization, so runs are sequential.
        For parallel execution, use ParallelRunner with multiple ApexEngine instances.

        Args:
            specs: List of run specifications
            data: Optional dict of symbol -> OHLCV DataFrame (not used)

        Returns:
            List of RunResults in same order as specs
        """
        results = []
        for i, spec in enumerate(specs):
            logger.info(f"ApexEngine run {i+1}/{len(specs)}: {spec.symbol}")
            result = self.run(spec, data.get(spec.symbol) if data else None)
            results.append(result)
        return results
