"""
VectorBT backtest engine for fast strategy screening.

VectorBT uses NumPy broadcasting to run thousands of parameter combinations
simultaneously, making it ideal for the screening stage of the two-stage pipeline.

Key advantages:
- 100-1000x faster than event-driven backtesting for simple strategies
- Native parameter vectorization (run 10,000 variants in one pass)
- Memory-efficient via chunking

Limitations:
- Simpler execution model (no complex order types)
- No intraday data support
- Limited slippage/commission models

Use for: Fast screening, parameter space exploration
Use Apex for: Final validation, complex strategies, realistic execution
"""

from dataclasses import dataclass
from datetime import date, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import yaml

from ...core import RunSpec, RunResult, RunMetrics, RunStatus
from ...analysis import MetricsCalculator, Trade
from .interface import BaseEngine, EngineConfig, EngineType


@dataclass
class VectorBTConfig(EngineConfig):
    """VectorBT-specific configuration."""

    engine_type: EngineType = EngineType.VECTORBT

    # VectorBT settings
    freq: str = "1D"  # Data frequency
    init_cash: float = 100000.0
    size_type: str = "amount"  # amount, value, percent

    # Strategy-specific
    strategy_type: str = "ma_cross"  # ma_cross, rsi, custom

    # Performance
    use_numba: bool = True
    chunk_size: int = 1000  # For large parameter sweeps


class VectorBTEngine(BaseEngine):
    """
    VectorBT-based backtest engine.

    Provides fast vectorized backtesting for strategy screening.
    Supports common strategy patterns (MA crossover, RSI, etc.)
    and custom signal functions.

    Example:
        engine = VectorBTEngine(VectorBTConfig(strategy_type="ma_cross"))
        result = engine.run(run_spec)

        # Batch with parameter vectorization
        results = engine.run_batch(specs)  # Vectorized for speed
    """

    def __init__(self, config: Optional[VectorBTConfig] = None):
        super().__init__(config or VectorBTConfig())
        self._vbt_config = config or VectorBTConfig()
        self._manifest_cache: Optional[Dict[str, Any]] = None

    @property
    def engine_type(self) -> EngineType:
        return EngineType.VECTORBT

    @property
    def supports_vectorization(self) -> bool:
        return True

    def _load_manifest(self) -> Dict[str, Any]:
        """Load and cache strategy manifest.yaml."""
        if self._manifest_cache is None:
            manifest_path = Path(__file__).parents[3] / "domain/strategy/manifest.yaml"
            with manifest_path.open("r", encoding="utf-8") as f:
                self._manifest_cache = yaml.safe_load(f) or {}
        return self._manifest_cache

    def _get_signal_generator(self, strategy_name: str) -> Type:
        """
        Lazy import SignalGenerator from manifest.

        Args:
            strategy_name: Strategy name to look up

        Returns:
            SignalGenerator class

        Raises:
            KeyError: If strategy not found in manifest
            ValueError: If strategy is apex_only or has no signals
        """
        manifest = self._load_manifest()
        strategies = manifest.get("strategies", {})

        if strategy_name not in strategies:
            raise KeyError(f"Strategy '{strategy_name}' not found in manifest")

        strategy_entry = strategies[strategy_name]

        if strategy_entry.get("apex_only"):
            raise ValueError(
                f"Strategy '{strategy_name}' is apex_only and cannot run in VectorBT"
            )

        signals_path = strategy_entry.get("signals")
        if signals_path is None:
            raise ValueError(
                f"Strategy '{strategy_name}' has no SignalGenerator defined"
            )

        # Lazy import: "src.domain.strategy.signals.ma_cross:MACrossSignalGenerator"
        module_path, class_name = signals_path.rsplit(":", 1)
        module = import_module(module_path)
        return getattr(module, class_name)

    def run(self, spec: RunSpec, data: Optional[pd.DataFrame] = None) -> RunResult:
        """
        Execute a single backtest run.

        Args:
            spec: Run specification
            data: Optional pre-loaded OHLCV data

        Returns:
            RunResult with computed metrics
        """
        import vectorbt as vbt

        started_at = datetime.now()

        try:
            # Load data if not provided
            if data is None:
                data = self.load_data(spec.symbol, spec.start_date, spec.end_date)

            if data is None or data.empty:
                return self._create_error_result(
                    spec, RunStatus.FAIL_DATA, "No data available", started_at
                )

            # Filter to date range
            data = self._filter_date_range(data, spec.start_date, spec.end_date)
            if data.empty:
                return self._create_error_result(
                    spec, RunStatus.FAIL_DATA, "No data in date range", started_at
                )

            # Get strategy signal generator from manifest
            strategy_type = spec.params.get("strategy_type", self._vbt_config.strategy_type)

            try:
                signal_generator_cls = self._get_signal_generator(strategy_type)
                signal_generator = signal_generator_cls()
            except (KeyError, ValueError) as e:
                return self._create_error_result(
                    spec, RunStatus.FAIL_STRATEGY, str(e), started_at
                )

            # Generate signals
            entries, exits = signal_generator.generate(data, spec.params)

            # Run backtest
            close = data["close"]
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=entries,
                exits=exits,
                init_cash=spec.initial_capital,
                fees=spec.commission_per_share / close.mean(),  # Approximate
                slippage=spec.slippage_bps / 10000,
                freq=self._vbt_config.freq,
            )

            # Extract metrics
            metrics = self._extract_metrics(pf, data)

            completed_at = datetime.now()
            return RunResult(
                run_id=spec.run_id,
                trial_id=spec.trial_id,
                experiment_id=spec.experiment_id or "",
                symbol=spec.symbol,
                window_id=spec.window.window_id,
                profile_version=spec.profile_version,
                data_version=spec.data_version,
                status=RunStatus.SUCCESS,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                metrics=metrics,
                is_train=spec.window.is_train,
                is_oos=spec.window.is_oos,
                params=spec.params,
            )

        except Exception as e:
            return self._create_error_result(
                spec, RunStatus.FAIL_EXECUTION, str(e), started_at
            )

    def run_batch(
        self,
        specs: List[RunSpec],
        data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[RunResult]:
        """
        Execute multiple runs with vectorization where possible.

        Groups specs by symbol and strategy, then vectorizes parameter sweeps.

        Args:
            specs: List of run specifications
            data: Optional dict of symbol -> OHLCV DataFrame

        Returns:
            List of RunResults in same order as specs
        """
        if not specs:
            return []

        # Group by symbol for efficient data loading
        symbol_groups: Dict[str, List[Tuple[int, RunSpec]]] = {}
        for i, spec in enumerate(specs):
            if spec.symbol not in symbol_groups:
                symbol_groups[spec.symbol] = []
            symbol_groups[spec.symbol].append((i, spec))

        # Process each symbol group
        results: List[Optional[RunResult]] = [None] * len(specs)

        for symbol, indexed_specs in symbol_groups.items():
            # Load data once per symbol
            symbol_data = data.get(symbol) if data else None
            if symbol_data is None and indexed_specs:
                first_spec = indexed_specs[0][1]
                symbol_data = self.load_data(
                    symbol, first_spec.start_date, first_spec.end_date
                )

            # Check if we can vectorize (same strategy, same date range)
            can_vectorize = self._can_vectorize(indexed_specs)

            if can_vectorize and len(indexed_specs) > 1:
                batch_results = self._run_vectorized(indexed_specs, symbol_data)
                for (idx, _), result in zip(indexed_specs, batch_results):
                    results[idx] = result
            else:
                # Fall back to sequential
                for idx, spec in indexed_specs:
                    results[idx] = self.run(spec, symbol_data)

        return results  # type: ignore

    def _can_vectorize(self, indexed_specs: List[Tuple[int, RunSpec]]) -> bool:
        """Check if a group of specs can be vectorized together."""
        if len(indexed_specs) <= 1:
            return False

        # Must have same strategy type and date range
        first_spec = indexed_specs[0][1]
        first_strategy = first_spec.params.get("strategy_type", self._vbt_config.strategy_type)
        first_range = (first_spec.start_date, first_spec.end_date)

        for _, spec in indexed_specs[1:]:
            strategy = spec.params.get("strategy_type", self._vbt_config.strategy_type)
            date_range = (spec.start_date, spec.end_date)
            if strategy != first_strategy or date_range != first_range:
                return False

        return True

    def _run_vectorized(
        self,
        indexed_specs: List[Tuple[int, RunSpec]],
        data: Optional[pd.DataFrame]
    ) -> List[RunResult]:
        """
        Run multiple specs with vectorized parameter sweep.

        This is the key performance optimization - VectorBT can evaluate
        thousands of parameter combinations in a single pass.
        """
        import vectorbt as vbt

        if data is None or data.empty:
            return [
                self._create_error_result(
                    spec, RunStatus.FAIL_DATA, "No data", datetime.now()
                )
                for _, spec in indexed_specs
            ]

        started_at = datetime.now()
        first_spec = indexed_specs[0][1]

        try:
            # Filter data to date range
            data = self._filter_date_range(data, first_spec.start_date, first_spec.end_date)
            if data.empty:
                return [
                    self._create_error_result(
                        spec, RunStatus.FAIL_DATA, "No data in range", started_at
                    )
                    for _, spec in indexed_specs
                ]

            # Get strategy
            strategy_type = first_spec.params.get("strategy_type", self._vbt_config.strategy_type)

            # Extract parameter arrays for vectorization
            close = data["close"]

            # Build vectorized signals based on strategy
            if strategy_type == "ma_cross":
                results = self._vectorized_ma_cross(indexed_specs, data, close)
            else:
                # Fall back to sequential for unsupported strategies
                return [self.run(spec, data) for _, spec in indexed_specs]

            return results

        except Exception as e:
            return [
                self._create_error_result(
                    spec, RunStatus.FAIL_EXECUTION, str(e), started_at
                )
                for _, spec in indexed_specs
            ]

    def _vectorized_ma_cross(
        self,
        indexed_specs: List[Tuple[int, RunSpec]],
        data: pd.DataFrame,
        close: pd.Series
    ) -> List[RunResult]:
        """Vectorized MA crossover strategy execution using SignalGenerator."""
        import vectorbt as vbt
        from src.domain.strategy.signals import MACrossSignalGenerator

        started_at = datetime.now()
        signal_generator = MACrossSignalGenerator()

        results = []
        for idx, spec in indexed_specs:
            try:
                # Generate signals using SignalGenerator (TA-Lib based)
                entries, exits = signal_generator.generate(data, spec.params)

                # Run backtest
                pf = vbt.Portfolio.from_signals(
                    close=close,
                    entries=entries,
                    exits=exits,
                    init_cash=spec.initial_capital,
                    fees=spec.commission_per_share / close.mean(),
                    slippage=spec.slippage_bps / 10000,
                    freq=self._vbt_config.freq,
                )

                metrics = self._extract_metrics(pf, data)
                completed_at = datetime.now()

                results.append(RunResult(
                    run_id=spec.run_id,
                    trial_id=spec.trial_id,
                    experiment_id=spec.experiment_id or "",
                    symbol=spec.symbol,
                    window_id=spec.window.window_id,
                    profile_version=spec.profile_version,
                    data_version=spec.data_version,
                    status=RunStatus.SUCCESS,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=(completed_at - started_at).total_seconds(),
                    metrics=metrics,
                    is_train=spec.window.is_train,
                    is_oos=spec.window.is_oos,
                    params=spec.params,
                ))
            except Exception as e:
                results.append(self._create_error_result(
                    spec, RunStatus.FAIL_EXECUTION, str(e), started_at
                ))

        return results

    def _extract_metrics(self, pf, data: pd.DataFrame) -> RunMetrics:
        """
        Extract comprehensive metrics from VectorBT portfolio.

        Uses MetricsCalculator for extended metrics (tail risk, stability,
        statistical, time-based, trading extended).
        """
        try:
            # Get returns series from portfolio
            returns = pf.returns()
            if returns.empty:
                return RunMetrics()

            # Extract trades from VectorBT for trade-level metrics
            trades = self._extract_trades(pf)

            # Use MetricsCalculator for comprehensive metrics
            calc = MetricsCalculator(risk_free_rate=0.0)
            metrics = calc.compute_all(
                returns=returns,
                trades=trades if trades else None,
                benchmark_returns=None,  # Could add SPY benchmark later
            )

            # Supplement with VectorBT-specific stats
            try:
                stats = pf.stats()

                def safe_get(key: str, default: float = 0.0) -> float:
                    try:
                        val = stats.get(key, default)
                        return default if pd.isna(val) else float(val)
                    except (TypeError, ValueError):
                        return default

                # VectorBT's profit factor and expectancy calculations
                if metrics.profit_factor == 0:
                    metrics.profit_factor = safe_get("Profit Factor", 0)
                if metrics.expectancy == 0:
                    metrics.expectancy = safe_get("Expectancy", 0)
                metrics.exposure_pct = safe_get("Exposure Time [%]", 0) / 100

            except Exception:
                pass  # VectorBT stats extraction is supplementary

            return metrics

        except Exception:
            return RunMetrics()

    def _extract_trades(self, pf) -> List[Trade]:
        """
        Extract trade records from VectorBT portfolio.

        Converts VectorBT's trade records to our Trade dataclass
        for use with MetricsCalculator.
        """
        try:
            # VectorBT stores trades in a DataFrame
            if not hasattr(pf, 'trades') or pf.trades.count() == 0:
                return []

            trades_df = pf.trades.records_readable

            trades = []
            for _, row in trades_df.iterrows():
                try:
                    trade = Trade(
                        entry_date=pd.Timestamp(row.get('Entry Timestamp', row.get('entry_idx', 0))),
                        exit_date=pd.Timestamp(row.get('Exit Timestamp', row.get('exit_idx', 0))),
                        return_pct=float(row.get('Return', row.get('return', 0))),
                        bars_held=int(row.get('Duration', row.get('exit_idx', 0) - row.get('entry_idx', 0))),
                    )
                    trades.append(trade)
                except Exception:
                    continue

            return trades

        except Exception:
            return []

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio from returns."""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        downside = returns[returns < 0]

        if len(downside) == 0:
            return 0.0

        downside_std = downside.std()
        if downside_std == 0:
            return 0.0

        # Annualize
        return (mean_return * 252) / (downside_std * np.sqrt(252))

    def _filter_date_range(
        self, data: pd.DataFrame, start: date, end: date
    ) -> pd.DataFrame:
        """Filter DataFrame to date range."""
        if data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        return data[(data.index >= start_ts) & (data.index <= end_ts)]

    def _create_error_result(
        self,
        spec: RunSpec,
        status: RunStatus,
        error: str,
        started_at: datetime
    ) -> RunResult:
        """Create an error RunResult."""
        return RunResult(
            run_id=spec.run_id,
            trial_id=spec.trial_id,
            experiment_id=spec.experiment_id or "",
            symbol=spec.symbol,
            window_id=spec.window.window_id,
            profile_version=spec.profile_version,
            data_version=spec.data_version,
            status=status,
            error=error,
            started_at=started_at,
            completed_at=datetime.now(),
            duration_seconds=(datetime.now() - started_at).total_seconds(),
            metrics=RunMetrics(),
            is_train=spec.window.is_train,
            is_oos=spec.window.is_oos,
            params=spec.params,
        )
