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

import logging
from dataclasses import dataclass
from datetime import date, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

from ...analysis import MetricsCalculator, Trade
from ...core import RunMetrics, RunResult, RunSpec, RunStatus
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
            raise ValueError(f"Strategy '{strategy_name}' is apex_only and cannot run in VectorBT")

        signals_path = strategy_entry.get("signals")
        if signals_path is None:
            raise ValueError(f"Strategy '{strategy_name}' has no SignalGenerator defined")

        # Lazy import: "src.domain.strategy.signals.ma_cross:MACrossSignalGenerator"
        module_path, class_name = signals_path.rsplit(":", 1)
        module = import_module(module_path)
        return getattr(module, class_name)  # type: ignore[no-any-return]

    def run(
        self,
        spec: RunSpec,
        data: Optional[pd.DataFrame] = None,
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> RunResult:
        """
        Execute a single backtest run.

        Args:
            spec: Run specification
            data: Optional pre-loaded primary timeframe OHLCV data
            secondary_data: Optional secondary timeframe data {timeframe: DataFrame}

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

            # Filter to date range, including lookback buffer for warmup
            # Signal generators need warmup bars before the scoring window
            # to compute indicators (EMA, ATR, etc.). Without this buffer,
            # OOS windows shorter than warmup produce zero signals.
            signal_generator_cls = self._get_signal_generator(
                spec.params.get("strategy_type", self._vbt_config.strategy_type)
            )
            warmup_bars_attr = getattr(signal_generator_cls, "warmup_bars", 0)
            _prebuilt_generator = None
            if isinstance(warmup_bars_attr, property):
                _prebuilt_generator = signal_generator_cls()
                warmup_bars = int(getattr(_prebuilt_generator, "warmup_bars", 0))
            else:
                warmup_bars = int(warmup_bars_attr)

            # Include warmup buffer before start_date for indicator computation
            if warmup_bars > 0:
                data_with_buffer = self._filter_with_lookback(
                    data, spec.start_date, spec.end_date, warmup_bars
                )
            else:
                data_with_buffer = self._filter_date_range(data, spec.start_date, spec.end_date)

            if data_with_buffer.empty:
                return self._create_error_result(
                    spec, RunStatus.FAIL_DATA, "No data in date range", started_at
                )

            # Also get the scoring-only slice for metrics
            scoring_data = self._filter_date_range(data, spec.start_date, spec.end_date)
            # Number of buffer bars prepended
            n_buffer = len(data_with_buffer) - len(scoring_data)

            data = data_with_buffer

            # Filter secondary data to date range if provided
            filtered_secondary = None
            if secondary_data:
                filtered_secondary = {}
                for tf, tf_data in secondary_data.items():
                    if warmup_bars > 0:
                        filtered_tf = self._filter_with_lookback(
                            tf_data, spec.start_date, spec.end_date, warmup_bars
                        )
                    else:
                        filtered_tf = self._filter_date_range(
                            tf_data, spec.start_date, spec.end_date
                        )
                    if not filtered_tf.empty:
                        filtered_secondary[tf] = filtered_tf

            # Reuse generator if already instantiated for warmup detection
            try:
                signal_generator = _prebuilt_generator or signal_generator_cls()
            except (KeyError, ValueError) as e:
                return self._create_error_result(spec, RunStatus.FAIL_STRATEGY, str(e), started_at)

            # Generate signals on full data (including warmup buffer)
            # then slice to scoring window for portfolio construction
            if hasattr(signal_generator, "generate_directional"):
                long_entries, long_exits, short_entries, short_exits = (
                    signal_generator.generate_directional(
                        data, spec.params, secondary_data=filtered_secondary
                    )
                )
                # Slice to scoring window
                if n_buffer > 0:
                    long_entries = long_entries.iloc[n_buffer:]
                    long_exits = long_exits.iloc[n_buffer:]
                    short_entries = short_entries.iloc[n_buffer:]
                    short_exits = short_exits.iloc[n_buffer:]
                    close = scoring_data["close"]
                else:
                    close = data["close"]

                pf = vbt.Portfolio.from_signals(
                    close=close,
                    entries=long_entries,
                    exits=long_exits,
                    short_entries=short_entries,
                    short_exits=short_exits,
                    init_cash=spec.initial_capital,
                    fees=spec.commission_per_share / close.mean(),
                    slippage=spec.slippage_bps / 10000,
                    freq=self._vbt_config.freq,
                )
            else:
                entries, exits = signal_generator.generate(
                    data, spec.params, secondary_data=filtered_secondary
                )

                # Support confidence-based sizing via entry_sizes attribute
                size_kwargs: Dict[str, Any] = {}
                if hasattr(signal_generator, "entry_sizes"):
                    sizes = signal_generator.entry_sizes
                    assert (sizes >= 0).all() and (
                        sizes <= 1
                    ).all(), "entry_sizes must be in [0, 1]"
                    assert len(sizes) == len(data), "entry_sizes must align to data.index"
                    if n_buffer > 0:
                        sizes = sizes.iloc[n_buffer:]
                    size_kwargs["size"] = sizes
                    size_kwargs["size_type"] = "percent"

                # Slice signals to scoring window
                if n_buffer > 0:
                    entries = entries.iloc[n_buffer:]
                    exits = exits.iloc[n_buffer:]
                    close = scoring_data["close"]
                else:
                    close = data["close"]

                pf = vbt.Portfolio.from_signals(
                    close=close,
                    entries=entries,
                    exits=exits,
                    init_cash=spec.initial_capital,
                    fees=spec.commission_per_share / close.mean(),
                    slippage=spec.slippage_bps / 10000,
                    freq=self._vbt_config.freq,
                    **size_kwargs,
                )

            # Extract metrics
            metrics = self._extract_metrics(pf, data)

            completed_at = datetime.now()
            return RunResult(
                run_id=spec.run_id or "",
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
            return self._create_error_result(spec, RunStatus.FAIL_EXECUTION, str(e), started_at)

    def run_batch(
        self,
        specs: List[RunSpec],
        data: Optional[Dict[str, pd.DataFrame]] = None,
        secondary_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> List[RunResult]:
        """
        Execute multiple runs with vectorization where possible.

        Groups specs by symbol and strategy, then vectorizes parameter sweeps.

        Args:
            specs: List of run specifications
            data: Optional dict of symbol -> OHLCV DataFrame
            secondary_data: Optional dict of symbol -> {timeframe: DataFrame}

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
                symbol_data = self.load_data(symbol, first_spec.start_date, first_spec.end_date)

            # Get secondary data for this symbol
            symbol_secondary = secondary_data.get(symbol) if secondary_data else None

            # Check if we can vectorize (same strategy, same date range)
            can_vectorize = self._can_vectorize(indexed_specs)

            if can_vectorize and len(indexed_specs) > 1:
                batch_results = self._run_vectorized(indexed_specs, symbol_data, symbol_secondary)
                for (idx, _), result in zip(indexed_specs, batch_results):
                    results[idx] = result
            else:
                # Fall back to sequential
                for idx, spec in indexed_specs:
                    results[idx] = self.run(spec, symbol_data, symbol_secondary)

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
        data: Optional[pd.DataFrame],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[RunResult]:
        """
        Run multiple specs with vectorized parameter sweep.

        This is the key performance optimization - VectorBT can evaluate
        thousands of parameter combinations in a single pass.

        Args:
            indexed_specs: List of (index, RunSpec) tuples
            data: Primary timeframe OHLCV data
            secondary_data: Optional secondary timeframe data {timeframe: DataFrame}
        """

        if data is None or data.empty:
            return [
                self._create_error_result(spec, RunStatus.FAIL_DATA, "No data", datetime.now())
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

            # Filter secondary data if provided
            filtered_secondary = None
            if secondary_data:
                filtered_secondary = {}
                for tf, tf_data in secondary_data.items():
                    filtered_tf = self._filter_date_range(
                        tf_data, first_spec.start_date, first_spec.end_date
                    )
                    if not filtered_tf.empty:
                        filtered_secondary[tf] = filtered_tf

            # Get strategy
            strategy_type = first_spec.params.get("strategy_type", self._vbt_config.strategy_type)

            # Extract parameter arrays for vectorization
            close = data["close"]

            # Build vectorized signals based on strategy
            if strategy_type == "ma_cross":
                results = self._vectorized_ma_cross(indexed_specs, data, close)
            else:
                # Fall back to sequential for unsupported strategies
                # Pass secondary data for MTF support
                return [self.run(spec, data, filtered_secondary) for _, spec in indexed_specs]

            return results

        except Exception as e:
            return [
                self._create_error_result(spec, RunStatus.FAIL_EXECUTION, str(e), started_at)
                for _, spec in indexed_specs
            ]

    def _vectorized_ma_cross(
        self, indexed_specs: List[Tuple[int, RunSpec]], data: pd.DataFrame, close: pd.Series
    ) -> List[RunResult]:
        """
        HIGH-011: True vectorized MA crossover using single Portfolio.

        Instead of creating separate portfolios per spec, we:
        1. Generate all signals into multi-column DataFrames
        2. Create ONE Portfolio with all columns
        3. Extract metrics per column

        This provides 10-20x speedup for parameter sweeps.
        """
        import vectorbt as vbt

        from src.domain.strategy.signals import MACrossSignalGenerator

        started_at = datetime.now()
        signal_generator = MACrossSignalGenerator()

        # Generate all signals first (into columns)
        all_entries = pd.DataFrame(index=close.index)
        all_exits = pd.DataFrame(index=close.index)
        spec_map = {}  # column_name -> (idx, spec)

        for i, (idx, spec) in enumerate(indexed_specs):
            try:
                entries, exits = signal_generator.generate(data, spec.params)
                col_name = f"param_{i}"
                all_entries[col_name] = entries
                all_exits[col_name] = exits
                spec_map[col_name] = (idx, spec, "")  # Empty string for no error
            except Exception as e:
                # Store error for later
                spec_map[f"error_{i}"] = (idx, spec, str(e))

        if all_entries.empty:
            return [
                self._create_error_result(
                    spec, RunStatus.FAIL_EXECUTION, "Signal generation failed", started_at
                )
                for _, spec in indexed_specs
            ]

        # HIGH-011 FIX: Validate that all specs have same fees/slippage before vectorization
        # If they differ, fall back to sequential to avoid silent incorrect results
        first_spec = indexed_specs[0][1]
        fees_slippage_consistent = all(
            spec.commission_per_share == first_spec.commission_per_share
            and spec.slippage_bps == first_spec.slippage_bps
            and spec.initial_capital == first_spec.initial_capital
            for _, spec in indexed_specs
        )
        if not fees_slippage_consistent:
            logger.warning(
                "HIGH-011: Specs have different fees/slippage/capital - falling back to sequential"
            )
            return self._vectorized_ma_cross_sequential(indexed_specs, data, close)

        # HIGH-011: Create SINGLE vectorized portfolio with all parameter combinations
        avg_close = close.mean()

        try:
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=all_entries,
                exits=all_exits,
                init_cash=first_spec.initial_capital,
                fees=first_spec.commission_per_share / avg_close if avg_close > 0 else 0,
                slippage=first_spec.slippage_bps / 10000,
                freq=self._vbt_config.freq,
            )
        except Exception:
            # Fallback to sequential if vectorized fails
            return self._vectorized_ma_cross_sequential(indexed_specs, data, close)

        # Extract metrics per column
        results = []
        completed_at = datetime.now()

        for col_name, value in spec_map.items():
            idx, spec, error_msg = value
            if col_name.startswith("error_"):
                results.append(
                    (
                        idx,
                        self._create_error_result(
                            spec, RunStatus.FAIL_EXECUTION, error_msg, started_at
                        ),
                    )
                )
            else:
                try:
                    # Extract metrics for this specific column
                    col_pf = pf[col_name] if hasattr(pf, "__getitem__") else pf
                    metrics = self._extract_metrics(col_pf, data)

                    results.append(
                        (
                            idx,
                            RunResult(
                                run_id=spec.run_id or "",
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
                            ),
                        )
                    )
                except Exception as e:
                    results.append(
                        (
                            idx,
                            self._create_error_result(
                                spec, RunStatus.FAIL_EXECUTION, str(e), started_at
                            ),
                        )
                    )

        # Sort by original index and return just results
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def _vectorized_ma_cross_sequential(
        self, indexed_specs: List[Tuple[int, RunSpec]], data: pd.DataFrame, close: pd.Series
    ) -> List[RunResult]:
        """Fallback sequential execution if vectorized fails."""
        import vectorbt as vbt

        from src.domain.strategy.signals import MACrossSignalGenerator

        started_at = datetime.now()
        signal_generator = MACrossSignalGenerator()

        results = []
        for idx, spec in indexed_specs:
            try:
                entries, exits = signal_generator.generate(data, spec.params)
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

                results.append(
                    RunResult(
                        run_id=spec.run_id or "",
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
                )
            except Exception as e:
                results.append(
                    self._create_error_result(spec, RunStatus.FAIL_EXECUTION, str(e), started_at)
                )

        return results

    def _extract_metrics(self, pf: Any, data: pd.DataFrame) -> RunMetrics:
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

            # HIGH-014: Extract comprehensive VectorBT-specific stats
            try:
                stats = pf.stats()

                def safe_get(key: str, default: float = 0.0) -> float:
                    try:
                        val = stats.get(key, default)
                        return default if pd.isna(val) else float(val)
                    except (TypeError, ValueError):
                        return default

                # Risk-adjusted metrics from VectorBT
                if metrics.sortino == 0:
                    metrics.sortino = safe_get("Sortino Ratio", 0)
                if metrics.calmar == 0:
                    metrics.calmar = safe_get("Calmar Ratio", 0)

                # Trade metrics from VectorBT
                if metrics.win_rate == 0:
                    metrics.win_rate = safe_get("Win Rate [%]", 0) / 100
                if metrics.profit_factor == 0:
                    metrics.profit_factor = safe_get("Profit Factor", 0)
                if metrics.expectancy == 0:
                    metrics.expectancy = safe_get("Expectancy", 0)

                # Additional VectorBT metrics
                metrics.exposure_pct = safe_get("Exposure Time [%]", 0) / 100
                metrics.total_trades = int(safe_get("Total Trades", 0))
                metrics.best_trade_pct = safe_get("Best Trade [%]", 0) / 100
                metrics.worst_trade_pct = safe_get("Worst Trade [%]", 0) / 100
                metrics.avg_win_pct = safe_get("Avg Winning Trade [%]", 0) / 100
                metrics.avg_loss_pct = safe_get("Avg Losing Trade [%]", 0) / 100

            except Exception:
                pass  # VectorBT stats extraction is supplementary

            return metrics

        except Exception:
            return RunMetrics()

    def _extract_trades(self, pf: Any) -> List[Trade]:
        """
        Extract trade records from VectorBT portfolio.

        Converts VectorBT's trade records to our Trade dataclass
        for use with MetricsCalculator.
        """
        try:
            # VectorBT stores trades in a DataFrame
            if not hasattr(pf, "trades") or pf.trades.count() == 0:
                return []

            trades_df = pf.trades.records_readable

            trades = []
            for _, row in trades_df.iterrows():
                try:
                    trade = Trade(
                        entry_date=pd.Timestamp(
                            row.get("Entry Timestamp", row.get("entry_idx", 0))
                        ),
                        exit_date=pd.Timestamp(row.get("Exit Timestamp", row.get("exit_idx", 0))),
                        return_pct=float(row.get("Return", row.get("return", 0))),
                        bars_held=int(
                            row.get("Duration", row.get("exit_idx", 0) - row.get("entry_idx", 0))
                        ),
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
        return float((mean_return * 252) / (downside_std * np.sqrt(252)))

    def _filter_date_range(self, data: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Filter DataFrame to date range."""
        if data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        return data[(data.index >= start_ts) & (data.index <= end_ts)]

    def _filter_with_lookback(
        self, data: pd.DataFrame, start: date, end: date, lookback_bars: int
    ) -> pd.DataFrame:
        """Filter DataFrame to date range with lookback buffer for indicator warmup.

        Includes `lookback_bars` bars before `start` so signal generators can
        compute indicators (EMA, ATR, etc.) before the scoring window begins.
        """
        if data.index.tz is not None:
            data = data.copy()
            data.index = data.index.tz_localize(None)

        end_ts = pd.Timestamp(end)
        start_ts = pd.Timestamp(start)

        # Find the index position of the first bar >= start
        mask_after_start = data.index >= start_ts
        if not mask_after_start.any():
            return data.iloc[0:0]  # empty

        first_scoring_pos = mask_after_start.argmax()
        buffer_start_pos = max(0, first_scoring_pos - lookback_bars)

        return data.iloc[buffer_start_pos:][(data.iloc[buffer_start_pos:].index <= end_ts)]

    def _create_error_result(
        self, spec: RunSpec, status: RunStatus, error: str, started_at: datetime
    ) -> RunResult:
        """Create an error RunResult."""
        return RunResult(
            run_id=spec.run_id or "",
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
