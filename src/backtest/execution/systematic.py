"""
Systematic runner - main experiment orchestrator.

Executes backtests across parameter space, symbols, and time windows.
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from ..core import ExperimentSpec, TrialSpec, RunSpec, TimeWindow
from ..core import RunResult, RunMetrics, RunStatus, TrialResult, ExperimentResult
from ..data import DatabaseManager, ExperimentRepository, TrialRepository, RunRepository
from ..data import WalkForwardSplitter, SplitConfig
from ..analysis import Aggregator, AggregationConfig
from ..analysis import ConstraintValidator, Constraint
from ..optimization import GridOptimizer
from .parallel import ParallelRunner, ParallelConfig, ExecutionProgress

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the systematic runner."""

    db_path: Union[str, Path] = ":memory:"
    parallel_workers: int = 1
    skip_existing: bool = True
    save_equity_curves: bool = False
    log_progress_interval: int = 10


@dataclass
class SystematicRunner:
    """
    Systematic backtest runner.

    Orchestrates the complete experiment workflow:
    1. Parse experiment specification
    2. Generate parameter combinations (grid/Bayesian)
    3. Split data into train/test windows
    4. Execute backtests across symbols and windows
    5. Aggregate results with robust statistics
    6. Apply constraint validation
    7. Store results in DuckDB

    Example:
        runner = SystematicRunner(db_path="results/backtest.db")
        experiment_id = runner.run(spec)
        top_trials = runner.get_top_trials(experiment_id, limit=10)
    """

    config: RunnerConfig = field(default_factory=RunnerConfig)
    _db: Optional[DatabaseManager] = field(default=None, repr=False)
    _aggregator: Aggregator = field(default_factory=Aggregator)

    def __post_init__(self):
        self._db = DatabaseManager(self.config.db_path)
        self._db.initialize_schema()
        self._exp_repo = ExperimentRepository(self._db)
        self._trial_repo = TrialRepository(self._db)
        self._run_repo = RunRepository(self._db)

    def run(
        self,
        spec: ExperimentSpec,
        backtest_fn: Optional[Callable[[RunSpec], RunResult]] = None,
        on_trial_complete: Optional[Callable[[TrialResult], None]] = None,
    ) -> str:
        """
        Run a complete experiment.

        Args:
            spec: Experiment specification
            backtest_fn: Function to execute a single backtest run
            on_trial_complete: Optional callback after each trial

        Returns:
            Experiment ID
        """
        start_time = datetime.now()
        logger.info(f"Starting experiment: {spec.name} ({spec.experiment_id})")

        # Check if experiment already exists
        if self._exp_repo.exists(spec.experiment_id):
            if self.config.skip_existing:
                logger.info(f"Experiment {spec.experiment_id} already exists, skipping (use skip_existing=False to re-run)")
                return spec.experiment_id
            else:
                raise ValueError(
                    f"Experiment {spec.experiment_id} already exists. "
                    "Delete the database file or use a different config to generate a new ID."
                )

        # Create experiment record
        self._exp_repo.create(spec.experiment_id, spec.model_dump())

        # Generate parameter combinations
        optimizer = GridOptimizer(spec) # can use bayesian?
        param_combinations = optimizer.generate_params_list()
        logger.info(f"Generated {len(param_combinations)} parameter combinations")

        # Create walk-forward splitter
        splitter = WalkForwardSplitter(
            SplitConfig(
                train_days=spec.temporal.train_days,
                test_days=spec.temporal.test_days,
                folds=spec.temporal.folds,
                purge_days=spec.temporal.purge_days,
                embargo_days=spec.temporal.embargo_days,
            )
        )

        # Get symbols
        symbols = spec.get_symbols()
        if not symbols:
            raise ValueError("No symbols in universe")

        # Get time windows - now returns (train_window, test_window) tuples
        start_date = spec.temporal.start_date or "2020-01-01"
        end_date = spec.temporal.end_date or "2024-12-31"
        window_pairs = list(splitter.split(start_date, end_date))
        logger.info(f"Created {len(window_pairs)} walk-forward folds")

        # Prepare constraint validator
        constraints = [
            Constraint(**c) for c in spec.optimization.constraints
        ]
        validator = ConstraintValidator(constraints) if constraints else None

        # Process trials
        all_trials = []
        trial_count = 0

        # Decide execution mode
        use_parallel = self.config.parallel_workers > 1 and backtest_fn is not None

        for params in param_combinations:
            trial_spec = TrialSpec(
                experiment_id=spec.experiment_id,
                params=params,
                trial_index=trial_count,
            )

            # Create trial stub first (for FK constraint)
            self._trial_repo.create_stub(
                trial_spec.trial_id,
                spec.experiment_id,
                params,
                trial_count,
            )

            if use_parallel:
                trial_result = self._run_trial_parallel(
                    spec=spec,
                    trial_spec=trial_spec,
                    symbols=symbols,
                    window_pairs=window_pairs,
                    backtest_fn=backtest_fn,
                )
            else:
                trial_result = self._run_trial(
                    spec=spec,
                    trial_spec=trial_spec,
                    symbols=symbols,
                    window_pairs=window_pairs,
                    backtest_fn=backtest_fn,
                )

            # Apply constraints
            if validator:
                passed, violations = validator.validate_with_details(trial_result)
                trial_result.constraints_met = passed
                trial_result.constraint_violations = violations

            # Update trial with aggregated results
            self._trial_repo.update(trial_result)
            all_trials.append(trial_result)

            if on_trial_complete:
                on_trial_complete(trial_result)

            trial_count += 1
            if trial_count % self.config.log_progress_interval == 0:
                logger.info(f"Completed {trial_count}/{len(param_combinations)} trials")

        # Complete experiment
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._exp_repo.update_status(
            spec.experiment_id, "completed", completed_at=end_time
        )

        logger.info(
            f"Experiment complete: {spec.experiment_id} "
            f"({trial_count} trials in {duration:.1f}s)"
        )

        return spec.experiment_id

    def _build_run_params(self, spec: ExperimentSpec, trial_spec: TrialSpec) -> Dict[str, Any]:
        params = dict(trial_spec.params)
        params.setdefault("strategy", spec.strategy)
        params.setdefault("strategy_type", spec.strategy)
        return params

    def _run_trial(
        self,
        spec: ExperimentSpec,
        trial_spec: TrialSpec,
        symbols: List[str],
        window_pairs: List[tuple[TimeWindow, TimeWindow]],
        backtest_fn: Optional[Callable[[RunSpec], RunResult]],
    ) -> TrialResult:
        """
        Run a single trial across all symbols and windows (sequential).

        Args:
            spec: Experiment specification
            trial_spec: Trial specification
            symbols: List of symbols to test
            window_pairs: List of (train_window, test_window) tuples
            backtest_fn: Backtest execution function

        Returns:
            Aggregated trial result
        """
        trial_start = datetime.now()
        runs: List[RunResult] = []

        profile_version = (
            next(iter(spec.profiles.values())).version
            if spec.profiles
            else "default"
        )

        run_index = 0
        for symbol in symbols:
            for train_window, test_window in window_pairs:
                # Create run spec using the test window (for OOS validation)
                run_params = self._build_run_params(spec, trial_spec)
                run_spec = RunSpec(
                    trial_id=trial_spec.trial_id,
                    symbol=symbol,
                    window=test_window,  # Use test window for execution
                    profile_version=profile_version,
                    data_version=spec.reproducibility.data_version,
                    params=run_params,
                    run_index=run_index,
                    experiment_id=spec.experiment_id,
                )

                # Check if run already exists
                if self.config.skip_existing and self._run_repo.exists(run_spec.run_id):
                    logger.debug(f"Skipping existing run: {run_spec.run_id}")
                    continue

                # Execute backtest
                if backtest_fn:
                    try:
                        result = backtest_fn(run_spec)
                    except Exception as e:
                        logger.error(f"Run {run_spec.run_id} failed: {e}")
                        result = RunResult(
                            run_id=run_spec.run_id,
                            trial_id=trial_spec.trial_id,
                            experiment_id=spec.experiment_id,
                            symbol=symbol,
                            window_id=test_window.window_id,
                            profile_version=profile_version,
                            data_version=spec.reproducibility.data_version,
                            status=RunStatus.FAIL_STRATEGY,
                            error=str(e),
                            is_train=test_window.is_train,
                            is_oos=test_window.is_oos,
                        )
                else:
                    # Create placeholder result for testing
                    logger.warning(f"Run {run_spec.run_id}, result is MOCKED!!!!")
                    result = self._create_mock_result(run_spec, test_window)

                runs.append(result)
                run_index += 1

        # Store runs in batch
        self._run_repo.create_batch(runs)

        # Aggregate trial
        trial_result = self._aggregator.aggregate_trial(
            trial_id=trial_spec.trial_id,
            experiment_id=spec.experiment_id,
            params=trial_spec.params,
            runs=runs,
        )

        trial_result.started_at = trial_start
        trial_result.completed_at = datetime.now()
        trial_result.total_duration_seconds = (
            trial_result.completed_at - trial_start
        ).total_seconds()
        trial_result.trial_index = trial_spec.trial_index

        return trial_result

    def _run_trial_parallel(
        self,
        spec: ExperimentSpec,
        trial_spec: TrialSpec,
        symbols: List[str],
        window_pairs: List[tuple[TimeWindow, TimeWindow]],
        backtest_fn: Callable[[RunSpec], RunResult],
    ) -> TrialResult:
        """
        Run a single trial across all symbols and windows (parallel).

        Uses ParallelRunner for multi-core execution.

        Args:
            spec: Experiment specification
            trial_spec: Trial specification
            symbols: List of symbols to test
            window_pairs: List of (train_window, test_window) tuples
            backtest_fn: Backtest execution function (required for parallel)

        Returns:
            Aggregated trial result
        """
        trial_start = datetime.now()

        profile_version = (
            next(iter(spec.profiles.values())).version
            if spec.profiles
            else "default"
        )

        # Build all run specs upfront
        run_specs: List[RunSpec] = []
        run_index = 0

        for symbol in symbols:
            for train_window, test_window in window_pairs:
                run_params = self._build_run_params(spec, trial_spec)
                run_spec = RunSpec(
                    trial_id=trial_spec.trial_id,
                    symbol=symbol,
                    window=test_window,  # Use test window for execution
                    profile_version=profile_version,
                    data_version=spec.reproducibility.data_version,
                    params=run_params,
                    run_index=run_index,
                    experiment_id=spec.experiment_id,
                )

                # Check if run already exists
                if self.config.skip_existing and self._run_repo.exists(run_spec.run_id):
                    logger.debug(f"Skipping existing run: {run_spec.run_id}")
                    continue

                run_specs.append(run_spec)
                run_index += 1

        # Execute in parallel
        parallel_config = ParallelConfig(max_workers=self.config.parallel_workers)
        parallel_runner = ParallelRunner(parallel_config)

        runs = parallel_runner.run_all(run_specs, backtest_fn)

        # Store runs in batch
        self._run_repo.create_batch(runs)

        # Aggregate trial
        trial_result = self._aggregator.aggregate_trial(
            trial_id=trial_spec.trial_id,
            experiment_id=spec.experiment_id,
            params=trial_spec.params,
            runs=runs,
        )

        trial_result.started_at = trial_start
        trial_result.completed_at = datetime.now()
        trial_result.total_duration_seconds = (
            trial_result.completed_at - trial_start
        ).total_seconds()
        trial_result.trial_index = trial_spec.trial_index

        return trial_result

    def _create_mock_result(
        self, run_spec: RunSpec, window: TimeWindow
    ) -> RunResult:
        """Create a mock result for testing without actual backtest."""
        import random

        # Generate random-ish metrics based on params
        base_sharpe = sum(run_spec.params.values()) / 100 if run_spec.params else 0.5
        sharpe = base_sharpe + random.gauss(0, 0.3)
        total_return = sharpe * 0.1 + random.gauss(0, 0.05)

        return RunResult(
            run_id=run_spec.run_id,
            trial_id=run_spec.trial_id,
            experiment_id=run_spec.experiment_id,
            symbol=run_spec.symbol,
            window_id=window.window_id,
            profile_version=run_spec.profile_version,
            data_version=run_spec.data_version,
            status=RunStatus.SUCCESS,
            is_train=window.is_train,
            is_oos=window.is_oos,
            metrics=RunMetrics(
                sharpe=sharpe,
                total_return=total_return,
                max_drawdown=abs(random.gauss(0.1, 0.05)),
                total_trades=random.randint(10, 100),
                win_rate=random.uniform(0.4, 0.6),
                profit_factor=max(0.5, random.gauss(1.2, 0.3)),
            ),
            params=run_spec.params,
        )

    def get_top_trials(
        self,
        experiment_id: str,
        limit: int = 10,
        order_by: str = "trial_score DESC",
    ) -> List[Dict[str, Any]]:
        """Get top performing trials for an experiment."""
        return self._trial_repo.get_top_trials(experiment_id, limit, order_by)

    def get_experiment_result(self, experiment_id: str) -> ExperimentResult:
        """Get complete experiment result with summary statistics."""
        exp_data = self._exp_repo.get(experiment_id)
        if not exp_data:
            raise ValueError(f"Experiment not found: {experiment_id}")

        top_trials_data = self.get_top_trials(experiment_id, limit=10)

        # Convert to TrialResult objects (simplified)
        top_trials = []
        for t in top_trials_data:
            from ..core import TrialAggregates

            agg = TrialAggregates(
                median_sharpe=t.get("median_sharpe", 0),
                median_return=t.get("median_return", 0),
                median_max_dd=t.get("median_max_dd", 0),
                p10_sharpe=t.get("p10_sharpe", 0),
                oos_median_sharpe=t.get("oos_median_sharpe", 0),
            )
            trial = TrialResult(
                trial_id=t["trial_id"],
                experiment_id=experiment_id,
                params=t.get("params", {}),
                aggregates=agg,
                trial_score=t.get("trial_score", 0),
                constraints_met=t.get("constraints_met", True),
            )
            top_trials.append(trial)

        # Get best trial
        best = top_trials_data[0] if top_trials_data else None

        # Query trial and run counts
        trial_counts = self._db.fetchone(
            "SELECT COUNT(*), SUM(CASE WHEN trial_score IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM trials WHERE experiment_id = ?",
            (experiment_id,),
        )
        total_trials = trial_counts[0] if trial_counts else 0
        successful_trials = trial_counts[1] if trial_counts and trial_counts[1] else 0

        run_counts = self._db.fetchone(
            "SELECT COUNT(*), SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) "
            "FROM runs WHERE experiment_id = ?",
            (experiment_id,),
        )
        total_runs = run_counts[0] if run_counts else 0
        successful_runs = run_counts[1] if run_counts and run_counts[1] else 0

        # Extract configuration from stored experiment data
        universe_data = exp_data.get("universe", {})
        temporal_data = exp_data.get("temporal", {})
        parameters_data = exp_data.get("parameters", {})
        reproducibility_data = exp_data.get("reproducibility", {})

        # Get symbols
        symbols = universe_data.get("symbols", []) if isinstance(universe_data, dict) else []

        # Count parameter combinations (from stored parameters)
        param_combinations = 0
        if parameters_data:
            # Try to expand if we have the parameter definitions
            from ..core import ParameterDef
            try:
                combo_count = 1
                for name, defn in parameters_data.items():
                    if isinstance(defn, dict):
                        pdef = ParameterDef(name=name, **defn)
                        combo_count *= len(pdef.expand())
                param_combinations = combo_count
            except Exception:
                param_combinations = total_trials  # Fallback to actual trial count

        return ExperimentResult(
            experiment_id=experiment_id,
            name=exp_data.get("name", ""),
            strategy=exp_data.get("strategy", ""),
            total_parameter_combinations=param_combinations,
            symbols_tested=symbols,
            temporal_folds=temporal_data.get("folds", 0) if isinstance(temporal_data, dict) else 0,
            total_trials=total_trials,
            successful_trials=successful_trials,
            total_runs=total_runs,
            successful_runs=successful_runs,
            top_trials=top_trials,
            best_trial_id=best["trial_id"] if best else None,
            best_params=best.get("params") if best else None,
            best_sharpe=best.get("median_sharpe") if best else None,
            best_return=best.get("median_return") if best else None,
            data_version=reproducibility_data.get("data_version", "") if isinstance(reproducibility_data, dict) else "",
            random_seed=reproducibility_data.get("random_seed", 42) if isinstance(reproducibility_data, dict) else 42,
        )

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
