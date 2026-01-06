"""
Systematic runner - main experiment orchestrator.

Executes backtests across parameter space, symbols, and time windows.
"""

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from ..core import ExperimentSpec, TrialSpec, RunSpec, TimeWindow
from ..core import RunResult, RunMetrics, RunStatus, TrialResult, ExperimentResult
from ..core.hashing import get_next_version
from ..data import DatabaseManager, ExperimentRepository, TrialRepository, RunRepository
from ..data import WalkForwardSplitter, SplitConfig
from ..analysis import Aggregator, AggregationConfig
from ..analysis import ConstraintValidator, Constraint
from ..analysis import PBOCalculator, DSRCalculator
from ..optimization import GridOptimizer, BayesianOptimizer
from .parallel import ParallelRunner, ParallelConfig, ExecutionProgress

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the systematic runner.

    Args:
        db_path: Path to DuckDB database for results storage.
        parallel_workers: Number of parallel workers. 0 = auto-scale based on workload.
        max_workers_cap: Maximum workers when auto-scaling (prevents oversubscription).
        skip_existing: Skip runs that already exist in the database.
        save_equity_curves: Store per-run equity curves (increases storage).
        log_progress_interval: Log trial progress every N trials (for grid search).
    """

    db_path: Union[str, Path] = ":memory:"
    parallel_workers: int = 0  # 0 = auto-scale based on runs_per_trial
    max_workers_cap: int = 16  # Cap for auto-scaling
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

        Each run auto-increments version (v1, v2, v3...) allowing the same
        experiment config to be executed multiple times.

        Args:
            spec: Experiment specification
            backtest_fn: Function to execute a single backtest run
            on_trial_complete: Optional callback after each trial

        Returns:
            Versioned experiment ID (e.g., "exp_name_dv1_abc123_v2")
        """
        # Generate versioned experiment ID with retry for concurrent runs
        # Note: We use a local variable instead of mutating spec to avoid issues
        # when the same ExperimentSpec instance is reused across runs
        base_experiment_id = spec.experiment_id
        start_time = datetime.now()

        versioned_experiment_id = ""
        run_version = 0
        max_retries = 3

        for attempt in range(max_retries):
            run_version = get_next_version(self._db, base_experiment_id)
            versioned_experiment_id = f"{base_experiment_id}_v{run_version}"
            try:
                # Create experiment record with version tracking
                self._exp_repo.create(
                    versioned_experiment_id,
                    spec.model_dump(),
                    base_experiment_id=base_experiment_id,
                    run_version=run_version,
                )
                break
            except Exception as exc:
                message = str(exc)
                is_conflict = any(
                    token in message
                    for token in ("Constraint Error", "Duplicate key", "PRIMARY KEY", "UNIQUE")
                )
                if is_conflict and attempt < max_retries - 1:
                    logger.warning(
                        "Experiment version collision for %s (v%s), retrying...",
                        base_experiment_id,
                        run_version,
                    )
                    continue
                raise

        logger.info(f"Starting experiment: {spec.name} ({versioned_experiment_id}) [v{run_version}]")

        # Select optimizer based on spec.optimization.method
        grid_optimizer = GridOptimizer(spec)
        grid_size = grid_optimizer.get_total_combinations()

        # Determine optimization method (Bayesian is model default)
        method = spec.optimization.method  # Already defaults to "bayesian" in OptimizationConfig

        # Smart n_trials scaling based on grid size
        if spec.optimization.n_trials:
            n_trials = spec.optimization.n_trials
        else:
            # Scale based on search space:
            # - 10% of grid size (reasonable coverage)
            # - sqrt(grid) × 10 (diminishing returns for huge spaces)
            # - Capped at 500 (practical limit)
            # - Minimum 30 (enough for TPE to learn)
            n_trials = min(
                max(30, int(grid_size * 0.1)),  # 10% of grid, min 30
                int(grid_size ** 0.5 * 10),     # sqrt scaling
                500,                             # upper cap
            )
            logger.info(f"Auto-scaled n_trials to {n_trials} (grid_size={grid_size})")

        # Auto-fallback to grid if search space is smaller than n_trials
        if method == "bayesian" and grid_size <= n_trials:
            logger.info(f"Auto-fallback: grid size ({grid_size}) <= n_trials ({n_trials}), using grid search")
            method = "grid"

        if method == "bayesian":
            seed = spec.reproducibility.random_seed if spec.reproducibility else 42
            optimizer = BayesianOptimizer(spec, n_trials=n_trials, seed=seed)
            use_bayesian = True
            total_trials = n_trials
            logger.info(f"Using Bayesian optimization (TPE) with {n_trials} trials")
        else:
            optimizer = grid_optimizer
            use_bayesian = False
            total_trials = grid_size
            logger.info(f"Using grid search with {grid_size} combinations")

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

        # Calculate runs per trial for auto-scaling
        # Each trial = symbols × folds × 2 (IS + OOS)
        runs_per_trial = len(symbols) * len(window_pairs) * 2

        # Auto-scale workers based on workload
        # Treat negative values as auto (0)
        parallel_workers = max(0, self.config.parallel_workers)
        max_workers_cap = max(1, self.config.max_workers_cap)

        if parallel_workers == 0:
            # Auto mode: scale to runs_per_trial, capped by CPU and max_workers_cap
            cpu_limit = max(1, (mp.cpu_count() or 1) - 1)
            effective_workers = min(runs_per_trial, cpu_limit, max_workers_cap)
            effective_workers = max(1, effective_workers)  # At least 1

        else:
            effective_workers = parallel_workers

        # Log experiment structure summary
        logger.info(
            f"Experiment: {total_trials} trials × {len(symbols)} symbols × "
            f"{len(window_pairs)} folds × 2 (IS/OOS) = {total_trials * runs_per_trial} total runs"
        )
        logger.info(f"Workers: {effective_workers} (parallel runs per trial)")

        # Prepare constraint validator
        constraints = [
            Constraint(**c) for c in spec.optimization.constraints
        ]
        validator = ConstraintValidator(constraints) if constraints else None

        # Process trials
        all_trials = []
        trial_count = 0
        best_score = float("-inf")
        trial_times: List[float] = []  # Track trial durations for ETA

        # Decide execution mode: parallel when multiple runs AND backtest_fn provided
        use_parallel = runs_per_trial > 1 and effective_workers > 1 and backtest_fn is not None

        # Create shared parallel runner ONCE with persistent executor (reused across all trials)
        parallel_runner: Optional[ParallelRunner] = None
        if use_parallel:
            parallel_config = ParallelConfig(max_workers=effective_workers)
            parallel_runner = ParallelRunner(parallel_config, persistent=True)
            parallel_runner.start(backtest_fn)  # Create executor once

        # Get parameter iterator (generator for Bayesian, list for Grid)
        if use_bayesian:
            param_iterator = optimizer.generate_params()
        else:
            param_iterator = optimizer.generate_params_list()

        for params in param_iterator:
            trial_start_time = time.time()

            trial_spec = TrialSpec(
                experiment_id=versioned_experiment_id,
                params=params,
                trial_index=trial_count,
            )

            # Create trial stub first (for FK constraint)
            self._trial_repo.create_stub(
                trial_spec.trial_id,
                versioned_experiment_id,
                params,
                trial_count,
            )

            if use_parallel and parallel_runner:
                trial_result = self._run_trial_parallel(
                    spec=spec,
                    trial_spec=trial_spec,
                    symbols=symbols,
                    window_pairs=window_pairs,
                    backtest_fn=backtest_fn,
                    parallel_runner=parallel_runner,
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

            # Report result to Bayesian optimizer for learning
            if use_bayesian:
                score = trial_result.trial_score or 0.0
                optimizer.report_result(params, score)

            # Update trial with aggregated results
            self._trial_repo.update(trial_result)
            all_trials.append(trial_result)

            if on_trial_complete:
                on_trial_complete(trial_result)

            # Track timing and best score
            trial_count += 1
            trial_duration = time.time() - trial_start_time
            trial_times.append(trial_duration)
            if trial_result.trial_score and trial_result.trial_score > best_score:
                best_score = trial_result.trial_score

            # Log progress with ETA (every N trials)
            if trial_count % self.config.log_progress_interval == 0 or trial_count == total_trials:
                pct = trial_count / total_trials * 100
                avg_time = sum(trial_times) / len(trial_times)
                remaining = total_trials - trial_count
                eta_seconds = remaining * avg_time
                eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                logger.info(
                    f"Trial {trial_count}/{total_trials} ({pct:.0f}%) | "
                    f"best={best_score:.3f} | ETA: {eta_str}"
                )

        # Cleanup persistent executor
        if parallel_runner:
            parallel_runner.stop()

        # Complete experiment
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self._exp_repo.update_status(
            versioned_experiment_id, "completed", completed_at=end_time
        )

        logger.info(
            f"Experiment complete: {versioned_experiment_id} "
            f"({trial_count} trials in {duration:.1f}s)"
        )

        return versioned_experiment_id

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
                run_params = self._build_run_params(spec, trial_spec)

                # Execute BOTH train (IS) and test (OOS) windows
                # This enables proper overfitting detection via IS/OOS comparison
                for window, is_train in [(train_window, True), (test_window, False)]:
                    run_spec = RunSpec(
                        trial_id=trial_spec.trial_id,
                        symbol=symbol,
                        window=window,
                        profile_version=profile_version,
                        data_version=spec.reproducibility.data_version,
                        params=run_params,
                        run_index=run_index,
                        experiment_id=trial_spec.experiment_id,  # Use versioned ID
                        bar_size=spec.data.primary_timeframe,
                        secondary_timeframes=spec.data.secondary_timeframes,
                    )

                    # Check if run already exists
                    if self.config.skip_existing and self._run_repo.exists(run_spec.run_id):
                        logger.debug(f"Skipping existing run: {run_spec.run_id}")
                        continue

                    # Execute backtest
                    if backtest_fn:
                        try:
                            result = backtest_fn(run_spec)
                            # Ensure IS/OOS flags are set correctly
                            result.is_train = is_train
                            result.is_oos = not is_train
                        except Exception as e:
                            logger.error(f"Run {run_spec.run_id} failed: {e}")
                            result = RunResult(
                                run_id=run_spec.run_id,
                                trial_id=trial_spec.trial_id,
                                experiment_id=trial_spec.experiment_id,
                                symbol=symbol,
                                window_id=window.window_id,
                                profile_version=profile_version,
                                data_version=spec.reproducibility.data_version,
                                status=RunStatus.FAIL_STRATEGY,
                                error=str(e),
                                is_train=is_train,
                                is_oos=not is_train,
                            )
                    else:
                        # Create placeholder result for testing
                        logger.warning(f"Run {run_spec.run_id}, result is MOCKED!!!!")
                        result = self._create_mock_result(run_spec, window)
                        result.is_train = is_train
                        result.is_oos = not is_train

                    runs.append(result)
                    run_index += 1

        # Store runs in batch
        self._run_repo.create_batch(runs)

        # Aggregate trial
        trial_result = self._aggregator.aggregate_trial(
            trial_id=trial_spec.trial_id,
            experiment_id=trial_spec.experiment_id,  # Use versioned ID
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
        parallel_runner: ParallelRunner,
    ) -> TrialResult:
        """
        Run a single trial across all symbols and windows (parallel).

        Uses shared ParallelRunner for multi-core execution (reused across trials).

        Args:
            spec: Experiment specification
            trial_spec: Trial specification
            symbols: List of symbols to test
            window_pairs: List of (train_window, test_window) tuples
            backtest_fn: Backtest execution function (required for parallel)
            parallel_runner: Shared parallel runner (reused to avoid executor overhead)

        Returns:
            Aggregated trial result
        """
        trial_start = datetime.now()

        profile_version = (
            next(iter(spec.profiles.values())).version
            if spec.profiles
            else "default"
        )

        # Build all run specs upfront (both IS and OOS windows)
        run_specs: List[RunSpec] = []
        run_is_train: Dict[str, bool] = {}  # Track IS/OOS per run_id
        run_index = 0

        for symbol in symbols:
            for train_window, test_window in window_pairs:
                run_params = self._build_run_params(spec, trial_spec)

                # Execute BOTH train (IS) and test (OOS) windows
                for window, is_train in [(train_window, True), (test_window, False)]:
                    run_spec = RunSpec(
                        trial_id=trial_spec.trial_id,
                        symbol=symbol,
                        window=window,
                        profile_version=profile_version,
                        data_version=spec.reproducibility.data_version,
                        params=run_params,
                        run_index=run_index,
                        experiment_id=trial_spec.experiment_id,  # Use versioned ID
                        bar_size=spec.data.primary_timeframe,
                        secondary_timeframes=spec.data.secondary_timeframes,
                    )

                    # Check if run already exists
                    if self.config.skip_existing and self._run_repo.exists(run_spec.run_id):
                        logger.debug(f"Skipping existing run: {run_spec.run_id}")
                        continue

                    run_specs.append(run_spec)
                    run_is_train[run_spec.run_id] = is_train
                    run_index += 1

        # Execute in parallel using shared runner
        runs = parallel_runner.run_all(run_specs, backtest_fn)

        # Set IS/OOS flags on results
        for result in runs:
            is_train = run_is_train.get(result.run_id, False)
            result.is_train = is_train
            result.is_oos = not is_train

        # Store runs in batch
        self._run_repo.create_batch(runs)

        # Aggregate trial
        trial_result = self._aggregator.aggregate_trial(
            trial_id=trial_spec.trial_id,
            experiment_id=trial_spec.experiment_id,  # Use versioned ID
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

        # Get temporal folds for DSR calculation
        temporal_folds = temporal_data.get("folds", 1) if isinstance(temporal_data, dict) else 1

        # Compute PBO/DSR from trial IS/OOS Sharpe ratios
        pbo = None
        dsr = None
        dsr_p_value = None

        # Query paired IS/OOS sharpes from trials table
        paired_sharpes = self._db.fetchall(
            """
            SELECT is_median_sharpe, oos_median_sharpe
            FROM trials
            WHERE experiment_id = ?
              AND is_median_sharpe IS NOT NULL
              AND oos_median_sharpe IS NOT NULL
            """,
            (experiment_id,),
        )

        if paired_sharpes and len(paired_sharpes) >= 2:
            is_sharpes = [float(row[0]) for row in paired_sharpes]
            oos_sharpes = [float(row[1]) for row in paired_sharpes]

            # Calculate PBO (Probability of Backtest Overfit)
            # PBO uses paired IS/OOS sharpes - requires matching trial data
            pbo_calc = PBOCalculator()
            pbo = pbo_calc.calculate(is_sharpes, oos_sharpes)

            # Calculate DSR (Deflated Sharpe Ratio)
            # DSR corrects for multiple testing using ALL trials tested
            if is_sharpes and total_trials >= 2:
                best_is_sharpe = max(is_sharpes)
                # Use total_trials for multiple testing penalty, not just paired count
                # DSR penalizes for the total number of strategies tested
                n_trials_for_dsr = total_trials
                # n_observations: trading days for annualized Sharpe (252 per year)
                # Use folds × 252 as each fold represents ~1 year of test data
                n_observations = max(temporal_folds, 1) * 252

                dsr_calc = DSRCalculator()
                dsr, dsr_p_value = dsr_calc.calculate(
                    observed_sharpe=best_is_sharpe,
                    n_trials=n_trials_for_dsr,
                    n_observations=n_observations,
                )

            logger.debug(
                f"Statistical validation: PBO={pbo:.3f}, DSR={dsr:.3f}" if pbo and dsr else
                f"Statistical validation: insufficient data (paired_trials={len(paired_sharpes) if paired_sharpes else 0})"
            )

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
            temporal_folds=temporal_folds,
            total_trials=total_trials,
            successful_trials=successful_trials,
            total_runs=total_runs,
            successful_runs=successful_runs,
            top_trials=top_trials,
            # Statistical validation
            pbo=pbo,
            dsr=dsr,
            dsr_p_value=dsr_p_value,
            # Best trial details
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
