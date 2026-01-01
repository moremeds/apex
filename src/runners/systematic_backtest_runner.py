#!/usr/bin/env python3
"""
Systematic Backtest Runner

Main entry point for running systematic backtesting experiments.

Usage:
    # Run from YAML spec
    python -m src.runners.systematic_backtest_runner --spec config/backtest/examples/ta_metrics_experiment.yaml

    # Run with options
    python -m src.runners.systematic_backtest_runner \\
        --spec config/backtest/examples/ta_metrics_experiment.yaml \\
        --output results/experiments \\
        --parallel 4

    # Run with HTML report generation
    python -m src.runners.systematic_backtest_runner \\
        --spec config/backtest/examples/ta_metrics_experiment.yaml \\
        --report

    # List available strategies
    python -m src.runners.systematic_backtest_runner --list-strategies
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Ensure project root is in path."""
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def list_strategies():
    """List available strategies."""
    from src.domain.strategy.registry import get_all_strategies

    strategies = get_all_strategies()

    print("\nAvailable Strategies:")
    print("-" * 60)
    for name, info in sorted(strategies.items()):
        desc = info.get("description", "No description")
        print(f"  {name:20s} - {desc}")
    print()


def generate_html_report(
    runner: "SystematicRunner",
    experiment_id: str,
    spec: "ExperimentSpec",
    output_dir: Path,
) -> Path:
    """
    Generate an HTML report for the completed experiment.

    Args:
        runner: SystematicRunner instance with results
        experiment_id: ID of the completed experiment
        spec: Experiment specification
        output_dir: Directory for output

    Returns:
        Path to generated HTML report
    """
    from src.backtest.analysis.reporting import HTMLReportGenerator, ReportConfig, ReportData

    logger.info("Generating HTML report...")

    # Get experiment result and top trials
    result = runner.get_experiment_result(experiment_id)
    top_trials = runner.get_top_trials(experiment_id, limit=50)
    symbols = spec.get_symbols()

    # Build aggregate metrics (median across all trials)
    agg_metrics = _aggregate_trial_metrics(top_trials)

    # Build per-symbol metrics
    per_symbol_metrics = _build_per_symbol_metrics(runner, experiment_id, symbols)

    # Build parameters summary
    param_grid = spec.expand_parameter_grid()
    best_params = top_trials[0]["params"] if top_trials else {}
    best_score = top_trials[0].get("trial_score", 0.0) if top_trials else 0.0

    # Create ReportData matching the actual dataclass fields
    report_data = ReportData(
        experiment_id=experiment_id,
        strategy_name=spec.strategy,
        code_version=spec.reproducibility.code_version if spec.reproducibility else "",
        data_version=spec.reproducibility.data_version if spec.reproducibility else "",
        start_date=str(spec.temporal.start_date) if spec.temporal.start_date else "auto",
        end_date=str(spec.temporal.end_date) if spec.temporal.end_date else "auto",
        symbols=symbols,
        n_folds=spec.temporal.folds,
        train_days=spec.temporal.train_days,
        test_days=spec.temporal.test_days,
        total_trials=result.total_trials,
        best_params=best_params,
        best_trial_score=best_score,
        metrics=agg_metrics,
        validation={
            "successful_trials": result.successful_trials,
            "success_rate": result.successful_trials / result.total_trials if result.total_trials > 0 else 0,
            "total_runs": result.total_runs,
            "successful_runs": result.successful_runs,
            "pbo": result.pbo if result.pbo is not None else 0.0,
            "dsr": result.dsr if result.dsr is not None else 0.0,
        },
        per_symbol=per_symbol_metrics,
        per_window=[],  # Could be populated from trial data
        equity_curve=[],  # Could be populated from best trial
    )

    # Generate report
    report_config = ReportConfig(
        title=f"Backtest Report: {spec.name}",
        theme="light",
    )
    generator = HTMLReportGenerator(config=report_config)

    report_path = output_dir / f"{experiment_id}_report.html"
    generated_path = generator.generate(report_data, report_path)

    logger.info(f"HTML report generated: {generated_path}")
    return generated_path


def _aggregate_trial_metrics(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate metrics across trials (median values)."""
    import numpy as np

    if not trials:
        return {}

    # Collect all available metric values
    metric_values: Dict[str, List[float]] = {}

    for trial in trials:
        # Common metric keys from trial data
        metric_mappings = {
            "sharpe": "median_sharpe",
            "max_drawdown": "median_max_dd",
            "total_return": "median_total_return",
            "win_rate": "median_win_rate",
            "profit_factor": "median_profit_factor",
            "total_trades": "median_total_trades",
        }

        for our_key, trial_key in metric_mappings.items():
            if trial_key in trial and trial[trial_key] is not None:
                if our_key not in metric_values:
                    metric_values[our_key] = []
                metric_values[our_key].append(float(trial[trial_key]))

    # Compute median for each metric
    return {k: float(np.median(v)) for k, v in metric_values.items() if v}


def _build_per_symbol_metrics(
    runner: "SystematicRunner",
    experiment_id: str,
    symbols: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Build per-symbol metrics from experiment top trials.

    Since SystematicRunner doesn't expose per-run queries,
    we use the top trials data and the experiment result.
    """
    import numpy as np

    per_symbol: Dict[str, Dict[str, float]] = {}

    # Get top trials which contain aggregated metrics
    try:
        top_trials = runner.get_top_trials(experiment_id, limit=100)
        result = runner.get_experiment_result(experiment_id)
    except Exception:
        return {s: {} for s in symbols}

    # If we have best trial metrics, use them as aggregate
    if result.best_sharpe is not None:
        # For now, apply aggregate metrics to all symbols
        # (proper per-symbol breakdown would require database queries)
        for symbol in symbols:
            per_symbol[symbol] = {
                "sharpe": result.best_sharpe or 0.0,
                "total_return": result.best_return or 0.0,
                "max_drawdown": 0.0,  # Not in ExperimentResult
                "win_rate": 0.0,
                "total_trades": 0,
            }
    else:
        # Use median from top trials if available
        if top_trials:
            sharpes = [t.get("median_sharpe", 0) for t in top_trials if t.get("median_sharpe")]
            returns = [t.get("median_total_return", 0) for t in top_trials if t.get("median_total_return")]
            drawdowns = [t.get("median_max_dd", 0) for t in top_trials if t.get("median_max_dd")]

            median_sharpe = float(np.median(sharpes)) if sharpes else 0.0
            median_return = float(np.median(returns)) if returns else 0.0
            median_dd = float(np.median(drawdowns)) if drawdowns else 0.0

            for symbol in symbols:
                per_symbol[symbol] = {
                    "sharpe": median_sharpe,
                    "total_return": median_return,
                    "max_drawdown": median_dd,
                }
        else:
            for symbol in symbols:
                per_symbol[symbol] = {}

    return per_symbol


def prefetch_data(symbols: List[str], start_date, end_date, max_retries: int = 3) -> Dict[str, Any]:
    """
    Pre-fetch all data from IB once before running the experiment.

    This avoids creating a new IB connection for each of the thousands of runs.
    """
    import time
    from src.backtest.data.providers import IbBacktestDataProvider
    from datetime import datetime as dt

    logger.info(f"Pre-fetching data for {len(symbols)} symbols...")

    # Convert dates to datetime
    if hasattr(start_date, 'isoformat'):
        start_dt = dt.combine(start_date, dt.min.time())
    else:
        start_dt = dt.fromisoformat(str(start_date)) if start_date else dt(2020, 1, 1)

    if hasattr(end_date, 'isoformat'):
        end_dt = dt.combine(end_date, dt.max.time())
    else:
        end_dt = dt.fromisoformat(str(end_date)) if end_date else dt.now()

    # Initial wait to let IB recover from any previous connection attempts
    logger.info("Waiting 5s for IB connection readiness...")
    time.sleep(5)

    # Retry logic for IB connection issues
    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 15 * attempt  # Exponential backoff: 15s, 30s, 45s
                logger.info(f"Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)

            # Single IB connection to fetch all data
            provider = IbBacktestDataProvider(
                host="127.0.0.1",
                port=4001,
                client_id=4,
                rate_limit=True,
            )

            data = provider.fetch_bars_sync(
                symbols=symbols,
                start=start_dt,
                end=end_dt,
                timeframe="1d",
            )

            successful = sum(1 for df in data.values() if not df.empty)
            logger.info(f"Pre-fetch complete: {successful}/{len(symbols)} symbols with data")
            return data

        except Exception as e:
            last_error = e
            logger.warning(f"Pre-fetch attempt {attempt + 1} failed: {e}")

    # All retries failed
    raise RuntimeError(f"Failed to pre-fetch data after {max_retries} attempts: {last_error}")


def create_backtest_fn(cached_data: Optional[Dict[str, Any]] = None):
    """
    Create a backtest function using VectorBT engine.

    Args:
        cached_data: Pre-fetched data dict (symbol -> DataFrame)
                    If provided, avoids IB connections during backtests.

    Returns a function that takes a RunSpec and returns a RunResult.
    """
    from src.backtest.execution.engines import VectorBTEngine, VectorBTConfig

    # If we have cached data, don't use IB (use local cache)
    if cached_data:
        config = VectorBTConfig(
            data_source="local",  # Will use provided data
        )
    else:
        config = VectorBTConfig(
            data_source="ib",
            ib_port=4001,
        )

    engine = VectorBTEngine(config)

    def backtest_fn(spec):
        """Execute backtest for a single run spec."""
        # Pass cached data if available
        symbol_data = cached_data.get(spec.symbol) if cached_data else None
        return engine.run(spec, data=symbol_data)

    return backtest_fn


def run_experiment(
    spec_path: str,
    output_dir: str = "results/experiments",
    parallel: int = 1,
    dry_run: bool = False,
    generate_report: bool = False,
):
    """
    Run a systematic backtest experiment.

    Args:
        spec_path: Path to experiment YAML specification
        output_dir: Directory for results
        parallel: Number of parallel workers
        dry_run: If True, only show what would be run
        generate_report: If True, generate HTML report after completion
    """
    from src.backtest import ExperimentSpec, SystematicRunner, RunnerConfig

    # Load specification
    spec = ExperimentSpec.from_yaml(spec_path)
    logger.info(f"Loaded experiment: {spec.name}")
    logger.info(f"  Strategy: {spec.strategy}")
    logger.info(f"  Experiment ID: {spec.experiment_id}")

    # Count parameter combinations
    param_combinations = spec.expand_parameter_grid()
    symbols = spec.get_symbols()

    logger.info(f"  Parameter combinations: {len(param_combinations)}")
    logger.info(f"  Symbols: {len(symbols)} ({', '.join(symbols)})")
    logger.info(f"  Folds: {spec.temporal.folds}")

    total_runs = (
        len(param_combinations) *
        len(symbols) *
        spec.temporal.folds
    )
    logger.info(f"  Total runs: {total_runs}")

    if dry_run:
        logger.info("Dry run - would execute the above")
        print("\nFirst 5 parameter combinations:")
        for i, params in enumerate(param_combinations[:5]):
            print(f"  {i+1}: {params}")
        if len(param_combinations) > 5:
            print(f"  ... and {len(param_combinations) - 5} more")
        return

    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / f"{spec.experiment_id}.db"

    logger.info(f"  Output: {db_path}")

    # Create runner
    config = RunnerConfig(
        db_path=str(db_path),
        parallel_workers=parallel,
        skip_existing=True,
    )
    runner = SystematicRunner(config=config)

    # Pre-fetch all data from IB ONCE (instead of per-run)
    cached_data = prefetch_data(
        symbols=symbols,
        start_date=spec.temporal.start_date,
        end_date=spec.temporal.end_date,
    )

    # Create backtest function with cached data
    backtest_fn = create_backtest_fn(cached_data=cached_data)

    # Run experiment
    start_time = datetime.now()
    logger.info("Starting experiment execution...")

    try:
        experiment_id = runner.run(
            spec,
            backtest_fn=backtest_fn,
            on_trial_complete=lambda t: logger.debug(
                f"  Trial {t.trial_index}: score={t.trial_score:.3f}"
            ),
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Get results
        result = runner.get_experiment_result(experiment_id)
        result.total_duration_seconds = duration
        result.print_summary()

        # Show top trials
        print("\nTop 5 Trials by Score:")
        print("-" * 80)
        top_trials = runner.get_top_trials(experiment_id, limit=5)
        for i, trial in enumerate(top_trials, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in trial["params"].items())
            print(
                f"  {i}. Score: {trial['trial_score']:.3f} | "
                f"Sharpe: {trial['median_sharpe']:.2f} | "
                f"MaxDD: {trial['median_max_dd']:.1%}"
            )
            print(f"     Params: {params_str}")

        logger.info(f"Experiment complete! Results saved to: {db_path}")

        # Generate HTML report if requested
        if generate_report:
            try:
                report_path = generate_html_report(
                    runner=runner,
                    experiment_id=experiment_id,
                    spec=spec,
                    output_dir=output_path,
                )
                print(f"\nHTML Report: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")

    finally:
        runner.close()


def run_example():
    """
    Run a quick example with the TA metrics strategy.

    This demonstrates the complete workflow without requiring a YAML file.
    """
    from src.backtest import (
        ExperimentSpec,
        SystematicRunner,
        TemporalConfig,
        UniverseConfig,
        OptimizationConfig,
        ReproducibilityConfig,
    )
    print("\n" + "=" * 60)
    print("TA Metrics Strategy - Example Experiment")
    print("=" * 60)

    # Build spec programmatically
    spec = ExperimentSpec(
        name="TA_Metrics_Example",
        strategy="ta_metrics",
        parameters={
            "sma_fast": {"type": "range", "min": 15, "max": 25, "step": 5},
            "sma_slow": {"type": "range", "min": 45, "max": 55, "step": 5},
            "rsi_period": {"type": "fixed", "value": 14},
            "min_score": {"type": "categorical", "values": [2, 3]},
        },
        universe=UniverseConfig(
            type="static",
            symbols=["AAPL", "MSFT"],
        ),
        temporal=TemporalConfig(
            train_days=252,
            test_days=63,
            folds=3,
            purge_days=5,
            embargo_days=2,
        ),
        optimization=OptimizationConfig(
            method="grid",
            metric="sharpe",
            direction="maximize",
            constraints=[
                {"metric": "p10_sharpe", "operator": ">=", "value": -0.5},
            ],
        ),
        reproducibility=ReproducibilityConfig(
            random_seed=42,
            data_version="example_v1",
        ),
    )

    print(f"\nExperiment: {spec.name}")
    print(f"ID: {spec.experiment_id}")

    # Expand parameters
    param_grid = spec.expand_parameter_grid()
    print(f"\nParameter combinations: {len(param_grid)}")
    for i, params in enumerate(param_grid[:3]):
        print(f"  {i+1}: {params}")
    if len(param_grid) > 3:
        print(f"  ... and {len(param_grid) - 3} more")

    # Pre-fetch data from IB
    cached_data = prefetch_data(
        symbols=["AAPL", "MSFT"],
        start_date=None,  # Will use default
        end_date=None,
    )

    # Run experiment (in-memory) with real VectorBT engine
    print("\nRunning experiment...")
    config = RunnerConfig(db_path=":memory:")
    runner = SystematicRunner(config=config)
    backtest_fn = create_backtest_fn(cached_data=cached_data)

    try:
        experiment_id = runner.run(spec, backtest_fn=backtest_fn)
        result = runner.get_experiment_result(experiment_id)
        result.print_summary()
    finally:
        runner.close()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    setup_paths()

    parser = argparse.ArgumentParser(
        description="Systematic Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from YAML spec
  python -m src.runners.systematic_backtest_runner --spec config/backtest/examples/ta_metrics_experiment.yaml

  # Dry run to see what would execute
  python -m src.runners.systematic_backtest_runner --spec config/backtest/examples/ta_metrics_experiment.yaml --dry-run

  # Run quick example
  python -m src.runners.systematic_backtest_runner --example

  # List strategies
  python -m src.runners.systematic_backtest_runner --list-strategies
        """,
    )

    parser.add_argument(
        "--spec",
        type=str,
        help="Path to experiment YAML specification",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiments",
        help="Output directory for results (default: results/experiments)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report after experiment completion",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run a quick example with TA metrics strategy",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies",
    )

    args = parser.parse_args()

    if args.list_strategies:
        list_strategies()
    elif args.example:
        run_example()
    elif args.spec:
        run_experiment(
            spec_path=args.spec,
            output_dir=args.output,
            parallel=args.parallel,
            dry_run=args.dry_run,
            generate_report=args.report,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
