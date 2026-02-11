"""
Optuna-based strategy parameter optimization runner.

Downloads daily OHLCV data, computes regime series, and uses StrategyObjective
to find optimal strategy parameters via TPE sampling. Results are saved as JSON
and can optionally update the strategy YAML config.

Usage:
    # Basic optimization
    python -m src.runners.optimize_runner --strategy trend_pulse \
        --symbols SPY QQQ AAPL NVDA --n-trials 50

    # With custom lookback and output
    python -m src.runners.optimize_runner --strategy regime_flex \
        --symbols SPY AAPL MSFT --years 3 --n-trials 100 \
        --output out/optimization

    # Auto-update YAML params after optimization
    python -m src.runners.optimize_runner --strategy sector_pulse \
        --symbols SPY QQQ AAPL --n-trials 50 --update-yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import optuna
import pandas as pd

from src.backtest.optimization.strategy_objective import BacktestResult, StrategyObjective
from src.domain.strategy.param_loader import (
    get_strategy_metadata,
    list_strategies,
    load_strategy_config,
    update_strategy_params,
)
from src.runners.strategy_compare_runner import (
    _compute_regime_series,
    _download_data,
    _get_strategy_class,
)

logger = logging.getLogger(__name__)


def _download_all_data(
    symbols: List[str],
    start: date,
    end: date,
) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV data for all symbols.

    Args:
        symbols: List of ticker symbols.
        start: Start date for data download.
        end: End date for data download.

    Returns:
        Dict mapping symbol to OHLCV DataFrame.
    """
    all_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = _download_data(symbol, start, end)
            if not df.empty:
                df.attrs["symbol"] = symbol
                all_data[symbol] = df
                print(f"  {symbol}: {len(df)} bars")
            else:
                print(f"  {symbol}: no data (skipped)")
        except Exception as e:
            logger.error(f"Download failed for {symbol}: {e}")
            print(f"  {symbol}: download failed ({e})")
    return all_data


def _build_run_fn(
    strategy_name: str,
    strategy_class: Type,
    all_data: Dict[str, pd.DataFrame],
    regime_series: pd.Series,
    is_portfolio: bool,
    init_cash: float = 100_000.0,
) -> Callable[[Dict[str, Any]], BacktestResult]:
    """Build the run_fn callback for StrategyObjective.

    The returned function takes a params dict, runs the strategy across all symbols
    via BacktestEngine (event-driven), and returns an aggregated BacktestResult.

    Args:
        strategy_name: Name of the strategy.
        strategy_class: Strategy class from registry.
        all_data: Dict of symbol -> OHLCV DataFrame.
        regime_series: Pre-computed regime labels from SPY.
        is_portfolio: Whether this is a portfolio-level strategy.
        init_cash: Initial capital per symbol.

    Returns:
        Callable that takes params dict and returns BacktestResult.
    """

    def run_fn(params: Dict[str, Any]) -> BacktestResult:
        """Run strategy with given params and return aggregated result."""
        from src.runners.strategy_compare_runner import (
            _run_portfolio_strategy,
            _run_strategy_event_driven,
        )

        per_symbol: Dict[str, Dict[str, Any]] = {}
        if is_portfolio:
            result = asyncio.run(
                _run_portfolio_strategy(
                    strategy_name=strategy_name,
                    strategy_class=strategy_class,
                    all_data=all_data,
                    params=params,
                    regime_series=regime_series,
                    init_cash=init_cash,
                )
            )
            if result:
                per_symbol["PORTFOLIO"] = result
        else:
            for symbol, data in all_data.items():
                result = asyncio.run(
                    _run_strategy_event_driven(
                        strategy_name=strategy_name,
                        strategy_class=strategy_class,
                        data=data,
                        params=params,
                        regime_series=regime_series,
                        init_cash=init_cash,
                    )
                )
                if result:
                    per_symbol[symbol] = result

        return _aggregate_to_backtest_result(per_symbol)

    return run_fn


def _aggregate_to_backtest_result(
    per_symbol: Dict[str, Dict[str, Any]],
) -> BacktestResult:
    """Aggregate per-symbol metrics into a single BacktestResult.

    Args:
        per_symbol: Dict of symbol -> metrics dict from _run_strategy_event_driven.

    Returns:
        Aggregated BacktestResult for StrategyObjective scoring.
    """
    import numpy as np

    valid = {s: m for s, m in per_symbol.items() if m}
    if not valid:
        return BacktestResult()

    sharpes = [m["sharpe"] for m in valid.values()]
    returns = [m["total_return"] for m in valid.values()]
    drawdowns = [m["max_drawdown"] for m in valid.values()]
    win_rates = [m["win_rate"] for m in valid.values()]
    trade_counts = [m["total_trades"] for m in valid.values()]

    # Build composite equity curve from all symbols
    equity_series: Optional[pd.Series] = None
    norm_series: Dict[str, pd.Series] = {}
    for sym, m in valid.items():
        if m.get("equity_values") and m.get("equity_index"):
            eq_vals = m["equity_values"]
            eq_idx = m["equity_index"]
            if eq_vals and eq_vals[0] > 0:
                base = eq_vals[0]
                idx = pd.Index([pd.Timestamp(t, unit="s") for t in eq_idx])
                norm_series[sym] = pd.Series([v / base for v in eq_vals], index=idx)

    if norm_series:
        combined = pd.DataFrame(norm_series).ffill().bfill()
        equity_series = combined.mean(axis=1)

    # Compute exposure from equity curve: bars where equity differs from initial
    # capital indicate the strategy had a position. This is more reliable than
    # trade_log (which may not be populated in _extract_metrics).
    total_bars_all = 0
    bars_in_position_all = 0
    for sym, m in valid.items():
        eq_vals = m.get("equity_values", [])
        if len(eq_vals) < 2:
            continue
        total_bars_sym = len(eq_vals)
        total_bars_all += total_bars_sym
        # A bar is "in position" if equity changed from the initial value
        # or if there's a non-zero daily return (equity[i] != equity[i-1]).
        init_val = eq_vals[0]
        for i in range(1, total_bars_sym):
            if abs(eq_vals[i] - init_val[i - 1]) > 0.01:  # More than 1 cent change
                bars_in_position_all += 1

    if total_bars_all > 0:
        exposure_pct = bars_in_position_all / total_bars_all
    else:
        exposure_pct = 0.0

    return BacktestResult(
        sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        total_return=float(np.mean(returns)) if returns else 0.0,
        max_drawdown=float(np.mean(drawdowns)) if drawdowns else 0.0,
        win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
        trade_count=int(np.sum(trade_counts)) if trade_counts else 0,
        total_cost=0.0,
        equity_curve=equity_series,
        exposure_pct=exposure_pct,
    )


def _save_study_results(
    study: optuna.Study,
    strategy_name: str,
    output_dir: str,
) -> str:
    """Save Optuna study results to JSON.

    Args:
        study: Completed Optuna study.
        strategy_name: Name of the optimized strategy.
        output_dir: Directory to save results.

    Returns:
        Path to the saved JSON file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    filename = f"{strategy_name}_{today}.json"
    filepath = out_path / filename

    # Build results dict
    best = study.best_trial
    all_trials: List[Dict[str, Any]] = []
    for trial in study.trials:
        trial_data: Dict[str, Any] = {
            "number": trial.number,
            "score": trial.value,  # None for pruned trials
            "params": trial.params,
            "state": trial.state.name,
            "user_attrs": trial.user_attrs,
        }
        all_trials.append(trial_data)

    results: Dict[str, Any] = {
        "strategy": strategy_name,
        "date": today,
        "n_trials": len(study.trials),
        "best_trial": {
            "number": best.number,
            "score": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
        "all_trials": all_trials,
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return str(filepath)


def run_optimization(
    strategy_name: str,
    symbols: List[str],
    n_trials: int = 50,
    years: int = 3,
    output_dir: str = "out/optimization",
    update_yaml: bool = False,
    init_cash: float = 100_000.0,
) -> Dict[str, Any]:
    """Run Optuna optimization for a strategy.

    Args:
        strategy_name: Name of the strategy to optimize.
        symbols: List of ticker symbols for backtesting.
        n_trials: Number of Optuna trials.
        years: Lookback period in years.
        output_dir: Directory for output JSON.
        update_yaml: If True, update strategy YAML with best params.
        init_cash: Initial capital per symbol.

    Returns:
        Dict with best params and study summary.
    """
    # Validate strategy exists
    available = list_strategies()
    if strategy_name not in available:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {sorted(available)}")

    # Get current params and strategy class
    _module_path, _class_name, current_params, tier = get_strategy_metadata(strategy_name)
    strategy_class = _get_strategy_class(strategy_name)
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy_name}' not found in @register_strategy registry")

    # Check if portfolio-level strategy
    try:
        strat_config = load_strategy_config(strategy_name)
        is_portfolio = strat_config.get("execution_mode") == "portfolio"
    except KeyError:
        is_portfolio = False

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years)

    print(f"Optimization: {strategy_name} ({tier})")
    print(f"Period: {start_date} to {end_date} ({years}yr)")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Trials: {n_trials}")
    print(f"Mode: {'portfolio' if is_portfolio else 'per-symbol'}")
    print()

    # Step 1: Download data
    print("Downloading data...")
    all_data = _download_all_data(symbols, start_date, end_date)
    if not all_data:
        raise RuntimeError("No data available for any symbol")
    print(f"  {len(all_data)}/{len(symbols)} symbols loaded\n")

    # Step 2: Compute regime series from SPY
    regime_symbol = "SPY" if "SPY" in all_data else list(all_data.keys())[0]
    regime_series = _compute_regime_series(all_data[regime_symbol])
    regime_counts = regime_series.value_counts()
    print(f"Regime computed from {regime_symbol}:")
    for r in ["R0", "R1", "R2", "R3"]:
        count = regime_counts.get(r, 0)
        pct = count / len(regime_series) * 100 if len(regime_series) > 0 else 0
        print(f"  {r}: {count} days ({pct:.0f}%)")
    print()

    # Step 3: Build run_fn and objective
    run_fn = _build_run_fn(
        strategy_name=strategy_name,
        strategy_class=strategy_class,
        all_data=all_data,
        regime_series=regime_series,
        is_portfolio=is_portfolio,
        init_cash=init_cash,
    )

    # Per-strategy gate overrides: portfolio/exposure strategies inherit
    # market drawdowns, so they need a wider MaxDD cap than signal-based ones.
    gate_overrides: Dict[str, Dict[str, Any]] = {
        "regime_flex": {"max_drawdown_cap": -0.50, "min_trades": 10},
        "sector_pulse": {"max_drawdown_cap": -0.50, "min_trades": 10},
    }
    gates = gate_overrides.get(strategy_name, {})

    objective = StrategyObjective(
        strategy_name=strategy_name,
        run_fn=run_fn,
        strategy_class=strategy_class,
        **gates,
    )

    # Step 4: Create and run Optuna study
    print(f"Starting Optuna optimization ({n_trials} trials)...")
    print("-" * 60)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{strategy_name}_optimization",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Step 5: Report results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    completed = [t for t in study.trials if t.value is not None]
    pruned = len(study.trials) - len(completed)

    print(f"\nTrials: {len(study.trials)} total, {len(completed)} completed, {pruned} pruned")

    if not completed:
        print("\nNo trials completed successfully. All were pruned by gates.")
        print("Consider relaxing gate thresholds or adjusting parameter ranges.")
        return {"strategy": strategy_name, "best_params": None, "error": "all_pruned"}

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"  Score: {best.value:.4f}")
    print(f"\nBest parameters:")
    for k, v in sorted(best.params.items()):
        # Show comparison with current YAML value
        current = current_params.get(k, "N/A")
        changed = " (changed)" if current != "N/A" and current != v else ""
        print(f"  {k}: {v}  (was: {current}){changed}")

    print(f"\nBest trial metrics:")
    for attr in [
        "total_return",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "calmar",
        "exposure_pct",
    ]:
        val = best.user_attrs.get(attr)
        if val is not None:
            if attr in ("total_return", "max_drawdown", "win_rate", "exposure_pct"):
                print(f"  {attr}: {val:.1%}")
            else:
                print(f"  {attr}: {val}")

    # Step 6: Save results
    result_path = _save_study_results(study, strategy_name, output_dir)
    print(f"\nResults saved to {result_path}")

    # Step 6b: Build experiment tracking page
    try:
        from src.infrastructure.reporting.experiment_tracker.builder import build_experiment_page

        tracker_path = build_experiment_page(optimization_dir=output_dir)
        logger.info(f"Experiment tracker updated: {tracker_path}")
    except Exception as e:
        logger.debug(f"Experiment tracker build skipped: {e}")

    # Step 7: Optionally update YAML
    if update_yaml:
        # Merge best params with current (best params override, keep non-optimized params)
        merged = {**current_params, **best.params}

        # Build rich history metadata from best trial
        opt_results: Dict[str, Any] = {}
        for attr in (
            "total_return",
            "max_drawdown",
            "trade_count",
            "win_rate",
            "calmar",
            "exposure_pct",
        ):
            val = best.user_attrs.get(attr)
            if val is not None:
                opt_results[attr] = round(val, 4) if isinstance(val, float) else val
        if best.value is not None:
            opt_results["composite_score"] = round(best.value, 4)

        opt_changes = [
            f"{k}: {current_params.get(k, 'N/A')} -> {v}"
            for k, v in sorted(best.params.items())
            if current_params.get(k) != v
        ]

        update_strategy_params(
            strategy_name,
            merged,
            source=f"Optuna TPE ({len(completed)}/{len(study.trials)} trials completed)",
            reason=f"Best composite score {best.value:.4f} (trial #{best.number})",
            results=opt_results,
            changes=opt_changes or None,
        )
        print(f"\nYAML updated: config/strategy/{strategy_name}.yaml")
        print("  Previous params moved to history with metrics and change log.")

    return {
        "strategy": strategy_name,
        "best_params": best.params,
        "best_score": best.value,
        "best_metrics": best.user_attrs,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "result_path": result_path,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize strategy parameters using Optuna TPE sampler",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help=f"Strategy to optimize. Available: {', '.join(list_strategies())}",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMZN", "JPM", "XOM", "UNH", "HD"],
        help="Symbols to test (default: 10 diverse stocks)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Lookback period in years (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="out/optimization",
        help="Output directory for results JSON (default: out/optimization)",
    )
    parser.add_argument(
        "--update-yaml",
        action="store_true",
        help="Update strategy YAML config with best params",
    )
    parser.add_argument(
        "--init-cash",
        type=float,
        default=100_000.0,
        help="Initial capital per symbol (default: 100000)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Suppress noisy Optuna logging unless verbose
    if not args.verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    run_optimization(
        strategy_name=args.strategy,
        symbols=args.symbols,
        n_trials=args.n_trials,
        years=args.years,
        output_dir=args.output,
        update_yaml=args.update_yaml,
        init_cash=args.init_cash,
    )


if __name__ == "__main__":
    main()
