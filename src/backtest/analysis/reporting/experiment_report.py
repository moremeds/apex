"""
Experiment report generation helpers.

Query functions for extracting report data from SystematicRunner
and generating HTML experiment reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .html_report import HTMLReportGenerator, ReportConfig, ReportData

logger = logging.getLogger(__name__)


def query_per_symbol_metrics(
    runner: Any, experiment_id: str, best_trial_id: str
) -> Dict[str, Dict[str, Any]]:
    """Query per-symbol aggregated metrics from the runs table for the best trial."""
    rows = runner._db.fetchall(
        """
        SELECT
            symbol,
            AVG(sharpe) as sharpe,
            AVG(sortino) as sortino,
            AVG(calmar) as calmar,
            AVG(total_return) as total_return,
            AVG(cagr) as cagr,
            AVG(max_drawdown) as max_drawdown,
            SUM(total_trades) as total_trades,
            AVG(win_rate) as win_rate,
            AVG(profit_factor) as profit_factor,
            AVG(expectancy) as expectancy,
            AVG(best_trade_pct) as best_trade_pct,
            AVG(worst_trade_pct) as worst_trade_pct,
            AVG(avg_win_pct) as avg_win_pct,
            AVG(avg_loss_pct) as avg_loss_pct,
            AVG(avg_trade_duration_days) as avg_trade_duration_days,
            COUNT(*) as run_count
        FROM runs
        WHERE experiment_id = ? AND trial_id = ?
        GROUP BY symbol
        ORDER BY symbol
        """,
        (experiment_id, best_trial_id),
    )

    per_symbol = {}
    for row in rows:
        per_symbol[row[0]] = {
            "sharpe": row[1] or 0,
            "sortino": row[2] or 0,
            "calmar": row[3] or 0,
            "total_return": row[4] or 0,
            "cagr": row[5] or 0,
            "max_drawdown": row[6] or 0,
            "total_trades": int(row[7] or 0),
            "win_rate": row[8] or 0,
            "profit_factor": row[9] or 0,
            "expectancy": row[10] or 0,
            "best_trade_pct": row[11] or 0,
            "worst_trade_pct": row[12] or 0,
            "avg_win_pct": row[13] or 0,
            "avg_loss_pct": row[14] or 0,
            "avg_trade_duration_days": row[15] or 0,
            "run_count": int(row[16] or 0),
        }

    return per_symbol


def query_per_window_metrics(
    runner: Any, experiment_id: str, best_trial_id: str
) -> List[Dict[str, Any]]:
    """Query per-window (fold) metrics from the runs table for the best trial."""
    rows = runner._db.fetchall(
        """
        SELECT
            window_id,
            AVG(CASE WHEN is_train = true THEN sharpe END) as is_sharpe,
            AVG(CASE WHEN is_oos = true THEN sharpe END) as oos_sharpe,
            AVG(CASE WHEN is_train = true THEN total_return END) as is_return,
            AVG(CASE WHEN is_oos = true THEN total_return END) as oos_return,
            AVG(CASE WHEN is_train = true THEN max_drawdown END) as is_max_dd,
            AVG(CASE WHEN is_oos = true THEN max_drawdown END) as oos_max_dd,
            MIN(started_at) as start_time,
            MAX(completed_at) as end_time
        FROM runs
        WHERE experiment_id = ? AND trial_id = ?
        GROUP BY window_id
        ORDER BY window_id
        """,
        (experiment_id, best_trial_id),
    )

    per_window = []
    for row in rows:
        is_sharpe = row[1] or 0
        oos_sharpe = row[2] or 0
        degradation = 0.0
        if is_sharpe != 0:
            degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe) if is_sharpe else 0

        per_window.append(
            {
                "window_id": row[0],
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "is_return": row[3] or 0,
                "oos_return": row[4] or 0,
                "is_max_dd": row[5] or 0,
                "oos_max_dd": row[6] or 0,
                "degradation": degradation,
                "start_time": str(row[7]) if row[7] else "",
                "end_time": str(row[8]) if row[8] else "",
            }
        )

    return per_window


def build_equity_curve(
    runner: Any, experiment_id: str, best_trial_id: str, initial_capital: float = 100000.0
) -> List[Dict[str, Any]]:
    """Build a simulated equity curve from OOS returns per window."""
    rows = runner._db.fetchall(
        """
        SELECT
            window_id,
            AVG(total_return) as avg_return,
            MAX(completed_at) as end_date
        FROM runs
        WHERE experiment_id = ? AND trial_id = ? AND is_oos = true
        GROUP BY window_id
        ORDER BY window_id
        """,
        (experiment_id, best_trial_id),
    )

    equity_curve = []
    equity = initial_capital

    for row in rows:
        avg_return = row[1] or 0
        end_date = row[2]
        equity = equity * (1 + avg_return)
        equity_curve.append(
            {
                "date": str(end_date)[:10] if end_date else row[0],
                "equity": round(equity, 2),
                "return": round(avg_return * 100, 2),
            }
        )

    return equity_curve


def query_trade_summary(
    runner: Any, experiment_id: str, best_trial_id: str
) -> List[Dict[str, Any]]:
    """Query trade summary per run (individual trades not stored, return run-level summaries)."""
    rows = runner._db.fetchall(
        """
        SELECT
            run_id,
            symbol,
            window_id,
            is_oos,
            total_trades,
            win_rate,
            profit_factor,
            total_return,
            best_trade_pct,
            worst_trade_pct,
            avg_win_pct,
            avg_loss_pct,
            avg_trade_duration_days
        FROM runs
        WHERE experiment_id = ? AND trial_id = ? AND total_trades > 0
        ORDER BY window_id, symbol
        LIMIT 200
        """,
        (experiment_id, best_trial_id),
    )

    trades = []
    for row in rows:
        trades.append(
            {
                "run_id": row[0][:12] if row[0] else "",
                "symbol": row[1],
                "window": row[2],
                "is_oos": "OOS" if row[3] else "IS",
                "trade_count": int(row[4] or 0),
                "win_rate": round((row[5] or 0) * 100, 1),
                "profit_factor": round(row[6] or 0, 2),
                "return_pct": round((row[7] or 0) * 100, 2),
                "best_pct": round((row[8] or 0) * 100, 2),
                "worst_pct": round((row[9] or 0) * 100, 2),
                "avg_win_pct": round((row[10] or 0) * 100, 2),
                "avg_loss_pct": round((row[11] or 0) * 100, 2),
                "avg_duration": round(row[12] or 0, 1),
            }
        )

    return trades


def generate_experiment_report(
    runner: Any, experiment_id: str, spec: Any, output_dir: Path
) -> Path:
    """Generate HTML report for completed experiment."""
    logger.info("Generating HTML report...")

    result = runner.get_experiment_result(experiment_id)
    top_trials = runner.get_top_trials(experiment_id, limit=50)
    symbols = spec.get_symbols()

    # Get best trial ID for querying run-level data
    best_trial_id = top_trials[0]["trial_id"] if top_trials else None

    # Aggregate metrics from top trials
    agg_metrics = {}
    if top_trials:
        metric_values: Dict[str, List[float]] = {}
        for trial in top_trials:
            for our_key, trial_key in [
                ("sharpe", "median_sharpe"),
                ("max_drawdown", "median_max_dd"),
                ("total_return", "median_total_return"),
            ]:
                if trial_key in trial and trial[trial_key] is not None:
                    if our_key not in metric_values:
                        metric_values[our_key] = []
                    metric_values[our_key].append(float(trial[trial_key]))
        agg_metrics = {k: float(np.median(v)) for k, v in metric_values.items() if v}

    best_params = top_trials[0]["params"] if top_trials else {}
    best_score = top_trials[0].get("trial_score", 0.0) if top_trials else 0.0

    # Query per-symbol metrics from runs table
    per_symbol = {}
    per_window = []
    equity_curve = []
    trades = []

    if best_trial_id:
        try:
            per_symbol = query_per_symbol_metrics(runner, experiment_id, best_trial_id)
            per_window = query_per_window_metrics(runner, experiment_id, best_trial_id)
            equity_curve = build_equity_curve(runner, experiment_id, best_trial_id)
            trades = query_trade_summary(runner, experiment_id, best_trial_id)
            logger.info(
                f"Loaded report data: {len(per_symbol)} symbols, {len(per_window)} windows, {len(trades)} trade summaries"
            )
        except Exception as e:
            logger.warning(f"Failed to load detailed report data: {e}")

    # Ensure all symbols have entries (even if no data)
    for s in symbols:
        if s not in per_symbol:
            per_symbol[s] = {}

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
            "success_rate": (
                result.successful_trials / result.total_trials if result.total_trials > 0 else 0
            ),
            "total_runs": result.total_runs,
            "successful_runs": result.successful_runs,
            "pbo": result.pbo if result.pbo is not None else 0.0,
            "dsr": result.dsr if result.dsr is not None else 0.0,
        },
        per_symbol=per_symbol,
        per_window=per_window,
        equity_curve=equity_curve,
        trades=trades,
    )

    report_config = ReportConfig(title=f"Backtest Report: {spec.name}", theme="light")
    generator = HTMLReportGenerator(config=report_config)

    report_path = output_dir / f"{experiment_id}_report.html"
    generated_path = generator.generate(report_data, report_path)

    logger.info(f"HTML report generated: {generated_path}")
    return generated_path
