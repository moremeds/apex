"""
Repository pattern for database operations.

Provides clean CRUD interfaces for experiments, trials, and runs.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core import RunMetrics, RunResult, RunStatus, TrialAggregates, TrialResult
from ...core.hashing import canonical_json
from .database import DatabaseManager


class ExperimentRepository:
    """Repository for experiment operations."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(
        self,
        experiment_id: str,
        spec_dict: Dict[str, Any],
        base_experiment_id: Optional[str] = None,
        run_version: int = 1,
    ) -> None:
        """Create a new experiment record with version tracking."""
        self.db.execute(
            """
            INSERT INTO experiments (
                experiment_id, base_experiment_id, run_version,
                name, strategy, parameters, universe,
                temporal, optimization, profiles, reproducibility, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                base_experiment_id or experiment_id,
                run_version,
                spec_dict.get("name"),
                spec_dict.get("strategy"),
                canonical_json(spec_dict.get("parameters")),
                canonical_json(spec_dict.get("universe")),
                canonical_json(spec_dict.get("temporal")),
                canonical_json(spec_dict.get("optimization")),
                canonical_json(spec_dict.get("profiles")),
                canonical_json(spec_dict.get("reproducibility")),
                "running",
            ),
        )

    def update_status(
        self, experiment_id: str, status: str, completed_at: Optional[datetime] = None
    ) -> None:
        """Update experiment status."""
        if completed_at:
            self.db.execute(
                "UPDATE experiments SET status = ?, completed_at = ? WHERE experiment_id = ?",
                (status, completed_at, experiment_id),
            )
        else:
            self.db.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                (status, experiment_id),
            )

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        row = self.db.fetchone(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        if row is None:
            return None

        cols = [
            "experiment_id", "base_experiment_id", "run_version",
            "name", "strategy", "parameters", "universe",
            "temporal", "optimization", "profiles", "reproducibility",
            "created_at", "completed_at", "status"
        ]
        result = dict(zip(cols, row))

        # Parse JSON fields
        json_fields = ["parameters", "universe", "temporal", "optimization", "profiles", "reproducibility"]
        for field in json_fields:
            if field in result and result[field] and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass

        return result

    def exists(self, experiment_id: str) -> bool:
        """Check if experiment exists."""
        result = self.db.fetchone(
            "SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        return result is not None


class TrialRepository:
    """Repository for trial operations."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_stub(self, trial_id: str, experiment_id: str, params: dict, trial_index: Optional[int] = None) -> None:
        """Create a minimal trial record (for FK constraint before runs are inserted)."""
        self.db.execute(
            """
            INSERT INTO trials (trial_id, experiment_id, params, trial_index)
            VALUES (?, ?, ?, ?)
            """,
            (trial_id, experiment_id, json.dumps(params), trial_index),
        )

    def create(self, trial: TrialResult) -> None:
        """Create a new trial record."""
        agg = trial.aggregates
        self.db.execute(
            """
            INSERT INTO trials (
                trial_id, experiment_id, params, trial_index, suggested_by,
                median_sharpe, median_return, median_max_dd, median_win_rate,
                median_profit_factor, mad_sharpe, p10_sharpe, p90_sharpe,
                p10_max_dd, p90_max_dd, stability_score, degradation_ratio,
                is_median_sharpe, oos_median_sharpe, trial_score,
                total_runs, successful_runs, failed_runs,
                constraints_met, constraint_violations,
                started_at, completed_at, duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trial.trial_id,
                trial.experiment_id,
                json.dumps(trial.params),
                trial.trial_index,
                trial.suggested_by,
                agg.median_sharpe,
                agg.median_return,
                agg.median_max_dd,
                agg.median_win_rate,
                agg.median_profit_factor,
                agg.mad_sharpe,
                agg.p10_sharpe,
                agg.p90_sharpe,
                agg.p10_max_dd,
                agg.p90_max_dd,
                agg.stability_score,
                agg.degradation_ratio,
                agg.is_median_sharpe,
                agg.oos_median_sharpe,
                trial.trial_score,
                agg.total_runs,
                agg.successful_runs,
                agg.failed_runs,
                trial.constraints_met,
                json.dumps(trial.constraint_violations),
                trial.started_at,
                trial.completed_at,
                trial.total_duration_seconds,
            ),
        )

    def update(self, trial: TrialResult) -> None:
        """Update an existing trial."""
        agg = trial.aggregates
        self.db.execute(
            """
            UPDATE trials SET
                median_sharpe = ?, median_return = ?, median_max_dd = ?,
                median_win_rate = ?, median_profit_factor = ?, mad_sharpe = ?,
                p10_sharpe = ?, p90_sharpe = ?, p10_max_dd = ?, p90_max_dd = ?,
                stability_score = ?, degradation_ratio = ?,
                is_median_sharpe = ?, oos_median_sharpe = ?, trial_score = ?,
                total_runs = ?, successful_runs = ?, failed_runs = ?,
                constraints_met = ?, constraint_violations = ?,
                completed_at = ?, duration_seconds = ?
            WHERE trial_id = ?
            """,
            (
                agg.median_sharpe, agg.median_return, agg.median_max_dd,
                agg.median_win_rate, agg.median_profit_factor, agg.mad_sharpe,
                agg.p10_sharpe, agg.p90_sharpe, agg.p10_max_dd, agg.p90_max_dd,
                agg.stability_score, agg.degradation_ratio,
                agg.is_median_sharpe, agg.oos_median_sharpe, trial.trial_score,
                agg.total_runs, agg.successful_runs, agg.failed_runs,
                trial.constraints_met, json.dumps(trial.constraint_violations),
                trial.completed_at, trial.total_duration_seconds,
                trial.trial_id,
            ),
        )

    def get_top_trials(
        self,
        experiment_id: str,
        limit: int = 10,
        order_by: str = "trial_score DESC",
    ) -> List[Dict[str, Any]]:
        """Get top trials for an experiment."""
        rows = self.db.fetchall(
            f"""
            SELECT trial_id, params, trial_score, median_sharpe, median_return,
                   median_max_dd, p10_sharpe, oos_median_sharpe, constraints_met
            FROM trials
            WHERE experiment_id = ?
            ORDER BY {order_by}
            LIMIT ?
            """,
            (experiment_id, limit),
        )

        return [
            {
                "trial_id": row[0],
                "params": json.loads(row[1]) if row[1] else {},
                "trial_score": row[2],
                "median_sharpe": row[3],
                "median_return": row[4],
                "median_max_dd": row[5],
                "p10_sharpe": row[6],
                "oos_median_sharpe": row[7],
                "constraints_met": row[8],
            }
            for row in rows
        ]


class RunRepository:
    """Repository for run operations."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(self, run: RunResult) -> None:
        """Create a new run record."""
        m = run.metrics
        self.db.execute(
            """
            INSERT INTO runs (
                run_id, trial_id, experiment_id, symbol, window_id,
                profile_version, data_version, status, error,
                is_train, is_oos,
                total_return, cagr, annualized_return, sharpe, sortino, calmar,
                max_drawdown, avg_drawdown, max_dd_duration_days,
                total_trades, win_rate, profit_factor, expectancy, sqn,
                best_trade_pct, worst_trade_pct, avg_win_pct, avg_loss_pct,
                exposure_pct, avg_trade_duration_days,
                total_commission, total_slippage, total_costs,
                started_at, completed_at, duration_seconds, params
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id, run.trial_id, run.experiment_id, run.symbol, run.window_id,
                run.profile_version, run.data_version, run.status.value, run.error,
                run.is_train, run.is_oos,
                m.total_return, m.cagr, m.annualized_return, m.sharpe, m.sortino, m.calmar,
                m.max_drawdown, m.avg_drawdown, m.max_dd_duration_days,
                m.total_trades, m.win_rate, m.profit_factor, m.expectancy, m.sqn,
                m.best_trade_pct, m.worst_trade_pct, m.avg_win_pct, m.avg_loss_pct,
                m.exposure_pct, m.avg_trade_duration_days,
                m.total_commission, m.total_slippage, m.total_costs,
                run.started_at, run.completed_at, run.duration_seconds,
                json.dumps(run.params) if run.params else None,
            ),
        )

    def create_batch(self, runs: List[RunResult]) -> int:
        """Create multiple run records efficiently."""
        if not runs:
            return 0

        records = []
        for run in runs:
            m = run.metrics
            records.append({
                "run_id": run.run_id,
                "trial_id": run.trial_id,
                "experiment_id": run.experiment_id,
                "symbol": run.symbol,
                "window_id": run.window_id,
                "profile_version": run.profile_version,
                "data_version": run.data_version,
                "status": run.status.value,
                "error": run.error,
                "is_train": run.is_train,
                "is_oos": run.is_oos,
                "total_return": m.total_return,
                "cagr": m.cagr,
                "annualized_return": m.annualized_return,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "calmar": m.calmar,
                "max_drawdown": m.max_drawdown,
                "avg_drawdown": m.avg_drawdown,
                "max_dd_duration_days": m.max_dd_duration_days,
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "expectancy": m.expectancy,
                "sqn": m.sqn,
                "best_trade_pct": m.best_trade_pct,
                "worst_trade_pct": m.worst_trade_pct,
                "avg_win_pct": m.avg_win_pct,
                "avg_loss_pct": m.avg_loss_pct,
                "exposure_pct": m.exposure_pct,
                "avg_trade_duration_days": m.avg_trade_duration_days,
                "total_commission": m.total_commission,
                "total_slippage": m.total_slippage,
                "total_costs": m.total_costs,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "duration_seconds": run.duration_seconds,
                "params": json.dumps(run.params) if run.params else None,
            })

        return self.db.insert_batch("runs", records)

    def get_by_trial(self, trial_id: str) -> List[RunResult]:
        """Get all runs for a trial."""
        rows = self.db.fetchall(
            """
            SELECT run_id, trial_id, experiment_id, symbol, window_id,
                   profile_version, data_version, status, error,
                   is_train, is_oos,
                   total_return, cagr, annualized_return, sharpe, sortino, calmar,
                   max_drawdown, avg_drawdown, max_dd_duration_days,
                   total_trades, win_rate, profit_factor, expectancy, sqn,
                   best_trade_pct, worst_trade_pct, avg_win_pct, avg_loss_pct,
                   exposure_pct, avg_trade_duration_days,
                   total_commission, total_slippage, total_costs,
                   started_at, completed_at, duration_seconds, params
            FROM runs
            WHERE trial_id = ?
            """,
            (trial_id,),
        )

        results = []
        for row in rows:
            metrics = RunMetrics(
                total_return=row[11] or 0,
                cagr=row[12] or 0,
                annualized_return=row[13] or 0,
                sharpe=row[14] or 0,
                sortino=row[15] or 0,
                calmar=row[16] or 0,
                max_drawdown=row[17] or 0,
                avg_drawdown=row[18] or 0,
                max_dd_duration_days=row[19] or 0,
                total_trades=row[20] or 0,
                win_rate=row[21] or 0,
                profit_factor=row[22] or 0,
                expectancy=row[23] or 0,
                sqn=row[24] or 0,
                best_trade_pct=row[25] or 0,
                worst_trade_pct=row[26] or 0,
                avg_win_pct=row[27] or 0,
                avg_loss_pct=row[28] or 0,
                exposure_pct=row[29] or 0,
                avg_trade_duration_days=row[30] or 0,
                total_commission=row[31] or 0,
                total_slippage=row[32] or 0,
                total_costs=row[33] or 0,
            )

            results.append(RunResult(
                run_id=row[0],
                trial_id=row[1],
                experiment_id=row[2],
                symbol=row[3],
                window_id=row[4],
                profile_version=row[5] or "",
                data_version=row[6] or "",
                status=RunStatus(row[7]),
                error=row[8],
                is_train=row[9],
                is_oos=row[10],
                metrics=metrics,
                started_at=row[34],
                completed_at=row[35],
                duration_seconds=row[36] or 0,
                params=json.loads(row[37]) if row[37] else None,
            ))

        return results

    def exists(self, run_id: str) -> bool:
        """Check if run exists (for skip logic)."""
        result = self.db.fetchone(
            "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
        )
        return result is not None
