"""
File-based Experiment Tracker Adapter.

Infrastructure implementation of ExperimentTrackerPort using local JSON files.
Stores training runs and baselines in a simple JSON-based format.

Directory structure:
    experiments/
        runs/
            run_abc123.json
            run_def456.json
        baselines.json  # symbol -> baseline metrics

Usage:
    tracker = FileExperimentTracker(Path("experiments"))

    await tracker.record_run(record)
    baseline = await tracker.get_baseline("SPY")
    comparison = await tracker.compare_to_baseline("SPY", 0.75, 0.60, 0.15)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.domain.interfaces.experiment_tracker import (
    BaselineMetrics,
    ComparisonResult,
    ExperimentTrackerPortABC,
    TrainingRunRecord,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class FileExperimentTracker(ExperimentTrackerPortABC):
    """
    File-based implementation of ExperimentTrackerPort.

    Stores training runs as individual JSON files and baselines
    in a consolidated JSON file.
    """

    def __init__(self, base_dir: Path) -> None:
        """
        Initialize file tracker.

        Args:
            base_dir: Base directory for experiment storage.
        """
        self._base_dir = Path(base_dir)
        self._runs_dir = self._base_dir / "runs"
        self._baselines_file = self._base_dir / "baselines.json"

        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    async def record_run(self, record: TrainingRunRecord) -> None:
        """
        Record a training run to JSON file.

        Args:
            record: Training run record.
        """
        run_file = self._runs_dir / f"run_{record.run_id}_{record.symbol}.json"

        with open(run_file, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

        logger.debug(f"Recorded run {record.run_id} for {record.symbol}")

    async def get_baseline(self, symbol: str) -> Optional[BaselineMetrics]:
        """
        Get baseline metrics for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            BaselineMetrics or None if not found.
        """
        baselines = self._load_baselines()
        data = baselines.get(symbol.upper())

        if not data:
            return None

        return BaselineMetrics(
            symbol=data["symbol"],
            roc_auc=data["roc_auc"],
            pr_auc=data["pr_auc"],
            brier_score=data["brier_score"],
            recorded_at=datetime.fromisoformat(data["recorded_at"]),
            model_version=data["model_version"],
        )

    async def set_baseline(self, symbol: str, metrics: BaselineMetrics) -> None:
        """
        Set baseline metrics for a symbol.

        Args:
            symbol: Trading symbol.
            metrics: Baseline metrics.
        """
        baselines = self._load_baselines()

        baselines[symbol.upper()] = {
            "symbol": metrics.symbol,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "brier_score": metrics.brier_score,
            "recorded_at": metrics.recorded_at.isoformat(),
            "model_version": metrics.model_version,
        }

        self._save_baselines(baselines)
        logger.debug(f"Set baseline for {symbol}")

    async def compare_to_baseline(
        self,
        symbol: str,
        new_roc_auc: float,
        new_pr_auc: float,
        new_brier: float,
        improvement_threshold: float = 0.01,
    ) -> ComparisonResult:
        """
        Compare new model against baseline.

        Args:
            symbol: Trading symbol.
            new_roc_auc: New model's ROC-AUC.
            new_pr_auc: New model's PR-AUC.
            new_brier: New model's Brier score.
            improvement_threshold: Minimum improvement for promotion.

        Returns:
            ComparisonResult with decision.
        """
        baseline = await self.get_baseline(symbol)

        if baseline is None:
            return ComparisonResult(
                symbol=symbol,
                roc_auc_improvement=0.0,
                pr_auc_improvement=0.0,
                brier_improvement=0.0,
                decision="no_baseline",
                reason="No baseline exists for comparison",
                meets_threshold=True,
            )

        roc_improvement = new_roc_auc - baseline.roc_auc
        pr_improvement = new_pr_auc - baseline.pr_auc
        brier_improvement = baseline.brier_score - new_brier  # Lower is better

        meets_threshold = roc_improvement >= improvement_threshold

        decision: Literal["promote", "reject", "no_baseline"]
        if meets_threshold:
            decision = "promote"
            reason = f"ROC-AUC improved by {roc_improvement:.4f} (>= {improvement_threshold})"
        else:
            decision = "reject"
            reason = (
                f"ROC-AUC improvement {roc_improvement:.4f} below threshold {improvement_threshold}"
            )

        return ComparisonResult(
            symbol=symbol,
            roc_auc_improvement=roc_improvement,
            pr_auc_improvement=pr_improvement,
            brier_improvement=brier_improvement,
            decision=decision,
            reason=reason,
            meets_threshold=meets_threshold,
        )

    async def get_recent_runs(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> List[TrainingRunRecord]:
        """
        Get recent training runs.

        Args:
            symbol: Filter by symbol (None = all).
            limit: Maximum runs to return.

        Returns:
            List of runs, newest first.
        """
        runs: List[TrainingRunRecord] = []

        for run_file in sorted(self._runs_dir.glob("run_*.json"), reverse=True):
            if len(runs) >= limit:
                break

            try:
                with open(run_file) as f:
                    data = json.load(f)

                # Filter by symbol if specified
                if symbol and data["symbol"].upper() != symbol.upper():
                    continue

                # Parse the record (simplified - full implementation would parse all fields)
                record = self._parse_run_record(data)
                runs.append(record)

            except Exception as e:
                logger.warning(f"Failed to load run file {run_file}: {e}")

        return runs

    async def get_run(self, run_id: str) -> Optional[TrainingRunRecord]:
        """
        Get a specific run by ID.

        Args:
            run_id: Run identifier.

        Returns:
            TrainingRunRecord or None.
        """
        for run_file in self._runs_dir.glob(f"run_{run_id}_*.json"):
            try:
                with open(run_file) as f:
                    data = json.load(f)
                return self._parse_run_record(data)
            except Exception as e:
                logger.warning(f"Failed to load run {run_id}: {e}")

        return None

    def _load_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Load baselines from file."""
        if not self._baselines_file.exists():
            return {}

        try:
            with open(self._baselines_file) as f:
                result: Dict[str, Dict[str, Any]] = json.load(f)
                return result
        except Exception:
            return {}

    def _save_baselines(self, baselines: Dict[str, dict]) -> None:
        """Save baselines to file."""
        with open(self._baselines_file, "w") as f:
            json.dump(baselines, f, indent=2)

    def _parse_run_record(self, data: dict) -> TrainingRunRecord:
        """Parse run record from JSON data."""
        return TrainingRunRecord(
            run_id=data["run_id"],
            symbol=data["symbol"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            duration_seconds=data["duration_seconds"],
            model_type=data["model_type"],
            cv_splits=data["cv_splits"],
            label_horizon=data["label_horizon"],
            embargo=data["embargo"],
            dataset_start=datetime.fromisoformat(data["dataset_start"]),
            dataset_end=datetime.fromisoformat(data["dataset_end"]),
            dataset_hash=data["dataset_hash"],
            n_samples=data["n_samples"],
            n_positive_top=data["n_positive_top"],
            n_positive_bottom=data["n_positive_bottom"],
            roc_auc_top=data["roc_auc_top"],
            roc_auc_bottom=data["roc_auc_bottom"],
            pr_auc_top=data["pr_auc_top"],
            pr_auc_bottom=data["pr_auc_bottom"],
            brier_top=data["brier_top"],
            brier_bottom=data["brier_bottom"],
            cv_roc_auc_std=data["cv_roc_auc_std"],
            cv_pr_auc_std=data["cv_pr_auc_std"],
            was_promoted=data.get("was_promoted", False),
            model_path=data.get("model_path"),
            feature_importance=data.get("feature_importance", {}),
            notes=data.get("notes", ""),
        )
