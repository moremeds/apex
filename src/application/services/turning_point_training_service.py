"""
Turning Point Model Training Service (Hexagonal Architecture).

Application layer orchestration for training turning point prediction models.
This service:
1. Coordinates training across multiple symbols
2. Uses injected ports for all I/O (no direct file/network access)
3. Returns typed results (dataclasses, not Dict[str, Any])
4. Handles model versioning and promotion decisions

The service depends on:
- ModelRegistryPort: For saving/loading model artifacts
- ExperimentTrackerPort: For recording runs and baselines
- HistoricalDataPort: For fetching training data

Usage:
    # Production
    registry = FileModelRegistry(Path("models/turning_point"))
    tracker = FileExperimentTracker(Path("experiments"))
    service = TurningPointTrainingService(registry, tracker)

    config = TrainingConfig(symbols=["SPY", "QQQ"], days=750)
    result = await service.train(config)

    # Testing (with fakes)
    registry = FakeModelRegistry()
    tracker = FakeExperimentTracker()
    service = TurningPointTrainingService(registry, tracker)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    import pandas as pd

    from src.domain.interfaces.experiment_tracker import ExperimentTrackerPort
    from src.domain.interfaces.model_registry import ModelRegistryPort

from src.utils.logging_setup import get_logger

from .turning_point.models import (
    ModelComparisonResult,
    SymbolTrainingResult,
    TrainingConfig,
    TrainingRunResult,
)

logger = get_logger(__name__)


class TurningPointTrainingService:
    """
    Application layer service for training turning point models.

    This service:
    - Has NO direct I/O (filesystem, network, database)
    - Uses injected ports for all external interactions
    - Returns typed results for type safety
    - Can be tested with fake implementations

    Hexagonal Architecture:
    - Application layer: This service (orchestration)
    - Domain layer: TurningPointModel, feature extraction
    - Infrastructure layer: Ports (ModelRegistryPort, etc.)
    """

    # Promotion threshold: ROC-AUC improvement needed to promote
    IMPROVEMENT_THRESHOLD = 0.01  # 1% improvement required

    def __init__(
        self,
        model_registry: "ModelRegistryPort",
        experiment_tracker: Optional["ExperimentTrackerPort"] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the training service.

        Args:
            model_registry: Port for model artifact storage.
            experiment_tracker: Port for experiment tracking (optional).
            logger: Custom logger (optional).
        """
        self._registry = model_registry
        self._tracker = experiment_tracker
        self._logger = logger or get_logger(__name__)

    async def train(self, config: TrainingConfig) -> TrainingRunResult:
        """
        Train models for configured symbols.

        Steps:
        1. Fetch historical data for each symbol
        2. Train model (labels, features, CV)
        3. Compare against baseline
        4. Promote if better (unless eval_only)
        5. Record experiment

        Args:
            config: Training configuration.

        Returns:
            TrainingRunResult with all metrics and decisions.
        """
        run_id = str(uuid4())[:8]
        started_at = datetime.utcnow()

        self._logger.info(f"Starting training run {run_id} for {len(config.symbols)} symbols")

        results: Dict[str, SymbolTrainingResult] = {}
        comparisons: Dict[str, ModelComparisonResult] = {}
        promoted: List[str] = []
        rejected: List[str] = []
        failed: List[str] = []
        errors: Dict[str, str] = {}

        # Train symbols (optionally in parallel)
        if config.max_workers > 1 and len(config.symbols) > 1:
            results, errors = await self._train_parallel(config)
        else:
            for symbol in config.symbols:
                try:
                    result = await self._train_symbol(symbol, config)
                    results[symbol] = result
                except Exception as e:
                    self._logger.error(f"Training failed for {symbol}: {e}")
                    failed.append(symbol)
                    errors[symbol] = str(e)

        # Compare and promote
        for symbol, result in results.items():
            try:
                comparison = await self._compare_and_promote(
                    symbol=symbol,
                    result=result,
                    config=config,
                )
                comparisons[symbol] = comparison

                if comparison.should_promote:
                    promoted.append(symbol)
                else:
                    rejected.append(symbol)

            except Exception as e:
                self._logger.error(f"Comparison failed for {symbol}: {e}")
                rejected.append(symbol)
                errors[symbol] = str(e)

        completed_at = datetime.utcnow()

        run_result = TrainingRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            config=config,
            results=results,
            comparisons=comparisons,
            promoted=promoted,
            rejected=rejected,
            failed=failed,
            errors=errors,
        )

        # Record experiment
        if self._tracker:
            await self._record_experiment(run_result)

        self._logger.info(
            f"Training run {run_id} complete: "
            f"{len(promoted)} promoted, {len(rejected)} rejected, {len(failed)} failed"
        )

        return run_result

    async def _train_symbol(
        self,
        symbol: str,
        config: TrainingConfig,
    ) -> SymbolTrainingResult:
        """
        Train model for a single symbol.

        This method does the heavy lifting:
        1. Fetch historical data
        2. Generate labels
        3. Extract features
        4. Train model with CV
        5. Return typed result

        Args:
            symbol: Trading symbol to train on.
            config: Training configuration.

        Returns:
            SymbolTrainingResult with all metrics.
        """
        import time

        start_time = time.time()
        self._logger.info(f"Training {symbol}...")

        # Import training components (domain layer)
        from src.domain.signals.indicators.regime.turning_point import (
            TurningPointLabeler,
            TurningPointModel,
        )
        from src.domain.signals.indicators.regime.turning_point.features import (
            extract_features,
        )

        # 1. Fetch historical data
        df = await self._fetch_historical_data(symbol, config.days)
        dataset_hash = self._compute_dataset_hash(df)

        # 2. Generate labels
        labeler = TurningPointLabeler(
            atr_period=config.atr_period,
            zigzag_threshold=config.zigzag_threshold,
            risk_horizon=config.label_horizon,
            risk_threshold=config.risk_threshold,
        )
        y_top, y_bottom, _ = labeler.generate_combined_labels(df)

        # 3. Extract features
        features_df = extract_features(df)

        # 4. Align data (remove NaN rows, exclude last horizon bars)
        valid_mask = ~features_df.isna().any(axis=1)
        valid_idx = features_df.index[valid_mask]
        valid_idx = valid_idx[: -config.label_horizon]

        X = features_df.loc[valid_idx].values
        y_top_arr = y_top.loc[valid_idx].values
        y_bottom_arr = y_bottom.loc[valid_idx].values

        # 5. Train model
        model = TurningPointModel(
            model_type=config.model_type,
            confidence_threshold=0.7,
        )

        top_metrics, bottom_metrics = model.train(
            X=X,
            y_top=y_top_arr,
            y_bottom=y_bottom_arr,
            cv_splits=config.cv_splits,
            label_horizon=config.label_horizon,
            embargo=config.embargo,
        )

        # 6. Save candidate model
        from src.domain.interfaces.model_registry import ModelMetadata

        metadata = ModelMetadata(
            symbol=symbol,
            model_type=config.model_type,
            trained_at=datetime.utcnow(),
            dataset_start=df.index[0].to_pydatetime(),
            dataset_end=df.index[-1].to_pydatetime(),
            dataset_hash=dataset_hash,
            feature_version="1.0",
            roc_auc=(top_metrics.cv_roc_auc_mean + bottom_metrics.cv_roc_auc_mean) / 2,
            pr_auc=(top_metrics.cv_pr_auc_mean + bottom_metrics.cv_pr_auc_mean) / 2,
            brier_score=(
                (top_metrics.calibration.brier_score if top_metrics.calibration else 0)
                + (bottom_metrics.calibration.brier_score if bottom_metrics.calibration else 0)
            )
            / 2,
            cv_splits=config.cv_splits,
            label_horizon=config.label_horizon,
            embargo=config.embargo,
            feature_importance=top_metrics.feature_importance or {},
        )

        await self._registry.save_candidate(symbol, model, metadata)

        training_time = time.time() - start_time

        return SymbolTrainingResult(
            symbol=symbol,
            trained_at=datetime.utcnow(),
            dataset_hash=dataset_hash,
            n_samples=len(X),
            n_positive_top=int(y_top_arr.sum()),
            n_positive_bottom=int(y_bottom_arr.sum()),
            dataset_start=df.index[0].to_pydatetime(),
            dataset_end=df.index[-1].to_pydatetime(),
            roc_auc_top=top_metrics.cv_roc_auc_mean,
            roc_auc_top_std=top_metrics.cv_roc_auc_std,
            pr_auc_top=top_metrics.cv_pr_auc_mean,
            pr_auc_top_std=top_metrics.cv_pr_auc_std,
            brier_top=top_metrics.calibration.brier_score if top_metrics.calibration else 0,
            roc_auc_bottom=bottom_metrics.cv_roc_auc_mean,
            roc_auc_bottom_std=bottom_metrics.cv_roc_auc_std,
            pr_auc_bottom=bottom_metrics.cv_pr_auc_mean,
            pr_auc_bottom_std=bottom_metrics.cv_pr_auc_std,
            brier_bottom=(
                bottom_metrics.calibration.brier_score if bottom_metrics.calibration else 0
            ),
            feature_importance_top=dict(
                sorted(
                    (top_metrics.feature_importance or {}).items(),
                    key=lambda x: -x[1],
                )[:10]
            ),
            feature_importance_bottom=dict(
                sorted(
                    (bottom_metrics.feature_importance or {}).items(),
                    key=lambda x: -x[1],
                )[:10]
            ),
            ece_top=(
                top_metrics.calibration.expected_calibration_error
                if top_metrics.calibration
                else None
            ),
            ece_bottom=(
                bottom_metrics.calibration.expected_calibration_error
                if bottom_metrics.calibration
                else None
            ),
            training_seconds=training_time,
        )

    async def _train_parallel(
        self,
        config: TrainingConfig,
    ) -> tuple[Dict[str, SymbolTrainingResult], Dict[str, str]]:
        """
        Train symbols in parallel using thread pool.

        Args:
            config: Training configuration.

        Returns:
            Tuple of (results dict, errors dict).
        """
        results: Dict[str, SymbolTrainingResult] = {}
        errors: Dict[str, str] = {}

        # Run training in thread pool (CPU-bound operations)
        async def train_one(
            symbol: str,
        ) -> tuple[str, Optional[SymbolTrainingResult], Optional[str]]:
            try:
                result = await self._train_symbol(symbol, config)
                return symbol, result, None
            except Exception as e:
                self._logger.error(f"Training failed for {symbol}: {e}")
                return symbol, None, str(e)

        # Limit concurrency
        semaphore = asyncio.Semaphore(config.max_workers)

        async def train_with_semaphore(
            symbol: str,
        ) -> tuple[str, Optional[SymbolTrainingResult], Optional[str]]:
            async with semaphore:
                return await train_one(symbol)

        tasks = [train_with_semaphore(s) for s in config.symbols]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for outcome in outcomes:
            if isinstance(outcome, Exception):
                self._logger.error(f"Unexpected error: {outcome}")
                continue
            symbol, result, error = outcome
            if result:
                results[symbol] = result
            if error:
                errors[symbol] = error

        return results, errors

    async def _compare_and_promote(
        self,
        symbol: str,
        result: SymbolTrainingResult,
        config: TrainingConfig,
    ) -> ModelComparisonResult:
        """
        Compare trained model against baseline and potentially promote.

        Args:
            symbol: Trading symbol.
            result: Training result for the symbol.
            config: Training configuration.

        Returns:
            ModelComparisonResult with decision.
        """
        # Load baseline metadata
        baseline_metadata = await self._registry.load_metadata(symbol, "active")

        if baseline_metadata is None:
            # No baseline - promote automatically (unless eval_only)
            if not config.eval_only:
                await self._registry.promote_to_active(symbol)

            return ModelComparisonResult(
                symbol=symbol,
                candidate_roc_auc=result.roc_auc_combined,
                baseline_roc_auc=None,
                improvement_pct=0.0,
                decision="promote" if not config.eval_only else "no_baseline",
                reason="No baseline exists" + (" (eval_only)" if config.eval_only else ""),
                candidate_pr_auc=result.pr_auc_combined,
                candidate_brier=result.brier_combined,
            )

        # Compare metrics
        improvement = result.roc_auc_combined - baseline_metadata.roc_auc
        improvement_pct = (
            improvement / baseline_metadata.roc_auc if baseline_metadata.roc_auc > 0 else 0
        )

        # Decision logic
        if config.eval_only:
            decision = "reject"
            reason = "eval_only mode - no promotion"
        elif config.force_update:
            decision = "promote"
            reason = "force_update enabled"
        elif improvement >= self.IMPROVEMENT_THRESHOLD:
            decision = "promote"
            reason = f"ROC-AUC improved by {improvement_pct:.1%}"
        else:
            decision = "reject"
            reason = f"ROC-AUC improvement {improvement_pct:.1%} below threshold {self.IMPROVEMENT_THRESHOLD:.1%}"

        # Promote if decided
        if decision == "promote":
            await self._registry.promote_to_active(symbol)

        return ModelComparisonResult(
            symbol=symbol,
            candidate_roc_auc=result.roc_auc_combined,
            baseline_roc_auc=baseline_metadata.roc_auc,
            improvement_pct=improvement_pct,
            decision=decision,
            reason=reason,
            candidate_pr_auc=result.pr_auc_combined,
            baseline_pr_auc=baseline_metadata.pr_auc,
            candidate_brier=result.brier_combined,
            baseline_brier=baseline_metadata.brier_score,
        )

    async def _fetch_historical_data(
        self,
        symbol: str,
        days: int,
    ) -> "pd.DataFrame":
        """
        Fetch historical OHLCV data for training.

        Currently uses yfinance directly. In production, this should
        use an injected HistoricalDataPort for testability.

        Args:
            symbol: Trading symbol.
            days: Number of days of history.

        Returns:
            DataFrame with OHLCV columns.
        """
        import yfinance as yf

        self._logger.debug(f"Fetching {days} days of {symbol} data...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days + 50}d", interval="1d")
        df.columns = df.columns.str.lower()

        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}: only {len(df)} bars")

        return df

    def _compute_dataset_hash(self, df: "pd.DataFrame") -> str:
        """
        Compute hash of dataset for reproducibility tracking.

        Args:
            df: Training DataFrame.

        Returns:
            SHA256 hash of the data.
        """
        # Hash key columns
        data_str = df[["open", "high", "low", "close", "volume"]].to_csv()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    async def _record_experiment(self, result: TrainingRunResult) -> None:
        """
        Record experiment in tracker.

        Args:
            result: Training run result.
        """
        if not self._tracker:
            return

        from src.domain.interfaces.experiment_tracker import TrainingRunRecord

        for symbol, sym_result in result.results.items():
            comparison = result.comparisons.get(symbol)

            record = TrainingRunRecord(
                run_id=result.run_id,
                symbol=symbol,
                started_at=result.started_at,
                completed_at=result.completed_at,
                duration_seconds=sym_result.training_seconds,
                model_type=result.config.model_type,
                cv_splits=result.config.cv_splits,
                label_horizon=result.config.label_horizon,
                embargo=result.config.embargo,
                dataset_start=sym_result.dataset_start,
                dataset_end=sym_result.dataset_end,
                dataset_hash=sym_result.dataset_hash,
                n_samples=sym_result.n_samples,
                n_positive_top=sym_result.n_positive_top,
                n_positive_bottom=sym_result.n_positive_bottom,
                roc_auc_top=sym_result.roc_auc_top,
                roc_auc_bottom=sym_result.roc_auc_bottom,
                pr_auc_top=sym_result.pr_auc_top,
                pr_auc_bottom=sym_result.pr_auc_bottom,
                brier_top=sym_result.brier_top,
                brier_bottom=sym_result.brier_bottom,
                cv_roc_auc_std=(sym_result.roc_auc_top_std + sym_result.roc_auc_bottom_std) / 2,
                cv_pr_auc_std=(sym_result.pr_auc_top_std + sym_result.pr_auc_bottom_std) / 2,
                was_promoted=symbol in result.promoted,
                feature_importance=sym_result.feature_importance_top,
            )

            await self._tracker.record_run(record)
