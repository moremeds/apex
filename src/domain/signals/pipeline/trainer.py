"""
Turning Point Trainer.

Handles model training for turning point prediction.
Extracted from signal_runner.py for better modularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from src.utils.logging_setup import get_logger

from .config import SignalPipelineConfig

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def load_training_universe(subset: str = "quick_test") -> List[str]:
    """
    Load symbols for model training from the universe config.

    Uses the regime_verification_universe.yaml as single source of truth.

    Args:
        subset: Which subset to load. Options:
            - "quick_test": 11 core symbols for fast iteration (default)
            - "all": Full universe (~66 symbols)
            - "market": Market ETFs only (6 symbols)
            - "sector_etfs": Sector ETFs only (11 symbols)

    Returns:
        List of symbols for training.
    """
    from src.domain.services.regime.universe_loader import load_universe

    universe = load_universe()

    if subset == "quick_test":
        return universe.quick_test
    elif subset == "all":
        return universe.all_symbols
    elif subset == "market":
        return [m.symbol for m in universe.market_symbols]
    elif subset == "sector_etfs":
        return universe.sector_etfs
    else:
        logger.warning(f"Unknown subset '{subset}', using quick_test")
        return universe.quick_test


class TurningPointTrainer:
    """
    Handles turning point model training.

    Uses TurningPointTrainingService with hexagonal architecture:
    - FileModelRegistry for model storage
    - FileExperimentTracker for experiment recording

    Usage:
        trainer = TurningPointTrainer(config)
        exit_code = await trainer.train()
    """

    def __init__(self, config: SignalPipelineConfig) -> None:
        """
        Initialize trainer.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

    async def train(self) -> int:
        """
        Run model training phase.

        Returns:
            Exit code (0 for success, non-zero for errors).
        """
        from src.application.services.turning_point.models import TrainingConfig
        from src.application.services.turning_point_training_service import (
            TurningPointTrainingService,
        )
        from src.infrastructure.adapters.file_experiment_tracker import (
            FileExperimentTracker,
        )
        from src.infrastructure.adapters.file_model_registry import FileModelRegistry

        # Determine model output directory
        model_dir = Path(self.config.model_output_dir or "models/turning_point")
        experiment_dir = Path("experiments/turning_point")

        # Create registry and tracker
        registry = FileModelRegistry(model_dir)
        tracker = FileExperimentTracker(experiment_dir)

        # Create training service
        service = TurningPointTrainingService(
            model_registry=registry,
            experiment_tracker=tracker,
        )

        # Determine symbols for training
        # Priority: --model-symbols > quick_test universe (if training-only) > --symbols
        if self.config.model_symbols:
            model_symbols = self.config.model_symbols
        elif not self.config.live and not self.config.backfill:
            # Training-only mode: use quick_test universe from config
            model_symbols = load_training_universe("quick_test")
            print(f"Using quick_test universe: {', '.join(model_symbols)}")
        else:
            # Training + signals mode: use the signal symbols
            model_symbols = self.config.symbols

        # Determine effective eval_only (dry_run implies eval_only)
        effective_eval_only = self.config.eval_only or self.config.dry_run

        # Create training config
        training_config = TrainingConfig(
            symbols=model_symbols,
            days=self.config.model_days,
            model_type="logistic",
            cv_splits=5,
            force_update=self.config.force_retrain and not self.config.dry_run,
            eval_only=effective_eval_only,
            max_workers=self.config.train_concurrency,
        )

        print(f"Training Configuration:")
        print(f"  Symbols:         {', '.join(model_symbols)}")
        print(f"  Days of history: {self.config.model_days}")
        print(f"  Model type:      logistic")
        print(f"  Force update:    {self.config.force_retrain and not self.config.dry_run}")
        print(f"  Eval only:       {effective_eval_only}")
        print(f"  Dry run:         {self.config.dry_run}")
        print(f"  Output dir:      {model_dir}")
        print()

        try:
            result = await service.train(training_config)

            # Print summary
            print("\n" + "-" * 60)
            print("TRAINING RESULTS")
            print("-" * 60)
            print(result.summary())

            # Detailed per-symbol results
            if self.config.verbose:
                print("\nPer-symbol details:")
                for symbol, sym_result in result.results.items():
                    comparison = result.comparisons.get(symbol)
                    print(f"\n  {symbol}:")
                    print(
                        f"    ROC-AUC (top):    {sym_result.roc_auc_top:.4f} +/- {sym_result.roc_auc_top_std:.4f}"
                    )
                    print(
                        f"    ROC-AUC (bottom): {sym_result.roc_auc_bottom:.4f} +/- {sym_result.roc_auc_bottom_std:.4f}"
                    )
                    print(f"    PR-AUC (combined): {sym_result.pr_auc_combined:.4f}")
                    if comparison:
                        print(f"    Decision: {comparison.decision} - {comparison.reason}")

            # Check for failures
            if result.failed:
                print(f"\nWarning: {len(result.failed)} symbols failed to train")
                for symbol, error in result.errors.items():
                    print(f"  {symbol}: {error}")
                return 1

            return 0

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            print(f"\nError: Training failed - {e}")
            return 1
