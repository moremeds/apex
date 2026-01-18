"""
Turning Point Training Service Package.

Provides hexagonal architecture for ML model training:
- TrainingConfig: Configuration for training runs
- SymbolTrainingResult: Per-symbol training metrics
- TrainingRunResult: Complete training run output
- TurningPointTrainingService: Application layer orchestration
"""

from .models import (
    ModelComparisonResult,
    SymbolTrainingResult,
    TrainingConfig,
    TrainingRunResult,
)

__all__ = [
    "TrainingConfig",
    "SymbolTrainingResult",
    "ModelComparisonResult",
    "TrainingRunResult",
]
