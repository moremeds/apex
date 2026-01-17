"""
Turning Point Detection Module (Phase 4).

Predicts TOP_RISK and BOTTOM_RISK to gate regime decisions.

Components:
- labels: ZigZag pivot + reversal risk labeling (no leakage)
- features: Feature extraction from regime components
- cv: Purged + Embargo Cross-Validation
- calibration: Probability calibration evidence
- model: Model training and inference

Usage:
    from src.domain.signals.indicators.regime.turning_point import (
        TurningPointOutput,
        TurningPointLabeler,
        PurgedTimeSeriesSplit,
        CalibrationEvidence,
    )
"""

from .calibration import CalibrationEvidence
from .cv import PurgedTimeSeriesSplit
from .features import TurningPointFeatures, extract_features
from .labels import (
    CurrentTurningPointState,
    TurnType,
    TurningPointHistory,
    TurningPointLabeler,
    TurningPointRecord,
    ZigZagPivot,
)
from .model import TurningPointModel, TurningPointOutput

__all__ = [
    "TurningPointOutput",
    "TurningPointLabeler",
    "TurningPointFeatures",
    "extract_features",
    "PurgedTimeSeriesSplit",
    "CalibrationEvidence",
    "TurningPointModel",
    # History tracking for verification
    "TurningPointHistory",
    "TurningPointRecord",
    "CurrentTurningPointState",
    "TurnType",
    "ZigZagPivot",
]
