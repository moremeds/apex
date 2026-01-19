"""
Turning Point Model (Phase 4).

LogisticRegression or LightGBM model for predicting TOP_RISK/BOTTOM_RISK.
Target inference time: <1ms per bar.

Output includes:
- turn_state: NONE, TOP_RISK, BOTTOM_RISK
- turn_confidence: Calibrated probability [0.0, 1.0]
- top_features: Top 3 contributing features
"""

import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .calibration import CalibrationEvidence, compute_calibration_evidence, compute_roc_and_pr_auc
from .cv import PurgedTimeSeriesSplit
from .features import TurningPointFeatures


class TurnState(str, Enum):
    """Turning point state."""

    NONE = "none"
    TOP_RISK = "top_risk"  # Risk of market top / decline
    BOTTOM_RISK = "bottom_risk"  # Risk of market bottom / rally


@dataclass
class TurningPointOutput:
    """
    Output from turning point model inference.

    Used to gate regime decisions:
    - TOP_RISK with high confidence → Block R0, force R1 minimum
    - BOTTOM_RISK with high confidence → Accelerate R3 entry
    """

    turn_state: TurnState = TurnState.NONE
    turn_confidence: float = 0.0  # Calibrated probability [0, 1]

    # Top contributing features: [(feature_name, contribution)]
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    # Model metadata
    model_version: str = "turning_point@1.0"
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "turn_state": self.turn_state.value,
            "turn_confidence": self.turn_confidence,
            "top_features": [
                {"name": name, "contribution": contrib} for name, contrib in self.top_features
            ],
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurningPointOutput":
        """Deserialize from dictionary."""
        top_features = [(f["name"], f["contribution"]) for f in data.get("top_features", [])]
        return cls(
            turn_state=TurnState(data.get("turn_state", "none")),
            turn_confidence=data.get("turn_confidence", 0.0),
            top_features=top_features,
            model_version=data.get("model_version", "turning_point@1.0"),
            inference_time_ms=data.get("inference_time_ms", 0.0),
        )

    def should_block_r0(self, threshold: float = 0.7) -> bool:
        """Check if TOP_RISK should block R0 regime."""
        return self.turn_state == TurnState.TOP_RISK and self.turn_confidence >= threshold

    def should_accelerate_r3(self, threshold: float = 0.7) -> bool:
        """Check if BOTTOM_RISK should accelerate R3 entry."""
        return self.turn_state == TurnState.BOTTOM_RISK and self.turn_confidence >= threshold


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    # Cross-validation scores
    cv_roc_auc_mean: float = 0.0
    cv_roc_auc_std: float = 0.0
    cv_pr_auc_mean: float = 0.0
    cv_pr_auc_std: float = 0.0

    # Calibration evidence
    calibration: Optional[CalibrationEvidence] = None

    # Model coefficients (for interpretability)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "cv_roc_auc_mean": self.cv_roc_auc_mean,
            "cv_roc_auc_std": self.cv_roc_auc_std,
            "cv_pr_auc_mean": self.cv_pr_auc_mean,
            "cv_pr_auc_std": self.cv_pr_auc_std,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "feature_importance": self.feature_importance,
        }


class TurningPointModel:
    """
    Model for turning point prediction.

    Uses LogisticRegression by default (fast inference, interpretable).
    Can be upgraded to LightGBM for better accuracy if needed.
    """

    MODEL_VERSION = "turning_point@1.0"

    def __init__(
        self,
        model_type: Literal["logistic", "lightgbm"] = "logistic",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize model.

        Args:
            model_type: "logistic" or "lightgbm"
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold

        self.top_model: Any = None  # For TOP_RISK
        self.bottom_model: Any = None  # For BOTTOM_RISK
        self.scaler: StandardScaler = StandardScaler()  # Feature scaling for convergence
        self.feature_names: List[str] = TurningPointFeatures.feature_names()
        self.is_fitted = False
        self.training_metrics_top: Optional[TrainingMetrics] = None
        self.training_metrics_bottom: Optional[TrainingMetrics] = None

    def _create_model(self) -> Any:
        """Create a new model instance."""
        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
            )
        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb

                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    class_weight="balanced",
                    verbose=-1,
                )
            except ImportError:
                # Fall back to logistic if lightgbm not available
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: np.ndarray,
        y_top: np.ndarray,
        y_bottom: np.ndarray,
        cv_splits: int = 5,
        label_horizon: int = 10,
        embargo: int = 2,
    ) -> Tuple[TrainingMetrics, TrainingMetrics]:
        """
        Train models for TOP_RISK and BOTTOM_RISK.

        Uses Purged + Embargo CV to avoid leakage.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_top: TOP_RISK labels
            y_bottom: BOTTOM_RISK labels
            cv_splits: Number of CV folds
            label_horizon: Label horizon for purging
            embargo: Embargo period

        Returns:
            Tuple of (top_metrics, bottom_metrics)
        """

        cv = PurgedTimeSeriesSplit(
            n_splits=cv_splits,
            label_horizon=label_horizon,
            embargo=embargo,
        )

        # Fit scaler on all data and transform
        X_scaled = self.scaler.fit_transform(X)

        # Train TOP_RISK model
        self.top_model = self._create_model()
        self.training_metrics_top = self._train_single_model(
            self.top_model, X_scaled, y_top, cv, "top_risk"
        )

        # Train BOTTOM_RISK model
        self.bottom_model = self._create_model()
        self.training_metrics_bottom = self._train_single_model(
            self.bottom_model, X_scaled, y_bottom, cv, "bottom_risk"
        )

        self.is_fitted = True
        return self.training_metrics_top, self.training_metrics_bottom

    def _train_single_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: PurgedTimeSeriesSplit,
        label_name: str,
    ) -> TrainingMetrics:
        """Train a single model with CV evaluation."""
        roc_aucs = []
        pr_aucs = []
        all_preds = []
        all_true = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predict probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Compute metrics
            roc_auc, pr_auc = compute_roc_and_pr_auc(y_test, y_pred_proba)
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)

            all_preds.extend(y_pred_proba)
            all_true.extend(y_test)

        # Final fit on all data
        model.fit(X, y)

        # Calibration evidence
        y_pred_all = model.predict_proba(X)[:, 1]
        calibration = compute_calibration_evidence(y, y_pred_all)

        # Feature importance
        if hasattr(model, "coef_"):
            importance = dict(zip(self.feature_names, np.abs(model.coef_[0])))
        elif hasattr(model, "feature_importances_"):
            importance = dict(zip(self.feature_names, model.feature_importances_))
        else:
            importance = {}

        return TrainingMetrics(
            n_samples=len(y),
            n_positive=int(np.sum(y)),
            n_negative=int(np.sum(1 - y)),
            cv_roc_auc_mean=float(np.mean(roc_aucs)) if roc_aucs else 0.0,
            cv_roc_auc_std=float(np.std(roc_aucs)) if roc_aucs else 0.0,
            cv_pr_auc_mean=float(np.mean(pr_aucs)) if pr_aucs else 0.0,
            cv_pr_auc_std=float(np.std(pr_aucs)) if pr_aucs else 0.0,
            calibration=calibration,
            feature_importance=importance,
        )

    def predict(self, features: TurningPointFeatures) -> TurningPointOutput:
        """
        Predict turning point state for a single bar.

        Target: <1ms inference time.

        Args:
            features: Feature set for current bar

        Returns:
            TurningPointOutput with state, confidence, and top features
        """
        import time

        start = time.perf_counter()

        if not self.is_fitted:
            return TurningPointOutput(
                turn_state=TurnState.NONE,
                turn_confidence=0.0,
                model_version=self.MODEL_VERSION,
            )

        X = features.to_array().reshape(1, -1)

        # Handle missing or unfitted scaler (can happen with models saved by FileModelRegistry)
        if hasattr(self, 'scaler') and self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X)
            except Exception:
                # Scaler not fitted, use unscaled features
                X_scaled = X
        else:
            X_scaled = X

        # Predict probabilities
        top_prob = self.top_model.predict_proba(X_scaled)[0, 1]
        bottom_prob = self.bottom_model.predict_proba(X_scaled)[0, 1]

        # Determine state
        if top_prob >= self.confidence_threshold and top_prob > bottom_prob:
            turn_state = TurnState.TOP_RISK
            confidence = top_prob
            model = self.top_model
        elif bottom_prob >= self.confidence_threshold and bottom_prob > top_prob:
            turn_state = TurnState.BOTTOM_RISK
            confidence = bottom_prob
            model = self.bottom_model
        else:
            turn_state = TurnState.NONE
            confidence = max(top_prob, bottom_prob)
            model = self.top_model if top_prob > bottom_prob else self.bottom_model

        # Get top contributing features
        top_features = self._get_top_features(features, model)

        inference_time = (time.perf_counter() - start) * 1000  # ms

        return TurningPointOutput(
            turn_state=turn_state,
            turn_confidence=confidence,
            top_features=top_features,
            model_version=self.MODEL_VERSION,
            inference_time_ms=inference_time,
        )

    def _get_top_features(
        self,
        features: TurningPointFeatures,
        model: Any,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Get top contributing features for interpretability."""
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
        elif hasattr(model, "feature_importances_"):
            coefs = model.feature_importances_
        else:
            return []

        X = features.to_array()

        # Contribution = coefficient * feature value
        contributions = coefs * X
        indices = np.argsort(np.abs(contributions))[::-1][:top_k]

        return [(self.feature_names[i], float(contributions[i])) for i in indices]

    def predict_batch(self, X: np.ndarray) -> List[TurningPointOutput]:
        """Predict for multiple samples."""
        if not self.is_fitted:
            return [TurningPointOutput() for _ in range(len(X))]

        X_scaled = self.scaler.transform(X)
        outputs = []
        top_probs = self.top_model.predict_proba(X_scaled)[:, 1]
        bottom_probs = self.bottom_model.predict_proba(X_scaled)[:, 1]

        for i in range(len(X)):
            if top_probs[i] >= self.confidence_threshold and top_probs[i] > bottom_probs[i]:
                turn_state = TurnState.TOP_RISK
                confidence = top_probs[i]
            elif bottom_probs[i] >= self.confidence_threshold:
                turn_state = TurnState.BOTTOM_RISK
                confidence = bottom_probs[i]
            else:
                turn_state = TurnState.NONE
                confidence = max(top_probs[i], bottom_probs[i])

            outputs.append(
                TurningPointOutput(
                    turn_state=turn_state,
                    turn_confidence=float(confidence),
                    model_version=self.MODEL_VERSION,
                )
            )

        return outputs

    def save(self, path: Path) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_type": self.model_type,
            "confidence_threshold": self.confidence_threshold,
            "top_model": self.top_model,
            "bottom_model": self.bottom_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "training_metrics_top": (
                self.training_metrics_top.to_dict() if self.training_metrics_top else None
            ),
            "training_metrics_bottom": (
                self.training_metrics_bottom.to_dict() if self.training_metrics_bottom else None
            ),
            "model_version": self.MODEL_VERSION,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "TurningPointModel":
        """Load model from file.

        Handles both formats:
        - Old format (dict): Saved by TurningPointModel.save()
        - New format (TurningPointModel): Saved directly by FileModelRegistry
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Handle new format: FileModelRegistry saves TurningPointModel directly
        if isinstance(data, cls):
            # Ensure scaler exists (might be missing if saved by FileModelRegistry)
            if not hasattr(data, 'scaler') or data.scaler is None:
                data.scaler = StandardScaler()
                # Mark as needing refit on first predict
                data._scaler_needs_fit = True
            return data

        # Handle old format: dict with model components
        model = cls(
            model_type=data["model_type"],
            confidence_threshold=data["confidence_threshold"],
        )
        model.top_model = data["top_model"]
        model.bottom_model = data["bottom_model"]
        model.scaler = data.get("scaler", StandardScaler())  # Backward compat
        model.feature_names = data["feature_names"]
        model.is_fitted = data["is_fitted"]

        return model
