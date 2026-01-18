"""
Model Registry Port for ML model artifact storage.

This port defines the contract for persisting and retrieving trained models.
Domain/application services depend on this abstract port, not on concrete
implementations (file system, S3, MLflow, etc.).

The registry supports versioned models with candidate → active promotion:
1. Train model → save as candidate
2. Evaluate candidate vs baseline
3. Promote to active if better (or force_update)

Implementations:
- FileModelRegistry (local filesystem) - development/CI
- S3ModelRegistry - production (future)
- MLflowModelRegistry - experiment tracking (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@dataclass
class ModelMetadata:
    """
    Metadata for a trained model version.

    Captures training context for reproducibility and auditing.
    """

    symbol: str
    model_type: str  # "logistic", "lightgbm"
    trained_at: datetime
    dataset_start: datetime
    dataset_end: datetime
    dataset_hash: str  # Hash of training data for reproducibility
    feature_version: str  # Version of feature extraction code

    # Performance metrics
    roc_auc: float
    pr_auc: float
    brier_score: float

    # Training config
    cv_splits: int
    label_horizon: int
    embargo: int

    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Additional context
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "trained_at": self.trained_at.isoformat(),
            "dataset_start": self.dataset_start.isoformat(),
            "dataset_end": self.dataset_end.isoformat(),
            "dataset_hash": self.dataset_hash,
            "feature_version": self.feature_version,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "cv_splits": self.cv_splits,
            "label_horizon": self.label_horizon,
            "embargo": self.embargo,
            "feature_importance": self.feature_importance,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Deserialize from dictionary."""
        return cls(
            symbol=data["symbol"],
            model_type=data["model_type"],
            trained_at=datetime.fromisoformat(data["trained_at"]),
            dataset_start=datetime.fromisoformat(data["dataset_start"]),
            dataset_end=datetime.fromisoformat(data["dataset_end"]),
            dataset_hash=data["dataset_hash"],
            feature_version=data["feature_version"],
            roc_auc=data["roc_auc"],
            pr_auc=data["pr_auc"],
            brier_score=data["brier_score"],
            cv_splits=data["cv_splits"],
            label_horizon=data["label_horizon"],
            embargo=data["embargo"],
            feature_importance=data.get("feature_importance", {}),
            notes=data.get("notes", ""),
        )


@dataclass
class ModelVersion:
    """Information about a model version."""

    version: str  # "active", "candidate", or timestamp-based version
    metadata: ModelMetadata
    path: str  # Path/URI to model artifact
    is_active: bool


@runtime_checkable
class ModelRegistryPort(Protocol):
    """
    Port for model artifact storage and versioning.

    Domain services interact with this interface, not concrete implementations.
    This enables testing with fake registries and swapping storage backends.

    Version semantics:
    - "active": Currently deployed model
    - "candidate": Newly trained model pending evaluation
    - Timestamp versions: Historical models for rollback

    Usage:
        # In training service (application layer)
        class TrainingService:
            def __init__(self, registry: ModelRegistryPort):
                self._registry = registry

            async def train_and_save(self, symbol: str, model, metadata):
                version = await self._registry.save_candidate(symbol, model, metadata)
                # ... evaluate and potentially promote
    """

    async def load_model(
        self,
        symbol: str,
        version: str = "active",
    ) -> Optional[Any]:
        """
        Load a model artifact.

        Args:
            symbol: Trading symbol (e.g., "SPY").
            version: Model version ("active", "candidate", or specific version).

        Returns:
            Loaded model object, or None if not found.
        """
        ...

    async def load_metadata(
        self,
        symbol: str,
        version: str = "active",
    ) -> Optional[ModelMetadata]:
        """
        Load model metadata without loading the model itself.

        Useful for comparing metrics before loading heavy model artifacts.

        Args:
            symbol: Trading symbol.
            version: Model version.

        Returns:
            ModelMetadata or None if not found.
        """
        ...

    async def save_candidate(
        self,
        symbol: str,
        model: Any,
        metadata: ModelMetadata,
    ) -> str:
        """
        Save a newly trained model as a candidate.

        The model is saved in candidate namespace, not yet active.
        Use promote_to_active() after evaluation to make it live.

        Args:
            symbol: Trading symbol.
            model: Trained model object (must be picklable).
            metadata: Model metadata for tracking.

        Returns:
            Version string for the saved candidate.
        """
        ...

    async def promote_to_active(
        self,
        symbol: str,
        candidate_version: str = "candidate",
    ) -> None:
        """
        Promote a candidate model to active.

        The current active model is archived with a timestamp version.

        Args:
            symbol: Trading symbol.
            candidate_version: Version to promote (default: "candidate").

        Raises:
            ValueError: If candidate version doesn't exist.
        """
        ...

    async def list_versions(
        self,
        symbol: str,
    ) -> List[ModelVersion]:
        """
        List all available versions for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            List of ModelVersion objects, sorted by date (newest first).
        """
        ...

    async def delete_version(
        self,
        symbol: str,
        version: str,
    ) -> bool:
        """
        Delete a specific model version.

        Cannot delete "active" version (must promote another first).

        Args:
            symbol: Trading symbol.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If trying to delete active version.
        """
        ...


class ModelRegistryPortABC(ABC):
    """
    Abstract base class version of ModelRegistryPort.

    Use this for inheritance-based implementations.
    The Protocol version above is for structural typing.
    """

    @abstractmethod
    async def load_model(
        self,
        symbol: str,
        version: str = "active",
    ) -> Optional[Any]:
        """Load a model artifact."""

    @abstractmethod
    async def load_metadata(
        self,
        symbol: str,
        version: str = "active",
    ) -> Optional[ModelMetadata]:
        """Load model metadata."""

    @abstractmethod
    async def save_candidate(
        self,
        symbol: str,
        model: Any,
        metadata: ModelMetadata,
    ) -> str:
        """Save a newly trained model as candidate."""

    @abstractmethod
    async def promote_to_active(
        self,
        symbol: str,
        candidate_version: str = "candidate",
    ) -> None:
        """Promote candidate to active."""

    @abstractmethod
    async def list_versions(
        self,
        symbol: str,
    ) -> List[ModelVersion]:
        """List all versions."""

    @abstractmethod
    async def delete_version(
        self,
        symbol: str,
        version: str,
    ) -> bool:
        """Delete a version."""
