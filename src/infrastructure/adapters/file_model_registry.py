"""
File-based Model Registry Adapter.

Infrastructure implementation of ModelRegistryPort using local filesystem.
Stores models as pickle files with JSON metadata sidecar files.

Directory structure:
    models/turning_point/
        spy/
            active.pkl          # Current active model
            active.meta.json    # Active model metadata
            candidate.pkl       # Candidate awaiting evaluation
            candidate.meta.json # Candidate metadata
            2024-01-15T10-30-00.pkl      # Archived version
            2024-01-15T10-30-00.meta.json

Usage:
    registry = FileModelRegistry(Path("models/turning_point"))

    # Save candidate
    await registry.save_candidate("SPY", model, metadata)

    # Promote to active
    await registry.promote_to_active("SPY")

    # Load active model
    model = await registry.load_model("SPY", "active")
"""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.interfaces.model_registry import (
    ModelMetadata,
    ModelRegistryPortABC,
    ModelVersion,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class FileModelRegistry(ModelRegistryPortABC):
    """
    File-based implementation of ModelRegistryPort.

    Stores models as pickle files with JSON metadata sidecars.
    Thread-safe for single-writer scenarios (typical training use case).
    """

    def __init__(self, base_dir: Path) -> None:
        """
        Initialize file registry.

        Args:
            base_dir: Base directory for model storage.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _symbol_dir(self, symbol: str) -> Path:
        """Get directory for a symbol's models."""
        return self._base_dir / symbol.lower()

    def _model_path(self, symbol: str, version: str) -> Path:
        """Get path to model pickle file."""
        return self._symbol_dir(symbol) / f"{version}.pkl"

    def _meta_path(self, symbol: str, version: str) -> Path:
        """Get path to metadata JSON file."""
        return self._symbol_dir(symbol) / f"{version}.meta.json"

    async def load_model(
        self,
        symbol: str,
        version: str = "active",
    ) -> Optional[Any]:
        """
        Load a model artifact from disk.

        Args:
            symbol: Trading symbol.
            version: Model version.

        Returns:
            Loaded model object or None if not found.
        """
        model_path = self._model_path(symbol, version)

        if not model_path.exists():
            logger.debug(f"Model not found: {model_path}")
            return None

        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None

    async def load_metadata(
        self,
        symbol: str,
        version: str = "active",
    ) -> Optional[ModelMetadata]:
        """
        Load model metadata from disk.

        Args:
            symbol: Trading symbol.
            version: Model version.

        Returns:
            ModelMetadata or None if not found.
        """
        meta_path = self._meta_path(symbol, version)

        if not meta_path.exists():
            logger.debug(f"Metadata not found: {meta_path}")
            return None

        try:
            with open(meta_path) as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load metadata {meta_path}: {e}")
            return None

    async def save_candidate(
        self,
        symbol: str,
        model: Any,
        metadata: ModelMetadata,
    ) -> str:
        """
        Save a model as candidate.

        Args:
            symbol: Trading symbol.
            model: Model object (must be picklable).
            metadata: Model metadata.

        Returns:
            Version string ("candidate").
        """
        symbol_dir = self._symbol_dir(symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)

        model_path = self._model_path(symbol, "candidate")
        meta_path = self._meta_path(symbol, "candidate")

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Saved candidate model for {symbol} to {model_path}")
        return "candidate"

    async def promote_to_active(
        self,
        symbol: str,
        candidate_version: str = "candidate",
    ) -> None:
        """
        Promote candidate to active.

        Steps:
        1. Archive current active (if exists) with timestamp
        2. Move candidate to active

        Args:
            symbol: Trading symbol.
            candidate_version: Version to promote.

        Raises:
            ValueError: If candidate doesn't exist.
        """
        candidate_model = self._model_path(symbol, candidate_version)
        candidate_meta = self._meta_path(symbol, candidate_version)

        if not candidate_model.exists():
            raise ValueError(f"Candidate model not found for {symbol}")

        active_model = self._model_path(symbol, "active")
        active_meta = self._meta_path(symbol, "active")

        # Archive current active if exists
        if active_model.exists():
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
            archive_model = self._model_path(symbol, timestamp)
            archive_meta = self._meta_path(symbol, timestamp)

            shutil.move(str(active_model), str(archive_model))
            if active_meta.exists():
                shutil.move(str(active_meta), str(archive_meta))

            logger.info(f"Archived previous active to {archive_model}")

        # Promote candidate to active
        shutil.copy2(str(candidate_model), str(active_model))
        if candidate_meta.exists():
            shutil.copy2(str(candidate_meta), str(active_meta))

        # Remove candidate files
        candidate_model.unlink()
        if candidate_meta.exists():
            candidate_meta.unlink()

        logger.info(f"Promoted candidate to active for {symbol}")

    async def list_versions(
        self,
        symbol: str,
    ) -> List[ModelVersion]:
        """
        List all versions for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            List of ModelVersion, newest first.
        """
        symbol_dir = self._symbol_dir(symbol)

        if not symbol_dir.exists():
            return []

        versions: List[ModelVersion] = []

        # Find all .pkl files
        for model_path in symbol_dir.glob("*.pkl"):
            version = model_path.stem
            meta_path = self._meta_path(symbol, version)

            metadata = None
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        metadata = ModelMetadata.from_dict(json.load(f))
                except Exception:
                    pass

            if metadata:
                versions.append(
                    ModelVersion(
                        version=version,
                        metadata=metadata,
                        path=str(model_path),
                        is_active=(version == "active"),
                    )
                )

        # Sort by trained_at (newest first), active always first
        def sort_key(v: ModelVersion) -> tuple:
            if v.version == "active":
                return (0, datetime.max)
            if v.version == "candidate":
                return (1, datetime.max)
            return (2, v.metadata.trained_at if v.metadata else datetime.min)

        versions.sort(key=sort_key, reverse=True)
        return versions

    async def delete_version(
        self,
        symbol: str,
        version: str,
    ) -> bool:
        """
        Delete a model version.

        Args:
            symbol: Trading symbol.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ValueError: If trying to delete active.
        """
        if version == "active":
            raise ValueError("Cannot delete active version")

        model_path = self._model_path(symbol, version)
        meta_path = self._meta_path(symbol, version)

        if not model_path.exists():
            return False

        model_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        logger.info(f"Deleted version {version} for {symbol}")
        return True
