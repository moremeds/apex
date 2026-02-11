from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import pytest

from src.application.services.turning_point_training_service import TurningPointTrainingService
from src.domain.interfaces.model_registry import ModelMetadata


class _FakeModelRegistry:
    def __init__(self, metadata: Optional[ModelMetadata]) -> None:
        self._metadata = metadata

    async def load_model(self, symbol: str, version: str = "active") -> Optional[Any]:
        return None

    async def load_metadata(self, symbol: str, version: str = "active") -> Optional[ModelMetadata]:
        return self._metadata

    async def save_candidate(self, symbol: str, model: Any, metadata: ModelMetadata) -> str:
        return "candidate"

    async def promote_to_active(self, symbol: str, candidate_version: str = "candidate") -> None:
        return None

    async def list_versions(self, symbol: str) -> list[Any]:
        return []

    async def delete_version(self, symbol: str, version: str) -> bool:
        return False


def _metadata(dataset_hash: str, feature_version: str) -> ModelMetadata:
    now = datetime.utcnow()
    return ModelMetadata(
        symbol="SPY",
        model_type="logistic",
        trained_at=now,
        dataset_start=now - timedelta(days=100),
        dataset_end=now,
        dataset_hash=dataset_hash,
        feature_version=feature_version,
        roc_auc=0.6,
        pr_auc=0.2,
        brier_score=0.3,
        cv_splits=5,
        label_horizon=10,
        embargo=2,
        feature_importance={},
    )


@pytest.mark.asyncio
async def test_skip_reason_when_dataset_and_code_signature_unchanged() -> None:
    service = TurningPointTrainingService(
        model_registry=_FakeModelRegistry(_metadata("abc123", "tp-v1"))
    )

    reason = await service._get_skip_reason(
        symbol="SPY",
        dataset_hash="abc123",
        training_code_signature="tp-v1",
        force_update=False,
    )

    assert reason is not None
    assert "unchanged" in reason


@pytest.mark.asyncio
async def test_skip_reason_none_when_code_signature_changed() -> None:
    service = TurningPointTrainingService(
        model_registry=_FakeModelRegistry(_metadata("abc123", "tp-v1"))
    )

    reason = await service._get_skip_reason(
        symbol="SPY",
        dataset_hash="abc123",
        training_code_signature="tp-v2",
        force_update=False,
    )

    assert reason is None


@pytest.mark.asyncio
async def test_skip_reason_none_when_force_update_enabled() -> None:
    service = TurningPointTrainingService(
        model_registry=_FakeModelRegistry(_metadata("abc123", "tp-v1"))
    )

    reason = await service._get_skip_reason(
        symbol="SPY",
        dataset_hash="abc123",
        training_code_signature="tp-v1",
        force_update=True,
    )

    assert reason is None
