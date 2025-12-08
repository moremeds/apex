"""
Persistent layer for historical order/trade data.

Provides PostgreSQL-based storage for:
- Raw API payloads from Futu and IB brokers
- Normalized order/trade/fee records
- Strategy classification
- Position snapshots

Usage:
    from src.infrastructure.persistent import PostgresStore, PersistenceOrchestrator

    # Standalone backfill
    orchestrator = PersistenceOrchestrator(config, futu_adapter, ib_adapter)
    await orchestrator.run(full_reload=True)

    # Or use the CLI:
    python scripts/backfill.py --full-reload
"""

from .store import PostgresStore
from .orchestrator import PersistenceOrchestrator, run_backfill, load_config
from .reconciler import Reconciler, run_reconciliation, Anomaly, AnomalyType
from .normalizers import BaseNormalizer, FutuNormalizer, IbNormalizer
from .classify import StrategyClassifier, StrategyType, StrategyResult

__all__ = [
    # Core
    "PostgresStore",
    "PersistenceOrchestrator",
    "run_backfill",
    "load_config",
    # Reconciliation
    "Reconciler",
    "run_reconciliation",
    "Anomaly",
    "AnomalyType",
    # Normalizers
    "BaseNormalizer",
    "FutuNormalizer",
    "IbNormalizer",
    # Classification
    "StrategyClassifier",
    "StrategyType",
    "StrategyResult",
]
