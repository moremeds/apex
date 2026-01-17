"""Repository implementations for persistence layer."""

from src.infrastructure.persistence.repositories.backtest_repository import (
    Backtest,
    BacktestRepository,
)
from src.infrastructure.persistence.repositories.base import BaseRepository
from src.infrastructure.persistence.repositories.futu_deal_repository import (
    FutuDealRepository,
    FutuRawDeal,
)
from src.infrastructure.persistence.repositories.futu_fee_repository import (
    FutuFeeRepository,
    FutuRawFee,
)
from src.infrastructure.persistence.repositories.futu_order_repository import (
    FutuOrderRepository,
    FutuRawOrder,
)
from src.infrastructure.persistence.repositories.ib_commission_repository import (
    IbCommissionRepository,
    IbRawCommission,
)
from src.infrastructure.persistence.repositories.ib_execution_repository import (
    IbExecutionRepository,
    IbRawExecution,
)
from src.infrastructure.persistence.repositories.signal_repository import (
    RiskSignal,
    RiskSignalRepository,
    TradeSignal,
    TradeSignalRepository,
)
from src.infrastructure.persistence.repositories.snapshot_repositories import (
    AccountSnapshot,
    AccountSnapshotRepository,
    PositionSnapshot,
    PositionSnapshotRepository,
    RiskSnapshotRecord,
    RiskSnapshotRepository,
)
from src.infrastructure.persistence.repositories.sync_state_repository import (
    SyncState,
    SyncStateRepository,
)
from src.infrastructure.persistence.repositories.ta_signal_repository import (
    ConfluenceScoreEntity,
    IndicatorValueEntity,
    TASignalEntity,
    TASignalRepository,
)

__all__ = [
    "BaseRepository",
    "FutuOrderRepository",
    "FutuRawOrder",
    "FutuDealRepository",
    "FutuRawDeal",
    "FutuFeeRepository",
    "FutuRawFee",
    "IbExecutionRepository",
    "IbRawExecution",
    "IbCommissionRepository",
    "IbRawCommission",
    "SyncStateRepository",
    "SyncState",
    "RiskSignalRepository",
    "RiskSignal",
    "TradeSignalRepository",
    "TradeSignal",
    "BacktestRepository",
    "Backtest",
    "PositionSnapshotRepository",
    "PositionSnapshot",
    "AccountSnapshotRepository",
    "AccountSnapshot",
    "RiskSnapshotRepository",
    "RiskSnapshotRecord",
    # TA Signal persistence
    "TASignalRepository",
    "TASignalEntity",
    "IndicatorValueEntity",
    "ConfluenceScoreEntity",
]
