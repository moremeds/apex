"""
Reality modeling layer for backtesting.

This module provides realistic simulation of:
- Transaction costs (commissions, fees)
- Slippage (spread crossing, market impact)
- Fill behavior (partial fills, queue position)
- Latency (order/fill delays)

Models can be composed into RealityModelPack for use with SimulatedExecution.
"""

from .admin_fee_model import (
    AdminFeeModel,
    AdminFeeResult,
    ConstantAdminFeeModel,
    ZeroAdminFeeModel,
)
from .fee_model import (
    AssetType,
    ConstantFeeModel,
    FeeBreakdown,
    FeeModel,
    FutuFeeModel,
    IBFeeModel,
    PerShareFeeModel,
    ZeroFeeModel,
)
from .fill_model import (
    FillModel,
    FillResult,
    ImmediateFillModel,
    NextBarFillModel,
    OrderType,
    ProbabilisticFillModel,
)
from .latency_model import (
    ConstantLatencyModel,
    LatencyModel,
    LatencyResult,
    RandomLatencyModel,
    VenueLatencyModel,
    ZeroLatencyModel,
)
from .reality_pack import (
    PRESET_PACKS,
    RealityModelPack,
    create_conservative_pack,
    create_futu_pack,
    create_ib_pack,
    create_simple_pack,
    create_zero_cost_pack,
    get_preset_pack,
)
from .slippage_model import (
    ConstantSlippageModel,
    SlippageModel,
    SlippageResult,
    SpreadSlippageModel,
    VolumeSlippageModel,
    ZeroSlippageModel,
)

__all__ = [
    # Fee models
    "FeeModel",
    "FeeBreakdown",
    "AssetType",
    "ZeroFeeModel",
    "ConstantFeeModel",
    "PerShareFeeModel",
    "IBFeeModel",
    "FutuFeeModel",
    # Slippage models
    "SlippageModel",
    "SlippageResult",
    "ZeroSlippageModel",
    "ConstantSlippageModel",
    "SpreadSlippageModel",
    "VolumeSlippageModel",
    # Fill models
    "FillModel",
    "FillResult",
    "OrderType",
    "ImmediateFillModel",
    "NextBarFillModel",
    "ProbabilisticFillModel",
    # Latency models
    "LatencyModel",
    "LatencyResult",
    "ZeroLatencyModel",
    "ConstantLatencyModel",
    "RandomLatencyModel",
    "VenueLatencyModel",
    # Admin fee models
    "AdminFeeModel",
    "AdminFeeResult",
    "ZeroAdminFeeModel",
    "ConstantAdminFeeModel",
    # Pack
    "RealityModelPack",
    "create_zero_cost_pack",
    "create_simple_pack",
    "create_ib_pack",
    "create_futu_pack",
    "create_conservative_pack",
    "get_preset_pack",
    "PRESET_PACKS",
]
