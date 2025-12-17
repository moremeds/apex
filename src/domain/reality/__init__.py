"""
Reality modeling layer for backtesting.

This module provides realistic simulation of:
- Transaction costs (commissions, fees)
- Slippage (spread crossing, market impact)
- Fill behavior (partial fills, queue position)
- Latency (order/fill delays)

Models can be composed into RealityModelPack for use with SimulatedExecution.
"""

from .fee_model import (
    FeeModel,
    FeeBreakdown,
    AssetType,
    ZeroFeeModel,
    ConstantFeeModel,
    PerShareFeeModel,
    IBFeeModel,
    FutuFeeModel,
)

from .slippage_model import (
    SlippageModel,
    SlippageResult,
    ZeroSlippageModel,
    ConstantSlippageModel,
    SpreadSlippageModel,
    VolumeSlippageModel,
)

from .fill_model import (
    FillModel,
    FillResult,
    OrderType,
    ImmediateFillModel,
    NextBarFillModel,
    ProbabilisticFillModel,
)

from .latency_model import (
    LatencyModel,
    LatencyResult,
    ZeroLatencyModel,
    ConstantLatencyModel,
    RandomLatencyModel,
    VenueLatencyModel,
)

from .reality_pack import (
    RealityModelPack,
    create_zero_cost_pack,
    create_simple_pack,
    create_ib_pack,
    create_futu_pack,
    create_conservative_pack,
    get_preset_pack,
    PRESET_PACKS,
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
