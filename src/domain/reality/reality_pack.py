"""
Reality model pack - composition of all reality models.

Provides pre-configured packs for different simulation scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .fee_model import (
    FeeModel,
    ZeroFeeModel,
    PerShareFeeModel,
    IBFeeModel,
    FutuFeeModel,
)
from .slippage_model import (
    SlippageModel,
    ZeroSlippageModel,
    ConstantSlippageModel,
    SpreadSlippageModel,
    VolumeSlippageModel,
)
from .fill_model import (
    FillModel,
    ImmediateFillModel,
    NextBarFillModel,
    ProbabilisticFillModel,
)
from .latency_model import (
    LatencyModel,
    ZeroLatencyModel,
    ConstantLatencyModel,
    RandomLatencyModel,
    VenueLatencyModel,
)


@dataclass
class RealityModelPack:
    """
    Composition of all reality models for backtesting.

    A RealityModelPack contains:
    - FeeModel: Transaction costs
    - SlippageModel: Execution price impact
    - FillModel: Fill behavior
    - LatencyModel: Execution delays

    Use factory functions for pre-configured packs.
    """

    fee_model: FeeModel
    slippage_model: SlippageModel
    fill_model: FillModel
    latency_model: LatencyModel

    # Metadata
    name: str = "custom"
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pack configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "fee_model": type(self.fee_model).__name__,
            "slippage_model": type(self.slippage_model).__name__,
            "fill_model": type(self.fill_model).__name__,
            "latency_model": type(self.latency_model).__name__,
            "config": self.config,
        }


def create_zero_cost_pack() -> RealityModelPack:
    """
    Create zero-cost reality pack.

    No fees, no slippage, instant fills.
    Use for unit tests and quick validation.
    """
    return RealityModelPack(
        fee_model=ZeroFeeModel(),
        slippage_model=ZeroSlippageModel(),
        fill_model=ImmediateFillModel(),
        latency_model=ZeroLatencyModel(),
        name="zero_cost",
        description="Zero costs - for testing and quick validation",
    )


def create_simple_pack(
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
) -> RealityModelPack:
    """
    Create simple reality pack with basic costs.

    Constant fees and slippage, immediate fills.

    Args:
        commission_per_share: Commission per share.
        slippage_bps: Slippage in basis points.
    """
    return RealityModelPack(
        fee_model=PerShareFeeModel(per_share=commission_per_share),
        slippage_model=ConstantSlippageModel(slippage_bps=slippage_bps),
        fill_model=ImmediateFillModel(),
        latency_model=ZeroLatencyModel(),
        name="simple",
        description="Simple costs with constant fees and slippage",
        config={
            "commission_per_share": commission_per_share,
            "slippage_bps": slippage_bps,
        },
    )


def create_ib_pack(
    use_spread_slippage: bool = True,
    use_probabilistic_fills: bool = False,
    use_venue_latency: bool = False,
) -> RealityModelPack:
    """
    Create Interactive Brokers reality pack.

    Realistic IB fee structure with configurable slippage/fill models.

    Args:
        use_spread_slippage: Use spread-based slippage instead of constant.
        use_probabilistic_fills: Use probabilistic fill model.
        use_venue_latency: Use venue-based latency model.
    """
    # Slippage model
    if use_spread_slippage:
        slippage = SpreadSlippageModel(additional_bps=2.0)
    else:
        slippage = ConstantSlippageModel(slippage_bps=5.0)

    # Fill model
    if use_probabilistic_fills:
        fill = ProbabilisticFillModel()
    else:
        fill = ImmediateFillModel()

    # Latency model
    if use_venue_latency:
        latency = VenueLatencyModel()
    else:
        latency = ConstantLatencyModel(order_latency_ms=30, fill_latency_ms=80)

    return RealityModelPack(
        fee_model=IBFeeModel(),
        slippage_model=slippage,
        fill_model=fill,
        latency_model=latency,
        name="interactive_brokers",
        description="Interactive Brokers fee structure with realistic execution",
        config={
            "use_spread_slippage": use_spread_slippage,
            "use_probabilistic_fills": use_probabilistic_fills,
            "use_venue_latency": use_venue_latency,
        },
    )


def create_futu_pack(
    market: str = "US",
    use_spread_slippage: bool = True,
) -> RealityModelPack:
    """
    Create Futu/Moomoo reality pack.

    Realistic Futu fee structure for US or HK markets.

    Args:
        market: "US" or "HK".
        use_spread_slippage: Use spread-based slippage.
    """
    if use_spread_slippage:
        slippage = SpreadSlippageModel(additional_bps=3.0)
    else:
        slippage = ConstantSlippageModel(slippage_bps=8.0)

    # Futu has higher latency for HK market
    if market == "HK":
        latency = ConstantLatencyModel(order_latency_ms=80, fill_latency_ms=200)
    else:
        latency = ConstantLatencyModel(order_latency_ms=40, fill_latency_ms=100)

    return RealityModelPack(
        fee_model=FutuFeeModel(default_market=market),
        slippage_model=slippage,
        fill_model=ImmediateFillModel(),
        latency_model=latency,
        name=f"futu_{market.lower()}",
        description=f"Futu/Moomoo fee structure for {market} market",
        config={
            "market": market,
            "use_spread_slippage": use_spread_slippage,
        },
    )


def create_conservative_pack() -> RealityModelPack:
    """
    Create conservative reality pack.

    Higher costs for pessimistic backtesting.
    Use for stress testing strategy profitability.
    """
    return RealityModelPack(
        fee_model=IBFeeModel(
            stock_per_share=0.01,
            option_per_contract=1.00,
        ),
        slippage_model=VolumeSlippageModel(
            base_bps=10.0,
            impact_factor=0.2,
        ),
        fill_model=ProbabilisticFillModel(
            market_fill_prob=0.95,
            partial_fill_prob=0.3,
        ),
        latency_model=RandomLatencyModel(
            order_latency_min_ms=50,
            order_latency_max_ms=200,
            fill_latency_min_ms=100,
            fill_latency_max_ms=500,
        ),
        name="conservative",
        description="Conservative costs for stress testing",
    )


# Registry of preset packs
PRESET_PACKS = {
    "zero_cost": create_zero_cost_pack,
    "simple": create_simple_pack,
    "interactive_brokers": create_ib_pack,
    "ib": create_ib_pack,
    "futu_us": lambda: create_futu_pack("US"),
    "futu_hk": lambda: create_futu_pack("HK"),
    "conservative": create_conservative_pack,
}


def get_preset_pack(name: str, **kwargs) -> RealityModelPack:
    """
    Get a preset reality pack by name.

    Args:
        name: Pack name (zero_cost, simple, ib, futu_us, futu_hk, conservative).
        **kwargs: Additional arguments passed to pack factory.

    Returns:
        RealityModelPack instance.

    Raises:
        ValueError: If preset name not found.
    """
    if name not in PRESET_PACKS:
        available = ", ".join(PRESET_PACKS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")

    factory = PRESET_PACKS[name]
    if kwargs:
        return factory(**kwargs)
    return factory()
