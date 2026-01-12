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
    ConstantFeeModel,
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
    RandomSlippageModel,
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
from .admin_fee_model import (
    AdminFeeModel,
    ZeroAdminFeeModel,
    ConstantAdminFeeModel,
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
    - AdminFeeModel: Time-based administrative costs

    Use factory functions for pre-configured packs.
    """

    fee_model: FeeModel
    slippage_model: SlippageModel
    fill_model: FillModel
    latency_model: LatencyModel
    admin_fee_model: AdminFeeModel = field(default_factory=ZeroAdminFeeModel)

    # Metadata
    name: str = "custom"
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> RealityModelPack:
        """
        Create a reality pack from a configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - fee_model: dict with 'type' and 'params' (or flat params)
                - slippage_model: dict with 'type' and 'params' (or flat params)
                - fill_model: dict with 'type' and 'params' (or flat params)
                - latency_model: dict with 'type' and 'params' (or flat params)

        Returns:
            RealityModelPack instance.
        """
        # Helper to get params from config dict (either nested in 'params' or flat)
        def get_params(cfg):
            if not isinstance(cfg, dict):
                return {}
            params = dict(cfg.get("params", {}))
            # Include other keys as params too (flat structure)
            for k, v in cfg.items():
                if k not in ["type", "params"]:
                    params[k] = v
            return params

        # Fee model
        fee_cfg = config.get("fee_model", {"type": "zero"})
        fee_type = fee_cfg.get("type", "zero").lower()
        fee_params = get_params(fee_cfg)
        
        # Mapping aliases
        if fee_type == "fixed":
            fee_type = "per_share"
        elif fee_type == "none":
            fee_type = "zero"
            
        fee_map = {
            "zero": ZeroFeeModel,
            "constant": ConstantFeeModel,
            "per_share": PerShareFeeModel,
            "ib": IBFeeModel,
            "futu": FutuFeeModel,
        }
        
        # Handle param name differences (commission_per_share -> per_share)
        if "commission_per_share" in fee_params and "per_share" not in fee_params:
            fee_params["per_share"] = fee_params.pop("commission_per_share")
        if "min_commission" in fee_params and "minimum" not in fee_params:
            fee_params["minimum"] = fee_params.pop("min_commission")

        fee_model = fee_map.get(fee_type, ZeroFeeModel)(**fee_params)

        # Slippage model
        slip_cfg = config.get("slippage_model", {"type": "zero"})
        slip_type = slip_cfg.get("type", "zero").lower()
        slip_params = get_params(slip_cfg)
        
        slip_map = {
            "zero": ZeroSlippageModel,
            "constant": ConstantSlippageModel,
            "spread": SpreadSlippageModel,
            "volume": VolumeSlippageModel,
            "random": RandomSlippageModel,
        }
        slippage_model = slip_map.get(slip_type, ZeroSlippageModel)(**slip_params)

        # Fill model
        fill_cfg = config.get("fill_model", {"type": "immediate"})
        fill_type = fill_cfg.get("type", "immediate").lower()
        fill_params = get_params(fill_cfg)
        
        fill_map = {
            "immediate": ImmediateFillModel,
            "next_bar": NextBarFillModel,
            "probabilistic": ProbabilisticFillModel,
        }
        fill_model = fill_map.get(fill_type, ImmediateFillModel)(**fill_params)

        # Latency model
        lat_cfg = config.get("latency_model", {"type": "zero"})
        lat_type = lat_cfg.get("type", "zero").lower()
        lat_params = get_params(lat_cfg)
        
        lat_map = {
            "zero": ZeroLatencyModel,
            "constant": ConstantLatencyModel,
            "random": RandomLatencyModel,
            "venue": VenueLatencyModel,
        }
        latency_model = lat_map.get(lat_type, ZeroLatencyModel)(**lat_params)

        # Admin fee model
        adm_cfg = config.get("admin_fee_model", {"type": "zero"})
        adm_type = adm_cfg.get("type", "zero").lower()
        adm_params = get_params(adm_cfg)

        adm_map = {
            "zero": ZeroAdminFeeModel,
            "constant": ConstantAdminFeeModel,
        }
        admin_fee_model = adm_map.get(adm_type, ZeroAdminFeeModel)(**adm_params)

        return cls(
            fee_model=fee_model,
            slippage_model=slippage_model,
            fill_model=fill_model,
            latency_model=latency_model,
            admin_fee_model=admin_fee_model,
            name=config.get("name", "custom"),
            description=config.get("description", "Custom configured pack"),
            config=config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert pack configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "fee_model": type(self.fee_model).__name__,
            "slippage_model": type(self.slippage_model).__name__,
            "fill_model": type(self.fill_model).__name__,
            "latency_model": type(self.latency_model).__name__,
            "admin_fee_model": type(self.admin_fee_model).__name__,
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
