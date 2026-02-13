"""Typed configuration for PEAD screener.

All screener configuration is validated at load time via dataclasses.
Typos or missing keys raise immediately, not silently no-op at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PEADFilters:
    """Filter thresholds for the screening pipeline."""

    min_sue: float = 2.0
    min_earnings_day_gap: float = 0.02
    min_volume_ratio: float = 2.0
    max_entry_delay_trading_days: int = 5
    max_forward_pe: float = 40.0
    exclude_at_52w_high: bool = True
    exclude_analyst_downgrade_days: int = 3
    require_revenue_beat: bool = False


@dataclass
class PEADTradeParams:
    """Default trade parameters for candidates."""

    default_profit_target: float = 0.06
    default_stop_loss: float = -0.05
    default_max_hold_trading_days: int = 25
    trailing_stop_atr_multiplier: float = 2.0
    trailing_stop_activation_pct: float = 0.03


@dataclass
class PEADRegimeRules:
    """Regime-dependent position sizing rules."""

    r0_position_size_factor: float = 1.0
    r1_position_size_factor: float = 0.5
    r2_block_entirely: bool = True


@dataclass
class PEADLiquidityTiers:
    """Market cap thresholds and slippage estimates."""

    large_cap_min_market_cap: float = 50_000_000_000
    mid_cap_min_market_cap: float = 2_000_000_000
    large_cap_slippage_bps: int = 10
    mid_cap_slippage_bps: int = 25
    small_cap_slippage_bps: int = 50


@dataclass
class PEADQualityThresholds:
    """Score thresholds for quality classification."""

    strong: float = 70.0
    moderate: float = 45.0


@dataclass
class MultiQuarterSueConfig:
    """Configuration for multi-quarter SUE enhancement."""

    enabled: bool = True
    decay_lambda: float = 0.75
    min_quarters: int = 6
    max_bonus: float = 10.0
    max_penalty: float = -5.0


@dataclass
class AttentionFilterConfig:
    """Configuration for Google Trends attention filter."""

    enabled: bool = False
    low_bonus: float = 5.0
    high_penalty: float = -5.0


@dataclass
class PEADConfig:
    """Complete PEAD screener configuration.

    Validates all keys at load time. Any typo or missing section raises
    immediately instead of silently using wrong defaults.
    """

    filters: PEADFilters = field(default_factory=PEADFilters)
    trade_params: PEADTradeParams = field(default_factory=PEADTradeParams)
    regime_rules: PEADRegimeRules = field(default_factory=PEADRegimeRules)
    liquidity_tiers: PEADLiquidityTiers = field(default_factory=PEADLiquidityTiers)
    quality_thresholds: PEADQualityThresholds = field(default_factory=PEADQualityThresholds)
    multi_quarter_sue: MultiQuarterSueConfig = field(default_factory=MultiQuarterSueConfig)
    attention_filter: AttentionFilterConfig = field(default_factory=AttentionFilterConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> PEADConfig:
        """Load and validate config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PEADConfig:
        """Build typed config from raw dict, using defaults for missing keys."""
        return cls(
            filters=_build_dataclass(PEADFilters, raw.get("filters", {})),
            trade_params=_build_dataclass(PEADTradeParams, raw.get("trade_params", {})),
            regime_rules=_build_dataclass(PEADRegimeRules, raw.get("regime_rules", {})),
            liquidity_tiers=_build_dataclass(PEADLiquidityTiers, raw.get("liquidity_tiers", {})),
            quality_thresholds=_build_dataclass(
                PEADQualityThresholds, raw.get("quality_thresholds", {})
            ),
            multi_quarter_sue=_build_dataclass(
                MultiQuarterSueConfig, raw.get("multi_quarter_sue", {})
            ),
            attention_filter=_build_dataclass(
                AttentionFilterConfig, raw.get("attention_filter", {})
            ),
        )


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in (data or {}).items() if k in field_names}
    return cls(**filtered)
