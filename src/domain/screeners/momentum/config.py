"""Typed configuration for momentum screener.

All screener configuration is validated at load time via dataclasses.
Typos or missing keys raise immediately, not silently no-op at runtime.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UniverseConfig:
    """Universe source configuration."""

    indices: list[str] = field(default_factory=lambda: ["sp500", "nasdaq"])
    russell_proxy_enabled: bool = True
    russell_proxy_market_cap_min: float = 300_000_000
    russell_proxy_market_cap_max: float = 10_000_000_000
    source: str = "fmp"
    cache_path: str = "data/cache/index_constituents.json"


@dataclass
class DataSourceConfig:
    """Price data source configuration."""

    price_source: str = "yfinance"
    lookback_trading_days: int = 252
    skip_recent_trading_days: int = 21


@dataclass
class MomentumFilters:
    """Filter thresholds for the screening pipeline."""

    min_market_cap: float = 500_000_000
    min_avg_daily_dollar_volume: float = 10_000_000
    min_price: float = 10.0
    min_daily_turnover_rate: float = 0.002


@dataclass
class ScoringConfig:
    """Scoring weights and top-N selection."""

    momentum_weight: float = 0.5
    fip_weight: float = 0.5
    top_n: int = 30


@dataclass
class MomentumRegimeRules:
    """Regime-dependent position sizing and thresholds."""

    r0_min_composite_percentile: float = 0.0
    r0_position_size_factor: float = 1.0
    r1_min_composite_percentile: float = 0.70
    r1_position_size_factor: float = 0.5
    r2_block_entirely: bool = True


@dataclass
class MomentumLiquidityTiers:
    """Market cap thresholds and slippage estimates."""

    large_cap_min: float = 50_000_000_000
    mid_cap_min: float = 2_000_000_000
    large_cap_slippage_bps: int = 10
    mid_cap_slippage_bps: int = 25
    small_cap_slippage_bps: int = 50


@dataclass
class MomentumQualityThresholds:
    """Score thresholds for quality classification."""

    strong: float = 0.80
    moderate: float = 0.60


@dataclass
class BacktestConfig:
    """Backtest-specific settings."""

    start_date: str = "2023-01-01"
    end_date: str | None = None
    benchmark: str = "SPY"
    use_historical_constituents: bool = True
    holding_period_days: int = 5
    portfolio_top_n: int = 20


@dataclass
class MomentumConfig:
    """Complete momentum screener configuration.

    Validates all keys at load time. Any typo or missing section raises
    immediately instead of silently using wrong defaults.
    """

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    filters: MomentumFilters = field(default_factory=MomentumFilters)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    regime_rules: MomentumRegimeRules = field(default_factory=MomentumRegimeRules)
    liquidity_tiers: MomentumLiquidityTiers = field(default_factory=MomentumLiquidityTiers)
    quality_thresholds: MomentumQualityThresholds = field(default_factory=MomentumQualityThresholds)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> MomentumConfig:
        """Load and validate config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MomentumConfig:
        """Build typed config from raw dict, using defaults for missing keys."""
        universe_raw = raw.get("universe", {})
        rp = universe_raw.get("russell_proxy", {})
        universe = _build_dataclass(
            UniverseConfig,
            {
                **{k: v for k, v in universe_raw.items() if k != "russell_proxy"},
                "russell_proxy_enabled": rp.get("enabled", True),
                "russell_proxy_market_cap_min": rp.get("market_cap_min", 300_000_000),
                "russell_proxy_market_cap_max": rp.get("market_cap_max", 10_000_000_000),
            },
        )

        return cls(
            universe=universe,
            data_source=_build_dataclass(DataSourceConfig, raw.get("data_source", {})),
            filters=_build_dataclass(MomentumFilters, raw.get("filters", {})),
            scoring=_build_dataclass(ScoringConfig, raw.get("scoring", {})),
            regime_rules=_build_dataclass(MomentumRegimeRules, raw.get("regime_rules", {})),
            liquidity_tiers=_build_dataclass(
                MomentumLiquidityTiers, raw.get("liquidity_tiers", {})
            ),
            quality_thresholds=_build_dataclass(
                MomentumQualityThresholds, raw.get("quality_thresholds", {})
            ),
            backtest=_build_dataclass(BacktestConfig, raw.get("backtest", {})),
        )


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from dict, ignoring unknown keys."""
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in (data or {}).items() if k in field_names}
    return cls(**filtered)
