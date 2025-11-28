"""Configuration data models."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class IbConfig:
    """Interactive Brokers configuration."""
    enabled: bool
    host: str
    port: int
    client_id: int
    provides_market_data: bool = True


@dataclass
class FutuConfig:
    """Futu OpenD configuration."""
    enabled: bool
    host: str
    port: int
    security_firm: str
    trd_env: str
    filter_trdmarket: str
    provides_market_data: bool = False


@dataclass
class ManualPositionsConfig:
    """Manual positions file configuration."""
    file: str
    reload_interval_sec: int


@dataclass
class RiskLimitsConfig:
    """Risk limits configuration."""
    max_total_gross_notional: float
    max_per_underlying_notional: Dict[str, float]
    portfolio_delta_range: List[float]
    portfolio_vega_range: List[float]
    portfolio_theta_range: List[float]
    max_margin_utilization: float
    max_concentration_pct: float
    soft_breach_threshold: float


@dataclass
class ScenariosConfig:
    """Scenarios configuration."""
    enabled: bool
    spot_shocks: List[float]
    iv_shocks: List[float]
    combined: List[Dict[str, Any]]


@dataclass
class MdqcConfig:
    """Market Data Quality Control configuration."""
    stale_seconds: int
    ignore_zero_quotes: bool
    enforce_bid_ask_sanity: bool
    max_abs_delta: float
    fallback_order: List[str]


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    refresh_interval_sec: int
    show_positions: bool


@dataclass
class WatchdogConfig:
    """Watchdog configuration."""
    snapshot_stale_sec: int
    max_missing_md_ratio: float
    reconnect_backoff_sec: Dict[str, Any]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    json: bool
    file: str
    rotation: str  # "size" or "time"
    max_bytes: int  # For size-based rotation (e.g., 10MB = 10 * 1024 * 1024)
    backup_count: int  # Number of backup files to keep
    when: str  # For time-based rotation: "midnight", "H" (hourly), "D" (daily), "W0" (Monday)
    interval: int  # Interval for time-based rotation
    timezone: str  # Timezone for log timestamps (e.g., "America/New_York", "UTC", or "local")


@dataclass
class AppConfig:
    """Complete application configuration."""
    ibkr: IbConfig
    futu: FutuConfig
    manual_positions: ManualPositionsConfig
    risk_limits: RiskLimitsConfig
    scenarios: ScenariosConfig
    mdqc: MdqcConfig
    dashboard: DashboardConfig
    watchdog: WatchdogConfig
    logging: LoggingConfig
    raw: Dict[str, Any]  # Raw config dict for backward compatibility
