"""Configuration data models."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class IbClientIdsConfig:
    """Reserved IB client IDs per adapter type."""

    execution: int = 1
    monitoring: int = 2
    historical_pool: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8, 9, 10])


@dataclass
class IbConfig:
    """Interactive Brokers configuration."""
    enabled: bool
    host: str
    port: int
    client_ids: IbClientIdsConfig = field(default_factory=IbClientIdsConfig)
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
class DatabasePoolConfig:
    """Database connection pool configuration."""
    min_connections: int = 2
    max_connections: int = 10


@dataclass
class TimescaleConfig:
    """TimescaleDB-specific configuration."""
    enabled: bool = True
    chunk_interval: str = "1 month"
    compression_enabled: bool = True
    compression_after: str = "7 days"


@dataclass
class DatabaseConfig:
    """Database connection configuration for PostgreSQL/TimescaleDB."""
    type: str = "timescaledb"
    host: str = "localhost"
    port: int = 5432
    database: str = "apex_risk"
    user: str = "apex"
    password: str = ""
    pool: DatabasePoolConfig = field(default_factory=DatabasePoolConfig)
    timescale: TimescaleConfig = field(default_factory=TimescaleConfig)

    @property
    def dsn(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class DisplayConfig:
    """Display timezone and formatting configuration for UI."""
    timezone: str = "Asia/Hong_Kong"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    datetime_format: str = "%Y-%m-%d %H:%M:%S %Z"


@dataclass
class SnapshotConfig:
    """Snapshot persistence configuration for warm-start and history."""
    position_interval_sec: int = 60
    account_interval_sec: int = 60
    risk_interval_sec: int = 60
    capture_on_shutdown: bool = True
    retention_days: int = 365
    compression_after_days: int = 7


@dataclass
class RateLimitConfig:
    """Rate limit configuration for API calls."""
    requests_per_window: int = 10
    window_seconds: int = 30


@dataclass
class HistoryLoaderConfig:
    """History loader configuration."""
    default_lookback_days: int = 30
    batch_size: int = 100
    futu_rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(10, 30))


@dataclass
class BarCacheConfig:
    """Bar cache service configuration."""

    host: str = "127.0.0.1"
    port: int = 9001
    timeout_sec: int = 30
    max_cache_entries: int = 512


@dataclass
class HistoricalDataConfig:
    """Historical data services configuration."""

    bar_cache: BarCacheConfig = field(default_factory=BarCacheConfig)


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

    # Persistence layer configs (optional - have defaults)
    database: Optional[DatabaseConfig] = None
    display: Optional[DisplayConfig] = None
    snapshots: Optional[SnapshotConfig] = None
    history_loader: Optional[HistoryLoaderConfig] = None
    historical_data: Optional[HistoricalDataConfig] = None

    def __post_init__(self) -> None:
        """Initialize optional configs with defaults if not provided."""
        if self.database is None:
            self.database = DatabaseConfig()
        if self.display is None:
            self.display = DisplayConfig()
        if self.snapshots is None:
            self.snapshots = SnapshotConfig()
        if self.history_loader is None:
            self.history_loader = HistoryLoaderConfig()
        if self.historical_data is None:
            self.historical_data = HistoricalDataConfig()
