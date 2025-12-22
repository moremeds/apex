"""
Configuration manager with environment-based loading.

Supports:
- Base configuration (base.yaml)
- Environment-specific overrides (dev.yaml, prod.yaml)
- Secrets loading (secrets.yaml - gitignored)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml
import logging

from .models import (
    AppConfig,
    IbConfig,
    IbClientIdsConfig,
    FutuConfig,
    ManualPositionsConfig,
    RiskLimitsConfig,
    ScenariosConfig,
    MdqcConfig,
    DashboardConfig,
    WatchdogConfig,
    LoggingConfig,
    DatabaseConfig,
    DatabasePoolConfig,
    TimescaleConfig,
    DisplayConfig,
    SnapshotConfig,
    HistoryLoaderConfig,
    RateLimitConfig,
    BarCacheConfig,
    HistoricalDataConfig,
)


logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager with environment support.

    Loads configuration in this order:
    1. base.yaml (default config)
    2. {env}.yaml (environment-specific, e.g., dev.yaml)
    3. secrets.yaml (if exists, gitignored)

    Later configs override earlier ones.
    """

    def __init__(self, config_dir: str | Path = "config", env: str = "dev"):
        """
        Initialize config manager.

        Args:
            config_dir: Directory containing config files.
            env: Environment name (dev, prod, etc).
        """
        self.config_dir = Path(config_dir)
        self.env = env
        self.config: Dict[str, Any] = {}

    def load(self) -> AppConfig:
        """
        Load configuration from YAML files.

        Returns:
            AppConfig object.

        Raises:
            FileNotFoundError: If base config not found.
            ValueError: If config is invalid.
        """
        # Load base config (required) - no fallback, use base.yaml directly
        base_path = self.config_dir / "base.yaml"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Base config not found: {base_path}. "
                "Rename your risk_config.yaml to base.yaml or create a new base.yaml."
            )

        self.config = self._load_yaml(base_path)
        logger.info(f"Loaded base config from {base_path}")

        # Load environment-specific config (optional)
        env_path = self.config_dir / f"{self.env}.yaml"
        if env_path.exists():
            env_config = self._load_yaml(env_path)
            self.config = self._merge_dicts(self.config, env_config)
            logger.info(f"Loaded {self.env} config from {env_path}")

        # Load secrets (optional, gitignored)
        secrets_path = self.config_dir / "secrets.yaml"
        if secrets_path.exists():
            secrets = self._load_yaml(secrets_path)
            self.config = self._merge_dicts(self.config, secrets)
            logger.info("Loaded secrets")

        return self._parse_config()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dicts (override wins)."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _parse_config(self) -> AppConfig:
        """Parse raw dict into AppConfig."""
        try:
            # Support both old 'ibkr' and new 'brokers.ibkr' config structures
            brokers_raw = self.config.get("brokers", {})
            ibkr_raw = brokers_raw.get("ibkr", self.config.get("ibkr", {}))
            futu_raw = brokers_raw.get("futu", self.config.get("futu", {}))

            client_ids_raw = ibkr_raw.get("client_ids", {})
            ibkr = IbConfig(
                enabled=ibkr_raw.get("enabled", True),
                host=ibkr_raw.get("host", "127.0.0.1"),
                port=ibkr_raw.get("port", 4001),
                client_ids=IbClientIdsConfig(
                    execution=client_ids_raw.get("execution", 1),
                    monitoring=client_ids_raw.get("monitoring", 2),
                    historical_pool=client_ids_raw.get(
                        "historical_pool",
                        [3, 4, 5, 6, 7, 8, 9, 10],
                    ),
                ),
                provides_market_data=ibkr_raw.get("provides_market_data", True),
            )

            futu = FutuConfig(
                enabled=futu_raw.get("enabled", False),
                host=futu_raw.get("host", "127.0.0.1"),
                port=futu_raw.get("port", 11111),
                security_firm=futu_raw.get("security_firm", "FUTUSECURITIES"),
                trd_env=futu_raw.get("trd_env", "REAL"),
                filter_trdmarket=futu_raw.get("filter_trdmarket", "US"),
                provides_market_data=futu_raw.get("provides_market_data", False),
            )

            manual_raw = self.config.get("manual_positions", {})
            manual_positions = ManualPositionsConfig(
                file=manual_raw.get("file", "./data/positions/manual.yaml"),
                reload_interval_sec=manual_raw.get("reload_interval_sec", 60),
            )

            risk_raw = self.config.get("risk_limits", {})
            risk_limits = RiskLimitsConfig(
                max_total_gross_notional=risk_raw.get("max_total_gross_notional", 5_000_000),
                max_per_underlying_notional=risk_raw.get("max_per_underlying_notional", {}),
                portfolio_delta_range=risk_raw.get("portfolio_delta_range", [-50_000, 50_000]),
                portfolio_vega_range=risk_raw.get("portfolio_vega_range", [-15_000, 15_000]),
                portfolio_theta_range=risk_raw.get("portfolio_theta_range", [-5_000, 5_000]),
                max_margin_utilization=risk_raw.get("max_margin_utilization", 0.60),
                max_concentration_pct=risk_raw.get("max_concentration_pct", 0.30),
                soft_breach_threshold=risk_raw.get("soft_breach_threshold", 0.80),
            )

            scenarios_raw = self.config.get("scenarios", {})
            scenarios = ScenariosConfig(
                enabled=scenarios_raw.get("enabled", True),
                spot_shocks=scenarios_raw.get("spot_shocks", []),
                iv_shocks=scenarios_raw.get("iv_shocks", []),
                combined=scenarios_raw.get("combined", []),
            )

            mdqc_raw = self.config.get("mdqc", {})
            mdqc = MdqcConfig(
                stale_seconds=mdqc_raw.get("stale_seconds", 10),
                ignore_zero_quotes=mdqc_raw.get("ignore_zero_quotes", True),
                enforce_bid_ask_sanity=mdqc_raw.get("enforce_bid_ask_sanity", True),
                max_abs_delta=mdqc_raw.get("max_abs_delta", 1.0),
                fallback_order=mdqc_raw.get("fallback_order", ["mid", "last"]),
            )

            dashboard_raw = self.config.get("dashboard", {})
            dashboard = DashboardConfig(
                refresh_interval_sec=dashboard_raw.get("refresh_interval_sec", 2),
                show_positions=dashboard_raw.get("show_positions", True),
            )

            watchdog_raw = self.config.get("watchdog", {})
            watchdog = WatchdogConfig(
                snapshot_stale_sec=watchdog_raw.get("snapshot_stale_sec", 10),
                max_missing_md_ratio=watchdog_raw.get("max_missing_md_ratio", 0.2),
                reconnect_backoff_sec=watchdog_raw.get("reconnect_backoff_sec", {}),
            )

            logging_raw = self.config.get("logging", {})
            logging_config = LoggingConfig(
                level=logging_raw.get("level", "INFO"),
                json=logging_raw.get("json", True),
                file=logging_raw.get("file", "./logs/live_risk.log"),
                rotation=logging_raw.get("rotation", "time"),
                max_bytes=logging_raw.get("max_bytes", 10 * 1024 * 1024),  # 10MB default
                backup_count=logging_raw.get("backup_count", 7),  # Keep 7 backups
                when=logging_raw.get("when", "midnight"),  # Rotate at midnight
                interval=logging_raw.get("interval", 1),  # Every 1 day
                timezone=logging_raw.get("timezone", "local"),  # Default to local time
            )

            # Persistence layer configs (optional - have defaults)
            db_raw = self.config.get("database", {})
            pool_raw = db_raw.get("pool", {})
            ts_raw = db_raw.get("timescale", {})
            database = DatabaseConfig(
                type=db_raw.get("type", "timescaledb"),
                host=db_raw.get("host", "localhost"),
                port=db_raw.get("port", 5432),
                database=db_raw.get("database", "apex_risk"),
                user=db_raw.get("user", "apex"),
                password=db_raw.get("password", ""),
                pool=DatabasePoolConfig(
                    min_connections=pool_raw.get("min_connections", 2),
                    max_connections=pool_raw.get("max_connections", 10),
                ),
                timescale=TimescaleConfig(
                    enabled=ts_raw.get("enabled", True),
                    chunk_interval=ts_raw.get("chunk_interval", "1 month"),
                    compression_enabled=ts_raw.get("compression_enabled", True),
                    compression_after=ts_raw.get("compression_after", "7 days"),
                ),
            ) if db_raw else None

            display_raw = self.config.get("display", {})
            display = DisplayConfig(
                timezone=display_raw.get("timezone", "Asia/Hong_Kong"),
                date_format=display_raw.get("date_format", "%Y-%m-%d"),
                time_format=display_raw.get("time_format", "%H:%M:%S"),
                datetime_format=display_raw.get("datetime_format", "%Y-%m-%d %H:%M:%S %Z"),
            ) if display_raw else None

            snapshots_raw = self.config.get("snapshots", {})
            snapshots = SnapshotConfig(
                position_interval_sec=snapshots_raw.get("position_interval_sec", 60),
                account_interval_sec=snapshots_raw.get("account_interval_sec", 60),
                risk_interval_sec=snapshots_raw.get("risk_interval_sec", 60),
                capture_on_shutdown=snapshots_raw.get("capture_on_shutdown", True),
                retention_days=snapshots_raw.get("retention_days", 365),
                compression_after_days=snapshots_raw.get("compression_after_days", 7),
            ) if snapshots_raw else None

            history_raw = self.config.get("history_loader", {})
            rate_limit_raw = history_raw.get("futu_rate_limit", {})
            history_loader = HistoryLoaderConfig(
                default_lookback_days=history_raw.get("default_lookback_days", 30),
                batch_size=history_raw.get("batch_size", 100),
                futu_rate_limit=RateLimitConfig(
                    requests_per_window=rate_limit_raw.get("requests_per_window", 10),
                    window_seconds=rate_limit_raw.get("window_seconds", 30),
                ),
            ) if history_raw else None

            historical_raw = self.config.get("historical_data", {})
            bar_cache_raw = historical_raw.get("bar_cache", {})
            historical_data = HistoricalDataConfig(
                bar_cache=BarCacheConfig(
                    host=bar_cache_raw.get("host", "127.0.0.1"),
                    port=bar_cache_raw.get("port", 9001),
                    timeout_sec=bar_cache_raw.get("timeout_sec", 30),
                    max_cache_entries=bar_cache_raw.get("max_cache_entries", 512),
                )
            ) if historical_raw else None

            return AppConfig(
                ibkr=ibkr,
                futu=futu,
                manual_positions=manual_positions,
                risk_limits=risk_limits,
                scenarios=scenarios,
                mdqc=mdqc,
                dashboard=dashboard,
                watchdog=watchdog,
                logging=logging_config,
                raw=self.config,
                database=database,
                display=display,
                snapshots=snapshots,
                history_loader=history_loader,
                historical_data=historical_data,
            )

        except Exception as e:
            raise ValueError(f"Failed to parse config: {e}")
