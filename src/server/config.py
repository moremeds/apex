"""Server configuration loader — reads config/server.yaml into frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a single data provider."""

    enabled: bool = False
    sub_types: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ServerConfig:
    """Top-level server configuration."""

    # Server
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8080

    # Providers
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)

    # Symbols
    universe_path: str = "config/universe.yaml"
    from_screener: bool = True
    max_symbols: int = 100

    # Persistence
    duckdb_path: str = "data/server.duckdb"
    r2_flush_interval_sec: int = 300

    # Pipeline
    timeframes: List[str] = field(default_factory=lambda: ["30m", "1h", "4h", "1d"])
    indicators: List[str] = field(
        default_factory=lambda: ["rsi", "dual_macd", "supertrend", "bollinger"]
    )

    # Shared base config path (for AppContainer/broker/risk integration)
    base_config_path: str = "config/base.yaml"


def load_server_config(path: str = "config/server.yaml") -> ServerConfig:
    """Load server config from YAML file."""
    with open(Path(path)) as f:
        raw = yaml.safe_load(f)

    server = raw.get("server", {})
    providers_raw = raw.get("providers", {})
    symbols = raw.get("symbols", {})
    persistence = raw.get("persistence", {})
    pipeline = raw.get("pipeline", {})

    providers = {
        name: ProviderConfig(
            enabled=cfg.get("enabled", False),
            sub_types=cfg.get("sub_types", []),
        )
        for name, cfg in providers_raw.items()
    }

    return ServerConfig(
        host=server.get("host", "0.0.0.0"),  # nosec B104
        port=server.get("port", 8080),
        providers=providers,
        universe_path=symbols.get("universe_path", "config/universe.yaml"),
        from_screener=symbols.get("from_screener", True),
        max_symbols=symbols.get("max_total", 100),
        duckdb_path=persistence.get("duckdb_path", "data/server.duckdb"),
        r2_flush_interval_sec=persistence.get("r2_flush_interval_sec", 300),
        timeframes=pipeline.get("timeframes", ["30m", "1h", "4h", "1d"]),
        indicators=pipeline.get("indicators", []),
        base_config_path=raw.get("base_config", "config/base.yaml"),
    )
