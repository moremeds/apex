"""
Configuration loading utilities for backtest runner.

Provides functions to load:
- IB broker configuration from base.yaml
- Historical data storage configuration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "base.yaml"


def load_ib_config(config_path: Optional[Path] = None) -> Any:
    """Load IB config from base.yaml."""
    from config.models import IbClientIdsConfig, IbConfig

    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return None

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        ib_cfg = config.get("brokers", {}).get("ibkr", {})
        if not ib_cfg.get("enabled"):
            logger.warning("IB not enabled in config")
            return None

        client_ids_cfg = ib_cfg.get("client_ids", {})
        client_ids = IbClientIdsConfig(
            execution=client_ids_cfg.get("execution", 1),
            monitoring=client_ids_cfg.get("monitoring", 2),
            historical_pool=client_ids_cfg.get("historical_pool", [3, 4, 5, 6, 7, 8, 9, 10]),
        )

        return IbConfig(
            enabled=True,
            host=ib_cfg.get("host", "127.0.0.1"),
            port=ib_cfg.get("port", 7497),
            client_ids=client_ids,
            provides_market_data=ib_cfg.get("provides_market_data", True),
        )
    except Exception as e:
        logger.warning(f"Failed to load IB config: {e}")
        return None


def load_historical_data_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load historical data config from base.yaml."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        historical_cfg = config.get("historical_data", {})
        storage_cfg = historical_cfg.get("storage", {})

        return {
            "base_dir": storage_cfg.get("base_dir", "data/historical"),
            "source_priority": historical_cfg.get("source_priority", ["ib", "yahoo"]),
            "sources": historical_cfg.get("sources", {}),
        }
    except Exception as e:
        logger.warning(f"Failed to load historical data config: {e}")
        return {}
