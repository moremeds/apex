"""
YAML-based Universe Provider.

Loads ticker universe configuration from a YAML file with support for:
- Default settings for all symbols
- Group definitions with shared settings
- Individual symbol overrides
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logging_setup import get_logger

from .base import SymbolConfig, UniverseProviderBase

logger = get_logger(__name__)


class YamlUniverseProvider(UniverseProviderBase):
    """
    YAML file-based universe provider.

    Loads configuration from a YAML file with the following structure:

    ```yaml
    version: 1

    defaults:
      timeframes: ["1h", "4h", "1d"]
      min_volume_usd: 1000000

    groups:
      tech_megacap:
        symbols: [AAPL, MSFT, GOOGL]
        timeframes: ["15m", "1h", "4h", "1d"]

    overrides:
      AAPL:
        timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]
    ```

    Note:
        This provider does not implement automatic scheduled reloading.
        Callers are responsible for calling refresh() periodically if
        hot-reload of the configuration file is desired.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize YAML universe provider.

        Args:
            config_path: Path to YAML configuration file
        """
        super().__init__()
        self._config_path = Path(config_path)
        self._last_mtime: Optional[float] = None
        self._raw_config: Dict[str, Any] = {}

        # Load on init
        self._load()

    def _load(self) -> None:
        """Load and parse the YAML configuration file."""
        if not self._config_path.exists():
            logger.warning(f"Universe config not found: {self._config_path}")
            return

        try:
            with open(self._config_path) as f:
                self._raw_config = yaml.safe_load(f) or {}
            self._last_mtime = self._config_path.stat().st_mtime
        except Exception as e:
            logger.error(f"Failed to load universe config: {e}")
            return

        self._parse_config()
        self._rebuild_timeframe_index()
        logger.info(f"Loaded universe with {len(self._symbols)} symbols")

    def _parse_config(self) -> None:
        """Parse the raw YAML config into SymbolConfig objects."""
        self._symbols.clear()

        # Get defaults
        defaults = self._raw_config.get("defaults", {})
        default_timeframes = defaults.get("timeframes", ["1h", "4h", "1d"])
        default_min_volume = defaults.get("min_volume_usd")

        # Process groups
        groups = self._raw_config.get("groups", {})
        for group_name, group_config in groups.items():
            symbols = group_config.get("symbols", [])
            group_timeframes = group_config.get("timeframes", default_timeframes)
            group_min_volume = group_config.get("min_volume_usd", default_min_volume)
            group_enabled = group_config.get("enabled", True)

            for symbol in symbols:
                if symbol not in self._symbols:
                    self._symbols[symbol] = SymbolConfig(
                        symbol=symbol,
                        timeframes=list(group_timeframes),
                        min_volume_usd=group_min_volume,
                        enabled=group_enabled,
                        group=group_name,
                    )

        # Process standalone symbols (not in groups)
        standalone = self._raw_config.get("symbols", [])
        for symbol in standalone:
            if symbol not in self._symbols:
                self._symbols[symbol] = SymbolConfig(
                    symbol=symbol,
                    timeframes=list(default_timeframes),
                    min_volume_usd=default_min_volume,
                )

        # Apply overrides
        overrides = self._raw_config.get("overrides", {})
        for symbol, override_config in overrides.items():
            if symbol in self._symbols:
                config = self._symbols[symbol]
                if "timeframes" in override_config:
                    config.timeframes = list(override_config["timeframes"])
                if "min_volume_usd" in override_config:
                    config.min_volume_usd = override_config["min_volume_usd"]
                if "enabled" in override_config:
                    config.enabled = override_config["enabled"]
                if "custom_rules" in override_config:
                    config.custom_rules = list(override_config["custom_rules"])
            else:
                # Create new entry from override
                self._symbols[symbol] = SymbolConfig(
                    symbol=symbol,
                    timeframes=list(override_config.get("timeframes", default_timeframes)),
                    min_volume_usd=override_config.get("min_volume_usd", default_min_volume),
                    enabled=override_config.get("enabled", True),
                    custom_rules=list(override_config.get("custom_rules", [])),
                )

    async def refresh(self) -> None:
        """
        Refresh the universe from the YAML file.

        Only reloads if the file has been modified since last load.
        """
        if not self._config_path.exists():
            return

        current_mtime = self._config_path.stat().st_mtime
        if self._last_mtime is None or current_mtime > self._last_mtime:
            logger.info("Universe config changed, reloading...")
            self._load()

    def get_groups(self) -> List[str]:
        """
        Get all group names in the configuration.

        Returns:
            List of group names
        """
        return list(self._raw_config.get("groups", {}).keys())

    def get_symbols_in_group(self, group: str) -> List[str]:
        """
        Get symbols belonging to a specific group.

        Args:
            group: Group name

        Returns:
            List of symbols in the group
        """
        return [symbol for symbol, config in self._symbols.items() if config.group == group]

    def add_symbol(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        group: Optional[str] = None,
    ) -> None:
        """
        Dynamically add a symbol to the universe.

        Note: This does not persist to the YAML file.

        Args:
            symbol: Symbol to add
            timeframes: Timeframes to monitor (uses defaults if None)
            group: Optional group assignment
        """
        defaults = self._raw_config.get("defaults", {})
        default_timeframes = defaults.get("timeframes", ["1h", "4h", "1d"])

        self._symbols[symbol] = SymbolConfig(
            symbol=symbol,
            timeframes=timeframes or list(default_timeframes),
            group=group,
        )
        self._rebuild_timeframe_index()
        logger.info(f"Added symbol to universe: {symbol}")

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from the universe.

        Note: This does not persist to the YAML file.

        Args:
            symbol: Symbol to remove

        Returns:
            True if symbol was removed, False if not found
        """
        if symbol in self._symbols:
            del self._symbols[symbol]
            self._rebuild_timeframe_index()
            logger.info(f"Removed symbol from universe: {symbol}")
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Export current state as dictionary.

        Returns:
            Dictionary representation suitable for YAML serialization
        """
        groups: Dict[str, Dict[str, Any]] = {}
        standalone: List[str] = []

        for symbol, config in self._symbols.items():
            if config.group:
                if config.group not in groups:
                    groups[config.group] = {"symbols": [], "timeframes": []}
                groups[config.group]["symbols"].append(symbol)
                # Use first symbol's timeframes as group default
                if not groups[config.group]["timeframes"]:
                    groups[config.group]["timeframes"] = config.timeframes
            else:
                standalone.append(symbol)

        return {
            "version": 1,
            "defaults": self._raw_config.get("defaults", {}),
            "groups": groups,
            "symbols": standalone,
            "overrides": self._raw_config.get("overrides", {}),
        }
