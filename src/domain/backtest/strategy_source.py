"""
Unified Strategy Source Discovery.

Discovers strategies from:
1. StrategyRegistry - Class-based registered strategies
2. YAML spec files - Configured backtest specifications

Provides a unified StrategyItem interface for the TUI to display and execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import glob

logger = logging.getLogger(__name__)


@dataclass
class StrategyItem:
    """
    Unified strategy representation for TUI display.

    Combines information from both registry strategies and YAML specs.
    """

    name: str
    source_type: str  # "registry" or "yaml"
    source_path: Optional[str] = None  # Path to YAML file if applicable
    description: str = ""
    author: str = ""
    version: str = "1.0"
    params: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution settings (from spec or defaults)
    initial_capital: float = 100000.0
    data_source: str = "ib"
    bar_size: str = "1d"
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @property
    def display_name(self) -> str:
        """Get display name with source indicator."""
        icon = "📁" if self.source_type == "yaml" else "⚙️"
        return f"{icon} {self.name}"

    @property
    def short_description(self) -> str:
        """Get short description (first 60 chars)."""
        if not self.description:
            return ""
        return self.description[:60] + "..." if len(self.description) > 60 else self.description


class StrategySource:
    """
    Discovers and loads strategies from all sources.

    Combines:
    - Registered strategies from StrategyRegistry
    - YAML spec files from config/backtest/
    """

    DEFAULT_YAML_DIR = "config/backtest"

    @classmethod
    def list_all(cls, yaml_dir: Optional[str] = None) -> List[StrategyItem]:
        """
        List all strategies from registry and YAML files.

        Args:
            yaml_dir: Directory containing YAML spec files.
                     Defaults to config/backtest/

        Returns:
            List of StrategyItem objects, sorted by name.
        """
        items: List[StrategyItem] = []

        # Load from registry
        registry_items = cls._load_from_registry()
        items.extend(registry_items)

        # Load from YAML files
        yaml_dir = yaml_dir or cls.DEFAULT_YAML_DIR
        yaml_items = cls._load_from_yaml(yaml_dir)

        # Merge YAML items - if same strategy name exists, prefer YAML (has config)
        registry_names = {item.name for item in registry_items}
        for yaml_item in yaml_items:
            # Find matching registry item and merge
            if yaml_item.name in registry_names:
                # Update existing registry item with YAML config
                for i, item in enumerate(items):
                    if item.name == yaml_item.name:
                        # Merge: keep registry metadata but use YAML execution config
                        items[i] = cls._merge_items(item, yaml_item)
                        break
            else:
                # Add as new item
                items.append(yaml_item)

        # Sort by name
        items.sort(key=lambda x: x.name)

        logger.info(f"Discovered {len(items)} strategies ({len(registry_items)} registry, {len(yaml_items)} yaml)")
        return items

    @classmethod
    def _load_from_registry(cls) -> List[StrategyItem]:
        """Load strategies from StrategyRegistry."""
        items = []

        try:
            from ..strategy.registry import StrategyRegistry, get_strategy_info

            # Import example strategies to ensure they're registered
            try:
                from ..strategy.examples import (
                    MovingAverageCrossStrategy,
                    BuyAndHoldStrategy,
                )
            except ImportError:
                pass

            for name in StrategyRegistry.list_strategies():
                info = get_strategy_info(name)
                if info:
                    items.append(
                        StrategyItem(
                            name=name,
                            source_type="registry",
                            description=info.get("description", ""),
                            author=info.get("author", ""),
                            version=info.get("version", "1.0"),
                            params=info.get("default_params", {}),
                            metadata=info,
                        )
                    )

        except ImportError as e:
            logger.warning(f"Could not load from registry: {e}")

        return items

    @classmethod
    def _load_from_yaml(cls, yaml_dir: str) -> List[StrategyItem]:
        """Load strategies from YAML spec files."""
        items = []

        yaml_path = Path(yaml_dir)
        if not yaml_path.exists():
            logger.debug(f"YAML directory not found: {yaml_dir}")
            return items

        # Find all YAML files
        yaml_files = list(yaml_path.glob("**/*.yaml")) + list(yaml_path.glob("**/*.yml"))

        for yaml_file in yaml_files:
            try:
                from .backtest_spec import BacktestSpec

                spec = BacktestSpec.from_yaml(str(yaml_file))

                items.append(
                    StrategyItem(
                        name=spec.strategy.name,
                        source_type="yaml",
                        source_path=str(yaml_file),
                        description=spec.metadata.get("description", ""),
                        author=spec.metadata.get("author", ""),
                        version=spec.metadata.get("version", "1.0"),
                        params=spec.strategy.params,
                        symbols=spec.get_symbols(),
                        metadata=spec.metadata,
                        initial_capital=spec.execution.initial_capital,
                        data_source=spec.data.source,
                        bar_size=spec.data.bar_size,
                        start_date=spec.data.start_date,
                        end_date=spec.data.end_date,
                    )
                )
                logger.debug(f"Loaded spec from {yaml_file}: {spec.strategy.name}")

            except Exception as e:
                logger.warning(f"Failed to load spec from {yaml_file}: {e}")

        return items

    @classmethod
    def _merge_items(cls, registry_item: StrategyItem, yaml_item: StrategyItem) -> StrategyItem:
        """Merge registry and YAML items, preferring YAML config."""
        return StrategyItem(
            name=registry_item.name,
            source_type="yaml",  # Mark as YAML since it has config
            source_path=yaml_item.source_path,
            description=registry_item.description or yaml_item.description,
            author=registry_item.author or yaml_item.author,
            version=registry_item.version,
            params={**registry_item.params, **yaml_item.params},
            symbols=yaml_item.symbols or registry_item.symbols,
            metadata={**registry_item.metadata, **yaml_item.metadata},
            initial_capital=yaml_item.initial_capital,
            data_source=yaml_item.data_source,
            bar_size=yaml_item.bar_size,
            start_date=yaml_item.start_date,
            end_date=yaml_item.end_date,
        )

    @classmethod
    def get_strategy(cls, name: str, yaml_dir: Optional[str] = None) -> Optional[StrategyItem]:
        """
        Get a specific strategy by name.

        Args:
            name: Strategy name.
            yaml_dir: Optional YAML directory.

        Returns:
            StrategyItem or None if not found.
        """
        items = cls.list_all(yaml_dir)
        for item in items:
            if item.name == name:
                return item
        return None

    @classmethod
    def create_backtest_spec(cls, item: StrategyItem):
        """
        Create a BacktestSpec from a StrategyItem.

        Args:
            item: Strategy item to convert.

        Returns:
            BacktestSpec ready for execution.
        """
        from .backtest_spec import (
            BacktestSpec,
            StrategySpecConfig,
            DataSpecConfig,
            ExecutionSpecConfig,
        )

        # If item has a source path, load the full spec
        if item.source_path:
            return BacktestSpec.from_yaml(item.source_path)

        # Otherwise, create a minimal spec from item
        return BacktestSpec(
            strategy=StrategySpecConfig(
                name=item.name,
                id=f"tui-{item.name}",
                params=item.params,
            ),
            universe={"symbols": item.symbols or ["SPY"]},
            data=DataSpecConfig(
                source=item.data_source,
                bar_size=item.bar_size,
                start_date=item.start_date,
                end_date=item.end_date,
            ),
            execution=ExecutionSpecConfig(
                initial_capital=item.initial_capital,
            ),
            metadata={
                "author": item.author,
                "description": item.description,
                "version": item.version,
            },
        )
