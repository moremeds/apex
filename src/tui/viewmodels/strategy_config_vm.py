"""
Strategy Config ViewModel - framework agnostic.

Extracts strategy introspection logic from the widget layer.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StrategyParam:
    """A single strategy parameter."""

    name: str
    default: Any
    annotation: Optional[str] = None
    required: bool = False


@dataclass
class StrategyConfigData:
    """Strategy configuration display data."""

    name: str
    description: str
    version: str
    author: str
    params: Dict[str, StrategyParam] = field(default_factory=dict)


class StrategyConfigViewModel:
    """
    Framework-agnostic ViewModel for strategy configuration.

    All strategy introspection (parameter extraction, defaults)
    is performed here, not in the widget.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, StrategyConfigData] = {}

    def get_config(
        self,
        strategy_name: Optional[str],
        strategy_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[StrategyConfigData]:
        """
        Get strategy configuration data.

        Args:
            strategy_name: Name of the strategy
            strategy_info: Optional info dict with description, version, author

        Returns:
            StrategyConfigData with all introspected params, or None if invalid
        """
        if not strategy_name:
            return None

        # Check cache
        if strategy_name in self._cache:
            return self._cache[strategy_name]

        # Extract from registry
        info = strategy_info or {}
        params = self._extract_params(strategy_name)

        config = StrategyConfigData(
            name=strategy_name,
            description=info.get("description", ""),
            version=info.get("version", "1.0"),
            author=info.get("author", ""),
            params=params,
        )

        self._cache[strategy_name] = config
        return config

    def _extract_params(self, strategy_name: str) -> Dict[str, StrategyParam]:
        """
        Extract configurable parameters from strategy class.

        Uses introspection to find __init__ parameters and their defaults.
        """
        try:
            from ...domain.strategy.registry import StrategyRegistry

            strategy_class = StrategyRegistry.get(strategy_name)
            if not strategy_class:
                return {}

            excluded = {"strategy_id", "symbols", "context", "self"}
            params = {}

            sig = inspect.signature(strategy_class.__init__)
            for name, param in sig.parameters.items():
                if name in excluded:
                    continue

                required = param.default is inspect.Parameter.empty
                annotation = None
                if param.annotation != inspect.Parameter.empty:
                    annotation = str(param.annotation)

                params[name] = StrategyParam(
                    name=name,
                    default=None if required else param.default,
                    annotation=annotation,
                    required=required,
                )

            return params
        except Exception:
            return {}

    def invalidate(self, strategy_name: Optional[str] = None) -> None:
        """
        Clear cache for a strategy or all strategies.

        Args:
            strategy_name: Specific strategy to invalidate, or None for all
        """
        if strategy_name:
            self._cache.pop(strategy_name, None)
        else:
            self._cache.clear()
