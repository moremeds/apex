"""
Indicator Registry - Auto-discovery and management of indicators.

Provides:
- Auto-discovery of indicator classes from category packages
- Registration and lookup by name
- Filtering by category
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

from src.utils.logging_setup import get_logger

from ..models import SignalCategory
from .base import Indicator, IndicatorBase

logger = get_logger(__name__)


class IndicatorRegistry:
    """
    Registry for indicator discovery and management.

    Auto-discovers indicator classes from category packages (momentum/, trend/, etc.)
    and provides lookup by name or category.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._indicators: Dict[str, Indicator] = {}
        self._by_category: Dict[SignalCategory, Set[str]] = {
            cat: set() for cat in SignalCategory
        }

    def clear(self) -> None:
        """Clear all registered indicators."""
        self._indicators.clear()
        for cat in self._by_category:
            self._by_category[cat].clear()

    def discover(self) -> int:
        """
        Auto-discover indicators from category packages.

        Scans momentum/, trend/, volatility/, volume/, pattern/ packages
        for classes implementing the Indicator protocol.

        Returns:
            Number of indicators discovered
        """
        base_path = Path(__file__).parent
        categories = ["momentum", "trend", "volatility", "volume", "pattern", "regime"]

        discovered = 0
        for category in categories:
            category_path = base_path / category
            if not category_path.exists():
                logger.debug(f"Category package not found: {category}")
                continue

            discovered += self._discover_package(category)

        logger.info(f"Discovered {discovered} indicators across {len(categories)} categories")
        return discovered

    def _discover_package(self, category: str) -> int:
        """
        Discover indicators in a category package.

        Args:
            category: Package name (e.g., "momentum")

        Returns:
            Number of indicators discovered in this package
        """
        package_name = f"src.domain.signals.indicators.{category}"
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.warning(f"Failed to import {package_name}: {e}")
            return 0

        discovered = 0
        package_path = Path(package.__file__).parent

        for module_info in pkgutil.iter_modules([str(package_path)]):
            if module_info.name.startswith("_"):
                continue

            module_name = f"{package_name}.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                logger.warning(f"Failed to import {module_name}: {e}")
                continue

            # Find indicator classes in module
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue

                attr = getattr(module, attr_name)
                if self._is_indicator_class(attr):
                    try:
                        indicator = attr()
                        self.register(indicator)
                        discovered += 1
                        logger.debug(f"Discovered indicator: {indicator.name}")
                    except TypeError as e:
                        # Constructor requires arguments - skip this class
                        logger.warning(
                            f"Skipping {attr_name}: constructor requires arguments ({e})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to instantiate {attr_name}: {e}")

        return discovered

    def _is_indicator_class(self, obj: object) -> bool:
        """
        Check if object is an indicator class (not instance).

        Only returns True for concrete IndicatorBase subclasses.
        Protocol-only implementations are not auto-discovered to avoid
        false positives from test mocks or abstract classes.

        Args:
            obj: Object to check

        Returns:
            True if it's a concrete IndicatorBase subclass
        """
        if not isinstance(obj, type):
            return False

        # Must be a subclass of IndicatorBase but not IndicatorBase itself
        return issubclass(obj, IndicatorBase) and obj is not IndicatorBase

    def register(self, indicator: Indicator) -> None:
        """
        Register an indicator instance.

        If an indicator with the same name exists, it will be replaced
        and removed from its previous category index.

        Args:
            indicator: Indicator to register
        """
        name = indicator.name

        # Remove from old category if overwriting
        if name in self._indicators:
            old_indicator = self._indicators[name]
            old_category = old_indicator.category
            self._by_category[old_category].discard(name)
            logger.warning(f"Indicator {name} already registered, overwriting")

        self._indicators[name] = indicator
        self._by_category[indicator.category].add(name)
        logger.debug(f"Registered indicator: {name} ({indicator.category.value})")

    def get(self, name: str) -> Optional[Indicator]:
        """
        Get indicator by name.

        Args:
            name: Indicator name (e.g., "rsi")

        Returns:
            Indicator instance or None if not found
        """
        return self._indicators.get(name)

    def get_all(self) -> List[Indicator]:
        """
        Get all registered indicators.

        Returns:
            List of all indicator instances
        """
        return list(self._indicators.values())

    def get_by_category(self, category: SignalCategory) -> List[Indicator]:
        """
        Get indicators by category.

        Args:
            category: SignalCategory to filter by

        Returns:
            List of indicators in that category
        """
        names = self._by_category.get(category, set())
        return [self._indicators[n] for n in names]

    def get_names(self) -> List[str]:
        """
        Get all registered indicator names.

        Returns:
            List of indicator names
        """
        return list(self._indicators.keys())

    def __len__(self) -> int:
        """Return number of registered indicators."""
        return len(self._indicators)

    def __contains__(self, name: str) -> bool:
        """Check if indicator is registered."""
        return name in self._indicators


# Global registry instance
_global_registry: Optional[IndicatorRegistry] = None


def get_indicator_registry() -> IndicatorRegistry:
    """
    Get the global indicator registry.

    Creates and initializes the registry on first call.

    Returns:
        Global IndicatorRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = IndicatorRegistry()
        _global_registry.discover()
    return _global_registry
