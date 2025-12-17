"""
Strategy registry for dynamic strategy discovery and instantiation.

This module provides a central registry for all available trading strategies.
Strategies can be registered either via decorator or explicit registration.

Usage:
    # Register via decorator
    @register_strategy("ma_cross")
    class MovingAverageCrossStrategy(Strategy):
        pass

    # Register explicitly
    StrategyRegistry.register("momentum", MomentumStrategy)

    # Get strategy class by name
    strategy_class = get_strategy_class("ma_cross")

    # List all registered strategies
    for name in StrategyRegistry.list_strategies():
        print(name)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type, Any
import logging

from .base import Strategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Central registry for trading strategies.

    Provides:
    - Strategy registration by name
    - Strategy lookup by name
    - Strategy metadata storage
    - Validation of strategy classes
    """

    _strategies: Dict[str, Type[Strategy]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_class: Type[Strategy],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a strategy class.

        Args:
            name: Unique name for the strategy.
            strategy_class: Strategy class to register.
            metadata: Optional metadata (description, author, version, etc.).

        Raises:
            ValueError: If name already registered or class invalid.
        """
        if name in cls._strategies:
            logger.warning(f"Overwriting existing strategy: {name}")

        if not issubclass(strategy_class, Strategy):
            raise ValueError(
                f"{strategy_class.__name__} must be a subclass of Strategy"
            )

        cls._strategies[name] = strategy_class
        cls._metadata[name] = metadata or {}

        logger.debug(f"Registered strategy: {name} -> {strategy_class.__name__}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[Strategy]]:
        """
        Get a strategy class by name.

        Args:
            name: Name of the strategy to retrieve.

        Returns:
            Strategy class or None if not found.
        """
        return cls._strategies.get(name)

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a strategy.

        Args:
            name: Name of the strategy.

        Returns:
            Metadata dictionary (empty if not found).
        """
        return cls._metadata.get(name, {})

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List of strategy names.
        """
        return list(cls._strategies.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a strategy.

        Args:
            name: Name of strategy to unregister.

        Returns:
            True if strategy was found and removed.
        """
        if name in cls._strategies:
            del cls._strategies[name]
            cls._metadata.pop(name, None)
            logger.debug(f"Unregistered strategy: {name}")
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies."""
        cls._strategies.clear()
        cls._metadata.clear()
        logger.debug("Cleared strategy registry")

    @classmethod
    def get_all(cls) -> Dict[str, Type[Strategy]]:
        """
        Get all registered strategies.

        Returns:
            Dictionary of name -> strategy class.
        """
        return cls._strategies.copy()


def register_strategy(
    name: str,
    description: str = "",
    author: str = "",
    version: str = "1.0",
    **kwargs,
):
    """
    Decorator to register a strategy class.

    Usage:
        @register_strategy("ma_cross", description="MA crossover strategy")
        class MovingAverageCrossStrategy(Strategy):
            pass

    Args:
        name: Unique name for the strategy.
        description: Human-readable description.
        author: Strategy author.
        version: Strategy version.
        **kwargs: Additional metadata.

    Returns:
        Decorator function.
    """

    def decorator(cls: Type[Strategy]) -> Type[Strategy]:
        metadata = {
            "description": description,
            "author": author,
            "version": version,
            "class_name": cls.__name__,
            **kwargs,
        }
        StrategyRegistry.register(name, cls, metadata)
        return cls

    return decorator


def get_strategy_class(name: str) -> Optional[Type[Strategy]]:
    """
    Get a strategy class by name.

    Convenience function for StrategyRegistry.get().

    Args:
        name: Name of the strategy.

    Returns:
        Strategy class or None if not found.
    """
    return StrategyRegistry.get(name)


def list_strategies() -> List[str]:
    """
    List all registered strategy names.

    Convenience function for StrategyRegistry.list_strategies().

    Returns:
        List of strategy names.
    """
    return StrategyRegistry.list_strategies()


def get_strategy_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get full information about a strategy.

    Args:
        name: Name of the strategy.

    Returns:
        Dictionary with class and metadata, or None if not found.
    """
    strategy_class = StrategyRegistry.get(name)
    if strategy_class is None:
        return None

    metadata = StrategyRegistry.get_metadata(name)
    return {
        "name": name,
        "class": strategy_class,
        "class_name": strategy_class.__name__,
        **metadata,
    }
