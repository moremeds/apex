"""
Universe providers for configurable ticker universe management.

Provides:
- UniverseProvider: Protocol for universe providers
- SymbolConfig: Configuration for individual symbols
- YamlUniverseProvider: YAML file-based universe
"""

from .base import UniverseProvider, SymbolConfig
from .yaml_provider import YamlUniverseProvider

__all__ = [
    "UniverseProvider",
    "SymbolConfig",
    "YamlUniverseProvider",
]
