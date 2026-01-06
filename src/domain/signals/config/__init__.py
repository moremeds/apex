"""
Trading Signal Configuration Module.

Provides schema validation and loading for signal configuration files.
"""

from .schema import (
    ConfigError,
    ValidationResult,
    load_and_validate_rules,
    load_and_validate_universe,
    validate_rules_config,
    validate_universe_config,
)

__all__ = [
    "ConfigError",
    "ValidationResult",
    "load_and_validate_rules",
    "load_and_validate_universe",
    "validate_rules_config",
    "validate_universe_config",
]
