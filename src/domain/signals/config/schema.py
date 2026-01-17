"""
Configuration Schema and Validation for Trading Signal Engine.

Provides:
- Typed dataclasses for configuration sections
- Validation functions for YAML configs
- Error reporting with location context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.utils.logging_setup import get_logger

from ..models import ConditionType, SignalDirection, SignalPriority

logger = get_logger(__name__)


# Valid timeframe values (must match data provider capabilities)
VALID_TIMEFRAMES: Set[str] = {
    "1m",
    "2m",
    "3m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "8h",
    "12h",
    "1d",
    "1w",
    "1M",
}


class ConfigError(Exception):
    """Configuration validation error with context."""

    def __init__(self, message: str, path: str = "", value: Any = None):
        self.message = message
        self.path = path
        self.value = value
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.path:
            parts.append(f"at '{self.path}'")
        if self.value is not None:
            parts.append(f"(got: {self.value!r})")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    errors: List[ConfigError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str, path: str = "", value: Any = None) -> None:
        """Add a validation error."""
        self.errors.append(ConfigError(message, path, value))
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False


# --- Universe Schema ---


@dataclass
class UniverseDefaults:
    """Default settings for universe symbols."""

    timeframes: List[str] = field(default_factory=lambda: ["1d"])
    min_volume_usd: float = 0.0


@dataclass
class UniverseGroup:
    """Symbol group configuration."""

    symbols: List[str]
    timeframes: List[str]
    enabled: bool = True


@dataclass
class UniverseOverride:
    """Per-symbol override configuration."""

    timeframes: Optional[List[str]] = None
    custom_rules: Optional[List[str]] = None
    enabled: bool = True


@dataclass
class UniverseConfig:
    """Complete universe configuration."""

    version: int
    defaults: UniverseDefaults
    groups: Dict[str, UniverseGroup]
    overrides: Dict[str, UniverseOverride]
    provider: Dict[str, Any]


def validate_universe_config(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate universe configuration.

    Args:
        data: Raw YAML data

    Returns:
        ValidationResult with any errors/warnings
    """
    result = ValidationResult(valid=True)

    # Version check
    version = data.get("version")
    if version is None:
        result.add_error("Missing required field 'version'", "version")
    elif not isinstance(version, int):
        result.add_error("Must be an integer", "version", version)

    # Defaults validation
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        result.add_error("Must be a dictionary", "defaults", defaults)
    else:
        _validate_timeframes(defaults.get("timeframes", []), "defaults.timeframes", result)

        min_volume = defaults.get("min_volume_usd", 0)
        if not isinstance(min_volume, (int, float)) or min_volume < 0:
            result.add_error("Must be a non-negative number", "defaults.min_volume_usd", min_volume)

    # Groups validation
    groups = data.get("groups", {})
    if not isinstance(groups, dict):
        result.add_error("Must be a dictionary", "groups", groups)
    else:
        for group_name, group_data in groups.items():
            _validate_group(group_name, group_data, result)

    # Overrides validation
    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict):
        result.add_error("Must be a dictionary", "overrides", overrides)
    else:
        for symbol, override_data in overrides.items():
            _validate_override(symbol, override_data, result)

    return result


def _validate_timeframes(timeframes: Any, path: str, result: ValidationResult) -> None:
    """Validate a list of timeframes."""
    if not isinstance(timeframes, list):
        result.add_error("Must be a list", path, timeframes)
        return

    for i, tf in enumerate(timeframes):
        if not isinstance(tf, str):
            result.add_error("Timeframe must be a string", f"{path}[{i}]", tf)
        elif tf not in VALID_TIMEFRAMES:
            result.add_error(
                f"Invalid timeframe '{tf}', valid options: {sorted(VALID_TIMEFRAMES)}",
                f"{path}[{i}]",
                tf,
            )


def _validate_group(name: str, data: Any, result: ValidationResult) -> None:
    """Validate a symbol group."""
    path = f"groups.{name}"

    if not isinstance(data, dict):
        result.add_error("Must be a dictionary", path, data)
        return

    # Symbols required
    symbols = data.get("symbols")
    if symbols is None:
        result.add_error("Missing required field 'symbols'", f"{path}.symbols")
    elif not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
        result.add_error("Must be a list of strings", f"{path}.symbols", symbols)
    elif len(symbols) == 0:
        result.add_warning(f"Group '{name}' has no symbols")

    # Timeframes optional but validated if present
    timeframes = data.get("timeframes")
    if timeframes is not None:
        _validate_timeframes(timeframes, f"{path}.timeframes", result)


def _validate_override(symbol: str, data: Any, result: ValidationResult) -> None:
    """Validate a symbol override."""
    path = f"overrides.{symbol}"

    if not isinstance(data, dict):
        result.add_error("Must be a dictionary", path, data)
        return

    # Timeframes optional but validated if present
    timeframes = data.get("timeframes")
    if timeframes is not None:
        _validate_timeframes(timeframes, f"{path}.timeframes", result)

    # Custom rules optional but should be list of strings
    custom_rules = data.get("custom_rules")
    if custom_rules is not None:
        if not isinstance(custom_rules, list) or not all(isinstance(r, str) for r in custom_rules):
            result.add_error("Must be a list of strings", f"{path}.custom_rules", custom_rules)


# --- Rules Schema ---


VALID_DIRECTIONS: Set[str] = {"buy", "sell", "alert"}
VALID_PRIORITIES: Set[str] = {"high", "medium", "low"}
VALID_CONDITION_TYPES: Set[str] = {
    "threshold_cross_up",
    "threshold_cross_down",
    "state_change",
    "cross_up",
    "cross_down",
    "range_entry",
    "range_exit",
    "custom",
}


def validate_rules_config(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate rules configuration.

    Args:
        data: Raw YAML data

    Returns:
        ValidationResult with any errors/warnings
    """
    result = ValidationResult(valid=True)

    # Version check
    version = data.get("version")
    if version is None:
        result.add_error("Missing required field 'version'", "version")
    elif not isinstance(version, int):
        result.add_error("Must be an integer", "version", version)

    # Rule sections (momentum_rules, trend_rules, etc.)
    rule_sections = [k for k in data.keys() if k.endswith("_rules")]

    for section in rule_sections:
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            result.add_error("Must be a dictionary", section, section_data)
            continue

        for rule_name, rule_data in section_data.items():
            _validate_rule(rule_name, rule_data, section, result)

    return result


def _validate_rule(name: str, data: Any, section: str, result: ValidationResult) -> None:
    """Validate a single rule definition."""
    path = f"{section}.{name}"

    if not isinstance(data, dict):
        result.add_error("Must be a dictionary", path, data)
        return

    # Required fields
    required_fields = ["indicator", "direction", "strength", "priority", "condition_type"]
    for field_name in required_fields:
        if field_name not in data:
            result.add_error(f"Missing required field '{field_name}'", f"{path}.{field_name}")

    # Direction validation
    direction = data.get("direction")
    if direction is not None and direction not in VALID_DIRECTIONS:
        result.add_error(
            f"Invalid direction, must be one of {VALID_DIRECTIONS}",
            f"{path}.direction",
            direction,
        )

    # Strength validation (0-100)
    strength = data.get("strength")
    if strength is not None:
        if not isinstance(strength, (int, float)) or not (0 <= strength <= 100):
            result.add_error("Must be a number between 0 and 100", f"{path}.strength", strength)

    # Priority validation
    priority = data.get("priority")
    if priority is not None and priority not in VALID_PRIORITIES:
        result.add_error(
            f"Invalid priority, must be one of {VALID_PRIORITIES}",
            f"{path}.priority",
            priority,
        )

    # Condition type validation
    condition_type = data.get("condition_type")
    if condition_type is not None and condition_type not in VALID_CONDITION_TYPES:
        result.add_error(
            f"Invalid condition_type, must be one of {VALID_CONDITION_TYPES}",
            f"{path}.condition_type",
            condition_type,
        )

    # Condition config validation based on type
    condition_config = data.get("condition_config", {})
    if condition_type and isinstance(condition_type, str):
        if not isinstance(condition_config, dict):
            result.add_error("Must be a dictionary", f"{path}.condition_config", condition_config)
        else:
            _validate_condition_config(condition_type, condition_config, path, result)


def _validate_condition_config(
    condition_type: str,
    config: Dict[str, Any],
    path: str,
    result: ValidationResult,
) -> None:
    """Validate condition_config based on condition_type."""
    config_path = f"{path}.condition_config"

    if condition_type in ("threshold_cross_up", "threshold_cross_down"):
        # Requires: field, threshold
        if "threshold" not in config:
            result.add_error("Missing required 'threshold'", config_path)
        elif not isinstance(config["threshold"], (int, float)):
            result.add_error("'threshold' must be a number", config_path, config["threshold"])

    elif condition_type == "state_change":
        # Requires: field, from, to
        if "from" not in config:
            result.add_error("Missing required 'from' list", config_path)
        elif not isinstance(config["from"], list):
            result.add_error("'from' must be a list", config_path, config["from"])

        if "to" not in config:
            result.add_error("Missing required 'to' list", config_path)
        elif not isinstance(config["to"], list):
            result.add_error("'to' must be a list", config_path, config["to"])

    elif condition_type in ("cross_up", "cross_down"):
        # Requires: line_a, line_b
        if "line_a" not in config:
            result.add_error("Missing required 'line_a'", config_path)
        if "line_b" not in config:
            result.add_error("Missing required 'line_b'", config_path)

    elif condition_type in ("range_entry", "range_exit"):
        # Requires: lower, upper
        if "lower" not in config:
            result.add_error("Missing required 'lower'", config_path)
        if "upper" not in config:
            result.add_error("Missing required 'upper'", config_path)


def load_and_validate_universe(path: str) -> tuple[Dict[str, Any], ValidationResult]:
    """
    Load and validate a universe configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Tuple of (raw data, validation result)
    """
    import yaml

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        result = ValidationResult(valid=False)
        result.add_error(f"Configuration file not found: {path}")
        return {}, result
    except yaml.YAMLError as e:
        result = ValidationResult(valid=False)
        result.add_error(f"Invalid YAML syntax: {e}")
        return {}, result

    return data, validate_universe_config(data)


def load_and_validate_rules(path: str) -> tuple[Dict[str, Any], ValidationResult]:
    """
    Load and validate a rules configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Tuple of (raw data, validation result)
    """
    import yaml

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        result = ValidationResult(valid=False)
        result.add_error(f"Configuration file not found: {path}")
        return {}, result
    except yaml.YAMLError as e:
        result = ValidationResult(valid=False)
        result.add_error(f"Invalid YAML syntax: {e}")
        return {}, result

    return data, validate_rules_config(data)
