"""
Unit tests for configuration schema validation.

Tests:
- Universe config validation
- Rules config validation
- Error reporting with context
- Edge cases and invalid values
"""

import pytest
import tempfile
from pathlib import Path

import yaml

from src.domain.signals.config.schema import (
    ConfigError,
    ValidationResult,
    validate_universe_config,
    validate_rules_config,
    load_and_validate_universe,
    load_and_validate_rules,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_initial_state(self) -> None:
        """Test initial state is valid."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_invalidates(self) -> None:
        """Test adding error sets valid=False."""
        result = ValidationResult(valid=True)
        result.add_error("test error", "path.to.field", "bad_value")

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].message == "test error"
        assert result.errors[0].path == "path.to.field"
        assert result.errors[0].value == "bad_value"

    def test_add_warning_keeps_valid(self) -> None:
        """Test adding warning keeps valid=True."""
        result = ValidationResult(valid=True)
        result.add_warning("test warning")

        assert result.valid is True
        assert len(result.warnings) == 1

    def test_merge_results(self) -> None:
        """Test merging validation results."""
        result1 = ValidationResult(valid=True)
        result1.add_warning("warning 1")

        result2 = ValidationResult(valid=True)
        result2.add_error("error 1")
        result2.add_warning("warning 2")

        result1.merge(result2)

        assert result1.valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 2


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_format_message(self) -> None:
        """Test error message formatting."""
        error = ConfigError("Invalid value", "config.field", 123)
        assert "Invalid value" in str(error)
        assert "config.field" in str(error)
        assert "123" in str(error)

    def test_format_message_no_path(self) -> None:
        """Test error message without path."""
        error = ConfigError("Invalid value")
        assert str(error) == "Invalid value"


class TestUniverseConfigValidation:
    """Tests for universe configuration validation."""

    def test_valid_config(self) -> None:
        """Test validation of valid config."""
        config = {
            "version": 1,
            "defaults": {
                "timeframes": ["1h", "4h", "1d"],
                "min_volume_usd": 1000000,
            },
            "groups": {
                "tech": {
                    "symbols": ["AAPL", "MSFT"],
                    "timeframes": ["1h", "4h"],
                }
            },
            "overrides": {
                "AAPL": {
                    "timeframes": ["1m", "5m", "1h"],
                }
            },
        }

        result = validate_universe_config(config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_missing_version(self) -> None:
        """Test missing version field."""
        config = {"defaults": {}, "groups": {}}

        result = validate_universe_config(config)
        assert result.valid is False
        assert any("version" in str(e) for e in result.errors)

    def test_invalid_version_type(self) -> None:
        """Test non-integer version."""
        config = {"version": "1", "groups": {}}

        result = validate_universe_config(config)
        assert result.valid is False

    def test_invalid_timeframes(self) -> None:
        """Test invalid timeframe values."""
        config = {
            "version": 1,
            "defaults": {
                "timeframes": ["1h", "invalid_tf", "4h"],
            },
            "groups": {},
        }

        result = validate_universe_config(config)
        assert result.valid is False
        assert any("invalid_tf" in str(e) for e in result.errors)

    def test_group_missing_symbols(self) -> None:
        """Test group without symbols."""
        config = {
            "version": 1,
            "groups": {
                "empty_group": {
                    "timeframes": ["1h"],
                }
            },
        }

        result = validate_universe_config(config)
        assert result.valid is False
        assert any("symbols" in str(e) for e in result.errors)

    def test_group_empty_symbols_warning(self) -> None:
        """Test group with empty symbols list generates warning."""
        config = {
            "version": 1,
            "groups": {
                "empty_group": {
                    "symbols": [],
                    "timeframes": ["1h"],
                }
            },
        }

        result = validate_universe_config(config)
        assert result.valid is True  # Valid but warns
        assert any("no symbols" in w for w in result.warnings)

    def test_negative_min_volume(self) -> None:
        """Test negative min_volume_usd."""
        config = {
            "version": 1,
            "defaults": {"min_volume_usd": -100},
            "groups": {},
        }

        result = validate_universe_config(config)
        assert result.valid is False

    def test_override_invalid_custom_rules(self) -> None:
        """Test override with invalid custom_rules type."""
        config = {
            "version": 1,
            "groups": {},
            "overrides": {
                "AAPL": {
                    "custom_rules": "not_a_list",
                }
            },
        }

        result = validate_universe_config(config)
        assert result.valid is False
        assert any("custom_rules" in str(e) for e in result.errors)

    def test_defaults_not_a_dict(self) -> None:
        """Test defaults as non-dict type (robustness)."""
        config = {
            "version": 1,
            "defaults": ["list", "instead", "of", "dict"],
            "groups": {},
        }

        result = validate_universe_config(config)
        assert result.valid is False
        assert any("defaults" in str(e) and "dictionary" in str(e) for e in result.errors)

    def test_timeframe_not_a_string(self) -> None:
        """Test timeframe entry that isn't a string (robustness)."""
        config = {
            "version": 1,
            "defaults": {
                "timeframes": ["1h", {"nested": "dict"}, "4h"],
            },
            "groups": {},
        }

        result = validate_universe_config(config)
        assert result.valid is False
        assert any("must be a string" in str(e) for e in result.errors)


class TestRulesConfigValidation:
    """Tests for rules configuration validation."""

    def test_valid_config(self) -> None:
        """Test validation of valid rules config."""
        config = {
            "version": 1,
            "momentum_rules": {
                "rsi_oversold": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "high",
                    "condition_type": "state_change",
                    "condition_config": {
                        "field": "zone",
                        "from": ["oversold"],
                        "to": ["neutral"],
                    },
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is True

    def test_missing_required_fields(self) -> None:
        """Test rule missing required fields."""
        config = {
            "version": 1,
            "momentum_rules": {
                "incomplete_rule": {
                    "indicator": "rsi",
                    # Missing: direction, strength, priority, condition_type
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        # Should have errors for each missing field
        error_texts = [str(e) for e in result.errors]
        assert any("direction" in e for e in error_texts)
        assert any("strength" in e for e in error_texts)
        assert any("priority" in e for e in error_texts)
        assert any("condition_type" in e for e in error_texts)

    def test_invalid_direction(self) -> None:
        """Test invalid direction value."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "rsi",
                    "direction": "invalid",
                    "strength": 70,
                    "priority": "high",
                    "condition_type": "state_change",
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("direction" in str(e) and "invalid" in str(e) for e in result.errors)

    def test_strength_out_of_range(self) -> None:
        """Test strength outside 0-100 range."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 150,  # > 100
                    "priority": "high",
                    "condition_type": "state_change",
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("strength" in str(e) for e in result.errors)

    def test_invalid_priority(self) -> None:
        """Test invalid priority value."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "urgent",  # Invalid
                    "condition_type": "state_change",
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("priority" in str(e) for e in result.errors)

    def test_invalid_condition_type(self) -> None:
        """Test invalid condition_type value."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "high",
                    "condition_type": "magic_condition",  # Invalid
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("condition_type" in str(e) for e in result.errors)

    def test_threshold_condition_missing_threshold(self) -> None:
        """Test threshold_cross_up without threshold."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "high",
                    "condition_type": "threshold_cross_up",
                    "condition_config": {
                        "field": "value",
                        # Missing: threshold
                    },
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("threshold" in str(e) for e in result.errors)

    def test_state_change_missing_from_to(self) -> None:
        """Test state_change without from/to."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "high",
                    "condition_type": "state_change",
                    "condition_config": {
                        "field": "zone",
                        # Missing: from, to
                    },
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("from" in str(e) for e in result.errors)
        assert any("to" in str(e) for e in result.errors)

    def test_cross_condition_missing_lines(self) -> None:
        """Test cross_up without line_a/line_b."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "macd",
                    "direction": "buy",
                    "strength": 60,
                    "priority": "medium",
                    "condition_type": "cross_up",
                    "condition_config": {
                        # Missing: line_a, line_b
                    },
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("line_a" in str(e) for e in result.errors)
        assert any("line_b" in str(e) for e in result.errors)

    def test_condition_config_not_a_dict(self) -> None:
        """Test condition_config as non-dict type (robustness)."""
        config = {
            "version": 1,
            "momentum_rules": {
                "bad_rule": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "high",
                    "condition_type": "state_change",
                    "condition_config": ["list", "not", "dict"],
                }
            },
        }

        result = validate_rules_config(config)
        assert result.valid is False
        assert any("condition_config" in str(e) and "dictionary" in str(e) for e in result.errors)


class TestFileLoading:
    """Tests for file loading and validation."""

    def test_load_valid_universe(self, tmp_path: Path) -> None:
        """Test loading valid universe config file."""
        config = {
            "version": 1,
            "groups": {
                "test": {
                    "symbols": ["AAPL"],
                    "timeframes": ["1h"],
                }
            },
        }

        config_file = tmp_path / "universe.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        data, result = load_and_validate_universe(str(config_file))

        assert result.valid is True
        assert data["version"] == 1

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file."""
        data, result = load_and_validate_universe("/nonexistent/path.yaml")

        assert result.valid is False
        assert any("not found" in str(e) for e in result.errors)

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Test loading file with invalid YAML syntax."""
        config_file = tmp_path / "bad.yaml"
        with open(config_file, "w") as f:
            f.write("{ invalid yaml: [ unclosed")

        data, result = load_and_validate_universe(str(config_file))

        assert result.valid is False
        assert any("YAML" in str(e) for e in result.errors)

    def test_load_valid_rules(self, tmp_path: Path) -> None:
        """Test loading valid rules config file."""
        config = {
            "version": 1,
            "momentum_rules": {
                "test_rule": {
                    "indicator": "rsi",
                    "direction": "buy",
                    "strength": 70,
                    "priority": "high",
                    "condition_type": "state_change",
                    "condition_config": {
                        "from": ["oversold"],
                        "to": ["neutral"],
                    },
                }
            },
        }

        config_file = tmp_path / "rules.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        data, result = load_and_validate_rules(str(config_file))

        assert result.valid is True
        assert data["version"] == 1
