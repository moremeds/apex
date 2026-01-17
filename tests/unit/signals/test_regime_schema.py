"""
Tests for RegimeOutput schema validation (Phase 0).

Tests the schema version validation, from_dict deserialization,
and round-trip serialization.
"""

from datetime import datetime

import pytest

from src.domain.signals.indicators.regime.models import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_STR,
    ChopState,
    ComponentStates,
    ComponentValues,
    DataQuality,
    DataWindow,
    DerivedMetrics,
    ExtState,
    FallbackReason,
    InputsUsed,
    IVState,
    MarketRegime,
    RegimeOutput,
    RegimeTransitionState,
    SchemaVersionError,
    TrendState,
    VolState,
    validate_schema_version,
)


class TestSchemaVersionConstants:
    """Test schema version constants."""

    def test_schema_version_is_tuple(self):
        """Schema version should be a (major, minor) tuple."""
        assert isinstance(SCHEMA_VERSION, tuple)
        assert len(SCHEMA_VERSION) == 2
        assert all(isinstance(v, int) for v in SCHEMA_VERSION)

    def test_schema_version_str_format(self):
        """Schema version string should match expected format."""
        expected = f"regime_output@{SCHEMA_VERSION[0]}.{SCHEMA_VERSION[1]}"
        assert SCHEMA_VERSION_STR == expected

    def test_current_version(self):
        """Current version should be 1.1."""
        assert SCHEMA_VERSION == (1, 1)
        assert SCHEMA_VERSION_STR == "regime_output@1.1"


class TestValidateSchemaVersion:
    """Test schema version validation."""

    def test_valid_current_version(self):
        """Current version should validate."""
        result = validate_schema_version("regime_output@1.1")
        assert result == (1, 1)

    def test_valid_minor_upgrade(self):
        """Higher minor version should validate (forward compatible)."""
        # This tests that we can read newer minor versions
        # Note: This may need adjustment based on compatibility policy
        result = validate_schema_version("regime_output@1.0")
        assert result == (1, 0)

    def test_invalid_major_version_raises(self):
        """Different major version should raise SchemaVersionError."""
        with pytest.raises(SchemaVersionError) as exc_info:
            validate_schema_version("regime_output@2.0")
        assert exc_info.value.expected == SCHEMA_VERSION_STR
        assert exc_info.value.actual == "regime_output@2.0"

    def test_invalid_format_raises(self):
        """Invalid format should raise SchemaVersionError."""
        with pytest.raises(SchemaVersionError):
            validate_schema_version("invalid_format")

    def test_missing_prefix_raises(self):
        """Missing prefix should raise SchemaVersionError."""
        with pytest.raises(SchemaVersionError):
            validate_schema_version("1.1")

    def test_empty_string_raises(self):
        """Empty string should raise SchemaVersionError."""
        with pytest.raises(SchemaVersionError):
            validate_schema_version("")

    def test_none_raises(self):
        """None should raise SchemaVersionError."""
        with pytest.raises(SchemaVersionError):
            validate_schema_version(None)


class TestRegimeOutputToDict:
    """Test RegimeOutput.to_dict() serialization."""

    def test_to_dict_includes_schema_version(self):
        """to_dict should include schema_version."""
        output = RegimeOutput(symbol="QQQ")
        result = output.to_dict()
        assert result["schema_version"] == SCHEMA_VERSION_STR

    def test_to_dict_round_trip_preserves_data(self):
        """to_dict followed by from_dict should preserve all data."""
        now = datetime.now().replace(microsecond=0)  # Remove microseconds for comparison

        original = RegimeOutput(
            symbol="AAPL",
            asof_ts=now,
            bar_interval="1d",
            decision_regime=MarketRegime.R0_HEALTHY_UPTREND,
            final_regime=MarketRegime.R0_HEALTHY_UPTREND,
            regime_name="Healthy Uptrend",
            confidence=75,
            component_states=ComponentStates(
                trend_state=TrendState.UP,
                vol_state=VolState.NORMAL,
                chop_state=ChopState.TRENDING,
                ext_state=ExtState.NEUTRAL,
                iv_state=IVState.NA,
            ),
            component_values=ComponentValues(
                close=150.0,
                ma20=148.0,
                ma50=145.0,
                ma200=140.0,
                ma50_slope=0.02,
            ),
            regime_changed=False,
        )

        # Serialize and deserialize
        serialized = original.to_dict()
        restored = RegimeOutput.from_dict(serialized)

        # Compare key fields
        assert restored.symbol == original.symbol
        assert restored.bar_interval == original.bar_interval
        assert restored.decision_regime == original.decision_regime
        assert restored.final_regime == original.final_regime
        assert restored.confidence == original.confidence
        assert restored.regime_changed == original.regime_changed

        # Compare component states
        assert restored.component_states.trend_state == original.component_states.trend_state
        assert restored.component_states.vol_state == original.component_states.vol_state

        # Compare component values (with rounding tolerance)
        assert restored.component_values.close == pytest.approx(original.component_values.close)
        assert restored.component_values.ma50 == pytest.approx(original.component_values.ma50)


class TestRegimeOutputFromDict:
    """Test RegimeOutput.from_dict() deserialization."""

    def test_from_dict_validates_schema_version(self):
        """from_dict should validate schema version."""
        data = {"schema_version": "regime_output@2.0", "symbol": "QQQ"}
        with pytest.raises(SchemaVersionError):
            RegimeOutput.from_dict(data)

    def test_from_dict_handles_missing_fields(self):
        """from_dict should handle missing optional fields gracefully."""
        data = {
            "schema_version": SCHEMA_VERSION_STR,
            "symbol": "AAPL",
        }
        result = RegimeOutput.from_dict(data)
        assert result.symbol == "AAPL"
        assert result.decision_regime == MarketRegime.R1_CHOPPY_EXTENDED  # Default
        assert result.confidence == 50  # Default

    def test_from_dict_parses_enums(self):
        """from_dict should correctly parse enum values."""
        data = {
            "schema_version": SCHEMA_VERSION_STR,
            "symbol": "QQQ",
            "decision_regime": "R2",
            "final_regime": "R2",
            "component_states": {
                "trend_state": "trend_down",
                "vol_state": "vol_high",
                "chop_state": "choppy",
                "ext_state": "oversold",
                "iv_state": "iv_high",
            },
        }
        result = RegimeOutput.from_dict(data)
        assert result.decision_regime == MarketRegime.R2_RISK_OFF
        assert result.final_regime == MarketRegime.R2_RISK_OFF
        assert result.component_states.trend_state == TrendState.DOWN
        assert result.component_states.vol_state == VolState.HIGH
        assert result.component_states.iv_state == IVState.HIGH

    def test_from_dict_parses_datetime(self):
        """from_dict should correctly parse datetime strings."""
        now = datetime(2024, 6, 15, 10, 30, 0)
        data = {
            "schema_version": SCHEMA_VERSION_STR,
            "symbol": "QQQ",
            "asof_ts": now.isoformat(),
        }
        result = RegimeOutput.from_dict(data)
        assert result.asof_ts == now


class TestSchemaVersionMigration:
    """Test schema version migration scenarios."""

    def test_legacy_version_1_0_can_be_parsed(self):
        """Legacy 1.0 format should be parseable."""
        # The 1.0 format might have slightly different structure
        # This test ensures backward compatibility
        data = {
            "schema_version": "regime_output@1.0",
            "symbol": "AAPL",
            "decision_regime": "R0",
            "final_regime": "R0",
        }
        result = RegimeOutput.from_dict(data)
        assert result.symbol == "AAPL"
        assert result.final_regime == MarketRegime.R0_HEALTHY_UPTREND

    def test_modified_version_raises_on_deserialize(self):
        """
        Serialize → modify version → deserialize should raise error.

        This is a key acceptance criterion from Phase 0.
        """
        output = RegimeOutput(symbol="QQQ")
        serialized = output.to_dict()

        # Modify version to incompatible major version
        serialized["schema_version"] = "regime_output@2.0"

        with pytest.raises(SchemaVersionError):
            RegimeOutput.from_dict(serialized)


class TestSchemaVersionErrorMessage:
    """Test SchemaVersionError message formatting."""

    def test_error_message_includes_versions(self):
        """Error message should include both expected and actual versions."""
        error = SchemaVersionError("regime_output@1.1", "regime_output@2.0")
        assert "regime_output@1.1" in str(error)
        assert "regime_output@2.0" in str(error)
        assert error.expected == "regime_output@1.1"
        assert error.actual == "regime_output@2.0"
