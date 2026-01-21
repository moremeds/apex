"""
Unit tests for BarValidationReport schema (PR-01).

Tests the bar validation logic that solves the "350 vs 252" mystery.
"""

from datetime import datetime

import pytest

from src.domain.signals.schemas import (
    SCHEMA_VERSION_STR,
    BarReduction,
    BarReductionReason,
    BarValidationBuilder,
    BarValidationReport,
    SchemaVersionError,
    SizeBudget,
    validate_schema_version,
)


class TestBarValidationReport:
    """Tests for BarValidationReport frozen dataclass."""

    def test_create_basic_report(self) -> None:
        """Test creating a basic BarValidationReport."""
        report = BarValidationReport(
            symbol="AAPL",
            timeframe="1d",
            requested_bars=550,
            loaded_bars=389,
            usable_bars=340,
            validated_bars=140,
        )

        assert report.symbol == "AAPL"
        assert report.timeframe == "1d"
        assert report.requested_bars == 550
        assert report.loaded_bars == 389
        assert report.usable_bars == 340
        assert report.validated_bars == 140

    def test_report_is_frozen(self) -> None:
        """Test that BarValidationReport is immutable."""
        report = BarValidationReport(symbol="AAPL", timeframe="1d")

        with pytest.raises(AttributeError):
            report.symbol = "MSFT"  # type: ignore

    def test_coverage_pct(self) -> None:
        """Test coverage percentage calculation."""
        report = BarValidationReport(
            requested_bars=550,
            validated_bars=140,
        )

        assert report.coverage_pct == pytest.approx(25.45, rel=0.01)

    def test_coverage_pct_zero_requested(self) -> None:
        """Test coverage percentage when requested_bars is zero."""
        report = BarValidationReport(
            requested_bars=0,
            validated_bars=0,
        )

        assert report.coverage_pct == 0.0

    def test_is_valid_true(self) -> None:
        """Test is_valid returns True for valid report."""
        report = BarValidationReport(
            requested_bars=550,
            loaded_bars=389,
            usable_bars=340,
            validated_bars=88,
            warmup_satisfied=True,
        )

        assert report.is_valid is True

    def test_is_valid_false_invariant_violation(self) -> None:
        """Test is_valid returns False when invariants are violated."""
        # loaded_bars > requested_bars violates invariant
        report = BarValidationReport(
            requested_bars=100,
            loaded_bars=200,
            usable_bars=150,
            validated_bars=50,
            warmup_satisfied=True,
        )

        assert report.is_valid is False

    def test_is_valid_false_warmup_not_satisfied(self) -> None:
        """Test is_valid returns False when warmup not satisfied."""
        report = BarValidationReport(
            requested_bars=550,
            loaded_bars=389,
            usable_bars=340,
            validated_bars=140,
            warmup_satisfied=False,
        )

        assert report.is_valid is False

    def test_reasons_property(self) -> None:
        """Test reasons property returns formatted strings."""
        reductions = (
            BarReduction(
                reason=BarReductionReason.WEEKEND_HOLIDAY,
                bars_removed=161,
                description="Weekends/holidays removed",
            ),
            BarReduction(
                reason=BarReductionReason.NAN_GAP,
                bars_removed=49,
                description="NaN gaps trimmed",
            ),
        )

        report = BarValidationReport(
            symbol="AAPL",
            reductions=reductions,
        )

        reasons = report.reasons
        assert len(reasons) == 2
        assert "Weekends/holidays removed: 161 bars" in reasons
        assert "NaN gaps trimmed: 49 bars" in reasons

    def test_to_dict_serialization(self) -> None:
        """Test JSON serialization via to_dict."""
        now = datetime.now(tz=None)
        reductions = (
            BarReduction(
                reason=BarReductionReason.WARMUP,
                bars_removed=200,
                description="Warmup for SMA(200)",
            ),
        )

        report = BarValidationReport(
            symbol="AAPL",
            timeframe="1d",
            validated_at=now,
            requested_bars=550,
            loaded_bars=389,
            usable_bars=340,
            validated_bars=140,
            reductions=reductions,
            warmup_satisfied=True,
            warmup_required=252,
            warmup_indicator="SMA(200)",
        )

        data = report.to_dict()

        assert data["symbol"] == "AAPL"
        assert data["timeframe"] == "1d"
        assert data["requested_bars"] == 550
        assert data["loaded_bars"] == 389
        assert data["usable_bars"] == 340
        assert data["validated_bars"] == 140
        assert data["warmup_satisfied"] is True
        assert data["warmup_required"] == 252
        assert data["warmup_indicator"] == "SMA(200)"
        assert len(data["reductions"]) == 1
        assert data["reductions"][0]["reason"] == "warmup"

    def test_from_dict_deserialization(self) -> None:
        """Test JSON deserialization via from_dict."""
        data = {
            "schema_version": SCHEMA_VERSION_STR,
            "symbol": "SPY",
            "timeframe": "1d",
            "validated_at": "2025-01-15T10:30:00",
            "requested_bars": 550,
            "loaded_bars": 400,
            "usable_bars": 380,
            "validated_bars": 128,
            "reductions": [
                {
                    "reason": "weekend_holiday",
                    "bars_removed": 150,
                    "description": "Non-trading days",
                }
            ],
            "data_source": "ib",
            "start_date": "2023-07-01T00:00:00",
            "end_date": "2025-01-15T00:00:00",
            "warmup_satisfied": True,
            "warmup_required": 252,
            "warmup_indicator": "SMA(200)",
        }

        report = BarValidationReport.from_dict(data)

        assert report.symbol == "SPY"
        assert report.requested_bars == 550
        assert report.validated_bars == 128
        assert len(report.reductions) == 1
        assert report.reductions[0].bars_removed == 150

    def test_from_dict_invalid_schema(self) -> None:
        """Test from_dict raises error for invalid schema version."""
        data = {
            "schema_version": "invalid@999.0",
            "symbol": "AAPL",
        }

        with pytest.raises(SchemaVersionError):
            BarValidationReport.from_dict(data)

    def test_format_report(self) -> None:
        """Test human-readable format output."""
        reductions = (
            BarReduction(
                reason=BarReductionReason.WEEKEND_HOLIDAY,
                bars_removed=161,
                description="Weekends/holidays removed",
            ),
            BarReduction(
                reason=BarReductionReason.WARMUP,
                bars_removed=200,
                description="Warmup for SMA(200)",
            ),
        )

        report = BarValidationReport(
            symbol="AAPL",
            timeframe="1d",
            requested_bars=550,
            loaded_bars=389,
            usable_bars=340,
            validated_bars=88,
            reductions=reductions,
            warmup_satisfied=True,
            warmup_indicator="SMA(200)",
        )

        formatted = report.format_report()

        assert "AAPL" in formatted
        assert "550" in formatted
        assert "389" in formatted
        assert "340" in formatted
        assert "88" in formatted
        assert "Weekends/holidays removed: 161 bars" in formatted
        assert "Warmup for SMA(200): 200 bars" in formatted


class TestBarValidationBuilder:
    """Tests for BarValidationBuilder mutable builder."""

    def test_basic_builder_flow(self) -> None:
        """Test basic builder usage pattern."""
        builder = BarValidationBuilder(symbol="AAPL", timeframe="1d")
        builder.set_requested_bars(550)
        builder.set_loaded_bars(389, source="ib")
        builder.add_reduction(
            BarReductionReason.WEEKEND_HOLIDAY,
            161,
            "Weekends/holidays removed",
        )
        builder.set_usable_bars(340)
        builder.add_reduction(
            BarReductionReason.NAN_GAP,
            49,
            "NaN gaps trimmed",
        )
        builder.set_validated_bars(140, warmup_required=200, warmup_indicator="SMA(200)")
        builder.add_reduction(
            BarReductionReason.WARMUP,
            200,
            "Warmup for SMA(200)",
        )

        report = builder.build()

        assert report.symbol == "AAPL"
        assert report.timeframe == "1d"
        assert report.requested_bars == 550
        assert report.loaded_bars == 389
        assert report.usable_bars == 340
        assert report.validated_bars == 140
        assert report.data_source == "ib"
        assert len(report.reductions) == 3
        assert report.warmup_satisfied is True  # usable_bars >= warmup_required

    def test_builder_chaining(self) -> None:
        """Test builder methods return self for chaining."""
        builder = BarValidationBuilder(symbol="SPY", timeframe="1d")

        result = (
            builder.set_requested_bars(550)
            .set_loaded_bars(400)
            .set_usable_bars(380)
            .set_validated_bars(128)
        )

        assert result is builder
        report = builder.build()
        assert report.validated_bars == 128

    def test_builder_warmup_satisfied_calculation(self) -> None:
        """Test warmup_satisfied is calculated correctly in builder."""
        # Case 1: warmup satisfied
        builder = BarValidationBuilder(symbol="AAPL", timeframe="1d")
        builder.set_usable_bars(300)
        builder.set_validated_bars(48, warmup_required=252)
        report = builder.build()
        assert report.warmup_satisfied is True

        # Case 2: warmup not satisfied
        builder2 = BarValidationBuilder(symbol="SPY", timeframe="1d")
        builder2.set_usable_bars(200)
        builder2.set_validated_bars(0, warmup_required=252)
        report2 = builder2.build()
        assert report2.warmup_satisfied is False


class TestBarReduction:
    """Tests for BarReduction frozen dataclass."""

    def test_create_reduction(self) -> None:
        """Test creating a BarReduction."""
        reduction = BarReduction(
            reason=BarReductionReason.WARMUP,
            bars_removed=200,
            description="Warmup for SMA(200)",
        )

        assert reduction.reason == BarReductionReason.WARMUP
        assert reduction.bars_removed == 200
        assert reduction.description == "Warmup for SMA(200)"

    def test_reduction_is_frozen(self) -> None:
        """Test that BarReduction is immutable."""
        reduction = BarReduction(
            reason=BarReductionReason.WARMUP,
            bars_removed=200,
            description="Warmup",
        )

        with pytest.raises(AttributeError):
            reduction.bars_removed = 300  # type: ignore

    def test_reduction_to_dict(self) -> None:
        """Test serialization to dict."""
        reduction = BarReduction(
            reason=BarReductionReason.NAN_GAP,
            bars_removed=49,
            description="NaN gaps trimmed",
        )

        data = reduction.to_dict()
        assert data["reason"] == "nan_gap"
        assert data["bars_removed"] == 49
        assert data["description"] == "NaN gaps trimmed"

    def test_reduction_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "reason": "weekend_holiday",
            "bars_removed": 161,
            "description": "Non-trading days",
        }

        reduction = BarReduction.from_dict(data)
        assert reduction.reason == BarReductionReason.WEEKEND_HOLIDAY
        assert reduction.bars_removed == 161


class TestSchemaVersion:
    """Tests for schema version validation."""

    def test_validate_schema_version_valid(self) -> None:
        """Test validating a correct schema version."""
        major, minor = validate_schema_version(SCHEMA_VERSION_STR)
        assert major == 1
        assert minor == 0

    def test_validate_schema_version_invalid_prefix(self) -> None:
        """Test validation fails for wrong prefix."""
        with pytest.raises(SchemaVersionError) as exc_info:
            validate_schema_version("wrong_prefix@1.0")
        assert "mismatch" in str(exc_info.value)

    def test_validate_schema_version_incompatible_major(self) -> None:
        """Test validation fails for incompatible major version."""
        with pytest.raises(SchemaVersionError):
            validate_schema_version("signal_v2@99.0")


class TestSizeBudget:
    """Tests for SizeBudget constants."""

    def test_budget_constants(self) -> None:
        """Test size budget constants are defined."""
        assert SizeBudget.MAX_TOTAL_KB == 200
        assert SizeBudget.MARKET_BUDGET_KB == 8
        assert SizeBudget.SECTORS_BUDGET_KB == 20
        assert SizeBudget.TICKERS_BUDGET_KB == 100
        assert SizeBudget.HIGHLIGHTS_BUDGET_KB == 40
        assert SizeBudget.METADATA_BUDGET_KB == 32

    def test_validate_within_budget(self) -> None:
        """Test validate returns True for within budget."""
        assert SizeBudget.validate("market", 7 * 1024) is True
        assert SizeBudget.validate("tickers", 99 * 1024) is True

    def test_validate_over_budget(self) -> None:
        """Test validate returns False for over budget."""
        assert SizeBudget.validate("market", 10 * 1024) is False
        assert SizeBudget.validate("tickers", 150 * 1024) is False

    def test_validate_unknown_section(self) -> None:
        """Test validate returns False for unknown section."""
        assert SizeBudget.validate("unknown", 1024) is False


class TestBarReductionReason:
    """Tests for BarReductionReason enum."""

    def test_all_reasons_defined(self) -> None:
        """Test all expected reasons are defined."""
        reasons = {r.value for r in BarReductionReason}
        expected = {
            "weekend_holiday",
            "nan_gap",
            "warmup",
            "market_hours",
            "data_quality",
            "missing_data",
        }
        assert reasons == expected
