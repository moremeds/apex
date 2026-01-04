"""Tests for TradingRunner validation gate."""

from unittest.mock import patch

import pytest

from src.runners.trading_runner import (
    TradingRunner,
    StrategyNotValidatedError,
    ManifestLoadError,
    load_strategy_manifest,
)


class TestLoadStrategyManifest:
    """Tests for manifest loading."""

    def test_load_manifest_returns_dict(self):
        """Should load manifest.yaml and return dict."""
        manifest = load_strategy_manifest()
        assert isinstance(manifest, dict)
        assert "strategies" in manifest

    def test_load_manifest_contains_strategies(self):
        """Manifest should contain known strategies."""
        manifest = load_strategy_manifest()
        strategies = manifest.get("strategies", {})
        assert "ma_cross" in strategies
        assert "buy_and_hold" in strategies

    def test_load_manifest_has_validation_section(self):
        """Each strategy should have validation section."""
        manifest = load_strategy_manifest()
        strategies = manifest.get("strategies", {})
        for name, config in strategies.items():
            assert "validation" in config, f"Strategy {name} missing validation"


class TestValidationGateDryRun:
    """Tests for validation gate in dry-run mode."""

    def test_dry_run_always_allowed(self):
        """Dry run should always pass validation gate."""
        runner = TradingRunner(
            strategy_name="unknown_strategy",
            symbols=["AAPL"],
            dry_run=True,
        )
        # Should not raise even for unknown strategy
        runner._check_validation_gate()

    def test_dry_run_skips_manifest_check(self):
        """Dry run should not check manifest."""
        with patch(
            "src.runners.trading_runner.load_strategy_manifest"
        ) as mock_load:
            runner = TradingRunner(
                strategy_name="ma_cross",
                symbols=["AAPL"],
                dry_run=True,
            )
            runner._check_validation_gate()
            mock_load.assert_not_called()


class TestValidationGateLive:
    """Tests for validation gate in live mode."""

    def test_live_unvalidated_raises_error(self):
        """Live trading with unvalidated strategy should raise error."""
        runner = TradingRunner(
            strategy_name="ma_cross",
            symbols=["AAPL"],
            dry_run=False,
        )
        # Default manifest has validated_by_apex: false
        with pytest.raises(StrategyNotValidatedError) as exc_info:
            runner._check_validation_gate()

        assert "LIVE TRADING BLOCKED" in str(exc_info.value)
        assert "ma_cross" in str(exc_info.value)

    def test_live_validated_passes(self):
        """Live trading with validated strategy should pass."""
        # Create mock manifest with validated strategy
        mock_manifest = {
            "strategies": {
                "ma_cross": {
                    "strategy": "...",
                    "validation": {
                        "validated_by_apex": True,
                        "validation_date": "2026-01-04",
                    },
                }
            }
        }

        with patch(
            "src.runners.trading_runner.load_strategy_manifest",
            return_value=mock_manifest,
        ):
            runner = TradingRunner(
                strategy_name="ma_cross",
                symbols=["AAPL"],
                dry_run=False,
            )
            # Should not raise
            runner._check_validation_gate()

    def test_live_unknown_strategy_raises_error(self):
        """Live trading with unknown strategy should raise error."""
        runner = TradingRunner(
            strategy_name="nonexistent_strategy",
            symbols=["AAPL"],
            dry_run=False,
        )
        with pytest.raises(StrategyNotValidatedError):
            runner._check_validation_gate()

    def test_live_missing_validation_section_raises_error(self):
        """Strategy without validation section should raise error."""
        mock_manifest = {
            "strategies": {
                "ma_cross": {
                    "strategy": "...",
                    # No validation section
                }
            }
        }

        with patch(
            "src.runners.trading_runner.load_strategy_manifest",
            return_value=mock_manifest,
        ):
            runner = TradingRunner(
                strategy_name="ma_cross",
                symbols=["AAPL"],
                dry_run=False,
            )
            with pytest.raises(StrategyNotValidatedError):
                runner._check_validation_gate()


class TestValidationGateErrorMessages:
    """Tests for validation gate error messages."""

    def test_error_includes_strategy_name(self):
        """Error message should include strategy name."""
        runner = TradingRunner(
            strategy_name="my_custom_strategy",
            symbols=["AAPL"],
            dry_run=False,
        )
        with pytest.raises(StrategyNotValidatedError) as exc_info:
            runner._check_validation_gate()

        assert "my_custom_strategy" in str(exc_info.value)

    def test_error_includes_instructions(self):
        """Error message should include validation instructions."""
        runner = TradingRunner(
            strategy_name="ma_cross",
            symbols=["AAPL"],
            dry_run=False,
        )
        with pytest.raises(StrategyNotValidatedError) as exc_info:
            runner._check_validation_gate()

        error_msg = str(exc_info.value)
        assert "backtest.runner" in error_msg
        assert "validated_by_apex: true" in error_msg
        assert "--strategy" in error_msg


class TestValidationGateManifestMissing:
    """Tests for when manifest.yaml is missing or malformed."""

    def test_missing_manifest_blocks_live(self):
        """Missing manifest should block live trading (fail-closed)."""
        with patch(
            "src.runners.trading_runner.load_strategy_manifest",
            side_effect=ManifestLoadError("Not found"),
        ):
            runner = TradingRunner(
                strategy_name="ma_cross",
                symbols=["AAPL"],
                dry_run=False,
            )
            with pytest.raises(StrategyNotValidatedError) as exc_info:
                runner._check_validation_gate()

            assert "Cannot verify validation" in str(exc_info.value)

    def test_malformed_yaml_blocks_live(self):
        """Malformed YAML should block live trading."""
        with patch(
            "src.runners.trading_runner.load_strategy_manifest",
            side_effect=ManifestLoadError("Malformed YAML"),
        ):
            runner = TradingRunner(
                strategy_name="ma_cross",
                symbols=["AAPL"],
                dry_run=False,
            )
            with pytest.raises(StrategyNotValidatedError):
                runner._check_validation_gate()

    def test_invalid_strategies_section_blocks_live(self):
        """Non-dict strategies section should block live trading."""
        mock_manifest = {"strategies": "not a dict"}

        with patch(
            "src.runners.trading_runner.load_strategy_manifest",
            return_value=mock_manifest,
        ):
            runner = TradingRunner(
                strategy_name="ma_cross",
                symbols=["AAPL"],
                dry_run=False,
            )
            with pytest.raises(StrategyNotValidatedError) as exc_info:
                runner._check_validation_gate()

            assert "Invalid manifest format" in str(exc_info.value)

    def test_non_dict_validation_section_treated_as_unvalidated(self):
        """Non-dict validation section should be treated as unvalidated."""
        mock_manifest = {
            "strategies": {
                "ma_cross": {
                    "strategy": "...",
                    "validation": "not a dict",
                }
            }
        }

        with patch(
            "src.runners.trading_runner.load_strategy_manifest",
            return_value=mock_manifest,
        ):
            runner = TradingRunner(
                strategy_name="ma_cross",
                symbols=["AAPL"],
                dry_run=False,
            )
            with pytest.raises(StrategyNotValidatedError):
                runner._check_validation_gate()


class TestValidationGateIntegration:
    """Integration tests with actual manifest."""

    def test_all_strategies_have_validation_false_by_default(self):
        """All strategies should start with validated_by_apex: false."""
        manifest = load_strategy_manifest()
        strategies = manifest.get("strategies", {})

        for name, config in strategies.items():
            validation = config.get("validation", {})
            validated = validation.get("validated_by_apex", False)
            # By default, all should be false (safety check)
            assert validated is False, (
                f"Strategy {name} has validated_by_apex=true - "
                "should be false in source control"
            )


class TestStrategyNotValidatedError:
    """Tests for StrategyNotValidatedError exception."""

    def test_exception_is_exception_subclass(self):
        """StrategyNotValidatedError should be an Exception."""
        assert issubclass(StrategyNotValidatedError, Exception)

    def test_exception_can_be_raised_with_message(self):
        """Exception should accept and store message."""
        with pytest.raises(StrategyNotValidatedError) as exc_info:
            raise StrategyNotValidatedError("Test message")

        assert "Test message" in str(exc_info.value)


class TestManifestLoadError:
    """Tests for ManifestLoadError exception."""

    def test_exception_is_exception_subclass(self):
        """ManifestLoadError should be an Exception."""
        assert issubclass(ManifestLoadError, Exception)

    def test_exception_can_be_raised_with_message(self):
        """Exception should accept and store message."""
        with pytest.raises(ManifestLoadError) as exc_info:
            raise ManifestLoadError("Manifest not found")

        assert "Manifest not found" in str(exc_info.value)
