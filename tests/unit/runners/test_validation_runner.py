"""Tests for M2 Validation Runner."""

import json
import tempfile
from pathlib import Path

import pytest

from src.runners.validation_runner import (
    ValidationRunner,
    create_argument_parser,
    main,
)


class TestArgumentParser:
    """Tests for CLI argument parser."""

    def test_parser_creates(self):
        """Test parser is created."""
        parser = create_argument_parser()
        assert parser is not None

    def test_fast_mode_defaults(self):
        """Test fast mode default arguments."""
        parser = create_argument_parser()
        args = parser.parse_args(["fast", "--output", "test.json"])

        assert args.mode == "fast"
        assert "SPY" in args.symbols
        assert args.folds == 2
        assert args.timeframes == ["1d"]
        assert args.horizon_days == 20

    def test_fast_mode_custom_symbols(self):
        """Test fast mode with custom symbols."""
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "fast",
                "--output",
                "test.json",
                "--symbols",
                "AAPL",
                "MSFT",
                "GOOGL",
            ]
        )

        assert args.symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_full_mode_defaults(self):
        """Test full mode default arguments."""
        parser = create_argument_parser()
        args = parser.parse_args(["full", "--output", "test.json"])

        assert args.mode == "full"
        assert args.outer_folds == 5
        assert args.inner_folds == 3
        assert args.inner_trials == 20
        assert args.confirm_rule == "1d&4h"

    def test_full_mode_custom_folds(self):
        """Test full mode with custom fold counts."""
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "full",
                "--output",
                "test.json",
                "--outer-folds",
                "3",
                "--inner-folds",
                "2",
                "--inner-trials",
                "10",
            ]
        )

        assert args.outer_folds == 3
        assert args.inner_folds == 2
        assert args.inner_trials == 10

    def test_holdout_mode(self):
        """Test holdout mode arguments."""
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "holdout",
                "--output",
                "test.json",
                "--params",
                "params.yaml",
            ]
        )

        assert args.mode == "holdout"
        assert args.params == "params.yaml"

    def test_timeframes_argument(self):
        """Test timeframes argument parsing."""
        parser = create_argument_parser()
        args = parser.parse_args(
            [
                "fast",
                "--output",
                "test.json",
                "--timeframes",
                "1d",
                "4h",
                "2h",
                "1h",
            ]
        )

        assert args.timeframes == ["1d", "4h", "2h", "1h"]


class TestValidationRunnerFast:
    """Tests for fast validation mode."""

    def test_fast_mode_runs(self):
        """Test fast mode executes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fast_result.json"

            parser = create_argument_parser()
            args = parser.parse_args(
                [
                    "fast",
                    "--output",
                    str(output_path),
                    "--symbols",
                    "SPY",
                    "QQQ",
                ]
            )

            runner = ValidationRunner(args)
            exit_code = runner.run()

            assert exit_code == 0
            assert output_path.exists()

    def test_fast_mode_output_structure(self):
        """Test fast mode produces valid output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fast_result.json"

            parser = create_argument_parser()
            args = parser.parse_args(
                [
                    "fast",
                    "--output",
                    str(output_path),
                    "--symbols",
                    "AAPL",
                ]
            )

            runner = ValidationRunner(args)
            runner.run()

            with open(output_path) as f:
                output = json.load(f)

            assert output["version"] == "m2_v2.0"
            assert output["mode"] == "fast"
            assert "generated_at" in output
            assert "gate_results" in output
            assert "all_gates_passed" in output
            assert "universe" in output

    def test_fast_mode_gate_results(self):
        """Test fast mode produces expected gates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fast_result.json"

            parser = create_argument_parser()
            args = parser.parse_args(
                [
                    "fast",
                    "--output",
                    str(output_path),
                ]
            )

            runner = ValidationRunner(args)
            runner.run()

            with open(output_path) as f:
                output = json.load(f)

            gate_names = [g["gate_name"] for g in output["gate_results"]]
            assert "trending_r0" in gate_names
            assert "choppy_r0" in gate_names
            assert "causality_g7" in gate_names


class TestValidationRunnerFull:
    """Tests for full validation mode."""

    def test_full_mode_runs(self):
        """Test full mode executes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full_result.json"

            parser = create_argument_parser()
            args = parser.parse_args(
                [
                    "full",
                    "--output",
                    str(output_path),
                ]
            )

            runner = ValidationRunner(args)
            exit_code = runner.run()

            assert exit_code == 0
            assert output_path.exists()

    def test_full_mode_output_structure(self):
        """Test full mode produces valid output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full_result.json"

            parser = create_argument_parser()
            args = parser.parse_args(
                [
                    "full",
                    "--output",
                    str(output_path),
                ]
            )

            runner = ValidationRunner(args)
            runner.run()

            with open(output_path) as f:
                output = json.load(f)

            assert output["version"] == "m2_v2.0"
            assert output["mode"] == "full"
            assert "horizon_config" in output
            assert "split_config" in output
            assert "statistical_result" in output
            assert "earliness_stats_by_tf_pair" in output
            assert "confirmation_result" in output

    def test_full_mode_gate_results(self):
        """Test full mode produces expected gates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full_result.json"

            parser = create_argument_parser()
            args = parser.parse_args(
                [
                    "full",
                    "--output",
                    str(output_path),
                ]
            )

            runner = ValidationRunner(args)
            runner.run()

            with open(output_path) as f:
                output = json.load(f)

            gate_names = [g["gate_name"] for g in output["gate_results"]]
            assert "cohens_d" in gate_names
            assert "p_value" in gate_names
            assert "trending_ci_lower" in gate_names
            assert "choppy_ci_upper" in gate_names


class TestValidationRunnerHoldout:
    """Tests for holdout validation mode."""

    def test_holdout_mode_runs(self):
        """Test holdout mode executes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "holdout_result.json"

            parser = create_argument_parser()
            args = parser.parse_args(
                [
                    "holdout",
                    "--output",
                    str(output_path),
                ]
            )

            runner = ValidationRunner(args)
            exit_code = runner.run()

            assert exit_code == 0
            assert output_path.exists()


class TestMain:
    """Tests for main entry point."""

    def test_no_mode_shows_help(self):
        """Test that no mode argument returns error."""
        exit_code = main([])
        assert exit_code == 1

    def test_fast_mode_via_main(self):
        """Test fast mode via main function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "main_result.json"

            exit_code = main(
                [
                    "fast",
                    "--output",
                    str(output_path),
                    "--symbols",
                    "SPY",
                ]
            )

            assert exit_code == 0
            assert output_path.exists()
