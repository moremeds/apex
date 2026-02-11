"""
End-to-end tests for the signal pipeline validation workflow.

Tests the full flow: signal_runner --validate → validate_gates.py → pass/fail

These tests verify that:
1. signal_runner can generate a report package
2. validate_gates.py can check the package against quality gates
3. The --validate flag properly integrates both steps
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Mark entire module as integration tests
pytestmark = pytest.mark.integration


class TestSignalPipelineE2E:
    """End-to-end tests for the signal pipeline."""

    @pytest.fixture
    def test_output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory for test artifacts."""
        output_dir = tmp_path / "signal_report"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def mock_package(self, test_output_dir: Path) -> Path:
        """
        Create a mock signal package for testing validate_gates.py.

        This bypasses the actual signal generation to test the validation
        pipeline in isolation.
        """
        # Create package structure
        data_dir = test_output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Create summary.json (< 200KB)
        summary = {
            "version": "2.0.0",
            "generated_at": "2026-01-21T10:00:00Z",
            "symbols": ["SPY", "QQQ"],
            "tickers": [
                {
                    "symbol": "SPY",
                    "regime": "R0",
                    "turning_point": {"bars_to_event": 2},
                    "component_values": {"close": 450.0},
                },
                {
                    "symbol": "QQQ",
                    "regime": "R1",
                    "turning_point": {"bars_to_event": -1},
                    "component_values": {"close": 380.0},
                },
            ],
            "market": {"status": "open"},
        }
        (data_dir / "summary.json").write_text(json.dumps(summary))

        # Create symbol data files with required fields
        for symbol in ["SPY", "QQQ"]:
            data = {
                "symbol": symbol,
                "timeframe": "1d",
                "bar_count": 100,
                "chart_data": {"dates": [], "ohlc": []},
            }
            (data_dir / f"{symbol}.json").write_text(json.dumps(data))

        # Create index.html (minimal)
        (test_output_dir / "index.html").write_text("<html><body>Report</body></html>")

        # Create manifest
        manifest = {"config_hash": "abc12345", "version": "2.0.0"}
        (test_output_dir / "manifest.json").write_text(json.dumps(manifest))

        return test_output_dir

    def test_validate_gates_with_mock_package(self, mock_package: Path) -> None:
        """
        Test validate_gates.py can check a mock package.

        This tests the validation pipeline in isolation.
        """
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_gates.py",
                "--all",
                "--package",
                str(mock_package),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should pass (mock package meets requirements)
        assert result.returncode == 0, f"Validation failed: {result.stderr}\n{result.stdout}"
        assert "Overall: PASS" in result.stdout

    def test_validate_gates_single_gate(self, mock_package: Path) -> None:
        """Test running a single gate check."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_gates.py",
                "--gate",
                "G1",
                "--package",
                str(mock_package),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"G1 gate failed: {result.stderr}"
        assert "G1" in result.stdout

    def test_validate_gates_json_output(self, mock_package: Path, tmp_path: Path) -> None:
        """Test JSON output from validate_gates.py."""
        output_file = tmp_path / "results.json"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_gates.py",
                "--all",
                "--package",
                str(mock_package),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0
        assert output_file.exists()

        # Validate JSON structure
        with open(output_file) as f:
            report = json.load(f)

        assert "all_passed" in report
        assert "gates" in report
        assert isinstance(report["gates"], list)
        assert len(report["gates"]) == 13  # G3-G15 (G1/G2 removed)

    def test_validate_gates_fails_on_invalid_close(self, test_output_dir: Path) -> None:
        """Test that G11 fails when close values are <= 0."""
        data_dir = test_output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Create summary with invalid close (0 or negative)
        invalid_data = {"tickers": [{"symbol": "TEST", "component_values": {"close": 0.0}}]}
        (data_dir / "summary.json").write_text(json.dumps(invalid_data))
        (test_output_dir / "index.html").write_text("<html></html>")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_gates.py",
                "--gate",
                "G11",
                "--package",
                str(test_output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail on invalid close
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_validate_gates_fails_on_missing_sections(self, test_output_dir: Path) -> None:
        """Test that G4 fails when required sections are missing."""
        data_dir = test_output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Create summary missing required sections
        (data_dir / "summary.json").write_text("{}")
        (test_output_dir / "index.html").write_text("<html></html>")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_gates.py",
                "--gate",
                "G4",
                "--package",
                str(test_output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail on missing sections
        assert result.returncode == 1
        assert "missing" in result.stdout.lower() or "FAIL" in result.stdout

    def test_signal_runner_no_validate_flag_exists(self) -> None:
        """
        Test that --no-validate flag is recognized by the CLI.

        Validation is automatic with --format package, and --no-validate skips it.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.runners.signal_runner",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should show new report/cache controls
        assert result.returncode == 0
        assert "--no-validate" in result.stdout
        assert "--no-report-cache" in result.stdout
        assert "--training-code-signature" in result.stdout

    def test_makefile_targets_exist(self) -> None:
        """Test that new Makefile targets are defined."""
        makefile_path = Path("Makefile")
        assert makefile_path.exists(), "Makefile not found"

        # Read both Makefile and included validation.mk
        content = makefile_path.read_text()
        validation_mk = Path("make/validation.mk")
        if validation_mk.exists():
            content += validation_mk.read_text()

        expected_targets = [
            "validate-fast",
            "validate-full",
            "validate-holdout",
            "validate-all",
        ]

        for target in expected_targets:
            assert f"{target}:" in content, f"Missing Makefile target: {target}"


class TestValidateGatesImport:
    """Test that validate_gates.py can be imported."""

    def test_import_run_all_gates(self) -> None:
        """Test that run_all_gates function is importable."""
        # This tests that the import path works from signal_runner
        try:
            from scripts.validate_gates import ValidationReport, run_all_gates

            assert callable(run_all_gates)
            assert ValidationReport is not None
        except ImportError as e:
            pytest.fail(f"Failed to import validate_gates: {e}")

    def test_gate_result_dataclass(self) -> None:
        """Test GateResult dataclass structure."""
        from scripts.validate_gates import GateResult

        result = GateResult(
            gate_id="G1",
            gate_name="Summary Size",
            passed=True,
            value=50.0,
            threshold=200.0,
            severity="PASS",
            message="summary.json: 50.0KB",
        )

        assert result.gate_id == "G1"
        assert result.passed is True

        # Test to_dict
        d = result.to_dict()
        assert d["gate_id"] == "G1"
        assert d["passed"] is True
