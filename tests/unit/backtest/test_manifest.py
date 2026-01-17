"""
Tests for RunManifest and verification modules (Phase 6: Execution Evidence).

Tests cover:
1. RunManifest creation, serialization, and loading
2. SHA-256 checksum computation
3. Verification with dual tolerance
4. sha256sums.txt generation and verification
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.backtest.core.manifest import (
    MANIFEST_VERSION,
    RunManifest,
    compute_data_fingerprint,
    compute_sha256,
    create_manifest,
    generate_sha256sums,
    get_git_info,
    get_runner_info,
)
from src.backtest.core.verification import (
    VerificationResult,
    verify_checksums,
    verify_metrics_with_tolerance,
    verify_run_reproducibility,
    verify_sha256sums,
)


class TestRunManifest:
    """Tests for RunManifest dataclass."""

    def test_default_manifest(self):
        """Test creating a manifest with defaults."""
        manifest = RunManifest()
        assert manifest.manifest_version == MANIFEST_VERSION
        assert manifest.run_id == ""
        assert manifest.artifact_checksums == {}
        assert manifest.metrics_summary == {}

    def test_manifest_with_values(self):
        """Test creating a manifest with specific values."""
        manifest = RunManifest(
            run_id="test-run-001",
            git_commit="abc123",
            git_dirty=True,
            git_branch="feature/test",
            data_fingerprint="sha256:data123",
            params_fingerprint="sha256:params456",
            artifact_checksums={"trades.parquet": "sha256:trades789"},
            metrics_summary={"sharpe": 1.5, "max_drawdown": 0.1},
        )

        assert manifest.run_id == "test-run-001"
        assert manifest.git_commit == "abc123"
        assert manifest.git_dirty is True
        assert manifest.artifact_checksums["trades.parquet"] == "sha256:trades789"
        assert manifest.metrics_summary["sharpe"] == 1.5

    def test_manifest_to_dict(self):
        """Test manifest serialization to dict."""
        now = datetime.now()
        manifest = RunManifest(
            run_id="test-001",
            started_at=now,
            finished_at=now,
            duration_sec=10.5,
        )

        data = manifest.to_dict()

        assert data["run_id"] == "test-001"
        assert data["started_at"] == now.isoformat()
        assert data["duration_sec"] == 10.5
        assert "artifact_checksums" in data
        assert "metrics_summary" in data

    def test_manifest_from_dict(self):
        """Test manifest deserialization from dict."""
        data = {
            "manifest_version": MANIFEST_VERSION,
            "run_id": "test-002",
            "git_commit": "def456",
            "git_dirty": False,
            "git_branch": "main",
            "data_fingerprint": "sha256:abc",
            "params_fingerprint": "sha256:def",
            "artifact_checksums": {"file.csv": "sha256:file123"},
            "started_at": "2024-01-15T10:30:00",
            "finished_at": "2024-01-15T10:35:00",
            "duration_sec": 300.0,
            "runner_info": {"hostname": "test-host"},
            "metrics_summary": {"cagr": 0.15},
        }

        manifest = RunManifest.from_dict(data)

        assert manifest.run_id == "test-002"
        assert manifest.git_commit == "def456"
        assert manifest.started_at == datetime.fromisoformat("2024-01-15T10:30:00")
        assert manifest.metrics_summary["cagr"] == 0.15

    def test_manifest_save_and_load(self):
        """Test manifest save to file and load from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            original = RunManifest(
                run_id="save-load-test",
                git_commit="xyz789",
                metrics_summary={"sharpe": 2.0},
            )
            original.save(manifest_path)

            loaded = RunManifest.load(manifest_path)

            assert loaded.run_id == original.run_id
            assert loaded.git_commit == original.git_commit
            assert loaded.metrics_summary == original.metrics_summary

    def test_manifest_from_dict_with_none_dates(self):
        """Test manifest deserialization handles None dates."""
        data = {
            "run_id": "test-003",
            "started_at": None,
            "finished_at": None,
        }

        manifest = RunManifest.from_dict(data)

        assert manifest.started_at is None
        assert manifest.finished_at is None


class TestChecksumFunctions:
    """Tests for SHA-256 checksum functions."""

    def test_compute_sha256(self):
        """Test SHA-256 computation for a file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content for hashing")
            filepath = Path(f.name)

        try:
            checksum = compute_sha256(filepath)

            assert checksum.startswith("sha256:")
            assert len(checksum) == 7 + 64  # "sha256:" + 64 hex chars
        finally:
            filepath.unlink()

    def test_compute_sha256_deterministic(self):
        """Test that SHA-256 is deterministic for same content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("deterministic test")
            filepath = Path(f.name)

        try:
            checksum1 = compute_sha256(filepath)
            checksum2 = compute_sha256(filepath)

            assert checksum1 == checksum2
        finally:
            filepath.unlink()

    def test_compute_data_fingerprint(self):
        """Test data fingerprint computation."""
        data = {"key": "value", "number": 42}

        fingerprint = compute_data_fingerprint(data)

        assert fingerprint.startswith("sha256:")
        assert len(fingerprint) == 7 + 64

    def test_compute_data_fingerprint_deterministic(self):
        """Test that fingerprint is deterministic (key order independent)."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}

        fp1 = compute_data_fingerprint(data1)
        fp2 = compute_data_fingerprint(data2)

        assert fp1 == fp2  # sort_keys=True ensures same hash

    def test_generate_sha256sums(self):
        """Test sha256sums.txt generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files
            (tmpdir / "file1.txt").write_text("content 1")
            (tmpdir / "file2.csv").write_text("content 2")

            content = generate_sha256sums(tmpdir)

            assert "file1.txt" in content
            assert "file2.csv" in content
            assert (tmpdir / "sha256sums.txt").exists()

            # Verify format: "hash  filename"
            lines = content.strip().split("\n")
            for line in lines:
                parts = line.split("  ", 1)
                assert len(parts) == 2
                hash_part, filename = parts
                assert len(hash_part) == 64  # SHA-256 hex


class TestVerification:
    """Tests for verification functions."""

    def test_verify_checksums_pass(self):
        """Test checksum verification when all files match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create file
            test_file = tmpdir / "test.txt"
            test_file.write_text("test content")

            # Create manifest with correct checksum
            manifest = RunManifest(
                artifact_checksums={"test.txt": compute_sha256(test_file)}
            )

            passed, errors = verify_checksums(manifest, tmpdir)

            assert passed is True
            assert len(errors) == 0

    def test_verify_checksums_fail_mismatch(self):
        """Test checksum verification fails on mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create file
            test_file = tmpdir / "test.txt"
            test_file.write_text("test content")

            # Create manifest with wrong checksum
            manifest = RunManifest(
                artifact_checksums={"test.txt": "sha256:wronghash123"}
            )

            passed, errors = verify_checksums(manifest, tmpdir)

            assert passed is False
            assert len(errors) == 1
            assert "Checksum mismatch" in errors[0]

    def test_verify_checksums_fail_missing(self):
        """Test checksum verification fails when file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = RunManifest(
                artifact_checksums={"nonexistent.txt": "sha256:abc123"}
            )

            passed, errors = verify_checksums(manifest, Path(tmpdir))

            assert passed is False
            assert "Missing artifact" in errors[0]

    def test_verify_sha256sums_pass(self):
        """Test sha256sums.txt verification passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create files and generate checksums
            (tmpdir / "file1.txt").write_text("content 1")
            (tmpdir / "file2.txt").write_text("content 2")
            generate_sha256sums(tmpdir)

            passed, errors = verify_sha256sums(tmpdir)

            assert passed is True
            assert len(errors) == 0

    def test_verify_sha256sums_fail(self):
        """Test sha256sums.txt verification fails on tampering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create file and generate checksums
            (tmpdir / "file1.txt").write_text("original content")
            generate_sha256sums(tmpdir)

            # Tamper with file
            (tmpdir / "file1.txt").write_text("modified content")

            passed, errors = verify_sha256sums(tmpdir)

            assert passed is False
            assert any("FAILED" in e for e in errors)


class TestDualToleranceVerification:
    """Tests for dual tolerance metric verification (Go/No-Go Gate #7)."""

    def test_verify_metrics_exact_match(self):
        """Test metrics verification with exact match."""
        manifest_metrics = {"sharpe": 1.5, "cagr": 0.12}
        computed_metrics = {"sharpe": 1.5, "cagr": 0.12}

        passed, errors = verify_metrics_with_tolerance(
            manifest_metrics, computed_metrics
        )

        assert passed is True
        assert len(errors) == 0

    def test_verify_metrics_within_abs_tol(self):
        """Test metrics within absolute tolerance."""
        manifest_metrics = {"sharpe": 0.0}  # Near zero
        computed_metrics = {"sharpe": 5e-7}  # Within abs_tol=1e-6

        passed, errors = verify_metrics_with_tolerance(
            manifest_metrics, computed_metrics,
            abs_tol=1e-6, rel_tol=1e-6
        )

        assert passed is True

    def test_verify_metrics_within_rel_tol(self):
        """Test metrics within relative tolerance."""
        manifest_metrics = {"cagr": 0.15}
        # rel_tol=1e-6, so tolerance = 1e-6 + 1e-6 * 0.15 = 1.15e-6
        computed_metrics = {"cagr": 0.15 + 1e-6}

        passed, errors = verify_metrics_with_tolerance(
            manifest_metrics, computed_metrics,
            abs_tol=1e-6, rel_tol=1e-6
        )

        assert passed is True

    def test_verify_metrics_outside_tolerance(self):
        """Test metrics verification fails outside tolerance."""
        manifest_metrics = {"sharpe": 1.5}
        computed_metrics = {"sharpe": 1.6}  # 0.1 diff >> tolerance

        passed, errors = verify_metrics_with_tolerance(
            manifest_metrics, computed_metrics,
            abs_tol=1e-6, rel_tol=1e-6
        )

        assert passed is False
        assert len(errors) == 1
        assert "sharpe" in errors[0]

    def test_verify_metrics_missing_computed(self):
        """Test verification fails when computed metric missing."""
        manifest_metrics = {"sharpe": 1.5, "cagr": 0.12}
        computed_metrics = {"sharpe": 1.5}  # Missing cagr

        passed, errors = verify_metrics_with_tolerance(
            manifest_metrics, computed_metrics
        )

        assert passed is False
        assert "Missing computed metric: cagr" in errors[0]

    def test_dual_tolerance_formula(self):
        """Test the dual tolerance formula explicitly.

        Formula: |computed - manifest| <= abs_tol + rel_tol * |manifest|
        """
        abs_tol = 1e-6
        rel_tol = 1e-6

        # For manifest=100, tolerance = 1e-6 + 1e-6 * 100 = 0.000101
        manifest_metrics = {"value": 100.0}

        # Just within tolerance
        computed_within = {"value": 100.0 + 0.0001}  # diff = 0.0001 < 0.000101
        passed, _ = verify_metrics_with_tolerance(
            manifest_metrics, computed_within, abs_tol, rel_tol
        )
        assert passed is True

        # Just outside tolerance
        computed_outside = {"value": 100.0 + 0.0002}  # diff = 0.0002 > 0.000101
        passed, errors = verify_metrics_with_tolerance(
            manifest_metrics, computed_outside, abs_tol, rel_tol
        )
        assert passed is False


class TestVerifyRunReproducibility:
    """Tests for complete run reproducibility verification."""

    def test_verify_run_full_pass(self):
        """Test full run verification passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create artifacts
            (tmpdir / "trades.parquet").write_bytes(b"trade data")
            (tmpdir / "equity.parquet").write_bytes(b"equity data")

            # Generate checksums
            generate_sha256sums(tmpdir)

            # Create manifest
            manifest = RunManifest(
                run_id="test-run",
                artifact_checksums={
                    "trades.parquet": compute_sha256(tmpdir / "trades.parquet"),
                    "equity.parquet": compute_sha256(tmpdir / "equity.parquet"),
                },
                metrics_summary={"sharpe": 1.5},
            )
            manifest_path = tmpdir / "manifest.json"
            manifest.save(manifest_path)

            # Verify
            result = verify_run_reproducibility(
                manifest_path,
                recomputed_metrics={"sharpe": 1.5},
            )

            assert result.passed is True
            assert len(result.checksum_errors) == 0
            assert len(result.metric_errors) == 0

    def test_verify_run_checksum_fail(self):
        """Test run verification fails on checksum error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create artifact
            (tmpdir / "trades.parquet").write_bytes(b"original")

            # Create manifest with checksum
            manifest = RunManifest(
                artifact_checksums={
                    "trades.parquet": compute_sha256(tmpdir / "trades.parquet"),
                },
            )
            manifest_path = tmpdir / "manifest.json"
            manifest.save(manifest_path)

            # Tamper with artifact
            (tmpdir / "trades.parquet").write_bytes(b"tampered")

            result = verify_run_reproducibility(manifest_path)

            assert result.passed is False
            assert len(result.checksum_errors) > 0

    def test_verify_run_metric_fail(self):
        """Test run verification fails on metric error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            manifest = RunManifest(
                metrics_summary={"sharpe": 1.5},
            )
            manifest_path = tmpdir / "manifest.json"
            manifest.save(manifest_path)

            result = verify_run_reproducibility(
                manifest_path,
                recomputed_metrics={"sharpe": 2.5},  # Different value
            )

            assert result.passed is False
            assert len(result.metric_errors) > 0

    def test_verify_run_git_dirty_warning(self):
        """Test git dirty state produces warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            manifest = RunManifest(git_dirty=True)
            manifest_path = tmpdir / "manifest.json"
            manifest.save(manifest_path)

            result = verify_run_reproducibility(manifest_path)

            assert any("git dirty" in w.lower() for w in result.warnings)


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_verification_result_bool_pass(self):
        """Test VerificationResult bool conversion (pass)."""
        result = VerificationResult(
            passed=True,
            checksum_errors=[],
            metric_errors=[],
            warnings=[],
        )

        assert bool(result) is True

    def test_verification_result_bool_fail(self):
        """Test VerificationResult bool conversion (fail)."""
        result = VerificationResult(
            passed=False,
            checksum_errors=["error"],
            metric_errors=[],
            warnings=[],
        )

        assert bool(result) is False

    def test_verification_result_to_dict(self):
        """Test VerificationResult serialization."""
        result = VerificationResult(
            passed=True,
            checksum_errors=[],
            metric_errors=[],
            warnings=["warning 1"],
        )

        data = result.to_dict()

        assert data["passed"] is True
        assert data["warnings"] == ["warning 1"]


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_git_info(self):
        """Test git info retrieval."""
        info = get_git_info()

        # Should have keys even if git not available
        assert "commit" in info
        assert "dirty" in info
        assert "branch" in info

    def test_get_runner_info(self):
        """Test runner info retrieval."""
        info = get_runner_info()

        assert "hostname" in info
        assert "platform" in info
        assert "python_version" in info
        assert "pid" in info

    def test_create_manifest(self):
        """Test manifest creation from artifacts directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create some artifacts
            (tmpdir / "trades.csv").write_text("trade data")
            (tmpdir / "metrics.json").write_text('{"sharpe": 1.5}')

            manifest = create_manifest(
                run_id="test-run",
                artifacts_dir=tmpdir,
                params={"fast_period": 10, "slow_period": 30},
                metrics_summary={"sharpe": 1.5},
            )

            assert manifest.run_id == "test-run"
            assert "trades.csv" in manifest.artifact_checksums
            assert "metrics.json" in manifest.artifact_checksums
            assert manifest.params_fingerprint.startswith("sha256:")
            assert manifest.metrics_summary["sharpe"] == 1.5
