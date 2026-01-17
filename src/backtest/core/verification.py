"""
Run Verification - Recompute gate with dual tolerance.

Phase 6: Execution Evidence

Verifies that backtest artifacts are:
1. Exactly checksummed (sha256sums match)
2. Reproducible within tolerance (metrics recomputation)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_setup import get_logger

from .manifest import RunManifest, compute_sha256

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Result of manifest verification."""

    passed: bool
    checksum_errors: List[str]  # Files with mismatched checksums
    metric_errors: List[str]  # Metrics outside tolerance
    warnings: List[str]  # Non-fatal issues

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checksum_errors": self.checksum_errors,
            "metric_errors": self.metric_errors,
            "warnings": self.warnings,
        }


def verify_checksums(
    manifest: RunManifest,
    artifacts_dir: Path,
) -> Tuple[bool, List[str]]:
    """
    Verify all artifact checksums match the manifest.

    Args:
        manifest: RunManifest with expected checksums
        artifacts_dir: Directory containing artifacts

    Returns:
        Tuple of (all_matched, list_of_errors)
    """
    artifacts_dir = Path(artifacts_dir)
    errors = []

    for filename, expected_checksum in manifest.artifact_checksums.items():
        artifact_path = artifacts_dir / filename

        if not artifact_path.exists():
            errors.append(f"Missing artifact: {filename}")
            continue

        actual_checksum = compute_sha256(artifact_path)
        if actual_checksum != expected_checksum:
            errors.append(
                f"Checksum mismatch for {filename}: "
                f"expected {expected_checksum[:20]}..., got {actual_checksum[:20]}..."
            )

    return len(errors) == 0, errors


def verify_sha256sums(artifacts_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify artifacts against sha256sums.txt file.

    Equivalent to: sha256sum -c sha256sums.txt

    Args:
        artifacts_dir: Directory containing artifacts and sha256sums.txt

    Returns:
        Tuple of (all_passed, list_of_errors)
    """
    artifacts_dir = Path(artifacts_dir)
    checksums_file = artifacts_dir / "sha256sums.txt"

    if not checksums_file.exists():
        return False, ["sha256sums.txt not found"]

    errors = []

    with open(checksums_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse "hash  filename" format
            parts = line.split("  ", 1)
            if len(parts) != 2:
                errors.append(f"Invalid line in sha256sums.txt: {line}")
                continue

            expected_hash, filename = parts
            artifact_path = artifacts_dir / filename

            if not artifact_path.exists():
                errors.append(f"Missing: {filename}")
                continue

            actual_checksum = compute_sha256(artifact_path)
            actual_hash = actual_checksum.replace("sha256:", "")

            if actual_hash != expected_hash:
                errors.append(f"FAILED: {filename}")
            else:
                logger.debug(f"OK: {filename}")

    return len(errors) == 0, errors


def verify_metrics_with_tolerance(
    manifest_metrics: Dict[str, float],
    computed_metrics: Dict[str, float],
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """
    Verify computed metrics match manifest within dual tolerance.

    Uses the formula: |computed - manifest| <= abs_tol + rel_tol * |manifest|

    This dual tolerance handles:
    - Near-zero values (use abs_tol)
    - Larger values (use rel_tol)
    - Cross-platform/BLAS differences

    Args:
        manifest_metrics: Metrics from the manifest
        computed_metrics: Freshly computed metrics
        abs_tol: Absolute tolerance for near-zero values
        rel_tol: Relative tolerance for larger values

    Returns:
        Tuple of (all_within_tolerance, list_of_errors)
    """
    errors = []

    for metric_name, manifest_value in manifest_metrics.items():
        if metric_name not in computed_metrics:
            errors.append(f"Missing computed metric: {metric_name}")
            continue

        computed_value = computed_metrics[metric_name]

        # Dual tolerance formula
        tolerance = abs_tol + rel_tol * abs(manifest_value)
        diff = abs(computed_value - manifest_value)

        if diff > tolerance:
            errors.append(
                f"{metric_name}: manifest={manifest_value:.8f}, "
                f"computed={computed_value:.8f}, diff={diff:.2e}, "
                f"tolerance={tolerance:.2e}"
            )

    return len(errors) == 0, errors


def verify_run_reproducibility(
    manifest_path: Path,
    artifacts_dir: Optional[Path] = None,
    recomputed_metrics: Optional[Dict[str, float]] = None,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> VerificationResult:
    """
    Complete verification of a backtest run.

    Verifies:
    1. All artifact checksums match (exact)
    2. sha256sums.txt verification passes
    3. Recomputed metrics match within dual tolerance

    Args:
        manifest_path: Path to manifest.json
        artifacts_dir: Directory containing artifacts (defaults to manifest parent)
        recomputed_metrics: Optional freshly computed metrics for tolerance check
        abs_tol: Absolute tolerance for metric comparison
        rel_tol: Relative tolerance for metric comparison

    Returns:
        VerificationResult with pass/fail status and error details
    """
    manifest_path = Path(manifest_path)
    if artifacts_dir is None:
        artifacts_dir = manifest_path.parent

    checksum_errors = []
    metric_errors = []
    warnings = []

    # Load manifest
    try:
        manifest = RunManifest.load(manifest_path)
    except Exception as e:
        return VerificationResult(
            passed=False,
            checksum_errors=[f"Failed to load manifest: {e}"],
            metric_errors=[],
            warnings=[],
        )

    # Verify checksums against manifest
    checksums_ok, checksum_errs = verify_checksums(manifest, artifacts_dir)
    checksum_errors.extend(checksum_errs)

    # Verify sha256sums.txt
    sha256sums_path = artifacts_dir / "sha256sums.txt"
    if sha256sums_path.exists():
        sha256_ok, sha256_errs = verify_sha256sums(artifacts_dir)
        if not sha256_ok:
            checksum_errors.extend(sha256_errs)
    else:
        warnings.append("sha256sums.txt not found - skipping verification")

    # Verify metrics with dual tolerance
    if recomputed_metrics and manifest.metrics_summary:
        metrics_ok, metric_errs = verify_metrics_with_tolerance(
            manifest.metrics_summary,
            recomputed_metrics,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        metric_errors.extend(metric_errs)

    # Git dirty warning
    if manifest.git_dirty:
        warnings.append("Run was executed with uncommitted changes (git dirty)")

    passed = len(checksum_errors) == 0 and len(metric_errors) == 0

    result = VerificationResult(
        passed=passed,
        checksum_errors=checksum_errors,
        metric_errors=metric_errors,
        warnings=warnings,
    )

    if passed:
        logger.info("Verification PASSED")
    else:
        logger.error(
            f"Verification FAILED: {len(checksum_errors)} checksum errors, {len(metric_errors)} metric errors"
        )

    return result


def verify_golden_run(
    golden_dir: Path,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> VerificationResult:
    """
    Verify a golden run directory.

    Golden runs are reference runs used to detect regressions.
    They should have:
    - manifest.json
    - sha256sums.txt
    - All artifact files

    Args:
        golden_dir: Directory containing golden run artifacts
        abs_tol: Absolute tolerance for metric comparison
        rel_tol: Relative tolerance for metric comparison

    Returns:
        VerificationResult
    """
    golden_dir = Path(golden_dir)
    manifest_path = golden_dir / "manifest.json"

    if not manifest_path.exists():
        return VerificationResult(
            passed=False,
            checksum_errors=["manifest.json not found in golden directory"],
            metric_errors=[],
            warnings=[],
        )

    return verify_run_reproducibility(
        manifest_path=manifest_path,
        artifacts_dir=golden_dir,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
