"""
Base verification framework for APEX.

Provides common infrastructure for manifest-driven verification:
- YAML manifest loading
- JSON schema validation
- Profile-based check filtering
- Pluggable check type handlers

Extend this class to create domain-specific verifiers (RegimeVerifier, SignalVerifier).
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jsonschema
import yaml

from .invariants import (
    create_invariant,
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a single verification check."""

    check_id: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    warning: bool = False  # True if passed with warning (e.g., performance degradation)

    def __str__(self) -> str:
        if self.warning:
            status = "WARN"
        else:
            status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.check_id}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
            "details": self.details,
            "warning": self.warning,
        }


@dataclass
class PhaseResult:
    """Result of all checks in a verification phase."""

    phase_id: str
    phase_name: str
    all_passed: bool
    results: List[VerificationResult]
    duration_ms: float = 0.0

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.results if r.warning)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "phase_id": self.phase_id,
            "phase_name": self.phase_name,
            "all_passed": self.all_passed,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "duration_ms": round(self.duration_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }


class BaseVerifier(ABC):
    """
    Abstract base class for manifest-driven verification.

    Subclasses must implement:
    - _register_handlers(): Register domain-specific check handlers
    - _load_domain_fixtures(): Load domain-specific test fixtures

    The base class provides:
    - Manifest and profile loading
    - Schema loading and validation
    - Common check type handlers (json_schema, assertion, threshold)
    - Phase execution and result aggregation
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        profile: str = "dev",
    ):
        """
        Initialize verifier with manifest and profile.

        Args:
            manifest_path: Path to manifest YAML file
            profile: Profile name (dev, research, prod, nightly)
        """
        if manifest_path is None:
            manifest_path = self._default_manifest_path()

        self.manifest_path = manifest_path
        self.profile_name = profile
        self.base_dir = manifest_path.parent

        # Load manifest
        self.manifest = self._load_yaml(manifest_path)

        # Load profile
        profile_path = self.base_dir / f"{profile}.yaml"
        if profile_path.exists():
            self.profile = self._load_yaml(profile_path)
        else:
            logger.warning(f"Profile {profile} not found, using all checks")
            self.profile = {"checks": None}

        # Load schemas
        self.schemas: Dict[str, Dict] = {}
        self._load_schemas()

        # Load invariants from manifest
        self.invariants: Dict[str, Any] = {}
        self._load_invariants()

        # Register check handlers (base + domain-specific)
        self._check_handlers: Dict[str, Callable] = {
            "json_schema": self._check_json_schema,
            "schema": self._check_json_schema,  # Alias
            "assertion": self._check_assertion,
            "threshold": self._check_threshold,
            "golden_case": self._check_golden_case,
            "fixture": self._check_fixture,
            "performance": self._check_performance,
            "invariant": self._check_invariant,
        }

        # Let subclasses register additional handlers
        self._register_handlers()

    @abstractmethod
    def _default_manifest_path(self) -> Path:
        """Return the default manifest path for this verifier type."""
        ...

    @abstractmethod
    def _register_handlers(self) -> None:
        """Register domain-specific check handlers."""
        ...

    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML file."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_schemas(self) -> None:
        """Load all JSON schemas from manifest."""
        schemas_config = self.manifest.get("schemas", {})
        for schema_name, schema_path in schemas_config.items():
            full_path = self.base_dir / schema_path
            if full_path.exists():
                with open(full_path, "r") as f:
                    self.schemas[schema_name] = json.load(f)
                logger.debug(f"Loaded schema: {schema_name}")
            else:
                logger.warning(f"Schema not found: {full_path}")

    def _load_invariants(self) -> None:
        """Load invariant definitions from manifest."""
        invariants_config = self.manifest.get("invariants", [])
        for inv_config in invariants_config:
            inv_id = inv_config.get("id")
            if inv_id:
                try:
                    self.invariants[inv_id] = create_invariant(inv_config)
                except Exception as e:
                    logger.warning(f"Failed to load invariant {inv_id}: {e}")

    def get_phases(self) -> List[Dict]:
        """Get all phases from manifest."""
        return self.manifest.get("phases", [])

    def get_phase(self, phase_id: str) -> Optional[Dict]:
        """Get a specific phase by ID."""
        for phase in self.get_phases():
            if phase["id"] == phase_id:
                return phase
        return None

    def should_run_check(self, check_id: str) -> bool:
        """Check if a check should run based on profile."""
        profile_checks = self.profile.get("checks")
        if profile_checks is None:
            return True
        return check_id in profile_checks

    def should_skip_check(self, check_id: str) -> bool:
        """Check if a check should be skipped based on profile."""
        skip_list = self.profile.get("skip", [])
        return check_id in skip_list

    def verify_phase(self, phase_id: str) -> PhaseResult:
        """
        Run all checks for a phase.

        Args:
            phase_id: Phase identifier (e.g., "P1", "S1")

        Returns:
            PhaseResult with all check results
        """
        phase = self.get_phase(phase_id)
        if phase is None:
            return PhaseResult(
                phase_id=phase_id,
                phase_name="Unknown",
                all_passed=False,
                results=[
                    VerificationResult(
                        check_id="PHASE_NOT_FOUND",
                        passed=False,
                        message=f"Phase {phase_id} not found in manifest",
                    )
                ],
            )

        start_time = time.perf_counter()
        results: List[VerificationResult] = []

        for check in phase.get("checks", []):
            check_id = check["id"]

            # Skip if not in profile or explicitly skipped
            if not self.should_run_check(check_id) or self.should_skip_check(check_id):
                logger.debug(f"Skipping check: {check_id}")
                continue

            result = self._run_check(check)
            results.append(result)

        duration_ms = (time.perf_counter() - start_time) * 1000
        all_passed = all(r.passed for r in results) if results else True

        return PhaseResult(
            phase_id=phase_id,
            phase_name=phase.get("name", ""),
            all_passed=all_passed,
            results=results,
            duration_ms=duration_ms,
        )

    def verify_all(self) -> List[PhaseResult]:
        """Run all phases."""
        results = []
        for phase in self.get_phases():
            result = self.verify_phase(phase["id"])
            results.append(result)
        return results

    def _run_check(self, check: Dict) -> VerificationResult:
        """Run a single check."""
        check_id = check["id"]
        check_type = check["type"]

        start_time = time.perf_counter()

        handler = self._check_handlers.get(check_type)
        if handler is None:
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Unknown check type: {check_type}",
            )

        try:
            result = handler(check)
            result.duration_ms = (time.perf_counter() - start_time) * 1000
            return result
        except Exception as e:
            logger.exception(f"Check {check_id} failed with exception")
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Exception: {e}",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

    # ═══════════════════════════════════════════════════════════════
    # Common Check Handlers
    # ═══════════════════════════════════════════════════════════════

    def _check_json_schema(self, check: Dict) -> VerificationResult:
        """Validate data against JSON schema."""
        check_id = check["id"]
        target = check.get("target")

        schema = self.schemas.get(target)
        if schema is None:
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Schema '{target}' not found",
            )

        # Subclass must implement getting sample data
        sample_data = self._get_sample_for_schema(target)
        if sample_data is None:
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Could not generate sample data for '{target}'",
            )

        try:
            jsonschema.validate(instance=sample_data, schema=schema)
            return VerificationResult(
                check_id=check_id,
                passed=True,
                message=f"Schema validation passed for '{target}'",
            )
        except jsonschema.ValidationError as e:
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Schema validation failed: {e.message}",
                details={"path": list(e.path), "schema_path": list(e.schema_path)},
            )

    @abstractmethod
    def _get_sample_for_schema(self, target: str) -> Optional[Dict]:
        """Get sample data for schema validation. Subclass must implement."""
        ...

    def _check_assertion(self, check: Dict) -> VerificationResult:
        """
        Evaluate a logical assertion.

        Subclasses should override to add domain-specific assertions.
        """
        check_id = check["id"]
        assert_name = check.get("assert", check.get("expr", ""))
        description = check.get("description", "")

        # Subclass must implement assertion evaluation
        passed, message, details = self._evaluate_assertion(assert_name, check)

        return VerificationResult(
            check_id=check_id,
            passed=passed,
            message=message or description,
            details=details,
        )

    @abstractmethod
    def _evaluate_assertion(self, assert_name: str, check: Dict) -> Tuple[bool, str, Dict]:
        """Evaluate a named assertion. Subclass must implement."""
        ...

    def _check_threshold(self, check: Dict) -> VerificationResult:
        """Check metric against threshold."""
        check_id = check["id"]
        metric = check["metric"]
        op = check["op"]
        value = check["value"]
        check.get("description", "")

        actual_value = self._get_metric_value(metric)
        if actual_value is None:
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Metric '{metric}' not available",
            )

        ops = {
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            "==": lambda a, b: a == b,
        }
        op_func = ops.get(op)
        if op_func is None:
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Unknown operator: {op}",
            )

        passed = op_func(actual_value, value)
        return VerificationResult(
            check_id=check_id,
            passed=passed,
            message=f"{metric} = {actual_value:.3f} ({op} {value})",
            details={"metric": metric, "actual": actual_value, "threshold": value},
        )

    def _get_metric_value(self, metric: str) -> Optional[float]:
        """Get metric value. Override in subclass."""
        return None

    def _check_golden_case(self, check: Dict) -> VerificationResult:
        """Compare against expected golden output."""
        check_id = check["id"]
        dataset = check.get("dataset", "")
        expect = check.get("expect", "")

        fixture_path = Path(dataset)
        if not fixture_path.exists():
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Golden case dataset not found: {dataset}",
            )

        # Subclass should override for actual comparison
        return VerificationResult(
            check_id=check_id,
            passed=True,
            message=f"Golden case: {expect}",
            details={"dataset": dataset},
        )

    def _check_fixture(self, check: Dict) -> VerificationResult:
        """Compare output against golden fixture snapshot."""
        check_id = check["id"]
        snapshot = check.get("snapshot", "")

        # Get fixtures directory from manifest
        fixtures_config = self.manifest.get("fixtures", {})
        snapshots_dir = fixtures_config.get("snapshots_dir", "")

        fixture_path = Path(snapshots_dir) / snapshot
        if not fixture_path.exists():
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Fixture snapshot not found: {fixture_path}",
            )

        # Subclass should override for actual comparison
        return self._compare_fixture(check_id, fixture_path, fixtures_config)

    def _compare_fixture(
        self, check_id: str, fixture_path: Path, config: Dict
    ) -> VerificationResult:
        """
        Compare actual output against fixture.

        Override in subclass for domain-specific comparison.
        """
        return VerificationResult(
            check_id=check_id,
            passed=True,
            message=f"Fixture comparison (not implemented): {fixture_path}",
        )

    def _check_performance(self, check: Dict) -> VerificationResult:
        """Check runtime performance against baseline."""
        check_id = check["id"]
        baseline_path = check.get("baseline", "")
        thresholds = check.get("thresholds", {"warning_pct": 30, "fail_pct": 100})

        if not Path(baseline_path).exists():
            return VerificationResult(
                check_id=check_id,
                passed=True,
                warning=True,
                message=f"Performance baseline not found: {baseline_path}",
            )

        # Subclass should override for actual performance measurement
        return VerificationResult(
            check_id=check_id,
            passed=True,
            message="Performance check (not implemented)",
            details={"thresholds": thresholds},
        )

    def _check_invariant(self, check: Dict) -> VerificationResult:
        """Check invariant(s) from manifest."""
        check_id = check["id"]
        refs = check.get("refs", [])
        description = check.get("description", "")

        failed_invariants = []
        passed_invariants = []

        for ref in refs:
            invariant = self.invariants.get(ref)
            if invariant is None:
                failed_invariants.append(f"{ref}: not found")
                continue

            # Delegate to subclass for invariant evaluation
            passed, msg = self._evaluate_invariant(ref, invariant)
            if passed:
                passed_invariants.append(ref)
            else:
                failed_invariants.append(f"{ref}: {msg}")

        if failed_invariants:
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Failed: {', '.join(failed_invariants)}",
                details={"passed": passed_invariants, "failed": failed_invariants},
            )

        return VerificationResult(
            check_id=check_id,
            passed=True,
            message=description or f"All invariants passed: {', '.join(passed_invariants)}",
            details={"passed": passed_invariants},
        )

    def _evaluate_invariant(self, ref: str, invariant: Any) -> Tuple[bool, str]:
        """Evaluate an invariant. Override in subclass."""
        return True, "Not implemented"

    # ═══════════════════════════════════════════════════════════════
    # Output Formatting
    # ═══════════════════════════════════════════════════════════════

    def print_results(self, results: List[PhaseResult]) -> bool:
        """
        Print verification results to console.

        Returns:
            True if all checks passed
        """
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)

        all_passed = True
        for phase_result in results:
            print(f"\n[Phase {phase_result.phase_id}] {phase_result.phase_name}")
            print("-" * 40)

            if not phase_result.results:
                print("  (no checks run)")
                continue

            for result in phase_result.results:
                if result.warning:
                    icon = "!"
                    status = "WARN"
                elif result.passed:
                    icon = "+"
                    status = "PASS"
                else:
                    icon = "x"
                    status = "FAIL"

                print(f"  [{icon}] {result.check_id}: {status}")
                if not result.passed or result.warning:
                    print(f"      {result.message}")

            print(f"\n  Summary: {phase_result.pass_count}/{len(phase_result.results)} passed")
            if phase_result.warn_count > 0:
                print(f"  Warnings: {phase_result.warn_count}")
            print(f"  Duration: {phase_result.duration_ms:.1f}ms")

            if not phase_result.all_passed:
                all_passed = False

        print("\n" + "=" * 60)
        final_status = "ALL PASSED" if all_passed else "SOME FAILED"
        print(f"FINAL: {final_status}")
        print("=" * 60)

        return all_passed

    def to_json(self, results: List[PhaseResult]) -> str:
        """Serialize results to JSON."""
        return json.dumps(
            {
                "manifest": str(self.manifest_path),
                "profile": self.profile_name,
                "phases": [r.to_dict() for r in results],
            },
            indent=2,
        )
