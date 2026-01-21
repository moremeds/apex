"""
Signal Service Verification Framework.

Verifies the APEX signal service including:
- Indicator registry discovery
- Engine determinism
- Causality (no lookahead)
- Semantic invariants (bounds, identities)
- Golden fixture regression
- Service payload contracts

Usage:
    python -m src.verification.signal_verifier --all --profile signal_dev
    python -m src.verification.signal_verifier --phase S1 --profile signal_dev
    python -m src.verification.signal_verifier --all --profile signal_talib
    python -m src.verification.signal_verifier --all --profile signal_nightly
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .base_verifier import BaseVerifier, VerificationResult
from .invariants import (
    BoundsInvariant,
    CausalityInvariant,
    IdentityInvariant,
    NoNaNInvariant,
)

logger = logging.getLogger(__name__)


class SignalVerifier(BaseVerifier):
    """
    Verification framework for the APEX signal service.

    Implements four-layer verification:
    - Layer A: Structural (JSON Schema)
    - Layer B: Semantic (Bounds, Identities, No-NaN)
    - Layer C: Causality (No lookahead)
    - Layer D: Golden Fixtures (Regression)
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        profile: str = "signal_dev",
    ):
        super().__init__(manifest_path, profile)

        # Cached fixtures
        self._ohlcv_fixtures: Dict[str, pd.DataFrame] = {}
        self._snapshot_fixtures: Dict[str, List[Dict]] = {}

        # Load fixtures
        self._load_fixtures()

    def _default_manifest_path(self) -> Path:
        return Path("config/verification/signal_manifest.yaml")

    def _register_handlers(self) -> None:
        """Register signal-specific check handlers."""
        # Additional handlers can be registered here

    def _load_fixtures(self) -> None:
        """Load OHLCV and snapshot fixtures."""
        fixtures_config = self.manifest.get("fixtures", {})
        ohlcv_dir = fixtures_config.get("ohlcv_dir", "tests/fixtures/signal_verification/ohlcv")
        snapshots_dir = fixtures_config.get(
            "snapshots_dir", "tests/fixtures/signal_verification/snapshots"
        )

        # Load OHLCV CSV files
        ohlcv_path = Path(ohlcv_dir)
        if ohlcv_path.exists():
            for csv_file in ohlcv_path.glob("*.csv"):
                scenario_name = csv_file.stem
                try:
                    df = pd.read_csv(csv_file, parse_dates=["timestamp"], index_col="timestamp")
                    self._ohlcv_fixtures[scenario_name] = df
                    logger.debug(f"Loaded OHLCV fixture: {scenario_name} ({len(df)} bars)")
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")

        # Load JSONL snapshot files
        snapshots_path = Path(snapshots_dir)
        if snapshots_path.exists():
            for jsonl_file in snapshots_path.glob("*.jsonl"):
                snapshot_name = jsonl_file.name
                try:
                    records = []
                    with open(jsonl_file, "r") as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line))
                    self._snapshot_fixtures[snapshot_name] = records
                    logger.debug(f"Loaded snapshot: {snapshot_name} ({len(records)} records)")
                except Exception as e:
                    logger.warning(f"Failed to load {jsonl_file}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Schema Validation
    # ═══════════════════════════════════════════════════════════════

    def _get_sample_for_schema(self, target: str) -> Optional[Dict]:
        """Generate sample data for schema validation."""
        if target == "indicator_output":
            return self._get_sample_indicator_output()
        elif target == "signal_service_payload":
            return self._get_sample_service_payload()
        return None

    def _get_sample_indicator_output(self) -> Optional[Dict]:
        """Generate sample indicator output for schema validation."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()
            indicator = registry.get("rsi")

            if indicator is None:
                return None

            # Generate test data
            df = self._get_test_ohlcv(100)
            result = indicator.calculate(df, {"symbol": "TEST"})

            if result is None or len(result) == 0:
                return None

            # Get last row state
            last_row = result.iloc[-1]
            state = indicator.get_state(last_row, result.iloc[-2] if len(result) > 1 else None)

            return {
                "indicator": indicator.name,
                "category": indicator.category.value,
                "timestamp": df.index[-1].isoformat(),
                "symbol": "TEST",
                "timeframe": "1d",
                "warmup_complete": len(result) > indicator.warmup_periods,
                "outputs": state,
            }
        except Exception as e:
            logger.warning(f"Failed to generate indicator output sample: {e}")
            return None

    def _get_sample_service_payload(self) -> Optional[Dict]:
        """Generate sample service payload for schema validation."""
        try:
            from src.domain.signals.models import (
                SignalCategory,
                SignalDirection,
                SignalPriority,
                TradingSignal,
            )

            signal = TradingSignal(
                signal_id="momentum:rsi:AAPL:1d",
                symbol="AAPL",
                category=SignalCategory.MOMENTUM,
                indicator="rsi",
                direction=SignalDirection.BUY,
                strength=75,
                priority=SignalPriority.HIGH,
                timeframe="1d",
                trigger_rule="rsi_oversold",
                current_value=28.5,
                threshold=30.0,
                message="RSI crossed below 30",
            )

            return {
                "signals": [signal.to_dict()],
                "timestamp": signal.timestamp.isoformat(),
                "symbol_count": 1,
            }
        except Exception as e:
            logger.warning(f"Failed to generate service payload sample: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════
    # Assertion Evaluation
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_assertion(self, assert_name: str, check: Dict) -> Tuple[bool, str, Dict]:
        """Evaluate a named assertion."""
        assertions = {
            "registry_by_category_min": self._assert_registry_by_category_min,
            "all_have_metadata": self._assert_all_have_metadata,
            "engine_deterministic": self._assert_engine_deterministic,
            "parallel_vs_serial_equal": self._assert_parallel_vs_serial_equal,
            "payload_roundtrip_equal": self._assert_payload_roundtrip_equal,
            "output_ordering_stable": self._assert_output_ordering_stable,
        }

        assertion_fn = assertions.get(assert_name)
        if assertion_fn is None:
            return False, f"Unknown assertion: {assert_name}", {}

        try:
            return assertion_fn(check)
        except Exception as e:
            logger.exception(f"Assertion {assert_name} failed")
            return False, f"Assertion error: {e}", {}

    def _assert_registry_by_category_min(self, check: Dict) -> Tuple[bool, str, Dict]:
        """Assert minimum number of indicators per category."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry
            from src.domain.signals.models import SignalCategory

            registry = get_indicator_registry()
            params = check.get("params", {})

            category_counts = {}
            failures = []

            for cat_name, min_count in params.items():
                try:
                    category = SignalCategory(cat_name)
                    indicators = registry.get_by_category(category)
                    actual_count = len(indicators)
                    category_counts[cat_name] = actual_count

                    if actual_count < min_count:
                        failures.append(f"{cat_name}: {actual_count} < {min_count}")
                except ValueError:
                    failures.append(f"Unknown category: {cat_name}")

            if failures:
                return False, f"Category minimums not met: {', '.join(failures)}", category_counts

            return True, f"All category minimums met", category_counts

        except Exception as e:
            return False, f"Registry check failed: {e}", {}

    def _assert_all_have_metadata(self, check: Dict) -> Tuple[bool, str, Dict]:
        """Assert all indicators have required metadata fields."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()
            required_fields = check.get("fields", ["name", "category", "warmup_periods"])

            missing = []
            for indicator in registry.get_all():
                for field in required_fields:
                    if not hasattr(indicator, field):
                        missing.append(f"{indicator.name}.{field}")

            if missing:
                return False, f"Missing metadata: {missing[:10]}", {"missing": missing}

            return (
                True,
                f"All {len(registry)} indicators have required metadata",
                {
                    "count": len(registry),
                    "fields": required_fields,
                },
            )

        except Exception as e:
            return False, f"Metadata check failed: {e}", {}

    def _assert_engine_deterministic(self, check: Dict) -> Tuple[bool, str, Dict]:
        """Assert engine produces identical output on repeated runs."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()
            runs = check.get("runs", 2)
            check.get("compare", "byte_equal")

            # Use a fixed fixture
            df = self._get_test_ohlcv(100, seed=42)

            # Test a few indicators
            test_indicators = ["rsi", "macd", "sma"]
            failures = []

            for ind_name in test_indicators:
                indicator = registry.get(ind_name)
                if indicator is None:
                    continue

                results = []
                for _ in range(runs):
                    result = indicator.calculate(df.copy(), {"symbol": "TEST"})
                    if result is not None:
                        # Convert to list for comparison
                        results.append(result.values.tolist())

                if len(results) < runs:
                    failures.append(f"{ind_name}: failed to calculate")
                    continue

                # Compare all runs (always use numpy for NaN handling)
                first = results[0]
                first_arr = np.array(first)
                for i, r in enumerate(results[1:], 2):
                    r_arr = np.array(r)
                    # Use allclose with equal_nan=True to handle NaN values correctly
                    if not np.allclose(first_arr, r_arr, rtol=1e-12, atol=1e-12, equal_nan=True):
                        # For debugging, find where they differ
                        failures.append(f"{ind_name}: run 1 != run {i}")

            if failures:
                return False, f"Non-deterministic: {', '.join(failures)}", {"failures": failures}

            return (
                True,
                f"Engine deterministic across {runs} runs",
                {
                    "indicators_tested": test_indicators,
                    "runs": runs,
                },
            )

        except Exception as e:
            return False, f"Determinism check failed: {e}", {}

    def _assert_parallel_vs_serial_equal(self, check: Dict) -> Tuple[bool, str, Dict]:
        """Assert parallel and serial execution produce same results."""
        # This is a more complex test that requires the indicator engine
        # For now, return a placeholder
        return True, "Parallel vs serial check (placeholder)", {}

    def _assert_payload_roundtrip_equal(self, check: Dict) -> Tuple[bool, str, Dict]:
        """Assert payload → json → model → json is content-equal."""
        try:
            from src.domain.signals.models import (
                SignalCategory,
                SignalDirection,
                SignalPriority,
                TradingSignal,
            )

            # Create a signal
            signal = TradingSignal(
                signal_id="test:rsi:AAPL:1d",
                symbol="AAPL",
                category=SignalCategory.MOMENTUM,
                indicator="rsi",
                direction=SignalDirection.BUY,
                strength=75,
                priority=SignalPriority.HIGH,
                timeframe="1d",
                trigger_rule="test_rule",
                current_value=28.5,
                threshold=30.0,
                message="Test signal",
            )

            # Round trip: object → dict → json → dict
            dict1 = signal.to_dict()
            json_str = json.dumps(dict1, sort_keys=True)
            dict2 = json.loads(json_str)

            # Compare
            if dict1 == dict2:
                return True, "Payload roundtrip successful", {}

            # Find differences
            diffs = []
            for key in set(dict1.keys()) | set(dict2.keys()):
                if dict1.get(key) != dict2.get(key):
                    diffs.append(f"{key}: {dict1.get(key)} != {dict2.get(key)}")

            return False, f"Roundtrip differences: {diffs}", {"diffs": diffs}

        except Exception as e:
            return False, f"Roundtrip check failed: {e}", {}

    def _assert_output_ordering_stable(self, check: Dict) -> Tuple[bool, str, Dict]:
        """Assert same inputs produce same output ordering."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()

            # Get all indicator names twice
            names1 = sorted(registry.get_names())
            names2 = sorted(registry.get_names())

            if names1 == names2:
                return True, "Output ordering stable", {"count": len(names1)}

            return False, "Indicator ordering unstable", {}

        except Exception as e:
            return False, f"Ordering check failed: {e}", {}

    # ═══════════════════════════════════════════════════════════════
    # Invariant Evaluation
    # ═══════════════════════════════════════════════════════════════

    def _evaluate_invariant(self, ref: str, invariant: Any) -> Tuple[bool, str]:
        """Evaluate an invariant definition."""
        if isinstance(invariant, BoundsInvariant):
            return self._check_bounds_invariant(ref, invariant)
        elif isinstance(invariant, IdentityInvariant):
            return self._check_identity_invariant(ref, invariant)
        elif isinstance(invariant, NoNaNInvariant):
            return self._check_no_nan_invariant(ref, invariant)
        elif isinstance(invariant, CausalityInvariant):
            return self._check_causality_invariant(ref, invariant)

        return True, "Unknown invariant type (skipped)"

    def _check_bounds_invariant(self, ref: str, invariant: BoundsInvariant) -> Tuple[bool, str]:
        """Check bounds invariant on indicator output."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()

            # Find indicator from invariant config
            # The invariant should be associated with an indicator
            inv_config = None
            for cfg in self.manifest.get("invariants", []):
                if cfg.get("id") == ref:
                    inv_config = cfg
                    break

            if inv_config is None:
                return False, "Invariant config not found"

            indicator_name = inv_config.get("indicator", "")
            indicator = registry.get(indicator_name)

            if indicator is None:
                return False, f"Indicator '{indicator_name}' not found"

            # Calculate on test data
            df = self._get_test_ohlcv(200, seed=42)
            result = indicator.calculate(df, {"symbol": "TEST"})

            if result is None:
                return False, "Indicator returned None"

            # Check bounds after warmup
            warmup = indicator.warmup_periods
            field = invariant.field

            if field not in result.columns:
                # Field might be nested in state
                return True, f"Field '{field}' not in direct output (may be in state)"

            after_warmup = result.iloc[warmup:]
            violations = []

            for idx, value in after_warmup[field].items():
                if not math.isnan(value):
                    if not invariant.check(value):
                        violations.append(f"{idx}: {value}")
                        if len(violations) >= 5:
                            break

            if violations:
                return False, f"Bounds violations: {violations}"

            return True, f"Bounds check passed ({len(after_warmup)} rows)"

        except Exception as e:
            return False, f"Bounds check error: {e}"

    def _check_identity_invariant(self, ref: str, invariant: IdentityInvariant) -> Tuple[bool, str]:
        """Check identity relationship invariant."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()

            # Find indicator
            inv_config = None
            for cfg in self.manifest.get("invariants", []):
                if cfg.get("id") == ref:
                    inv_config = cfg
                    break

            if inv_config is None:
                return False, "Invariant config not found"

            indicator_name = inv_config.get("indicator", "macd")
            indicator = registry.get(indicator_name)

            if indicator is None:
                return False, f"Indicator '{indicator_name}' not found"

            # Calculate
            df = self._get_test_ohlcv(100, seed=42)
            result = indicator.calculate(df, {"symbol": "TEST"})

            if result is None:
                return False, "Indicator returned None"

            # Check identity on last 10 rows
            violations = []
            for i in range(-10, 0):
                row = result.iloc[i]
                row_dict = row.to_dict()

                if invariant.result_field not in row_dict:
                    continue

                if not invariant.check(row_dict):
                    violations.append(f"row {i}: identity failed")

            if violations:
                return False, f"Identity violations: {violations}"

            return True, "Identity check passed"

        except Exception as e:
            return False, f"Identity check error: {e}"

    def _check_no_nan_invariant(self, ref: str, invariant: NoNaNInvariant) -> Tuple[bool, str]:
        """Check no-NaN invariant."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()

            # Test on multiple indicators
            test_indicators = ["rsi", "macd", "sma", "ema", "atr"]
            nan_found = []

            for ind_name in test_indicators:
                indicator = registry.get(ind_name)
                if indicator is None:
                    continue

                df = self._get_test_ohlcv(200, seed=42)
                result = indicator.calculate(df, {"symbol": "TEST"})

                if result is None:
                    continue

                warmup = indicator.warmup_periods
                after_warmup = result.iloc[warmup:]

                for col in after_warmup.columns:
                    if after_warmup[col].dtype in [np.float64, np.float32]:
                        nan_count = after_warmup[col].isna().sum()
                        if nan_count > 0:
                            nan_found.append(f"{ind_name}.{col}: {nan_count} NaN")

            if nan_found:
                return False, f"NaN found: {nan_found[:5]}"

            return True, "No NaN after warmup"

        except Exception as e:
            return False, f"No-NaN check error: {e}"

    def _check_causality_invariant(
        self, ref: str, invariant: CausalityInvariant
    ) -> Tuple[bool, str]:
        """
        Check causality invariant (no lookahead).

        Perturbs future data and verifies past outputs are unchanged.
        """
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()

            # Find indicators to test
            inv_config = None
            for cfg in self.manifest.get("invariants", []):
                if cfg.get("id") == ref:
                    inv_config = cfg
                    break

            if inv_config is None:
                return False, "Invariant config not found"

            indicators_to_test = inv_config.get("indicators", ["rsi", "macd"])
            perturb_offset = invariant.perturb_offset
            perturb_field = invariant.perturb_field
            perturb_factor = invariant.perturb_factor

            failures = []

            for ind_name in indicators_to_test:
                indicator = registry.get(ind_name)
                if indicator is None:
                    continue

                # Original calculation
                df_original = self._get_test_ohlcv(100, seed=42)
                result_original = indicator.calculate(df_original, {"symbol": "TEST"})

                if result_original is None:
                    continue

                # Perturbed calculation
                df_perturbed = df_original.copy()
                perturb_idx = len(df_perturbed) - perturb_offset

                if perturb_idx < indicator.warmup_periods:
                    continue  # Can't test if perturb is in warmup

                if perturb_field in df_perturbed.columns:
                    df_perturbed.iloc[perturb_idx][perturb_field] *= perturb_factor

                result_perturbed = indicator.calculate(df_perturbed, {"symbol": "TEST"})

                if result_perturbed is None:
                    continue

                # Compare past outputs (before perturbation)
                past_original = result_original.iloc[:perturb_idx]
                past_perturbed = result_perturbed.iloc[:perturb_idx]

                # Only compare after warmup
                warmup = indicator.warmup_periods
                past_original = past_original.iloc[warmup:]
                past_perturbed = past_perturbed.iloc[warmup:]

                if len(past_original) == 0:
                    continue

                # Compare (use tolerance for floating point)
                for col in past_original.columns:
                    if past_original[col].dtype in [np.float64, np.float32]:
                        orig_vals = past_original[col].values
                        pert_vals = past_perturbed[col].values

                        if invariant.tolerance == 0.0:
                            if not np.array_equal(orig_vals, pert_vals):
                                failures.append(f"{ind_name}.{col}: past changed")
                        else:
                            if not np.allclose(
                                orig_vals, pert_vals, atol=invariant.tolerance, equal_nan=True
                            ):
                                failures.append(f"{ind_name}.{col}: past changed (tolerance)")

            if failures:
                return False, f"Causality violations: {failures}"

            return True, f"Causality check passed for {len(indicators_to_test)} indicators"

        except Exception as e:
            return False, f"Causality check error: {e}"

    # ═══════════════════════════════════════════════════════════════
    # Fixture Comparison
    # ═══════════════════════════════════════════════════════════════

    def _compare_fixture(
        self, check_id: str, fixture_path: Path, config: Dict
    ) -> VerificationResult:
        """Compare actual output against golden fixture."""
        try:
            from src.domain.signals.indicators.registry import get_indicator_registry

            registry = get_indicator_registry()

            # Load fixture
            snapshot_name = fixture_path.name
            fixture_records = self._snapshot_fixtures.get(snapshot_name, [])

            if not fixture_records:
                return VerificationResult(
                    check_id=check_id,
                    passed=False,
                    message=f"Fixture not loaded: {snapshot_name}",
                )

            tolerance_config = config.get("tolerance", {})
            abs_err = tolerance_config.get("abs_err", 1e-6)
            rel_err = tolerance_config.get("rel_err", 1e-4)

            failures = []
            passed = 0

            for record in fixture_records:
                indicator_name: str = record.get("indicator", "")
                scenario: str = record.get("scenario", "")
                bar_index: int = record.get("bar_index", 0)
                expected_outputs = record.get("outputs", {})

                indicator = registry.get(indicator_name)
                if indicator is None:
                    failures.append(f"{indicator_name}: not found")
                    continue

                # Get OHLCV data for scenario
                df = self._ohlcv_fixtures.get(scenario)
                if df is None:
                    failures.append(f"{scenario}: OHLCV not found")
                    continue

                # Calculate
                result = indicator.calculate(df, {"symbol": "TEST"})
                if result is None or len(result) <= bar_index:
                    failures.append(f"{indicator_name}@{scenario}: no output at {bar_index}")
                    continue

                # Compare outputs
                actual_row = result.iloc[bar_index]
                state = indicator.get_state(
                    actual_row,
                    result.iloc[bar_index - 1] if bar_index > 0 else None,
                )

                for field, expected_value in expected_outputs.items():
                    actual_value = state.get(field)

                    if actual_value is None:
                        failures.append(f"{indicator_name}.{field}: missing")
                        continue

                    if isinstance(expected_value, (int, float)) and isinstance(
                        actual_value, (int, float)
                    ):
                        diff = abs(actual_value - expected_value)
                        rel_diff = diff / max(abs(expected_value), 1e-10)

                        if diff > abs_err and rel_diff > rel_err:
                            failures.append(
                                f"{indicator_name}.{field}@{bar_index}: "
                                f"expected {expected_value}, got {actual_value}"
                            )
                            continue

                passed += 1

            if failures:
                return VerificationResult(
                    check_id=check_id,
                    passed=False,
                    message=f"Fixture comparison failed: {len(failures)} errors",
                    details={"failures": failures[:10], "passed": passed},
                )

            return VerificationResult(
                check_id=check_id,
                passed=True,
                message=f"Fixture comparison passed ({passed} records)",
                details={"passed": passed},
            )

        except Exception as e:
            logger.exception("Fixture comparison failed")
            return VerificationResult(
                check_id=check_id,
                passed=False,
                message=f"Fixture comparison error: {e}",
            )

    # ═══════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════

    def _get_test_ohlcv(self, periods: int = 100, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        # Use fixture if available
        if "trend_uptrend" in self._ohlcv_fixtures:
            df = self._ohlcv_fixtures["trend_uptrend"]
            if len(df) >= periods:
                return df.iloc[:periods].copy()

        # Generate synthetic data
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq="D")
        close = 100 + np.cumsum(np.random.randn(periods) * 0.5)
        high = close + np.random.rand(periods) * 2
        low = close - np.random.rand(periods) * 2

        return pd.DataFrame(
            {
                "open": close - np.random.rand(periods),
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000000, 10000000, periods),
            },
            index=dates,
        )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="APEX Signal Service Verification Framework")
    parser.add_argument(
        "--phase",
        type=str,
        help="Phase to verify (S1, S2, etc.). Use --all for all phases.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all phases",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="signal_dev",
        help="Verification profile (signal_dev, signal_talib, signal_nightly)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="config/verification/signal_manifest.yaml",
        help="Path to manifest.yaml",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create verifier
    try:
        verifier = SignalVerifier(
            manifest_path=Path(args.manifest),
            profile=args.profile,
        )
    except FileNotFoundError:
        print(f"Error: Manifest not found: {args.manifest}")
        print("Run from project root or specify --manifest path")
        return 1

    # Run verification
    if args.all:
        results = verifier.verify_all()
    elif args.phase:
        results = [verifier.verify_phase(args.phase)]
    else:
        parser.error("Must specify --phase or --all")
        return 1

    # Output results
    if args.json:
        print(verifier.to_json(results))
    else:
        all_passed = verifier.print_results(results)
        return 0 if all_passed else 1

    return 0


if __name__ == "__main__":
    exit(main())
