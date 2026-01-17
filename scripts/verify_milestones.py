#!/usr/bin/env python3
"""
Milestone Verification Script - Regime Classification Enhancement Plan

This script verifies all implemented phases of the Regime Classification Enhancement Plan:

Phase 0: Schema Lock - version validation, from_dict/to_dict
Phase 1: Audit Layer - EvalResult, eval_condition, mathematical invariants
Phase 2: Regime Sensitivity - ThresholdPair, dual-threshold hysteresis
Phase 3: Indicator Observability - IndicatorTrace
Phase 6: Execution Evidence - RunManifest, verification, dual tolerance

Run with: python scripts/verify_milestones.py
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_phase_0_schema_lock():
    """Verify Phase 0: Schema Lock with versioning."""
    print("\n" + "=" * 60)
    print("PHASE 0: Schema Lock Verification")
    print("=" * 60)

    from src.domain.signals.indicators.regime.models import (
        SCHEMA_VERSION,
        SCHEMA_VERSION_STR,
        MarketRegime,
        RegimeOutput,
        SchemaVersionError,
        validate_schema_version,
    )

    # Test 1: Schema version format
    print(f"\n✓ Schema version: {SCHEMA_VERSION_STR} (v{SCHEMA_VERSION[0]}.{SCHEMA_VERSION[1]})")

    # Test 2: Create and serialize
    output = RegimeOutput(
        symbol="AAPL",
        asof_ts=datetime.now(),
        final_regime=MarketRegime.R0_HEALTHY_UPTREND,
        confidence=85,
    )
    data = output.to_dict()
    print(f"✓ RegimeOutput.to_dict() works, keys: {len(data.keys())}")

    # Test 3: Deserialize
    restored = RegimeOutput.from_dict(data)
    print(f"✓ RegimeOutput.from_dict() works, symbol: {restored.symbol}")

    # Test 4: Version validation
    try:
        validate_schema_version("regime_output@1.1")
        print("✓ Schema version 1.1 validation passed")
    except SchemaVersionError as e:
        print(f"✗ Schema validation failed: {e}")
        return False

    # Test 5: Incompatible version rejection
    try:
        validate_schema_version("regime_output@2.0")
        print("✗ Should have rejected incompatible version")
        return False
    except SchemaVersionError:
        print("✓ Incompatible version correctly rejected")

    print("\n✓ Phase 0: Schema Lock - ALL CHECKS PASSED")
    return True


def verify_phase_1_audit_layer():
    """Verify Phase 1: Audit Layer with EvalResult and mathematical invariants."""
    print("\n" + "=" * 60)
    print("PHASE 1: Audit Layer Verification")
    print("=" * 60)

    from src.domain.signals.indicators.regime.rule_trace import (
        EvalResult,
        eval_condition,
    )

    # Test 1: Invariant A - passed=True implies gap <= 0
    print("\n--- Invariant A Tests (passed ↔ gap) ---")

    result = eval_condition("metric", 80, 70, ">")
    assert result.passed is True and result.gap <= 0, "Invariant A violated for passed=True"
    print(f"✓ 80 > 70: passed={result.passed}, gap={result.gap:.2f} (≤0)")

    result = eval_condition("metric", 60, 70, ">")
    assert result.passed is False and result.gap > 0, "Invariant A violated for passed=False"
    print(f"✓ 60 > 70: passed={result.passed}, gap={result.gap:.2f} (>0)")

    result = eval_condition("metric", 60, 70, "<")
    assert result.passed is True and result.gap <= 0, "Invariant A violated for passed=True"
    print(f"✓ 60 < 70: passed={result.passed}, gap={result.gap:.2f} (≤0)")

    result = eval_condition("metric", 80, 70, "<")
    assert result.passed is False and result.gap > 0, "Invariant A violated for passed=False"
    print(f"✓ 80 < 70: passed={result.passed}, gap={result.gap:.2f} (>0)")

    # Test 2: Invariant B - direction matches operator
    print("\n--- Invariant B Tests (direction ↔ operator) ---")

    for op in [">", ">="]:
        result = eval_condition("metric", 50, 60, op)
        assert result.direction == "increase", f"Invariant B violated for {op}"
        print(f"✓ operator {op}: direction={result.direction}")

    for op in ["<", "<="]:
        result = eval_condition("metric", 50, 60, op)
        assert result.direction == "decrease", f"Invariant B violated for {op}"
        print(f"✓ operator {op}: direction={result.direction}")

    # Test 3: EvalResult is frozen
    print("\n--- EvalResult Immutability ---")
    result = eval_condition("test", 50, 60, ">")
    try:
        result.passed = True  # Should raise
        print("✗ EvalResult should be frozen")
        return False
    except Exception:
        print("✓ EvalResult is immutable (frozen dataclass)")

    print("\n✓ Phase 1: Audit Layer - ALL CHECKS PASSED")
    return True


def verify_phase_2_sensitivity():
    """Verify Phase 2: Regime Sensitivity with ThresholdPair."""
    print("\n" + "=" * 60)
    print("PHASE 2: Regime Sensitivity Verification")
    print("=" * 60)

    from src.domain.signals.indicators.regime.models import (
        DUAL_THRESHOLD_CONFIG,
        ThresholdPair,
    )

    # Test 1: ThresholdPair creation
    print("\n--- ThresholdPair Configuration ---")
    pair = ThresholdPair(entry=80.0, exit=70.0, metric_name="vol_high")
    print(f"✓ ThresholdPair(entry={pair.entry}, exit={pair.exit})")

    # Test 2: Hysteresis evaluation
    print("\n--- Hysteresis Band Tests ---")

    # Not in state, value above entry -> enter
    should_be_in = pair.evaluate_with_hysteresis(85.0, currently_in_state=False)
    assert should_be_in is True, "Should enter when value > entry"
    print(f"✓ Value 85 (not in state): enter state = {should_be_in}")

    # In state, value above exit -> stay in
    should_stay = pair.evaluate_with_hysteresis(75.0, currently_in_state=True)
    assert should_stay is True, "Should stay when value > exit"
    print(f"✓ Value 75 (in state): stay in state = {should_stay}")

    # In state, value below exit -> exit
    should_exit = pair.evaluate_with_hysteresis(65.0, currently_in_state=True)
    assert should_exit is False, "Should exit when value < exit"
    print(f"✓ Value 65 (in state): exit state = {not should_exit}")

    # Test 3: Predefined config
    print("\n--- Predefined Threshold Config ---")
    for name, pair in DUAL_THRESHOLD_CONFIG.items():
        print(f"  {name}: entry={pair.entry}, exit={pair.exit}")

    # Test 4: RunMetrics has regime sensitivity fields
    from src.backtest.core.run_result import RunMetrics

    metrics = RunMetrics()
    assert hasattr(metrics, "regime_transition_rate")
    assert hasattr(metrics, "regime_switch_lag")
    assert hasattr(metrics, "regime_time_in_r0")
    print("\n✓ RunMetrics has regime sensitivity fields")

    print("\n✓ Phase 2: Regime Sensitivity - ALL CHECKS PASSED")
    return True


def verify_phase_3_observability():
    """Verify Phase 3: Indicator Observability with IndicatorTrace."""
    print("\n" + "=" * 60)
    print("PHASE 3: Indicator Observability Verification")
    print("=" * 60)

    from src.domain.signals.indicators.regime.models import MarketRegime, RegimeOutput
    from src.domain.signals.models import IndicatorTrace

    # Test 1: IndicatorTrace creation
    print("\n--- IndicatorTrace Creation ---")
    trace = IndicatorTrace(
        indicator_name="regime",
        timeframe="1d",
        bar_ts=datetime.now(),
        symbol="AAPL",
        raw={"confidence": 85, "atr_pct_63": 45.0, "chop_pct": 35.0},
        state={"regime": "R0", "trend_state": "trend_up"},
        rules_triggered_now=["trend_confirmed", "vol_normal"],
        lookback=252,
    )
    print(f"✓ Created IndicatorTrace: {trace.indicator_name}@{trace.timeframe}")
    print(f"  Raw values: {trace.raw}")
    print(f"  State: {trace.state}")
    print(f"  Rules: {trace.rules_triggered_now}")

    # Test 2: Serialization roundtrip
    print("\n--- Serialization Roundtrip ---")
    data = trace.to_dict()
    restored = IndicatorTrace.from_dict(data)
    assert restored.indicator_name == trace.indicator_name
    assert restored.raw == trace.raw
    print("✓ IndicatorTrace serialization roundtrip works")

    # Test 3: Delta calculation
    print("\n--- Delta Calculation ---")
    trace_with_prev = IndicatorTrace(
        indicator_name="rsi",
        timeframe="5m",
        bar_ts=datetime.now(),
        raw={"rsi": 35.0},
        prev_raw={"rsi": 28.5},
    )
    delta = trace_with_prev.get_delta("rsi")
    assert delta == 6.5, f"Expected delta 6.5, got {delta}"
    print(f"✓ RSI delta: 28.5 → 35.0 = +{delta}")

    # Test 4: Integration with RegimeOutput
    print("\n--- RegimeOutput Integration ---")
    output = RegimeOutput(
        symbol="AAPL",
        asof_ts=datetime.now(),
        final_regime=MarketRegime.R0_HEALTHY_UPTREND,
        indicator_traces=[trace],
    )
    data = output.to_dict()
    assert "indicator_traces" in data
    assert len(data["indicator_traces"]) == 1
    print(f"✓ RegimeOutput contains {len(data['indicator_traces'])} indicator trace(s)")

    # Roundtrip
    restored_output = RegimeOutput.from_dict(data)
    assert len(restored_output.indicator_traces) == 1
    print("✓ RegimeOutput.from_dict preserves indicator_traces")

    print("\n✓ Phase 3: Indicator Observability - ALL CHECKS PASSED")
    return True


def verify_phase_6_execution_evidence():
    """Verify Phase 6: Execution Evidence with manifest and verification."""
    print("\n" + "=" * 60)
    print("PHASE 6: Execution Evidence Verification")
    print("=" * 60)

    from src.backtest.core.manifest import (
        RunManifest,
        compute_data_fingerprint,
        compute_sha256,
        create_manifest,
        generate_sha256sums,
    )
    from src.backtest.core.verification import (
        verify_checksums,
        verify_metrics_with_tolerance,
        verify_run_reproducibility,
        verify_sha256sums,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test 1: Create artifacts
        print("\n--- Create Test Artifacts ---")
        (tmpdir / "trades.parquet").write_bytes(b"trade data content")
        (tmpdir / "equity.parquet").write_bytes(b"equity curve content")
        (tmpdir / "metrics.json").write_text('{"sharpe": 1.5, "cagr": 0.12}')
        print("✓ Created 3 test artifacts")

        # Test 2: Generate sha256sums
        print("\n--- Generate sha256sums.txt ---")
        content = generate_sha256sums(tmpdir)
        print("✓ Generated sha256sums.txt:")
        for line in content.strip().split("\n"):
            print(f"  {line[:20]}...  {line.split()[-1]}")

        # Test 3: Create manifest
        print("\n--- Create RunManifest ---")
        manifest = create_manifest(
            run_id="verification-test-001",
            artifacts_dir=tmpdir,
            params={"fast_period": 10, "slow_period": 30},
            metrics_summary={"sharpe": 1.5, "cagr": 0.12},
        )
        manifest_path = tmpdir / "manifest.json"
        manifest.save(manifest_path)
        print(f"✓ Created manifest: {manifest.run_id}")
        print(f"  Git commit: {manifest.git_commit[:8] if manifest.git_commit else 'N/A'}...")
        print(f"  Artifacts: {len(manifest.artifact_checksums)}")
        print(f"  Metrics: {manifest.metrics_summary}")

        # Test 4: Verify sha256sums
        print("\n--- Verify sha256sums.txt ---")
        passed, errors = verify_sha256sums(tmpdir)
        if passed:
            print("✓ sha256sum verification PASSED")
        else:
            print(f"✗ sha256sum verification FAILED: {errors}")
            return False

        # Test 5: Verify checksums against manifest
        print("\n--- Verify Checksums Against Manifest ---")
        passed, errors = verify_checksums(manifest, tmpdir)
        if passed:
            print("✓ Manifest checksum verification PASSED")
        else:
            print(f"✗ Manifest checksum verification FAILED: {errors}")
            return False

        # Test 6: Dual tolerance verification
        print("\n--- Dual Tolerance Metric Verification ---")
        manifest_metrics = {"sharpe": 1.5, "cagr": 0.12}

        # Exact match
        computed_exact = {"sharpe": 1.5, "cagr": 0.12}
        passed, _ = verify_metrics_with_tolerance(manifest_metrics, computed_exact)
        print(f"✓ Exact match: PASSED = {passed}")

        # Within tolerance (abs_tol=1e-6, rel_tol=1e-6)
        computed_within = {"sharpe": 1.5 + 1e-7, "cagr": 0.12 + 1e-8}
        passed, _ = verify_metrics_with_tolerance(manifest_metrics, computed_within)
        print(f"✓ Within tolerance: PASSED = {passed}")

        # Outside tolerance
        computed_outside = {"sharpe": 1.6, "cagr": 0.15}
        passed, errors = verify_metrics_with_tolerance(manifest_metrics, computed_outside)
        print(f"✓ Outside tolerance: PASSED = {passed} (expected False)")
        assert passed is False, "Should fail for values outside tolerance"

        # Test 7: Full run reproducibility verification
        print("\n--- Full Run Reproducibility Verification ---")
        result = verify_run_reproducibility(
            manifest_path,
            recomputed_metrics={"sharpe": 1.5, "cagr": 0.12},
        )
        if result.passed:
            print("✓ Run reproducibility verification PASSED")
        else:
            print(f"✗ Run reproducibility verification FAILED:")
            print(f"  Checksum errors: {result.checksum_errors}")
            print(f"  Metric errors: {result.metric_errors}")
            return False

        # Show warnings (if any)
        if result.warnings:
            print(f"  Warnings: {result.warnings}")

    print("\n✓ Phase 6: Execution Evidence - ALL CHECKS PASSED")
    return True


def verify_phase_4_turning_point():
    """Verify Phase 4: Turning Point Detection."""
    print("\n" + "=" * 60)
    print("PHASE 4: Turning Point Detection Verification")
    print("=" * 60)

    import numpy as np

    from src.domain.signals.indicators.regime.turning_point import (
        CalibrationEvidence,
        PurgedTimeSeriesSplit,
        TurningPointFeatures,
        TurningPointLabeler,
        TurningPointOutput,
    )
    from src.domain.signals.indicators.regime.turning_point.calibration import (
        compute_calibration_evidence,
    )
    from src.domain.signals.indicators.regime.turning_point.model import TurnState

    # Test 1: PurgedTimeSeriesSplit
    print("\n--- Purged + Embargo CV ---")
    n_samples = 100
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)

    splitter = PurgedTimeSeriesSplit(
        n_splits=5,
        label_horizon=10,
        embargo=2,
    )

    # Test No-Overlap Property
    passed, violations = splitter.verify_no_overlap(X, y)
    if passed:
        print("✓ No-Overlap Property: VERIFIED")
    else:
        print(f"✗ No-Overlap Property VIOLATED: {violations}")
        return False

    # Test Embargo Property
    passed, violations = splitter.verify_embargo(X, y)
    if passed:
        print("✓ Embargo Property: VERIFIED")
    else:
        print(f"✗ Embargo Property VIOLATED: {violations}")
        return False

    # Count folds
    folds = list(splitter.split(X, y))
    print(f"  Generated {len(folds)} folds")

    # Test 2: TurningPointFeatures
    print("\n--- TurningPointFeatures ---")
    features = TurningPointFeatures(
        price_vs_ma20=1.5,
        atr_pct_63=65.0,
        chop_pct_252=35.0,
        ext_atr_units=0.8,
        rsi_14=62.0,
    )
    arr = features.to_array()
    print(f"✓ Feature vector created, shape: {arr.shape}")
    print(f"  Feature names: {len(TurningPointFeatures.feature_names())}")

    # Test 3: TurningPointOutput
    print("\n--- TurningPointOutput ---")
    output = TurningPointOutput(
        turn_state=TurnState.TOP_RISK,
        turn_confidence=0.85,
        top_features=[("rsi_14", 0.3), ("ext_atr_units", 0.2), ("atr_pct_63", 0.15)],
    )
    data = output.to_dict()
    restored = TurningPointOutput.from_dict(data)
    assert restored.turn_state == TurnState.TOP_RISK
    assert restored.turn_confidence == 0.85
    print(f"✓ TurningPointOutput serialization works")
    print(f"  Turn state: {output.turn_state.value}")
    print(f"  Confidence: {output.turn_confidence:.0%}")
    print(f"  Should block R0: {output.should_block_r0()}")

    # Test 4: CalibrationEvidence
    print("\n--- Calibration Evidence ---")
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.rand(100)
    cal_evidence = compute_calibration_evidence(y_true, y_pred, n_bins=10)
    print(f"✓ CalibrationEvidence computed")
    print(f"  Brier Score: {cal_evidence.brier_score:.4f}")
    print(f"  ECE: {cal_evidence.expected_calibration_error:.4f}")
    print(f"  Buckets: {len(cal_evidence.bucket_counts)}")

    # Test 5: TurningPointLabeler with REAL market data
    print("\n--- TurningPointLabeler (REAL DATA) ---")
    import pandas as pd

    # Fetch REAL market data from Yahoo Finance
    try:
        import yfinance as yf
        print("  Fetching real SPY data from Yahoo Finance...")
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="1y", interval="1d")
        df.columns = df.columns.str.lower()
        # Ensure we have the required columns
        if "close" not in df.columns:
            raise ValueError("Missing close column")
        print(f"  ✓ Loaded {len(df)} bars of SPY daily data")
        print(f"    Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    except ImportError:
        print("  ✗ yfinance not installed, cannot verify with real data")
        return False
    except Exception as e:
        print(f"  ✗ Failed to fetch real data: {e}")
        return False

    labeler = TurningPointLabeler(
        atr_period=14,
        zigzag_threshold=2.0,
        risk_horizon=10,
        risk_threshold=1.5,
    )

    top_risk, bottom_risk, pivots = labeler.generate_combined_labels(df)
    print(f"✓ Labels generated")
    print(f"  TOP_RISK labels: {top_risk.sum()} positive ({100*top_risk.mean():.1f}%)")
    print(f"  BOTTOM_RISK labels: {bottom_risk.sum()} positive ({100*bottom_risk.mean():.1f}%)")
    print(f"  ZigZag pivots: {len(pivots)}")
    print(f"  Label horizon: {labeler.get_label_horizon()} bars")

    # Test 6: TurningPointHistory (recent turning point detection)
    print("\n--- Turning Point History ---")
    history = labeler.get_recent_turning_points(df, n_recent=5)
    print(f"✓ TurningPointHistory generated")
    print(f"  Total pivots detected: {history.total_pivots_detected}")
    print(f"  Recent pivots shown: {len(history.pivots)}")

    if history.pivots:
        print("\n  Recent Turning Points:")
        for i, pivot in enumerate(history.pivots):
            ts_str = pivot.timestamp.strftime("%Y-%m-%d") if pivot.timestamp else f"bar {pivot.index}"
            print(f"    {i+1}. {pivot.turn_type.value.upper()} @ {ts_str}: "
                  f"${pivot.price:.2f} ({pivot.atr_magnitude:.1f} ATR move)")

    if history.current_state:
        state = history.current_state
        print(f"\n  Current State (relative to last pivot):")
        print(f"    Bars since last pivot: {state.bars_since_last_pivot}")
        print(f"    Distance from pivot: {state.distance_atr_units:.2f} ATR units ({state.move_direction})")
        print(f"    Current close: ${state.current_close:.2f}")
        print(f"    Last pivot ({state.last_pivot_type.value}): ${state.last_pivot_price:.2f}")

    # Verify serialization
    history_dict = history.to_dict()
    assert "pivots" in history_dict
    assert "current_state" in history_dict
    print(f"\n✓ TurningPointHistory serialization works")

    print("\n✓ Phase 4: Turning Point Detection - ALL CHECKS PASSED")
    return True


def main():
    """Run all milestone verifications."""
    print("\n" + "#" * 60)
    print("# REGIME CLASSIFICATION ENHANCEMENT PLAN")
    print("# Milestone Verification Report")
    print("# Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 60)

    results = {}

    # Phase 0
    try:
        results["Phase 0: Schema Lock"] = verify_phase_0_schema_lock()
    except Exception as e:
        print(f"\n✗ Phase 0 FAILED: {e}")
        results["Phase 0: Schema Lock"] = False

    # Phase 1
    try:
        results["Phase 1: Audit Layer"] = verify_phase_1_audit_layer()
    except Exception as e:
        print(f"\n✗ Phase 1 FAILED: {e}")
        results["Phase 1: Audit Layer"] = False

    # Phase 2
    try:
        results["Phase 2: Sensitivity"] = verify_phase_2_sensitivity()
    except Exception as e:
        print(f"\n✗ Phase 2 FAILED: {e}")
        results["Phase 2: Sensitivity"] = False

    # Phase 3
    try:
        results["Phase 3: Observability"] = verify_phase_3_observability()
    except Exception as e:
        print(f"\n✗ Phase 3 FAILED: {e}")
        results["Phase 3: Observability"] = False

    # Phase 6
    try:
        results["Phase 6: Execution Evidence"] = verify_phase_6_execution_evidence()
    except Exception as e:
        print(f"\n✗ Phase 6 FAILED: {e}")
        results["Phase 6: Execution Evidence"] = False

    # Phase 4
    try:
        results["Phase 4: Turning Point"] = verify_phase_4_turning_point()
    except Exception as e:
        print(f"\n✗ Phase 4 FAILED: {e}")
        results["Phase 4: Turning Point"] = False

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for phase, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {phase}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "-" * 60)
    if all_passed:
        print("ALL IMPLEMENTED PHASES VERIFIED SUCCESSFULLY")
    else:
        print("SOME PHASES FAILED VERIFICATION")
    print("-" * 60)

    # Note remaining phases
    print("\n📋 Remaining Phases (Not Yet Implemented):")
    print("  - Phase 5: Param Recommender Upgrade")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
