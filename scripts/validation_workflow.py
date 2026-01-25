#!/usr/bin/env python3
"""
M2 Validation Workflow - Conditional parameter update based on gate results.

This script runs the full validation workflow:
1. Optimize detector parameters
2. Run full validation
3. Run holdout validation
4. Check if all gates pass
5. If passed: Update detector config with optimized params
6. Generate reports (with validated params)

Usage:
    python scripts/validation_workflow.py --mode full
    python scripts/validation_workflow.py --mode test  # minimal test
    python scripts/validation_workflow.py --mode test --force  # update params even if gates fail
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file if exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def check_gates_passed(validation_result: Dict[str, Any]) -> bool:
    """Check if all validation gates passed."""
    return validation_result.get("all_gates_passed", False)


def update_detector_params(optimized_params: Dict[str, Any], target_path: Path) -> None:
    """Update detector config with optimized parameters."""
    print(f"\n{'='*60}")
    print("UPDATING DETECTOR PARAMETERS")
    print(f"{'='*60}")

    best_params = optimized_params.get("best_params", {})
    if not best_params:
        print("  No optimized params found, skipping update")
        return

    # Create backup
    if target_path.exists():
        backup_path = target_path.with_suffix(".yaml.backup")
        shutil.copy(target_path, backup_path)
        print(f"  Backup created: {backup_path}")

    # Write updated params
    update_config = {
        "version": "v2.0_optimized",
        "updated_at": datetime.now().isoformat(),
        "params": best_params,
        "optimization_source": optimized_params.get("generated_at", "unknown"),
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f:
        yaml.dump(update_config, f, default_flow_style=False)

    print(f"  Updated detector params: {target_path}")
    print(f"  New params: {best_params}")


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 Validation Workflow")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                       help="Validation mode: test (minimal) or full")
    parser.add_argument("--force", action="store_true",
                       help="Update params even if gates fail")
    parser.add_argument("--skip-optimize", action="store_true",
                       help="Skip optimization step (use existing params)")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/validation"),
                       help="Reports output directory")
    parser.add_argument("--params-output", type=Path,
                       default=Path("config/validation/optimized_params.yaml"),
                       help="Optimized params output path")
    parser.add_argument("--detector-config", type=Path,
                       default=Path("config/validation/detector_params.yaml"),
                       help="Target detector config to update")

    args = parser.parse_args()

    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Mode-specific settings
    if args.mode == "test":
        opt_settings = ["--outer-folds", "2", "--inner-folds", "2", "--inner-trials", "5",
                       "--days", "400", "--max-symbols", "8"]
        val_settings = ["--outer-folds", "2", "--inner-folds", "2", "--inner-trials", "5",
                       "--days", "400", "--max-symbols", "10"]
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    else:
        opt_settings = ["--outer-folds", "3", "--inner-folds", "2", "--inner-trials", "20",
                       "--days", "600", "--max-symbols", "25"]
        val_settings = ["--outer-folds", "3", "--inner-folds", "2", "--inner-trials", "10",
                       "--days", "600", "--max-symbols", "40"]
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "AMZN", "META", "TSLA"]

    prefix = "test_" if args.mode == "test" else ""

    # Step 1: Optimize
    if not args.skip_optimize:
        opt_output = reports_dir / f"{prefix}optimization.json"
        run_command([
            sys.executable, "-m", "src.runners.validation_runner", "optimize",
            "--universe", "config/universe.yaml",
            "--horizon-days", "20",
            "--output", str(opt_output),
            "--params-output", str(args.params_output),
        ] + opt_settings, "Parameter Optimization (Optuna)")

    # Step 2: Full Validation
    val_output = reports_dir / f"{prefix}full_validation.json"
    run_command([
        sys.executable, "-m", "src.runners.validation_runner", "full",
        "--universe", "config/universe.yaml",
        "--timeframes", "1d",
        "--horizon-days", "20",
        "--output", str(val_output),
    ] + val_settings, "Full Validation")

    # Step 3: Holdout Validation
    holdout_output = reports_dir / f"{prefix}holdout_validation.json"
    run_command([
        sys.executable, "-m", "src.runners.validation_runner", "holdout",
        "--universe", "config/universe.yaml",
        "--horizon-days", "20",
        "--days", "500",
        "--output", str(holdout_output),
    ], "Holdout Validation")

    # Step 4: Check Gates
    print(f"\n{'='*60}")
    print("CHECKING GATE RESULTS")
    print(f"{'='*60}")

    validation_result = load_json(val_output)
    holdout_result = load_json(holdout_output)
    optimization_result = load_json(args.params_output.with_suffix(".json") if args.params_output.suffix == ".yaml"
                                    else reports_dir / f"{prefix}optimization.json")

    full_gates_passed = check_gates_passed(validation_result) if validation_result else False
    holdout_causality = holdout_result.get("causality_passed", False) if holdout_result else False

    all_passed = full_gates_passed and holdout_causality

    print(f"  Full validation gates: {'PASS' if full_gates_passed else 'FAIL'}")
    print(f"  Holdout causality: {'PASS' if holdout_causality else 'FAIL'}")
    print(f"  Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")

    # Step 5: Conditional Parameter Update
    if all_passed or args.force:
        if args.force and not all_passed:
            print("\n  WARNING: Forcing parameter update despite gate failures")

        opt_data = load_json(reports_dir / f"{prefix}optimization.json")
        if opt_data:
            update_detector_params(opt_data, args.detector_config)
    else:
        print("\n  Skipping parameter update (gates failed). Use --force to override.")

    # Step 6: Generate Validation Summary Report
    run_command([
        sys.executable, "-m", "src.infrastructure.reporting.validation_report",
        "--reports-dir", str(reports_dir),
        "--output", str(reports_dir / "validation_summary.html"),
    ], "Validation Summary Report")

    # Step 7: Generate Signal Report (with potentially updated params)
    signal_output = reports_dir / "signal_report.html"
    run_command([
        sys.executable, "-m", "src.runners.signal_runner", "--live",
        "--symbols", *symbols,
        "--timeframes", "1d", "4h", "1h",
        "--html-output", str(signal_output),
    ], "Signal Analysis Report")

    # Summary
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETE")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    for f in reports_dir.glob("*"):
        if f.is_file():
            print(f"  {f}")

    print(f"\nGate Status: {'PASSED' if all_passed else 'FAILED'}")
    if all_passed:
        print("  ✓ Detector parameters have been updated")
        print("  ✓ Signal report generated with validated params")
    elif args.force:
        print("  ⚠ Parameters updated despite gate failures (--force)")
    else:
        print("  ✗ Parameters NOT updated (gates failed)")
        print("  Tip: Review validation results or use --force to update anyway")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
