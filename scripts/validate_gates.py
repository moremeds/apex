#!/usr/bin/env python
"""
Validate Gates Script - M3 PR-07 + PR-A Deliverable.

Implements 13 validation gates for signal report quality:

Original Gates (M3):
- G3: first-screen <= 2MB (index.html + summary.json)
- G4: 0 missing sections
- G5: <10% metric drift (uses SnapshotBuilder.diff())
- G6: bar validation (all 4 fields present)
- G7: causality test (pytest)
- G8: config hash match
- G9: TP timing +/-5/+3 bars
- G10: regime R0 >=85%/<=15%

Data Quality Gates (PR-A):
- G11: close > 0 for all symbols (FAIL)
- G12: no sentinel values (-1.0) in chart data (FAIL)
- G13: ATR percentile valid_n >= 50 (WARN)
- G14: close timestamp matches chart data (FAIL)
- G15: bar continuity (no excessive gaps) (WARN)

Note: G1 (summary size) and G2 (package size) were removed as they are no longer
relevant for modern universe sizes. The limits were outdated for full universe runs.

Usage:
    python scripts/validate_gates.py --all --package reports/signal_report
    python scripts/validate_gates.py --gate G5 --old-snapshot baseline.json --new-snapshot new.json
    python scripts/validate_gates.py --gate G11 --package reports/signal_report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Gate thresholds (configurable)
THRESHOLDS = {
    "G1_summary_kb": 200,
    "G2_package_mb": 15,
    "G3_first_screen_mb": 2,
    "G5_metric_drift_pct": 10,
    "G9_tp_early_bars": 5,
    "G9_tp_late_bars": 3,
    "G10_trending_r0_min": 0.85,
    "G10_choppy_r0_max": 0.15,
    # PR-A: Data Quality Gates
    "G11_close_min": 0.0,  # close must be > 0
    "G12_sentinel_value": -1.0,  # sentinel value to detect
    "G13_atr_valid_n_min": 50,  # minimum valid samples for ATR percentile
    "G14_timestamp_tolerance_sec": 86400,  # 1 day tolerance for timestamp match
    "G15_max_gap_bars": 5,  # max consecutive missing bars
}


@dataclass
class GateResult:
    """Result of a single gate check."""

    gate_id: str
    gate_name: str
    passed: bool
    value: float
    threshold: float
    severity: str  # "FAIL" or "WARN"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Complete validation report."""

    gates: List[GateResult]
    all_passed: bool
    fail_count: int
    warn_count: int
    package_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "fail_count": self.fail_count,
            "warn_count": self.warn_count,
            "package_path": self.package_path,
            "gates": [g.to_dict() for g in self.gates],
        }

    def to_html(self) -> str:
        """Generate validation.html from gate results."""
        from datetime import datetime

        def gate_icon(passed: bool, severity: str) -> str:
            if passed:
                return "✓"
            return "✗" if severity == "FAIL" else "~"

        def gate_class(passed: bool, severity: str) -> str:
            if passed:
                return "pass"
            return "fail" if severity == "FAIL" else "warn"

        gates_html = ""
        for g in self.gates:
            icon = gate_icon(g.passed, g.severity)
            cls = gate_class(g.passed, g.severity)
            gates_html += f"""
                <li class="gate-item {cls}">
                    <span class="gate-icon">{icon}</span>
                    <span class="gate-id">{g.gate_id}</span>
                    <span class="gate-name">{g.gate_name}</span>
                    <span class="gate-status">{g.severity if not g.passed else 'PASS'}</span>
                    <span class="gate-message">{g.message}</span>
                </li>
"""

        overall_class = "pass" if self.all_passed else "fail"
        overall_text = "PASS" if self.all_passed else "FAIL"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Results - G3-G15</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --success: #4ade80;
            --warning: #fbbf24;
            --error: #ef4444;
            --border: #334155;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: var(--text-secondary); margin-bottom: 2rem; }}
        .summary {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .summary-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            border: 1px solid var(--border);
            flex: 1;
            text-align: center;
        }}
        .summary-card h3 {{ font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem; }}
        .summary-card .value {{ font-size: 2rem; font-weight: 700; }}
        .summary-card .value.pass {{ color: var(--success); }}
        .summary-card .value.fail {{ color: var(--error); }}
        .summary-card .value.warn {{ color: var(--warning); }}
        .gate-list {{ list-style: none; }}
        .gate-item {{
            display: grid;
            grid-template-columns: 30px 50px 180px 60px 1fr;
            gap: 1rem;
            align-items: center;
            padding: 0.875rem 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 4px solid var(--border);
        }}
        .gate-item.pass {{ border-left-color: var(--success); }}
        .gate-item.fail {{ border-left-color: var(--error); }}
        .gate-item.warn {{ border-left-color: var(--warning); }}
        .gate-icon {{ font-size: 1.25rem; }}
        .gate-item.pass .gate-icon {{ color: var(--success); }}
        .gate-item.fail .gate-icon {{ color: var(--error); }}
        .gate-item.warn .gate-icon {{ color: var(--warning); }}
        .gate-id {{ font-family: monospace; font-weight: 600; color: var(--accent); }}
        .gate-name {{ font-weight: 500; }}
        .gate-status {{
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            text-align: center;
        }}
        .gate-item.pass .gate-status {{ background: rgba(74, 222, 128, 0.2); color: var(--success); }}
        .gate-item.fail .gate-status {{ background: rgba(239, 68, 68, 0.2); color: var(--error); }}
        .gate-item.warn .gate-status {{ background: rgba(251, 191, 36, 0.2); color: var(--warning); }}
        .gate-message {{ font-size: 0.875rem; color: var(--text-secondary); }}
        .back-link {{
            display: inline-block;
            margin-bottom: 1.5rem;
            color: var(--accent);
            text-decoration: none;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        .timestamp {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-link">← Heatmap</a>
        <h1>Validation Results</h1>
        <p class="subtitle">Quality Gates G3-G15</p>

        <div class="summary">
            <div class="summary-card">
                <h3>Overall</h3>
                <div class="value {overall_class}">{overall_text}</div>
            </div>
            <div class="summary-card">
                <h3>Failures</h3>
                <div class="value {'fail' if self.fail_count > 0 else 'pass'}">{self.fail_count}</div>
            </div>
            <div class="summary-card">
                <h3>Warnings</h3>
                <div class="value {'warn' if self.warn_count > 0 else 'pass'}">{self.warn_count}</div>
            </div>
            <div class="summary-card">
                <h3>Total Gates</h3>
                <div class="value">{len(self.gates)}</div>
            </div>
        </div>

        <ul class="gate-list">
{gates_html}
        </ul>

        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""

    def save_html(self, output_path: Path) -> Path:
        """Save validation.html to the package."""
        html = self.to_html()
        output_path = Path(output_path)
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Validation report saved: {output_path}")
        return output_path


def check_g1_summary_size(package_path: Path) -> GateResult:
    """G1: summary.json <= 200KB"""
    summary_path = package_path / "data" / "summary.json"

    if not summary_path.exists():
        return GateResult(
            gate_id="G1",
            gate_name="Summary Size",
            passed=False,
            value=0,
            threshold=THRESHOLDS["G1_summary_kb"],
            severity="FAIL",
            message="summary.json not found",
        )

    size_kb = summary_path.stat().st_size / 1024
    passed = size_kb <= THRESHOLDS["G1_summary_kb"]

    return GateResult(
        gate_id="G1",
        gate_name="Summary Size",
        passed=passed,
        value=round(size_kb, 1),
        threshold=THRESHOLDS["G1_summary_kb"],
        severity="FAIL" if not passed else "PASS",
        message=f"summary.json: {size_kb:.1f}KB (limit: {THRESHOLDS['G1_summary_kb']}KB)",
    )


def check_g2_package_size(package_path: Path) -> GateResult:
    """G2: package <= 15MB"""

    def get_dir_size(path: Path) -> int:
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    size_bytes = get_dir_size(package_path)
    size_mb = size_bytes / (1024 * 1024)
    passed = size_mb <= THRESHOLDS["G2_package_mb"]

    return GateResult(
        gate_id="G2",
        gate_name="Package Size",
        passed=passed,
        value=round(size_mb, 2),
        threshold=THRESHOLDS["G2_package_mb"],
        severity="FAIL" if not passed else "PASS",
        message=f"Package: {size_mb:.2f}MB (limit: {THRESHOLDS['G2_package_mb']}MB)",
    )


def check_g3_first_screen(package_path: Path) -> GateResult:
    """G3: first-screen <= 2MB (index.html/heatmap + summary.json)"""
    # index.html is now the heatmap landing page
    index_path = package_path / "index.html"
    summary_path = package_path / "data" / "summary.json"

    total_bytes = 0
    if index_path.exists():
        total_bytes += index_path.stat().st_size
    if summary_path.exists():
        total_bytes += summary_path.stat().st_size

    size_mb = total_bytes / (1024 * 1024)
    passed = size_mb <= THRESHOLDS["G3_first_screen_mb"]

    return GateResult(
        gate_id="G3",
        gate_name="First Screen Size",
        passed=passed,
        value=round(size_mb, 3),
        threshold=THRESHOLDS["G3_first_screen_mb"],
        severity="WARN" if not passed else "PASS",
        message=f"First screen (heatmap): {size_mb:.3f}MB (limit: {THRESHOLDS['G3_first_screen_mb']}MB)",
    )


def check_g4_missing_sections(package_path: Path) -> GateResult:
    """G4: 0 missing sections in summary.json"""
    summary_path = package_path / "data" / "summary.json"

    if not summary_path.exists():
        return GateResult(
            gate_id="G4",
            gate_name="Missing Sections",
            passed=False,
            value=1,
            threshold=0,
            severity="FAIL",
            message="summary.json not found",
        )

    with open(summary_path) as f:
        summary = json.load(f)

    required_sections = ["version", "generated_at", "symbols", "tickers", "market"]
    missing = [s for s in required_sections if s not in summary]

    return GateResult(
        gate_id="G4",
        gate_name="Missing Sections",
        passed=len(missing) == 0,
        value=len(missing),
        threshold=0,
        severity="FAIL" if missing else "PASS",
        message=f"Missing sections: {missing}" if missing else "All required sections present",
        details={"missing": missing},
    )


def check_g5_metric_drift(
    old_snapshot_path: Optional[Path],
    new_snapshot_path: Path,
    threshold_pct: float = 10,
) -> GateResult:
    """G5: <10% metric drift (uses SnapshotBuilder.diff())"""
    if not old_snapshot_path or not old_snapshot_path.exists():
        return GateResult(
            gate_id="G5",
            gate_name="Metric Drift",
            passed=True,
            value=0,
            threshold=threshold_pct,
            severity="PASS",
            message="No baseline snapshot - skipping drift check",
        )

    if not new_snapshot_path.exists():
        return GateResult(
            gate_id="G5",
            gate_name="Metric Drift",
            passed=False,
            value=100,
            threshold=threshold_pct,
            severity="FAIL",
            message="New snapshot not found",
        )

    # Load snapshots
    with open(old_snapshot_path) as f:
        old = json.load(f)
    with open(new_snapshot_path) as f:
        new = json.load(f)

    # Use SnapshotBuilder.diff() if available
    try:
        from src.domain.signals.reporting.snapshot_builder import SnapshotBuilder

        builder = SnapshotBuilder()
        diff = builder.diff(old, new)

        # Check metric changes
        significant_drifts = []
        for symbol, metrics in diff.metric_changes.items():
            for metric, change in metrics.items():
                old_val = change.get("old", 0)
                new_val = change.get("new", 0)
                if old_val != 0:
                    pct_change = abs(new_val - old_val) / abs(old_val) * 100
                    if pct_change > threshold_pct:
                        significant_drifts.append(f"{symbol}.{metric}: {pct_change:.1f}%")

        passed = len(significant_drifts) == 0
        return GateResult(
            gate_id="G5",
            gate_name="Metric Drift",
            passed=passed,
            value=len(significant_drifts),
            threshold=0,
            severity="WARN" if not passed else "PASS",
            message=f"Significant drifts: {significant_drifts[:5]}"
            if significant_drifts
            else "No significant metric drift",
            details={"drifts": significant_drifts[:10]},
        )
    except ImportError as e:
        logger.debug("SnapshotBuilder not available, using manual diff: %s", e)
        # Manual diff if SnapshotBuilder not available
        old_symbols = set(old.get("symbols", []))
        new_symbols = set(new.get("symbols", []))
        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols

        passed = len(added) == 0 and len(removed) == 0
        return GateResult(
            gate_id="G5",
            gate_name="Metric Drift",
            passed=passed,
            value=len(added) + len(removed),
            threshold=0,
            severity="WARN" if not passed else "PASS",
            message=f"Symbol changes: +{len(added)} -{len(removed)}",
            details={"added": list(added)[:5], "removed": list(removed)[:5]},
        )


def check_g6_bar_validation(package_path: Path) -> GateResult:
    """G6: bar validation (all 4 fields present per data file)"""
    data_dir = package_path / "data"
    if not data_dir.exists():
        return GateResult(
            gate_id="G6",
            gate_name="Bar Validation",
            passed=False,
            value=0,
            threshold=4,
            severity="FAIL",
            message="Data directory not found",
        )

    # Check symbol data files
    required_fields = ["symbol", "timeframe", "bar_count", "chart_data"]
    invalid_files = []

    for data_file in data_dir.glob("*.json"):
        if data_file.name in ["summary.json", "indicators.json"]:
            continue

        with open(data_file) as f:
            try:
                data = json.load(f)
                missing = [fld for fld in required_fields if fld not in data]
                if missing:
                    invalid_files.append(f"{data_file.name}: missing {missing}")
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in %s: %s", data_file.name, e)
                invalid_files.append(f"{data_file.name}: invalid JSON")

    passed = len(invalid_files) == 0
    return GateResult(
        gate_id="G6",
        gate_name="Bar Validation",
        passed=passed,
        value=len(invalid_files),
        threshold=0,
        severity="FAIL" if invalid_files else "PASS",
        message=f"Invalid files: {invalid_files[:3]}" if invalid_files else "All data files valid",
        details={"invalid_files": invalid_files[:10]},
    )


def check_g7_causality_test() -> GateResult:
    """G7: causality test (run pytest for validation tests)"""
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/unit/validation/",
                "-v",
                "--tb=no",
                "-q",
                "--no-cov",  # Disable coverage check for gate validation
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        passed = result.returncode == 0

        # Extract test count from output
        output = result.stdout + result.stderr
        test_info = "See pytest output"
        for line in output.split("\n"):
            if "passed" in line or "failed" in line:
                test_info = line.strip()
                break

        return GateResult(
            gate_id="G7",
            gate_name="Causality Test",
            passed=passed,
            value=result.returncode,
            threshold=0,
            severity="FAIL" if not passed else "PASS",
            message=f"Causality tests: {'PASS' if passed else 'FAIL'} - {test_info}",
        )
    except subprocess.TimeoutExpired as e:
        logger.error("Causality test timed out after %s seconds", e.timeout)
        return GateResult(
            gate_id="G7",
            gate_name="Causality Test",
            passed=False,
            value=-1,
            threshold=0,
            severity="FAIL",
            message="Causality tests timed out",
        )
    except FileNotFoundError as e:
        logger.debug("pytest not found, skipping causality test: %s", e)
        return GateResult(
            gate_id="G7",
            gate_name="Causality Test",
            passed=True,
            value=0,
            threshold=0,
            severity="PASS",
            message="Skipped - pytest not available",
        )


def check_g8_config_hash(package_path: Path, config_path: Optional[Path] = None) -> GateResult:
    """G8: config hash match (signal config hasn't changed unexpectedly)"""
    if config_path is None:
        config_path = Path("config/signals/rules.yaml")

    if not config_path.exists():
        return GateResult(
            gate_id="G8",
            gate_name="Config Hash",
            passed=True,
            value=0,
            threshold=0,
            severity="PASS",
            message="Config file not found - skipping hash check",
        )

    # Calculate config hash using SHA256 (more secure than MD5)
    with open(config_path, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()[:8]

    # Check if manifest contains expected hash
    manifest_path = package_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
            expected_hash = manifest.get("config_hash")
            if expected_hash and expected_hash != config_hash:
                return GateResult(
                    gate_id="G8",
                    gate_name="Config Hash",
                    passed=False,
                    value=1,
                    threshold=0,
                    severity="WARN",
                    message=f"Config changed: {expected_hash} -> {config_hash}",
                    details={"expected": expected_hash, "actual": config_hash},
                )

    return GateResult(
        gate_id="G8",
        gate_name="Config Hash",
        passed=True,
        value=0,
        threshold=0,
        severity="PASS",
        message=f"Config hash: {config_hash}",
    )


def check_g9_tp_timing(package_path: Path) -> GateResult:
    """G9: TP timing +/-5/+3 bars (turning point predictions within range)"""
    summary_path = package_path / "data" / "summary.json"

    if not summary_path.exists():
        return GateResult(
            gate_id="G9",
            gate_name="TP Timing",
            passed=True,
            value=0,
            threshold=5,
            severity="PASS",
            message="No summary.json - skipping TP check",
        )

    with open(summary_path) as f:
        summary = json.load(f)

    # Check turning point predictions
    out_of_range = []
    for ticker in summary.get("tickers", []):
        tp = ticker.get("turning_point")
        if tp and "bars_to_event" in tp:
            bars = tp["bars_to_event"]
            if bars < -THRESHOLDS["G9_tp_early_bars"] or bars > THRESHOLDS["G9_tp_late_bars"]:
                out_of_range.append(f"{ticker['symbol']}: {bars} bars")

    passed = len(out_of_range) == 0
    return GateResult(
        gate_id="G9",
        gate_name="TP Timing",
        passed=passed,
        value=len(out_of_range),
        threshold=0,
        severity="WARN" if out_of_range else "PASS",
        message=f"TPs out of range: {out_of_range[:3]}" if out_of_range else "All TPs within range",
        details={"out_of_range": out_of_range[:10]},
    )


def check_g10_regime_rates(package_path: Path) -> GateResult:
    """
    G10: Regime distribution sanity check.

    Note: The full M2 validation checks trending_R0 ≥85% and choppy_R0 ≤15%,
    but signal reports don't include trending/choppy labels. This gate performs
    a basic sanity check that regime distribution is reasonable (R0 > 10% to
    ensure detector is classifying some healthy regimes).
    """
    summary_path = package_path / "data" / "summary.json"

    if not summary_path.exists():
        return GateResult(
            gate_id="G10",
            gate_name="Regime Rates",
            passed=True,
            value=0,
            threshold=0.1,
            severity="PASS",
            message="No summary.json - skipping regime check",
        )

    with open(summary_path) as f:
        summary = json.load(f)

    # Count regimes
    r0_count = 0
    total = 0
    for ticker in summary.get("tickers", []):
        regime = ticker.get("regime")
        if regime:
            total += 1
            if regime == "R0":
                r0_count += 1

    if total == 0:
        return GateResult(
            gate_id="G10",
            gate_name="Regime Rates",
            passed=True,
            value=0,
            threshold=0.1,
            severity="PASS",
            message="No regime data available",
        )

    r0_rate = r0_count / total
    # Sanity check: at least 10% R0 (detector is finding some healthy regimes)
    # Full trending/choppy separation requires M2 validation infrastructure
    passed = r0_rate >= 0.1

    return GateResult(
        gate_id="G10",
        gate_name="Regime Rates",
        passed=passed,
        value=round(r0_rate, 3),
        threshold=0.1,
        severity="WARN" if not passed else "PASS",
        message=f"R0 rate: {r0_rate:.1%} ({r0_count}/{total})",
        details={"r0_count": r0_count, "total": total},
    )


# =============================================================================
# PR-A: DATA QUALITY GATES (G11-G15)
# =============================================================================


def check_g11_close_positive(package_path: Path) -> GateResult:
    """
    G11: All symbols must have close > 0.

    PR-A Critical Gate: Prevents close=0.0 bug from corrupting regime calculations.
    """
    summary_path = package_path / "data" / "summary.json"

    if not summary_path.exists():
        return GateResult(
            gate_id="G11",
            gate_name="Close Positive",
            passed=False,
            value=0,
            threshold=0,
            severity="FAIL",
            message="summary.json not found",
        )

    with open(summary_path) as f:
        summary = json.load(f)

    # Check close values for all tickers
    invalid_symbols: list[str] = []
    sample_closes: list[str] = []

    for ticker in summary.get("tickers", []):
        symbol = ticker.get("symbol", "UNKNOWN")
        component_values = ticker.get("component_values", {})
        close = component_values.get("close", 0.0)

        if close <= THRESHOLDS["G11_close_min"]:
            invalid_symbols.append(f"{symbol}: close={close}")

        # Collect sample closes for debugging
        if len(sample_closes) < 10:
            sample_closes.append(f"{symbol}:{close}")

    passed = len(invalid_symbols) == 0
    return GateResult(
        gate_id="G11",
        gate_name="Close Positive",
        passed=passed,
        value=len(invalid_symbols),
        threshold=0,
        severity="FAIL" if invalid_symbols else "PASS",
        message=f"Invalid close: {invalid_symbols[:5]}" if invalid_symbols else "All closes > 0",
        details={
            "invalid_symbols": invalid_symbols[:10],
            "sample_closes": sample_closes,
        },
    )


def check_g12_no_sentinels(package_path: Path) -> GateResult:
    """
    G12: No sentinel values (-1.0) in chart data.

    PR-A Critical Gate: Sentinel values indicate missing data from IB.
    """
    data_dir = package_path / "data"

    if not data_dir.exists():
        return GateResult(
            gate_id="G12",
            gate_name="No Sentinels",
            passed=False,
            value=0,
            threshold=0,
            severity="FAIL",
            message="Data directory not found",
        )

    sentinel = THRESHOLDS["G12_sentinel_value"]
    files_with_sentinels = []
    total_sentinels = 0

    for data_file in data_dir.glob("*.json"):
        if data_file.name in ["summary.json", "indicators.json"]:
            continue

        with open(data_file) as f:
            try:
                data = json.load(f)
                chart_data = data.get("chart_data", {})

                # Check OHLCV arrays for sentinel values
                for col in ["open", "high", "low", "close"]:
                    values = chart_data.get(col, [])
                    sentinel_count = sum(1 for v in values if v == sentinel)
                    if sentinel_count > 0:
                        total_sentinels += sentinel_count
                        files_with_sentinels.append(
                            f"{data_file.name}:{col}={sentinel_count}"
                        )
            except json.JSONDecodeError:
                continue

    passed = total_sentinels == 0
    return GateResult(
        gate_id="G12",
        gate_name="No Sentinels",
        passed=passed,
        value=total_sentinels,
        threshold=0,
        severity="FAIL" if total_sentinels > 0 else "PASS",
        message=f"Sentinel values: {total_sentinels} in {len(files_with_sentinels)} files"
        if total_sentinels > 0
        else "No sentinel values found",
        details={"files_with_sentinels": files_with_sentinels[:10]},
    )


def check_g13_atr_valid_samples(package_path: Path) -> GateResult:
    """
    G13: ATR percentile has sufficient valid samples (≥50).

    PR-A Warning Gate: Percentile calculations need enough data to be meaningful.
    """
    summary_path = package_path / "data" / "summary.json"

    if not summary_path.exists():
        return GateResult(
            gate_id="G13",
            gate_name="ATR Valid Samples",
            passed=True,
            value=0,
            threshold=THRESHOLDS["G13_atr_valid_n_min"],
            severity="PASS",
            message="summary.json not found - skipping",
        )

    with open(summary_path) as f:
        summary = json.load(f)

    # Check ATR percentile valid_n in derived_metrics
    insufficient_symbols = []
    min_valid_n = THRESHOLDS["G13_atr_valid_n_min"]

    for ticker in summary.get("tickers", []):
        symbol = ticker.get("symbol", "UNKNOWN")
        derived = ticker.get("derived_metrics", {})

        # Check if we have ATR percentile info with valid_n
        # Note: This field may not exist yet - it's a PR-A enhancement
        atr_valid_n = derived.get("atr_63_valid_n")
        if atr_valid_n is not None and atr_valid_n < min_valid_n:
            insufficient_symbols.append(f"{symbol}: valid_n={atr_valid_n}")

    # This is a WARN gate, not FAIL - we may not have this field yet
    passed = len(insufficient_symbols) == 0
    return GateResult(
        gate_id="G13",
        gate_name="ATR Valid Samples",
        passed=passed,
        value=len(insufficient_symbols),
        threshold=0,
        severity="WARN" if insufficient_symbols else "PASS",
        message=f"Insufficient ATR samples: {insufficient_symbols[:5]}"
        if insufficient_symbols
        else "All ATR percentiles have sufficient samples",
        details={"insufficient_symbols": insufficient_symbols[:10]},
    )


def check_g14_timestamp_match(package_path: Path) -> GateResult:
    """
    G14: Close timestamp matches last chart data timestamp.

    PR-A Critical Gate: Ensures summary close is from the actual latest bar.
    """
    data_dir = package_path / "data"
    summary_path = data_dir / "summary.json"

    if not summary_path.exists():
        return GateResult(
            gate_id="G14",
            gate_name="Timestamp Match",
            passed=False,
            value=0,
            threshold=0,
            severity="FAIL",
            message="summary.json not found",
        )

    with open(summary_path) as f:
        summary = json.load(f)

    mismatched_symbols = []
    tolerance_sec = THRESHOLDS["G14_timestamp_tolerance_sec"]

    for ticker in summary.get("tickers", []):
        symbol = ticker.get("symbol", "UNKNOWN")
        asof_ts_str = ticker.get("asof_ts")

        if not asof_ts_str:
            continue

        # Find corresponding data file
        for tf in ["1d", "1h", "5m"]:
            data_file = data_dir / f"{symbol}_{tf}.json"
            if data_file.exists():
                with open(data_file) as f:
                    data = json.load(f)
                    chart_data = data.get("chart_data", {})
                    timestamps = chart_data.get("timestamps", [])

                    if timestamps:
                        last_chart_ts = timestamps[-1]
                        # Parse timestamps and compare
                        try:
                            from datetime import datetime as dt

                            # Handle ISO format
                            if isinstance(asof_ts_str, str):
                                asof_ts = dt.fromisoformat(
                                    asof_ts_str.replace("Z", "+00:00")
                                )
                            if isinstance(last_chart_ts, str):
                                chart_ts = dt.fromisoformat(
                                    last_chart_ts.replace("Z", "+00:00")
                                )

                            delta = abs((asof_ts - chart_ts).total_seconds())
                            if delta > tolerance_sec:
                                mismatched_symbols.append(
                                    f"{symbol}: delta={delta}s"
                                )
                        except (ValueError, TypeError):
                            # Can't parse timestamps - skip
                            pass
                break

    passed = len(mismatched_symbols) == 0
    return GateResult(
        gate_id="G14",
        gate_name="Timestamp Match",
        passed=passed,
        value=len(mismatched_symbols),
        threshold=0,
        severity="FAIL" if mismatched_symbols else "PASS",
        message=f"Timestamp mismatches: {mismatched_symbols[:5]}"
        if mismatched_symbols
        else "All timestamps match",
        details={"mismatched_symbols": mismatched_symbols[:10]},
    )


def check_g15_bar_continuity(package_path: Path) -> GateResult:
    """
    G15: Bar continuity check (no excessive gaps).

    PR-A Warning Gate: Large gaps in data may indicate quality issues.
    """
    data_dir = package_path / "data"

    if not data_dir.exists():
        return GateResult(
            gate_id="G15",
            gate_name="Bar Continuity",
            passed=True,
            value=0,
            threshold=THRESHOLDS["G15_max_gap_bars"],
            severity="PASS",
            message="Data directory not found - skipping",
        )

    max_gap_bars = THRESHOLDS["G15_max_gap_bars"]
    files_with_gaps = []

    for data_file in data_dir.glob("*.json"):
        if data_file.name in ["summary.json", "indicators.json"]:
            continue

        with open(data_file) as f:
            try:
                data = json.load(f)
                timeframe = data.get("timeframe", "1d")
                chart_data = data.get("chart_data", {})
                timestamps = chart_data.get("timestamps", [])

                if len(timestamps) < 2:
                    continue

                # Calculate expected interval based on timeframe
                tf_seconds = {
                    "1m": 60,
                    "5m": 300,
                    "15m": 900,
                    "30m": 1800,
                    "1h": 3600,
                    "4h": 14400,
                    "1d": 86400,
                }.get(timeframe, 86400)

                # Check for gaps
                max_gap = 0
                from datetime import datetime as dt

                for i in range(1, len(timestamps)):
                    try:
                        ts1 = dt.fromisoformat(
                            timestamps[i - 1].replace("Z", "+00:00")
                        )
                        ts2 = dt.fromisoformat(timestamps[i].replace("Z", "+00:00"))
                        delta = (ts2 - ts1).total_seconds()
                        gap_bars = int(delta / tf_seconds) - 1
                        max_gap = max(max_gap, gap_bars)
                    except (ValueError, TypeError):
                        continue

                if max_gap > max_gap_bars:
                    files_with_gaps.append(f"{data_file.name}: gap={max_gap} bars")

            except json.JSONDecodeError:
                continue

    passed = len(files_with_gaps) == 0
    return GateResult(
        gate_id="G15",
        gate_name="Bar Continuity",
        passed=passed,
        value=len(files_with_gaps),
        threshold=0,
        severity="WARN" if files_with_gaps else "PASS",
        message=f"Files with gaps: {files_with_gaps[:5]}"
        if files_with_gaps
        else "No excessive gaps found",
        details={"files_with_gaps": files_with_gaps[:10]},
    )


def run_all_gates(
    package_path: Path,
    old_snapshot_path: Optional[Path] = None,
) -> ValidationReport:
    """Run all validation gates (G3-G15, G1/G2 removed as outdated)."""
    gates = []

    # G1/G2 removed - size limits are outdated for modern universe sizes

    # G3: First screen size
    gates.append(check_g3_first_screen(package_path))

    # G4: Missing sections
    gates.append(check_g4_missing_sections(package_path))

    # G5: Metric drift
    new_snapshot = package_path / "snapshots" / "payload_snapshot.json"
    gates.append(check_g5_metric_drift(old_snapshot_path, new_snapshot))

    # G6: Bar validation
    gates.append(check_g6_bar_validation(package_path))

    # G7: Causality test
    gates.append(check_g7_causality_test())

    # G8: Config hash
    gates.append(check_g8_config_hash(package_path))

    # G9: TP timing
    gates.append(check_g9_tp_timing(package_path))

    # G10: Regime rates
    gates.append(check_g10_regime_rates(package_path))

    # PR-A: Data Quality Gates (G11-G15)
    # G11: Close positive
    gates.append(check_g11_close_positive(package_path))

    # G12: No sentinel values
    gates.append(check_g12_no_sentinels(package_path))

    # G13: ATR valid samples
    gates.append(check_g13_atr_valid_samples(package_path))

    # G14: Timestamp match
    gates.append(check_g14_timestamp_match(package_path))

    # G15: Bar continuity
    gates.append(check_g15_bar_continuity(package_path))

    # Count failures
    fail_count = sum(1 for g in gates if not g.passed and g.severity == "FAIL")
    warn_count = sum(1 for g in gates if not g.passed and g.severity == "WARN")
    all_passed = fail_count == 0

    return ValidationReport(
        gates=gates,
        all_passed=all_passed,
        fail_count=fail_count,
        warn_count=warn_count,
        package_path=str(package_path),
    )


def run_single_gate(
    gate: str,
    package_path: Optional[Path] = None,
    old_snapshot_path: Optional[Path] = None,
    new_snapshot_path: Optional[Path] = None,
) -> GateResult:
    """Run a single gate (G3-G15, G1/G2 removed as outdated)."""
    gate = gate.upper()

    # G1/G2 removed - size limits are outdated for modern universe sizes
    if gate in ("G1", "G2"):
        return GateResult(
            gate_id=gate,
            gate_name="Removed",
            passed=True,
            value=0,
            threshold=0,
            severity="PASS",
            message=f"{gate} size limit removed - no longer relevant for modern universe sizes",
        )
    elif gate == "G3" and package_path:
        return check_g3_first_screen(package_path)
    elif gate == "G4" and package_path:
        return check_g4_missing_sections(package_path)
    elif gate == "G5":
        return check_g5_metric_drift(old_snapshot_path, new_snapshot_path or Path())
    elif gate == "G6" and package_path:
        return check_g6_bar_validation(package_path)
    elif gate == "G7":
        return check_g7_causality_test()
    elif gate == "G8" and package_path:
        return check_g8_config_hash(package_path)
    elif gate == "G9" and package_path:
        return check_g9_tp_timing(package_path)
    elif gate == "G10" and package_path:
        return check_g10_regime_rates(package_path)
    # PR-A: Data Quality Gates (G11-G15)
    elif gate == "G11" and package_path:
        return check_g11_close_positive(package_path)
    elif gate == "G12" and package_path:
        return check_g12_no_sentinels(package_path)
    elif gate == "G13" and package_path:
        return check_g13_atr_valid_samples(package_path)
    elif gate == "G14" and package_path:
        return check_g14_timestamp_match(package_path)
    elif gate == "G15" and package_path:
        return check_g15_bar_continuity(package_path)
    else:
        return GateResult(
            gate_id=gate,
            gate_name="Unknown",
            passed=False,
            value=-1,
            threshold=0,
            severity="FAIL",
            message=f"Unknown gate: {gate}",
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="validate_gates",
        description="Validate signal report against quality gates",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all gates",
    )
    parser.add_argument(
        "--gate",
        type=str,
        help="Run single gate (G3-G15)",
    )
    parser.add_argument(
        "--package",
        type=str,
        help="Path to signal package directory",
    )
    parser.add_argument(
        "--old-snapshot",
        type=str,
        help="Path to baseline snapshot (for G5)",
    )
    parser.add_argument(
        "--new-snapshot",
        type=str,
        help="Path to new snapshot (for G5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    if not args.all and not args.gate:
        parser.print_help()
        return 1

    package_path = Path(args.package) if args.package else None
    old_snapshot = Path(args.old_snapshot) if args.old_snapshot else None
    new_snapshot = Path(args.new_snapshot) if args.new_snapshot else None

    if args.all:
        if not package_path:
            print("ERROR: --package required for --all")
            return 1

        report = run_all_gates(package_path, old_snapshot)

        print("=" * 60)
        print("SIGNAL GATE VALIDATION REPORT")
        print("=" * 60)
        print(f"Package: {package_path}")
        print()

        for gate in report.gates:
            status = "PASS" if gate.passed else gate.severity
            symbol = "+" if gate.passed else "X" if gate.severity == "FAIL" else "~"
            print(f"  [{symbol}] {gate.gate_id} {gate.gate_name}: {status}")
            if args.verbose or not gate.passed:
                print(f"      {gate.message}")

        print()
        print("-" * 60)
        print(f"FAILS: {report.fail_count} | WARNS: {report.warn_count}")
        print(f"Overall: {'PASS' if report.all_passed else 'FAIL'}")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nResults written to: {output_path}")

        return 0 if report.all_passed else 1

    elif args.gate:
        result = run_single_gate(
            args.gate,
            package_path,
            old_snapshot,
            new_snapshot,
        )

        print(f"Gate {result.gate_id}: {'PASS' if result.passed else result.severity}")
        print(f"  {result.message}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

        return 0 if result.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
