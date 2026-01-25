"""
Validation Report Generator - HTML report for M2 regime validation results.

Generates a standalone HTML report that includes:
- Validation methodology explanation
- Full validation statistics (Cohen's d, p-value, CIs)
- Holdout validation results
- Gate pass/fail status
- Optimized parameters
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _render_validation_section(
    optimization_result: Optional[Dict[str, Any]] = None,
    full_validation_result: Optional[Dict[str, Any]] = None,
    holdout_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Render validation section HTML (for embedding in other reports)."""
    stats = full_validation_result.get("statistical_result", {}) if full_validation_result else {}

    trending_mean = stats.get("trending_mean", 0) * 100
    choppy_mean = stats.get("choppy_mean", 0) * 100
    cohens_d = stats.get("effect_size_cohens_d", 0)
    p_value = stats.get("p_value", 1)

    holdout_trending = holdout_result.get("trending_r0_rate", 0) * 100 if holdout_result else 0
    holdout_choppy = holdout_result.get("choppy_r0_rate", 0) * 100 if holdout_result else 0
    holdout_causality = holdout_result.get("causality_passed", False) if holdout_result else False

    def gate_icon(passed: bool) -> str:
        return "✓" if passed else "✗"

    return f"""
    <div class="validation-section">
        <h3>Validation Summary</h3>
        <div class="validation-metrics">
            <div>Trending R0: {trending_mean:.1f}%</div>
            <div>Choppy R0: {choppy_mean:.1f}%</div>
            <div>Cohen's d: {cohens_d:.3f}</div>
            <div>p-value: {p_value:.4f}</div>
        </div>
        <h3>Holdout Results</h3>
        <div>Trending: {holdout_trending:.1f}%, Choppy: {holdout_choppy:.1f}%</div>
        <div>Causality: {gate_icon(holdout_causality)} {'PASS' if holdout_causality else 'FAIL'}</div>
    </div>
    """


def generate_validation_section_html(
    optimization_result: Optional[Dict[str, Any]] = None,
    full_validation_result: Optional[Dict[str, Any]] = None,
    holdout_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate HTML section for validation results (for embedding in signal report).

    Returns:
        HTML string to embed in another report
    """
    return _render_validation_section(
        optimization_result=optimization_result,
        full_validation_result=full_validation_result,
        holdout_result=holdout_result,
    )


def generate_validation_report(
    optimization_result: Optional[Dict[str, Any]] = None,
    full_validation_result: Optional[Dict[str, Any]] = None,
    holdout_result: Optional[Dict[str, Any]] = None,
    output_path: Path = Path("reports/validation/validation_report.html"),
) -> Path:
    """
    Generate comprehensive validation HTML report.

    Args:
        optimization_result: Results from optimize mode
        full_validation_result: Results from full validation mode
        holdout_result: Results from holdout validation mode
        output_path: Where to save HTML report

    Returns:
        Path to generated HTML file
    """
    html = _render_validation_html(
        optimization_result=optimization_result,
        full_validation_result=full_validation_result,
        holdout_result=holdout_result,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    logger.info(f"Validation report generated: {output_path}")
    return output_path


def _render_validation_html(
    optimization_result: Optional[Dict[str, Any]] = None,
    full_validation_result: Optional[Dict[str, Any]] = None,
    holdout_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Render complete validation HTML."""

    # Extract statistics
    stats = full_validation_result.get("statistical_result", {}) if full_validation_result else {}
    gates = full_validation_result.get("gate_results", []) if full_validation_result else []
    best_params = optimization_result.get("best_params", {}) if optimization_result else {}

    # Format values
    trending_mean = stats.get("trending_mean", 0) * 100
    choppy_mean = stats.get("choppy_mean", 0) * 100
    cohens_d = stats.get("effect_size_cohens_d", 0)
    p_value = stats.get("p_value", 1)
    trending_ci = (stats.get("trending_ci_lower", 0) * 100, stats.get("trending_ci_upper", 0) * 100)
    choppy_ci = (stats.get("choppy_ci_lower", 0) * 100, stats.get("choppy_ci_upper", 0) * 100)

    holdout_trending = holdout_result.get("trending_r0_rate", 0) * 100 if holdout_result else 0
    holdout_choppy = holdout_result.get("choppy_r0_rate", 0) * 100 if holdout_result else 0
    holdout_causality = holdout_result.get("causality_passed", False) if holdout_result else False
    holdout_n = holdout_result.get("n_symbols", 0) if holdout_result else 0

    # Gate status
    def gate_icon(passed: bool) -> str:
        return "✓" if passed else "✗"

    def gate_class(passed: bool) -> str:
        return "pass" if passed else "fail"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M2 Validation Report</title>
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

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 1rem;
            margin-bottom: 2rem;
        }}

        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }}

        .section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}

        .card {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1.25rem;
            border: 1px solid var(--border);
        }}

        .card h3 {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .card .value {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .card .detail {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .pass {{ color: var(--success); }}
        .fail {{ color: var(--error); }}
        .warning {{ color: var(--warning); }}

        .gate-list {{
            list-style: none;
        }}

        .gate-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: var(--bg-card);
            border-radius: 6px;
            margin-bottom: 0.5rem;
            border-left: 4px solid var(--border);
        }}

        .gate-item.pass {{
            border-left-color: var(--success);
        }}

        .gate-item.fail {{
            border-left-color: var(--error);
        }}

        .gate-name {{
            font-weight: 500;
        }}

        .gate-value {{
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.875rem;
        }}

        .methodology {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1.5rem;
        }}

        .methodology h3 {{
            color: var(--accent);
            margin-bottom: 1rem;
        }}

        .methodology p {{
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }}

        .methodology ul {{
            margin-left: 1.5rem;
            color: var(--text-secondary);
        }}

        .methodology li {{
            margin-bottom: 0.5rem;
        }}

        .methodology strong {{
            color: var(--text-primary);
        }}

        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 0.75rem;
        }}

        .param-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0.75rem;
            background: var(--bg-primary);
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.875rem;
        }}

        .param-name {{
            color: var(--text-secondary);
        }}

        .param-value {{
            color: var(--success);
            font-weight: 600;
        }}

        .diagram {{
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.75rem;
            white-space: pre;
            overflow-x: auto;
            color: var(--text-secondary);
            line-height: 1.4;
        }}

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
        <h1>M2 Validation Report</h1>
        <p class="subtitle">Regime Detection Model Validation Results</p>

        <!-- Overview Cards -->
        <div class="section">
            <h2>📊 Validation Summary</h2>
            <div class="grid">
                <div class="card">
                    <h3>Trending R0 Rate</h3>
                    <div class="value">{trending_mean:.1f}%</div>
                    <div class="detail">95% CI: [{trending_ci[0]:.1f}%, {trending_ci[1]:.1f}%]</div>
                </div>
                <div class="card">
                    <h3>Choppy R0 Rate</h3>
                    <div class="value">{choppy_mean:.1f}%</div>
                    <div class="detail">95% CI: [{choppy_ci[0]:.1f}%, {choppy_ci[1]:.1f}%]</div>
                </div>
                <div class="card">
                    <h3>Effect Size (Cohen's d)</h3>
                    <div class="value {'pass' if cohens_d >= 0.8 else 'fail'}">{cohens_d:.3f}</div>
                    <div class="detail">Threshold: ≥ 0.8 (large effect)</div>
                </div>
                <div class="card">
                    <h3>Statistical Significance</h3>
                    <div class="value {'pass' if p_value < 0.01 else 'fail'}">p = {p_value:.4f}</div>
                    <div class="detail">Threshold: p < 0.01</div>
                </div>
            </div>
        </div>

        <!-- Gate Results -->
        <div class="section">
            <h2>🚦 Gate Results</h2>
            <ul class="gate-list">
"""

    for gate in gates:
        passed = gate.get("passed", False)
        html += f"""
                <li class="gate-item {gate_class(passed)}">
                    <span class="gate-name">{gate_icon(passed)} {gate.get('gate_name', 'Unknown')}</span>
                    <span class="gate-value">{gate.get('message', '')}</span>
                </li>
"""

    html += f"""
            </ul>
        </div>

        <!-- Holdout Validation -->
        <div class="section">
            <h2>🔒 Holdout Validation (Release Gate)</h2>
            <div class="grid">
                <div class="card">
                    <h3>Holdout Symbols</h3>
                    <div class="value">{holdout_n}</div>
                    <div class="detail">Never seen during training</div>
                </div>
                <div class="card">
                    <h3>Trending R0 Rate</h3>
                    <div class="value">{holdout_trending:.1f}%</div>
                </div>
                <div class="card">
                    <h3>Choppy R0 Rate</h3>
                    <div class="value">{holdout_choppy:.1f}%</div>
                </div>
                <div class="card">
                    <h3>Causality Test</h3>
                    <div class="value {gate_class(holdout_causality)}">{gate_icon(holdout_causality)} {'PASS' if holdout_causality else 'FAIL'}</div>
                    <div class="detail">Trending R0 > Choppy R0 + 10%</div>
                </div>
            </div>
        </div>

        <!-- Optimized Parameters -->
        <div class="section">
            <h2>⚙️ Optimized Parameters</h2>
            <div class="params-grid">
"""

    for name, value in best_params.items():
        html += f"""
                <div class="param-item">
                    <span class="param-name">{name}</span>
                    <span class="param-value">{value}</span>
                </div>
"""

    html += f"""
            </div>
        </div>

        <!-- Methodology -->
        <div class="section">
            <h2>📖 Methodology</h2>
            <div class="methodology">
                <h3>Nested Walk-Forward Cross-Validation</h3>
                <p>
                    This validation framework implements a <strong>strict nested CV design</strong> to prevent overfitting:
                </p>
                <ul>
                    <li><strong>Outer CV (Evaluation Only)</strong>: 5 folds for final metric calculation. No parameter tuning allowed.</li>
                    <li><strong>Inner CV (Optimization)</strong>: Optuna optimizes parameters within each outer training fold only.</li>
                    <li><strong>Holdout Set</strong>: 30% of symbols reserved, never seen until release validation.</li>
                </ul>

                <h3 style="margin-top: 1.5rem;">Anti-Overfitting Measures</h3>
                <ul>
                    <li><strong>Frozen Labeler</strong>: Ground truth thresholds are immutable and versioned (v1.0)</li>
                    <li><strong>Symbol-Level Statistics</strong>: n≈140 symbols provides real statistical power</li>
                    <li><strong>Block Bootstrap CIs</strong>: Confidence intervals respect time correlation</li>
                    <li><strong>Purge/Embargo</strong>: Prevents look-ahead bias in train/test splits</li>
                </ul>

                <h3 style="margin-top: 1.5rem;">Key Metrics Explained</h3>
                <ul>
                    <li><strong>R0 Rate</strong>: Percentage of bars where detector predicts "Healthy Uptrend" (R0)</li>
                    <li><strong>Trending R0</strong>: R0 rate during labeled TRENDING periods (should be HIGH)</li>
                    <li><strong>Choppy R0</strong>: R0 rate during labeled CHOPPY periods (should be LOW)</li>
                    <li><strong>Cohen's d</strong>: Effect size measuring separation between trending and choppy R0 rates</li>
                    <li><strong>Causality</strong>: Trending R0 must exceed Choppy R0 by at least 10%</li>
                </ul>

                <h3 style="margin-top: 1.5rem;">Validation Flow</h3>
                <div class="diagram">
┌─────────────────────────────────────────────────────────────────────┐
│  NESTED WALK-FORWARD VALIDATION                                     │
│                                                                     │
│  Training Universe (140 symbols)                                    │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  OUTER FOLD 1    OUTER FOLD 2    OUTER FOLD 3    ...          │ │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐                  │ │
│  │  │ Inner   │     │ Inner   │     │ Inner   │                  │ │
│  │  │ Optuna  │────▶│ Optuna  │────▶│ Optuna  │                  │ │
│  │  │ 20 trials│     │ 20 trials│     │ 20 trials│                  │ │
│  │  └────┬────┘     └────┬────┘     └────┬────┘                  │ │
│  │       │               │               │                        │ │
│  │       ▼               ▼               ▼                        │ │
│  │  [Outer Test]   [Outer Test]   [Outer Test]                   │ │
│  │  (evaluate)     (evaluate)     (evaluate)                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│                    Aggregate Statistics                             │
│                    (Cohen's d, p-value, CIs)                        │
│                                                                     │
│  Holdout Universe (60 symbols) ─── FINAL RELEASE GATE              │
└─────────────────────────────────────────────────────────────────────┘
                </div>
            </div>
        </div>

        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
        </div>
    </div>
</body>
</html>
"""

    return html


def load_validation_results(reports_dir: Path) -> Dict[str, Any]:
    """
    Load validation results from JSON files.

    Args:
        reports_dir: Directory containing validation JSON files

    Returns:
        Dict with optimization_result, full_validation_result, holdout_result
    """
    results = {}

    # Try to load each result file
    files = {
        "optimization_result": ["test_optimization.json", "optimization_result.json"],
        "full_validation_result": ["test_full_validation.json", "full_validation.json"],
        "holdout_result": ["test_holdout_validation.json", "holdout_validation.json"],
    }

    for key, filenames in files.items():
        for filename in filenames:
            filepath = reports_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    results[key] = json.load(f)
                break

    return results


def main() -> int:
    """CLI entry point for generating validation report."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate M2 Validation HTML Report")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/validation"),
        help="Directory containing validation JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/validation/validation_report.html"),
        help="Output HTML file path",
    )

    args = parser.parse_args()

    # Load results
    results = load_validation_results(args.reports_dir)

    if not results:
        print(f"No validation results found in {args.reports_dir}")
        return 1

    # Generate report
    output_path = generate_validation_report(
        optimization_result=results.get("optimization_result"),
        full_validation_result=results.get("full_validation_result"),
        holdout_result=results.get("holdout_result"),
        output_path=args.output,
    )

    print(f"Validation report generated: {output_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
