"""
Regime Turning Point - Turning point detection section.

Shows turning point state, confidence, features, and gating actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.signals.indicators.regime import RegimeOutput

from ..value_card import escape_html, render_section


def _load_experiment_result(symbol: str) -> Optional[Dict[str, Any]]:
    """Load experiment result for a symbol if available."""
    exp_path = Path(f"experiments/turning_point/{symbol.lower()}_latest.json")
    if exp_path.exists():
        try:
            with open(exp_path) as f:
                result: Dict[str, Any] = json.load(f)
                return result
        except Exception:
            return None
    return None


def generate_turning_point_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """
    Generate HTML for Turning Point Detection section (Phase 4).

    Shows:
    - Current turning point state (NONE, TOP_RISK, BOTTOM_RISK)
    - Confidence level with visual indicator
    - Top contributing features
    - Gating actions (block R0, accelerate R3)
    - Walk-forward backtest metrics (if available)
    """
    turning_point = regime_output.turning_point
    symbol = regime_output.symbol

    # If no turning point data, show placeholder
    if turning_point is None:
        return render_section(
            title="Turning Point Detection",
            body="""
            <div class="turning-point-placeholder">
                <p class="muted">Turning point model not active. Train model to enable predictive gating.</p>
                <p class="muted" style="font-size: 11px; margin-top: 8px;">
                    Run <code>python -m src.runners.signal_runner --train-models</code> to train models.
                </p>
            </div>
            """,
            collapsed=False,
            icon="🔄",
            section_id="turning-point-section",
        )

    # Determine state styling
    state = turning_point.turn_state.value
    confidence = turning_point.turn_confidence

    state_colors = {
        "none": {"bg": "#374151", "text": "#9ca3af", "label": "NONE"},
        "top_risk": {"bg": "#dc2626", "text": "#ffffff", "label": "TOP RISK"},
        "bottom_risk": {"bg": "#16a34a", "text": "#ffffff", "label": "BOTTOM RISK"},
    }
    sc = state_colors.get(state, state_colors["none"])

    # Confidence bar
    conf_pct = int(confidence * 100)
    if conf_pct >= 70:
        conf_class = "high"
        conf_color = "#16a34a"
    elif conf_pct >= 50:
        conf_class = "medium"
        conf_color = "#ca8a04"
    else:
        conf_class = "low"
        conf_color = "#6b7280"

    # Top features
    features_html = ""
    if turning_point.top_features:
        features_html = """
        <div class="tp-features">
            <h4>Top Contributing Features</h4>
            <div class="feature-list">
        """
        for name, contrib in turning_point.top_features[:3]:
            direction = "+" if contrib >= 0 else ""
            features_html += f"""
                <div class="feature-item">
                    <span class="feature-name">{escape_html(name)}</span>
                    <span class="feature-contrib" style="color: {'#16a34a' if contrib >= 0 else '#dc2626'}">{direction}{contrib:.3f}</span>
                </div>
            """
        features_html += "</div></div>"

    # Gating actions
    gating_html = ""
    if state == "top_risk" and confidence >= 0.7:
        gating_html = """
        <div class="tp-gating warning">
            <span class="gating-icon">⚠️</span>
            <span class="gating-text"><strong>R0 BLOCKED</strong> - High TOP_RISK gates Healthy Uptrend entry</span>
        </div>
        """
    elif state == "bottom_risk" and confidence >= 0.7:
        gating_html = """
        <div class="tp-gating success">
            <span class="gating-icon">🚀</span>
            <span class="gating-text"><strong>R3 ACCELERATED</strong> - High BOTTOM_RISK accelerates Rebound entry</span>
        </div>
        """

    # Inference time
    inference_time = turning_point.inference_time_ms
    inference_status = "fast" if inference_time < 1.0 else "slow"

    # Load experiment/backtest results if available
    exp_result = _load_experiment_result(symbol)
    backtest_html = ""
    if exp_result:
        # Format metrics
        top_roc = exp_result.get("median_top_roc_auc", 0)
        top_pr = exp_result.get("median_top_pr_auc", 0)
        bottom_roc = exp_result.get("median_bottom_roc_auc", 0)
        bottom_pr = exp_result.get("median_bottom_pr_auc", 0)
        std_top = exp_result.get("std_top_roc_auc", 0)
        std_bottom = exp_result.get("std_bottom_roc_auc", 0)
        n_windows = len(exp_result.get("window_metrics", []))
        created_at = exp_result.get("created_at", "N/A")[:10]

        # Color code by quality
        def _metric_color(val: float) -> str:
            if val >= 0.7:
                return "#16a34a"  # Good
            elif val >= 0.55:
                return "#ca8a04"  # Medium
            else:
                return "#dc2626"  # Poor

        backtest_html = f"""
        <div class="tp-backtest">
            <h4>Walk-Forward Backtest ({n_windows} windows)</h4>
            <div class="backtest-grid">
                <div class="backtest-model">
                    <span class="model-label">TOP_RISK</span>
                    <div class="metric-row">
                        <span class="metric-name">ROC-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(top_roc)}">{top_roc:.3f} ± {std_top:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">PR-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(top_pr)}">{top_pr:.3f}</span>
                    </div>
                </div>
                <div class="backtest-model">
                    <span class="model-label">BOTTOM_RISK</span>
                    <div class="metric-row">
                        <span class="metric-name">ROC-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(bottom_roc)}">{bottom_roc:.3f} ± {std_bottom:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">PR-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(bottom_pr)}">{bottom_pr:.3f}</span>
                    </div>
                </div>
            </div>
            <div class="backtest-meta">
                Last evaluated: {created_at} |
                <a href="#" onclick="alert('Run: python scripts/retrain_turning_point_models.py --symbol {symbol}'); return false;">Retrain</a>
            </div>
        </div>
        """

    body = f"""
    <div class="turning-point-content">
        <div class="tp-summary">
            <div class="tp-state-row">
                <span class="label">Turn State:</span>
                <span class="tp-state-badge" style="background: {sc['bg']}; color: {sc['text']}">{sc['label']}</span>
            </div>
            <div class="tp-confidence-row">
                <span class="label">Confidence:</span>
                <div class="confidence-bar-container">
                    <div class="confidence-bar {conf_class}" style="width: {conf_pct}%; background: {conf_color}"></div>
                    <span class="confidence-value">{conf_pct}%</span>
                </div>
            </div>
            <div class="tp-meta-row">
                <span class="label">Model:</span>
                <span class="tp-version">{escape_html(turning_point.model_version)}</span>
                <span class="tp-inference {inference_status}">({inference_time:.2f}ms)</span>
            </div>
        </div>

        {gating_html}
        {features_html}
        {backtest_html}
    </div>
    """

    return render_section(
        title="Turning Point Detection",
        body=body,
        collapsed=False,
        icon="🔄",
        section_id="turning-point-section",
    )
