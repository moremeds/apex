"""
Self-contained HTML report for DualMACD behavioral gate validation.

Generates a single HTML file with Plotly charts (dark theme) showing:
1. Summary dashboard with metric cards
2. Price + entry comparison chart
3. DualMACD state timeline
4. Equity curve overlay
5. Blocked trade analysis table
6. Optimization results (if available)
7. Suggestions section
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .behavioral_models import BehavioralMetrics, TradeDecision

logger = logging.getLogger(__name__)

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


def generate_behavioral_report(
    symbol: str,
    start_date: date,
    end_date: date,
    close_prices: pd.Series,
    baseline_entries: pd.Series,
    gated_entries: pd.Series,
    decisions: List[TradeDecision],
    metrics: BehavioralMetrics,
    macd_states: Optional[List[Dict[str, Any]]] = None,
    baseline_equity: Optional[pd.Series] = None,
    gated_equity: Optional[pd.Series] = None,
    optimization_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a self-contained HTML behavioral report.

    Returns HTML string. If output_path is provided, also writes to file.
    """
    sections = [
        _render_header(symbol, start_date, end_date, params),
        _render_summary_dashboard(metrics),
        _render_price_chart(symbol, close_prices, baseline_entries, gated_entries, decisions),
        _render_macd_timeline(close_prices, macd_states, decisions) if macd_states else "",
        (
            _render_equity_overlay(baseline_equity, gated_equity)
            if baseline_equity is not None
            else ""
        ),
        _render_blocked_table(decisions),
        _render_optimization(optimization_results) if optimization_results else "",
        _render_suggestions(metrics, params),
        _render_footer(),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Behavioral Gate Report: {symbol} ({start_date} to {end_date})</title>
    <script src="{PLOTLY_CDN}"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 24px; margin-bottom: 8px; }}
        h2 {{ font-size: 18px; margin: 24px 0 12px; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .subtitle {{ color: #94a3b8; font-size: 13px; margin-bottom: 24px; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
        .card-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
        .card-sub {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}
        .pass {{ color: #10b981; }}
        .fail {{ color: #ef4444; }}
        .warn {{ color: #f59e0b; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .badge-pass {{ background: rgba(16,185,129,0.15); color: #10b981; }}
        .badge-fail {{ background: rgba(239,68,68,0.15); color: #ef4444; }}
        .badge-warn {{ background: rgba(245,158,11,0.15); color: #f59e0b; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th {{ background: #1e293b; padding: 8px; text-align: left; color: #64748b; font-size: 11px; text-transform: uppercase; }}
        td {{ padding: 6px 8px; border-bottom: 1px solid #1e293b; }}
        tr.blocked-loss {{ background: rgba(16,185,129,0.05); }}
        tr.blocked-win {{ background: rgba(239,68,68,0.05); }}
        .chart {{ margin-bottom: 24px; }}
        .suggestions {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-top: 24px; }}
        .suggestion-item {{ padding: 4px 0; font-size: 13px; }}
        .recommendation {{ background: #0f172a; border-radius: 6px; padding: 12px; margin-top: 12px; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
{''.join(sections)}
</div>
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Report written to {output_path}")

    return html


def _render_header(
    symbol: str, start_date: date, end_date: date, params: Optional[Dict[str, Any]]
) -> str:
    params_str = ""
    if params:
        params_str = " · ".join(f"{k}={v}" for k, v in params.items())
        params_str = f" · {params_str}"
    return f"""
    <h1>DualMACD Behavioral Gate: {symbol}</h1>
    <div class="subtitle">{start_date} to {end_date}{params_str}</div>
    """


def _render_summary_dashboard(m: BehavioralMetrics) -> str:
    def _badge(value: float, threshold: float, op: str = ">=") -> str:
        if op == ">=":
            ok = value >= threshold
            margin = abs(value - threshold) / max(abs(threshold), 0.01)
        else:
            ok = value <= threshold
            margin = abs(threshold - value) / max(abs(threshold), 0.01)

        if ok and margin < 0.1:
            return '<span class="badge badge-warn">MARGINAL</span>'
        elif ok:
            return '<span class="badge badge-pass">PASS</span>'
        else:
            return '<span class="badge badge-fail">FAIL</span>'

    return f"""
    <h2>Summary</h2>
    <div class="cards">
        <div class="card">
            <div class="card-label">Blocked Loss Ratio {_badge(m.blocked_trade_loss_ratio, 0.6)}</div>
            <div class="card-value">{m.blocked_trade_loss_ratio:.0%}</div>
            <div class="card-sub">of blocked trades would have lost (≥60% required)</div>
        </div>
        <div class="card">
            <div class="card-label">Blocked Avg PnL</div>
            <div class="card-value {'pass' if m.blocked_trade_avg_pnl < 0 else 'fail'}">{m.blocked_trade_avg_pnl:+.2%}</div>
            <div class="card-sub">avg virtual PnL of blocked trades</div>
        </div>
        <div class="card">
            <div class="card-label">Allowed Sharpe</div>
            <div class="card-value">{m.allowed_trade_sharpe:.2f}</div>
            <div class="card-sub">vs baseline {m.baseline_sharpe:.2f}</div>
        </div>
        <div class="card">
            <div class="card-label">Trade Freedom {_badge(m.allowed_trade_ratio, 0.7)}</div>
            <div class="card-value">{m.allowed_trade_ratio:.0%}</div>
            <div class="card-sub">{m.allowed_trade_count}/{m.baseline_trade_count} trades allowed (≥70%)</div>
        </div>
        <div class="card">
            <div class="card-label">Max DD Reduction {_badge(m.max_dd_ratio, 0.85, "<=")}</div>
            <div class="card-value">{m.max_dd_gated:.1%}</div>
            <div class="card-sub">vs baseline {m.max_dd_baseline:.1%}</div>
        </div>
        <div class="card">
            <div class="card-label">Blocked Count</div>
            <div class="card-value">{m.blocked_trade_count}</div>
            <div class="card-sub">trades filtered out</div>
        </div>
    </div>
    """


def _render_price_chart(
    symbol: str,
    close: pd.Series,
    baseline_entries: pd.Series,
    gated_entries: pd.Series,
    decisions: List[TradeDecision],
) -> str:
    dates = [str(d) for d in close.index]
    prices = close.tolist()

    # Baseline entries (grey dots)
    bl_dates = [str(d) for d, v in zip(close.index, baseline_entries) if v]
    bl_prices = [float(close.loc[d]) for d, v in zip(close.index, baseline_entries) if v]

    # Gated allowed entries (green dots)
    ga_dates = [str(d) for d, v in zip(close.index, gated_entries) if v]
    ga_prices = [float(close.loc[d]) for d, v in zip(close.index, gated_entries) if v]

    # Blocked entries (red X)
    blocked = [d for d in decisions if not d.allowed]
    bk_dates = [str(d.timestamp) for d in blocked]
    bk_prices = [d.virtual_entry_price for d in blocked]

    return f"""
    <h2>Price + Entry Comparison</h2>
    <div class="chart" id="price-chart"></div>
    <script>
    Plotly.newPlot('price-chart', [
        {{
            x: {json.dumps(dates)},
            y: {json.dumps(prices)},
            type: 'scatter', mode: 'lines',
            name: '{symbol}', line: {{color: '#94a3b8', width: 1}},
        }},
        {{
            x: {json.dumps(bl_dates)},
            y: {json.dumps(bl_prices)},
            type: 'scatter', mode: 'markers',
            name: 'Baseline Entry', marker: {{color: '#64748b', size: 6, symbol: 'circle'}},
        }},
        {{
            x: {json.dumps(ga_dates)},
            y: {json.dumps(ga_prices)},
            type: 'scatter', mode: 'markers',
            name: 'Gated Allowed', marker: {{color: '#10b981', size: 8, symbol: 'circle'}},
        }},
        {{
            x: {json.dumps(bk_dates)},
            y: {json.dumps(bk_prices)},
            type: 'scatter', mode: 'markers',
            name: 'Blocked', marker: {{color: '#ef4444', size: 8, symbol: 'x'}},
        }},
    ], {{
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font: {{color: '#e2e8f0'}},
        xaxis: {{gridcolor: '#1e293b'}},
        yaxis: {{gridcolor: '#1e293b', title: 'Price'}},
        legend: {{x: 0, y: 1.1, orientation: 'h'}},
        margin: {{t: 10, r: 20, b: 40, l: 60}},
        height: 400,
    }});
    </script>
    """


def _render_macd_timeline(
    close: pd.Series,
    states: List[Dict[str, Any]],
    decisions: List[TradeDecision],
) -> str:
    dates = [str(d) for d in close.index]
    n = min(len(states), len(dates))

    slow_hist = [s.get("slow_histogram", 0) for s in states[:n]]
    fast_hist = [s.get("fast_histogram", 0) for s in states[:n]]

    # Trend state as numeric for colored band
    trend_map = {"BULLISH": 3, "IMPROVING": 2, "DETERIORATING": 1, "BEARISH": 0}
    trend_vals = [trend_map.get(s.get("trend_state", "BEARISH"), 0) for s in states[:n]]

    return f"""
    <h2>DualMACD State Timeline</h2>
    <div class="chart" id="macd-chart"></div>
    <script>
    Plotly.newPlot('macd-chart', [
        {{
            x: {json.dumps(dates[:n])},
            y: {json.dumps(slow_hist)},
            type: 'bar', name: 'Slow Histogram',
            marker: {{color: {json.dumps(slow_hist)},
                      colorscale: [[0, '#ef4444'], [0.5, '#334155'], [1, '#10b981']],
                      cmid: 0}},
            yaxis: 'y1',
        }},
        {{
            x: {json.dumps(dates[:n])},
            y: {json.dumps(fast_hist)},
            type: 'scatter', mode: 'lines', name: 'Fast Histogram',
            line: {{color: '#818cf8', width: 1.5}},
            yaxis: 'y1',
        }},
        {{
            x: {json.dumps(dates[:n])},
            y: {json.dumps(trend_vals)},
            type: 'scatter', mode: 'markers',
            name: 'Trend State',
            marker: {{
                color: {json.dumps(trend_vals)},
                colorscale: [[0, '#ef4444'], [0.33, '#f59e0b'], [0.67, '#06b6d4'], [1, '#10b981']],
                size: 4,
                cmin: 0, cmax: 3,
            }},
            yaxis: 'y2',
        }},
    ], {{
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font: {{color: '#e2e8f0'}},
        grid: {{rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom'}},
        yaxis: {{gridcolor: '#1e293b', title: 'Histogram', domain: [0.35, 1]}},
        yaxis2: {{gridcolor: '#1e293b', title: 'Trend', domain: [0, 0.3],
                  tickvals: [0, 1, 2, 3],
                  ticktext: ['BEARISH', 'DETERIORATING', 'IMPROVING', 'BULLISH']}},
        xaxis: {{gridcolor: '#1e293b'}},
        xaxis2: {{gridcolor: '#1e293b', anchor: 'y2'}},
        showlegend: true,
        legend: {{x: 0, y: 1.05, orientation: 'h'}},
        margin: {{t: 10, r: 20, b: 40, l: 60}},
        height: 450,
    }});
    </script>
    """


def _render_equity_overlay(
    baseline: pd.Series,
    gated: pd.Series,
) -> str:
    dates = [str(d) for d in baseline.index]
    bl_vals = baseline.tolist()
    ga_vals = gated.tolist()

    return f"""
    <h2>Equity Curve Overlay</h2>
    <div class="chart" id="equity-chart"></div>
    <script>
    Plotly.newPlot('equity-chart', [
        {{
            x: {json.dumps(dates)},
            y: {json.dumps(bl_vals)},
            type: 'scatter', mode: 'lines',
            name: 'Baseline', line: {{color: '#64748b', width: 1.5}},
        }},
        {{
            x: {json.dumps(dates)},
            y: {json.dumps(ga_vals)},
            type: 'scatter', mode: 'lines',
            name: 'Gated', line: {{color: '#3b82f6', width: 2}},
        }},
    ], {{
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font: {{color: '#e2e8f0'}},
        xaxis: {{gridcolor: '#1e293b'}},
        yaxis: {{gridcolor: '#1e293b', title: 'Equity'}},
        legend: {{x: 0, y: 1.1, orientation: 'h'}},
        margin: {{t: 10, r: 20, b: 40, l: 60}},
        height: 350,
    }});
    </script>
    """


def _render_blocked_table(decisions: List[TradeDecision]) -> str:
    blocked = [d for d in decisions if not d.allowed]
    if not blocked:
        return "<h2>Blocked Trade Analysis</h2><p style='color:#64748b;'>No blocked trades.</p>"

    rows = []
    for d in blocked:
        pnl = d.virtual_pnl_pct
        pnl_str = f"{pnl:+.2%}" if pnl is not None else "—"
        exit_str = f"{d.virtual_exit_price:.2f}" if d.virtual_exit_price else "—"
        row_class = "blocked-loss" if (pnl is not None and pnl < 0) else "blocked-win"
        rows.append(f"""
            <tr class="{row_class}">
                <td>{d.timestamp}</td>
                <td>{d.strategy_direction}</td>
                <td>{d.trend_state}</td>
                <td>{d.block_reason or '—'}</td>
                <td>{d.gate_strength:.2f}</td>
                <td>{d.virtual_entry_price:.2f}</td>
                <td>{exit_str}</td>
                <td>{pnl_str}</td>
            </tr>
        """)

    # Summary row
    resolved = [d for d in blocked if d.virtual_pnl_pct is not None]
    if resolved:
        avg_pnl = sum(d.virtual_pnl_pct for d in resolved) / len(resolved)  # type: ignore[misc]
        losses = sum(1 for d in resolved if (d.virtual_pnl_pct or 0) < 0)
        loss_ratio = losses / len(resolved)
        summary = f"<tr style='font-weight:600;border-top:2px solid #334155;'><td colspan='7'>Summary: {len(resolved)} resolved, loss_ratio={loss_ratio:.0%}</td><td>{avg_pnl:+.2%}</td></tr>"
    else:
        summary = ""

    return f"""
    <h2>Blocked Trade Analysis</h2>
    <div style="overflow-x:auto;">
    <table>
        <thead>
            <tr>
                <th>Date</th><th>Direction</th><th>Trend</th><th>Block Reason</th>
                <th>Gate Str.</th><th>Entry</th><th>Virtual Exit</th><th>Virtual PnL</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
            {summary}
        </tbody>
    </table>
    </div>
    """


def _render_optimization(results: Dict[str, Any]) -> str:
    """Render optimization results section (heatmap + walk-forward stability)."""
    if not results:
        return ""

    # Parameter heatmap data
    heatmap = results.get("heatmap")
    wf_rounds = results.get("walk_forward_rounds", [])

    sections = ["<h2>Optimization Results</h2>"]

    if heatmap:
        sections.append(f"""
        <div class="chart" id="param-heatmap"></div>
        <script>
        Plotly.newPlot('param-heatmap', [{{
            z: {json.dumps(heatmap['z'])},
            x: {json.dumps(heatmap['x'])},
            y: {json.dumps(heatmap['y'])},
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: {{title: 'Score'}},
        }}], {{
            paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
            font: {{color: '#e2e8f0'}},
            xaxis: {{title: 'hist_norm_window'}},
            yaxis: {{title: 'slope_lookback'}},
            margin: {{t: 10, r: 60, b: 60, l: 80}},
            height: 300,
        }});
        </script>
        """)

    if wf_rounds:
        round_rows = []
        for r in wf_rounds:
            round_rows.append(f"""
                <tr>
                    <td>{r.get('round', '—')}</td>
                    <td>{r.get('role', '—')}</td>
                    <td>{r.get('blocked_loss_ratio', 0):.2f}</td>
                    <td>{r.get('allowed_ratio', 0):.2f}</td>
                    <td>{r.get('score', 0):.3f}</td>
                </tr>
            """)
        sections.append(f"""
        <h3 style="color:#94a3b8;margin-top:16px;">Walk-Forward Stability</h3>
        <table>
            <thead><tr><th>Round</th><th>Role</th><th>Loss Ratio</th><th>Allow Ratio</th><th>Score</th></tr></thead>
            <tbody>{''.join(round_rows)}</tbody>
        </table>
        """)

    return "\n".join(sections)


def _render_suggestions(
    metrics: BehavioralMetrics,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate actionable suggestions based on metrics."""
    items = []

    # Check constraints
    def _check(label: str, value: float, threshold: float, op: str = ">=") -> None:
        if op == ">=":
            ok = value >= threshold
            margin = abs(value - threshold) / max(abs(threshold), 0.01)
        else:
            ok = value <= threshold
            margin = abs(threshold - value) / max(abs(threshold), 0.01)

        if ok and margin < 0.1:
            items.append(
                f'<div class="suggestion-item warn">⚠️ WARN: {label} = {value:.2f} ({op}{threshold}, marginal)</div>'
            )
        elif ok:
            items.append(
                f'<div class="suggestion-item pass">✅ PASS: {label} = {value:.2f} ({op}{threshold})</div>'
            )
        else:
            items.append(
                f'<div class="suggestion-item fail">❌ FAIL: {label} = {value:.2f} ({op}{threshold})</div>'
            )

    _check("blocked_trade_loss_ratio", metrics.blocked_trade_loss_ratio, 0.6)
    _check("allowed_trade_ratio", metrics.allowed_trade_ratio, 0.7)
    if metrics.max_dd_baseline > 0:
        _check("max_dd_ratio", metrics.max_dd_ratio, 0.85, "<=")

    # Recommendations
    recs = []
    if metrics.blocked_trade_loss_ratio >= 0.6:
        recs.append(
            f"Gate is effective: {metrics.blocked_trade_loss_ratio:.0%} of blocked trades "
            f"would have lost money (avg {metrics.blocked_trade_avg_pnl:+.1%})"
        )
    else:
        recs.append(
            "Gate is not selective enough — too many blocked trades would have been profitable. "
            "Consider tightening the gate or reviewing base strategy quality."
        )

    if metrics.allowed_trade_ratio >= 0.7:
        recs.append(
            f"Trade freedom preserved: {metrics.allowed_trade_ratio:.0%} of baseline entries still allowed"
        )
    else:
        recs.append(
            "Gate is too restrictive — blocking too many trades. "
            "Consider loosening gate thresholds or reviewing trend state criteria."
        )

    if params:
        recs.append(f"Current parameters: {', '.join(f'{k}={v}' for k, v in params.items())}")

    all_pass = (
        metrics.blocked_trade_loss_ratio >= 0.6
        and metrics.allowed_trade_ratio >= 0.7
        and (metrics.max_dd_baseline == 0 or metrics.max_dd_ratio <= 0.85)
    )

    if not all_pass:
        recs.append("Not all constraints pass — gate too aggressive or base strategy needs review.")

    rec_html = "\n".join(f"<div>• {r}</div>" for r in recs)

    return f"""
    <div class="suggestions">
        <h2 style="margin-top:0;">Suggestions</h2>
        {''.join(items)}
        <div class="recommendation">
            <strong>Recommendations:</strong>
            {rec_html}
        </div>
    </div>
    """


def _render_footer() -> str:
    return """
    <div style="margin-top:32px;padding-top:16px;border-top:1px solid #334155;color:#475569;font-size:11px;">
        Generated by APEX DualMACD Behavioral Gate Validator
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
# Summary Report — one page across all symbols
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SymbolResult:
    """Per-symbol result for the summary report."""

    symbol: str
    metrics: BehavioralMetrics
    decisions: List[TradeDecision]


def generate_summary_report(
    results: List[SymbolResult],
    start_date: date,
    end_date: date,
    params: Dict[str, Any],
    optimization_results: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    symbol_to_sector: Optional[Dict[str, str]] = None,
    gate_policies: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a single-page summary report across all symbols.

    Sections:
    1. Aggregate metrics cards
    2. Cross-symbol comparison table (sortable)
    3. Blocked trade loss ratio bar chart
    4. Per-symbol blocked trade PnL scatter
    5. Optimization heatmap (if available)
    6. Aggregate suggestions
    """
    agg = _aggregate_metrics(results)

    sections = [
        _summary_header(start_date, end_date, params, len(results)),
        _summary_aggregate_cards(agg),
        _summary_symbol_table(results, symbol_to_sector, gate_policies),
        _summary_policy_breakdown(results),
        _summary_loss_ratio_chart(results),
        _summary_blocked_pnl_scatter(results),
        _render_optimization(optimization_results) if optimization_results else "",
        _summary_suggestions(results, agg, params),
        _render_footer(),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Behavioral Gate Summary ({start_date} to {end_date})</title>
    <script src="{PLOTLY_CDN}"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 24px; margin-bottom: 8px; }}
        h2 {{ font-size: 18px; margin: 24px 0 12px; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .subtitle {{ color: #94a3b8; font-size: 13px; margin-bottom: 24px; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
        .card-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
        .card-sub {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}
        .pass {{ color: #10b981; }}
        .fail {{ color: #ef4444; }}
        .warn {{ color: #f59e0b; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
        .badge-pass {{ background: rgba(16,185,129,0.15); color: #10b981; }}
        .badge-fail {{ background: rgba(239,68,68,0.15); color: #ef4444; }}
        .badge-warn {{ background: rgba(245,158,11,0.15); color: #f59e0b; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th {{ background: #1e293b; padding: 8px; text-align: left; color: #64748b; font-size: 11px; text-transform: uppercase; cursor: pointer; }}
        th:hover {{ color: #e2e8f0; }}
        td {{ padding: 6px 8px; border-bottom: 1px solid #1e293b; }}
        .chart {{ margin-bottom: 24px; }}
        .suggestions {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-top: 24px; }}
        .suggestion-item {{ padding: 4px 0; font-size: 13px; }}
        .recommendation {{ background: #0f172a; border-radius: 6px; padding: 12px; margin-top: 12px; font-size: 12px; }}
        .symbol-link {{ color: #3b82f6; text-decoration: none; }}
        .symbol-link:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
<div class="container">
{''.join(sections)}
</div>
</body>
</html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Summary report written to {output_path}")

    return html


@dataclass
class _AggMetrics:
    total_symbols: int = 0
    total_baseline: int = 0
    total_allowed: int = 0
    total_blocked: int = 0
    avg_loss_ratio: float = 0.0
    avg_blocked_pnl: float = 0.0
    avg_trade_freedom: float = 0.0
    symbols_passing: int = 0
    symbols_failing: int = 0


def _aggregate_metrics(results: List[SymbolResult]) -> _AggMetrics:
    agg = _AggMetrics(total_symbols=len(results))
    if not results:
        return agg

    loss_ratios = []
    blocked_pnls = []
    freedoms = []

    for r in results:
        m = r.metrics
        agg.total_baseline += m.baseline_trade_count
        agg.total_allowed += m.allowed_trade_count
        agg.total_blocked += m.blocked_trade_count

        if m.blocked_trade_count > 0:
            loss_ratios.append(m.blocked_trade_loss_ratio)
            blocked_pnls.append(m.blocked_trade_avg_pnl)
        freedoms.append(m.allowed_trade_ratio)

        # Pass = loss_ratio >= 0.6 AND freedom >= 0.7 (only count if has blocks)
        if m.blocked_trade_count == 0:
            agg.symbols_passing += 1  # no blocks = pass (gate not triggered)
        elif m.blocked_trade_loss_ratio >= 0.6 and m.allowed_trade_ratio >= 0.7:
            agg.symbols_passing += 1
        else:
            agg.symbols_failing += 1

    agg.avg_loss_ratio = sum(loss_ratios) / len(loss_ratios) if loss_ratios else 0.0
    agg.avg_blocked_pnl = sum(blocked_pnls) / len(blocked_pnls) if blocked_pnls else 0.0
    agg.avg_trade_freedom = sum(freedoms) / len(freedoms) if freedoms else 0.0

    return agg


def _summary_header(
    start_date: date, end_date: date, params: Dict[str, Any], n_symbols: int
) -> str:
    params_str = " · ".join(f"{k}={v}" for k, v in params.items())
    return f"""
    <h1>DualMACD Behavioral Gate — Summary</h1>
    <div class="subtitle">{start_date} to {end_date} · {n_symbols} symbols · {params_str}</div>
    """


def _summary_aggregate_cards(a: _AggMetrics) -> str:
    pass_pct = a.symbols_passing / max(a.total_symbols, 1)
    pass_class = "pass" if pass_pct >= 0.8 else "warn" if pass_pct >= 0.5 else "fail"
    blk_class = "pass" if a.avg_blocked_pnl < 0 else "fail"
    lr_class = "pass" if a.avg_loss_ratio >= 0.6 else "fail"

    return f"""
    <h2>Aggregate Metrics</h2>
    <div class="cards">
        <div class="card">
            <div class="card-label">Symbols</div>
            <div class="card-value">{a.total_symbols}</div>
            <div class="card-sub {pass_class}">{a.symbols_passing} pass / {a.symbols_failing} fail</div>
        </div>
        <div class="card">
            <div class="card-label">Total Trades</div>
            <div class="card-value">{a.total_baseline}</div>
            <div class="card-sub">{a.total_allowed} allowed · {a.total_blocked} blocked</div>
        </div>
        <div class="card">
            <div class="card-label">Avg Loss Ratio</div>
            <div class="card-value {lr_class}">{a.avg_loss_ratio:.0%}</div>
            <div class="card-sub">across symbols with blocks (≥60% target)</div>
        </div>
        <div class="card">
            <div class="card-label">Avg Blocked PnL</div>
            <div class="card-value {blk_class}">{a.avg_blocked_pnl:+.2%}</div>
            <div class="card-sub">negative = gate is working</div>
        </div>
        <div class="card">
            <div class="card-label">Avg Trade Freedom</div>
            <div class="card-value">{a.avg_trade_freedom:.0%}</div>
            <div class="card-sub">≥70% target</div>
        </div>
    </div>
    """


def _summary_symbol_table(
    results: List[SymbolResult],
    symbol_to_sector: Optional[Dict[str, str]] = None,
    gate_policies: Optional[Dict[str, Any]] = None,
) -> str:
    s2s = symbol_to_sector or {}
    gp = gate_policies or {}

    rows = []
    for r in results:
        m = r.metrics
        lr = m.blocked_trade_loss_ratio
        lr_class = "pass" if (m.blocked_trade_count == 0 or lr >= 0.6) else "fail"
        pnl_class = "pass" if m.blocked_trade_avg_pnl <= 0 else "fail"
        fr_class = "pass" if m.allowed_trade_ratio >= 0.7 else "fail"

        sector = s2s.get(r.symbol.upper(), "—")
        policy_obj = gp.get(sector, gp.get("default"))
        if policy_obj and hasattr(policy_obj, "action_on_block"):
            policy_str = policy_obj.action_on_block
        elif isinstance(policy_obj, dict):
            policy_str = policy_obj.get("action_on_block", "BLOCK")
        else:
            policy_str = "BLOCK"

        policy_badge_class = {
            "BLOCK": "badge-fail",
            "SIZE_DOWN": "badge-warn",
            "BYPASS": "badge-pass",
        }.get(policy_str, "badge-fail")

        # Overall verdict
        if m.blocked_trade_count == 0:
            verdict = '<span class="badge badge-pass">NO BLOCKS</span>'
        elif lr >= 0.6 and m.allowed_trade_ratio >= 0.7:
            verdict = '<span class="badge badge-pass">PASS</span>'
        else:
            verdict = '<span class="badge badge-fail">FAIL</span>'

        rows.append(f"""
            <tr>
                <td><a class="symbol-link" href="{r.symbol}_{{link_stub}}.html">{r.symbol}</a></td>
                <td>{sector}</td>
                <td><span class="badge {policy_badge_class}">{policy_str}</span></td>
                <td>{m.baseline_trade_count}</td>
                <td>{m.blocked_trade_count}</td>
                <td>{m.size_down_count}</td>
                <td>{m.bypass_count}</td>
                <td class="{lr_class}">{lr:.0%}</td>
                <td class="{pnl_class}">{m.blocked_trade_avg_pnl:+.2%}</td>
                <td class="{fr_class}">{m.allowed_trade_ratio:.0%}</td>
                <td>{verdict}</td>
            </tr>
        """)

    return f"""
    <h2>Cross-Symbol Comparison</h2>
    <table id="symbol-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Sector</th>
                <th>Policy</th>
                <th>Baseline</th>
                <th>Blocked</th>
                <th>Size Down</th>
                <th>Bypass</th>
                <th>Loss Ratio</th>
                <th>Blocked PnL</th>
                <th>Freedom</th>
                <th>Verdict</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _summary_policy_breakdown(results: List[SymbolResult]) -> str:
    """Show BLOCK vs SIZE_DOWN vs BYPASS behavior comparison with resolved PnL."""
    # Aggregate by action type
    action_stats: Dict[str, Dict[str, float]] = {}
    for r in results:
        for d in r.decisions:
            action = d.action
            if action not in ("BLOCK", "SIZE_DOWN", "BYPASS"):
                continue
            if action not in action_stats:
                action_stats[action] = {
                    "count": 0,
                    "pnl_sum": 0.0,
                    "pnl_count": 0,
                    "resolved": 0,
                }
            action_stats[action]["count"] += 1
            if d.virtual_pnl_pct is not None:
                action_stats[action]["pnl_sum"] += d.virtual_pnl_pct
                action_stats[action]["pnl_count"] += 1
                action_stats[action]["resolved"] += 1

    if not action_stats:
        return ""

    rows = []
    for action in ["BLOCK", "SIZE_DOWN", "BYPASS"]:
        stats = action_stats.get(action)
        if not stats:
            continue
        avg_pnl = stats["pnl_sum"] / stats["pnl_count"] if stats["pnl_count"] > 0 else 0.0
        resolved = int(stats["resolved"])
        total = int(stats["count"])
        pnl_class = "pass" if avg_pnl <= 0 else "fail"
        note = {
            "BLOCK": "Entry fully prevented",
            "SIZE_DOWN": f"Entry at reduced size ({resolved}/{total} resolved)",
            "BYPASS": f"Gate bypassed ({resolved}/{total} resolved)",
        }.get(action, "")
        rows.append(f"""
            <tr>
                <td><strong>{action}</strong></td>
                <td>{total}</td>
                <td class="{pnl_class}">{avg_pnl:+.2%}</td>
                <td style="color:#94a3b8;">{note}</td>
            </tr>
        """)

    return f"""
    <h2>Policy Action Breakdown</h2>
    <table>
        <thead>
            <tr><th>Action</th><th>Count</th><th>Avg Virtual PnL</th><th>Note</th></tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    """


def _summary_loss_ratio_chart(results: List[SymbolResult]) -> str:
    symbols = [r.symbol for r in results]
    loss_ratios = [r.metrics.blocked_trade_loss_ratio for r in results]
    colors = ["#10b981" if lr >= 0.6 else "#ef4444" for lr in loss_ratios]

    return f"""
    <h2>Blocked Trade Loss Ratio by Symbol</h2>
    <div class="chart" id="loss-ratio-chart"></div>
    <script>
    Plotly.newPlot('loss-ratio-chart', [{{
        x: {json.dumps(symbols)},
        y: {json.dumps(loss_ratios)},
        type: 'bar',
        marker: {{color: {json.dumps(colors)}}},
    }}], {{
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font: {{color: '#e2e8f0'}},
        xaxis: {{gridcolor: '#1e293b'}},
        yaxis: {{gridcolor: '#1e293b', title: 'Loss Ratio', range: [0, 1.05]}},
        shapes: [{{
            type: 'line', x0: -0.5, x1: {len(symbols) - 0.5},
            y0: 0.6, y1: 0.6,
            line: {{color: '#f59e0b', width: 2, dash: 'dash'}},
        }}],
        annotations: [{{
            x: {len(symbols) - 1}, y: 0.62,
            text: '60% threshold', showarrow: false,
            font: {{color: '#f59e0b', size: 10}},
        }}],
        margin: {{t: 10, r: 20, b: 60, l: 60}},
        height: 300,
    }});
    </script>
    """


def _summary_blocked_pnl_scatter(results: List[SymbolResult]) -> str:
    """Scatter: x=symbol, y=blocked trade virtual PnL, color=correct/incorrect block."""
    x_vals: List[str] = []
    y_vals: List[float] = []
    colors: List[str] = []
    texts: List[str] = []

    for r in results:
        for d in r.decisions:
            if d.allowed or d.virtual_pnl_pct is None:
                continue
            x_vals.append(r.symbol)
            y_vals.append(d.virtual_pnl_pct)
            correct = d.virtual_pnl_pct < 0
            colors.append("#10b981" if correct else "#ef4444")
            texts.append(
                f"{d.timestamp}<br>PnL: {d.virtual_pnl_pct:+.2%}<br>"
                f"{'Correct block' if correct else 'Missed opportunity'}"
            )

    if not x_vals:
        return ""

    return f"""
    <h2>Blocked Trade Virtual PnL</h2>
    <div class="chart" id="blocked-pnl-scatter"></div>
    <script>
    Plotly.newPlot('blocked-pnl-scatter', [{{
        x: {json.dumps(x_vals)},
        y: {json.dumps(y_vals)},
        text: {json.dumps(texts)},
        type: 'scatter', mode: 'markers',
        marker: {{color: {json.dumps(colors)}, size: 10}},
        hovertemplate: '%{{text}}<extra></extra>',
    }}], {{
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font: {{color: '#e2e8f0'}},
        xaxis: {{gridcolor: '#1e293b', title: 'Symbol'}},
        yaxis: {{gridcolor: '#1e293b', title: 'Virtual PnL', tickformat: '.1%'}},
        shapes: [{{
            type: 'line', x0: -0.5, x1: {len(set(x_vals)) - 0.5},
            y0: 0, y1: 0,
            line: {{color: '#475569', width: 1, dash: 'dash'}},
        }}],
        margin: {{t: 10, r: 20, b: 60, l: 60}},
        height: 350,
    }});
    </script>
    """


def _summary_suggestions(
    results: List[SymbolResult],
    agg: _AggMetrics,
    params: Dict[str, Any],
) -> str:
    items = []
    recs = []

    # Aggregate pass/fail
    if agg.symbols_passing == agg.total_symbols:
        items.append('<div class="suggestion-item pass">All symbols pass constraints</div>')
    else:
        failing = [
            r.symbol
            for r in results
            if r.metrics.blocked_trade_count > 0
            and (r.metrics.blocked_trade_loss_ratio < 0.6 or r.metrics.allowed_trade_ratio < 0.7)
        ]
        items.append(
            f'<div class="suggestion-item fail">'
            f'{agg.symbols_failing} symbols fail: {", ".join(failing)}</div>'
        )

    if agg.avg_loss_ratio >= 0.6:
        items.append(
            f'<div class="suggestion-item pass">'
            f"Avg loss ratio: {agg.avg_loss_ratio:.0%} (≥60%)</div>"
        )
    else:
        items.append(
            f'<div class="suggestion-item fail">'
            f"Avg loss ratio: {agg.avg_loss_ratio:.0%} (&lt;60%)</div>"
        )

    # Symbols with zero blocks
    no_blocks = [r.symbol for r in results if r.metrics.blocked_trade_count == 0]
    if no_blocks:
        recs.append(
            f"Gate never triggered for: {', '.join(no_blocks)}. "
            f"These symbols may not enter DETERIORATING/BEARISH trend states "
            f"at MA cross entry points."
        )

    # Symbols where gate blocked profitable trades
    bad_blocks = [
        r.symbol
        for r in results
        if r.metrics.blocked_trade_count > 0 and r.metrics.blocked_trade_avg_pnl > 0
    ]
    if bad_blocks:
        recs.append(
            f"Gate blocked profitable trades for: {', '.join(bad_blocks)}. "
            f"Consider these symbols may not suit the DualMACD gate with current params."
        )

    recs.append(f"Parameters: {', '.join(f'{k}={v}' for k, v in params.items())}")

    rec_html = "\n".join(f"<div>• {r}</div>" for r in recs)

    return f"""
    <div class="suggestions">
        <h2 style="margin-top:0;">Suggestions</h2>
        {''.join(items)}
        <div class="recommendation">
            <strong>Recommendations:</strong>
            {rec_html}
        </div>
    </div>
    """
