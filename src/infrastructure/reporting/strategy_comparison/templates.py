"""
HTML templates for strategy comparison dashboard.

Renders a self-contained interactive HTML file with Plotly charts.
Uses the same dark theme as the existing signal report package.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def render_comparison_html(data: Dict[str, Any]) -> str:
    """
    Render the full comparison dashboard HTML.

    Args:
        data: Dashboard data with strategies, metrics, etc.

    Returns:
        Complete HTML string.
    """
    strategies_json = json.dumps(data["strategies"], indent=2)
    strategy_names = list(data["strategies"].keys())

    # Build scorecard HTML
    scorecards = ""
    for name, m in data["strategies"].items():
        tier_badge = f'<span class="tier-badge">{m.get("tier", "")}</span>' if m.get("tier") else ""
        scorecards += f"""
        <div class="scorecard">
            <h3>{name} {tier_badge}</h3>
            <div class="metric">
                <span class="label">Sharpe</span>
                <span class="value {'positive' if m['sharpe'] > 0 else 'negative'}">{m['sharpe']:.2f}</span>
            </div>
            <div class="metric">
                <span class="label">Return</span>
                <span class="value {'positive' if m['total_return'] > 0 else 'negative'}">{m['total_return']:+.1%}</span>
            </div>
            <div class="metric">
                <span class="label">MaxDD</span>
                <span class="value negative">{m['max_drawdown']:.1%}</span>
            </div>
            <div class="metric">
                <span class="label">WinRate</span>
                <span class="value">{m['win_rate']:.0%}</span>
            </div>
            <div class="metric">
                <span class="label">Trades</span>
                <span class="value">{m['trade_count']}</span>
            </div>
        </div>"""

    # Build metrics table rows
    metrics_rows = ""
    for name, m in data["strategies"].items():
        metrics_rows += f"""
        <tr>
            <td class="strategy-name">{name}</td>
            <td class="{'best' if m['sharpe'] == max(s['sharpe'] for s in data['strategies'].values()) else ''}">{m['sharpe']:.2f}</td>
            <td>{m['sortino']:.2f}</td>
            <td>{m['calmar']:.2f}</td>
            <td>{m['max_drawdown']:.1%}</td>
            <td>{m['win_rate']:.0%}</td>
            <td>{m['profit_factor']:.2f}</td>
            <td>{m['trade_count']}</td>
        </tr>"""

    # Build stress table
    stress_rows = ""
    stress_windows = [
        "covid_crash",
        "bear_2022",
        "ai_meltup_2023",
        "regional_bank_2023",
        "aug_2024_unwind",
    ]
    stress_labels = {
        "covid_crash": "COVID (2020-03)",
        "bear_2022": "Bear 2022",
        "ai_meltup_2023": "AI Melt-up 2023",
        "regional_bank_2023": "Regional Bank 2023",
        "aug_2024_unwind": "Aug 2024 Unwind",
    }
    for window in stress_windows:
        row = f"<tr><td>{stress_labels.get(window, window)}</td>"
        for name, m in data["strategies"].items():
            stress = m.get("stress_results", {}).get(window, {})
            dd = stress.get("max_drawdown", 0)
            ret = stress.get("total_return", 0)
            val = f"{ret:+.1%}" if ret != 0 else "N/A"
            row += f'<td class="{"positive" if ret > 0 else "negative"}">{val}</td>'
        row += "</tr>"
        stress_rows += row

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{data['title']}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --positive: #3fb950;
    --negative: #f85149;
    --warning: #d29922;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
}}

.header {{
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header .meta {{ color: var(--text-secondary); font-size: 13px; }}

.tabs {{
    display: flex;
    gap: 4px;
    padding: 8px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
}}
.tab {{
    padding: 8px 16px;
    cursor: pointer;
    border-radius: 6px 6px 0 0;
    color: var(--text-secondary);
    font-size: 14px;
    font-weight: 500;
    transition: all 0.15s;
}}
.tab:hover {{ color: var(--text-primary); background: var(--bg-card); }}
.tab.active {{ color: var(--accent); background: var(--bg-card); border-bottom: 2px solid var(--accent); }}

.tab-content {{ display: none; padding: 24px; }}
.tab-content.active {{ display: block; }}

.scorecards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}
.scorecard {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}}
.scorecard h3 {{
    font-size: 16px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.tier-badge {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--accent);
    color: var(--bg-primary);
    font-weight: 600;
}}
.metric {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 14px;
}}
.metric .label {{ color: var(--text-secondary); }}
.metric .value.positive {{ color: var(--positive); }}
.metric .value.negative {{ color: var(--negative); }}

.chart-container {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
}}

.charts-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border-radius: 8px;
    overflow: hidden;
}}
th, td {{
    padding: 10px 16px;
    text-align: right;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
}}
th {{
    background: var(--bg-secondary);
    color: var(--text-secondary);
    font-weight: 600;
    text-align: right;
}}
th:first-child, td:first-child {{ text-align: left; }}
td.strategy-name {{ font-weight: 600; color: var(--accent); }}
td.best {{ color: var(--positive); font-weight: 600; }}
td.positive {{ color: var(--positive); }}
td.negative {{ color: var(--negative); }}

.no-data {{
    color: var(--text-secondary);
    text-align: center;
    padding: 40px;
    font-size: 14px;
}}
</style>
</head>
<body>

<div class="header">
    <h1>{data['title']}</h1>
    <div class="meta">
        Generated: {data['generated_at']} |
        Universe: {data.get('universe_name', 'N/A')} ({data.get('strategy_count', 0)} strategies) |
        Period: {data.get('period', 'N/A')}
    </div>
</div>

<div class="tabs">
    <div class="tab active" onclick="switchTab('overview', this)">Overview</div>
    <div class="tab" onclick="switchTab('metrics', this)">Metrics</div>
    <div class="tab" onclick="switchTab('regime', this)">Regime Performance</div>
    <div class="tab" onclick="switchTab('heatmap', this)">Per-Symbol Heatmap</div>
    <div class="tab" onclick="switchTab('trades', this)">Trade Analysis</div>
</div>

<!-- TAB 1: OVERVIEW -->
<div id="overview" class="tab-content active">
    <div class="scorecards">{scorecards}</div>
    <div class="chart-container">
        <div id="equity-chart" style="height: 400px;"></div>
    </div>
    <div class="chart-container">
        <div id="drawdown-chart" style="height: 250px;"></div>
    </div>
</div>

<!-- TAB 2: METRICS TABLE -->
<div id="metrics" class="tab-content">
    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Sharpe</th>
                <th>Sortino</th>
                <th>Calmar</th>
                <th>MaxDD</th>
                <th>WinRate</th>
                <th>ProfitF</th>
                <th>Trades</th>
            </tr>
        </thead>
        <tbody>{metrics_rows}</tbody>
    </table>
</div>

<!-- TAB 3: REGIME PERFORMANCE -->
<div id="regime" class="tab-content">
    <div class="charts-row">
        <div class="chart-container">
            <div id="regime-sharpe-chart" style="height: 400px;"></div>
        </div>
        <div class="chart-container">
            <div id="regime-return-chart" style="height: 400px;"></div>
        </div>
    </div>
    <h3 style="margin: 16px 0 8px;">Stress Window Survival</h3>
    <table>
        <thead>
            <tr>
                <th>Window</th>
                {''.join(f'<th>{name}</th>' for name in data['strategies'].keys())}
            </tr>
        </thead>
        <tbody>{stress_rows}</tbody>
    </table>
</div>

<!-- TAB 4: PER-SYMBOL HEATMAP -->
<div id="heatmap" class="tab-content">
    <div class="chart-container">
        <div id="symbol-heatmap" style="height: 500px;"></div>
    </div>
</div>

<!-- TAB 5: TRADE ANALYSIS -->
<div id="trades" class="tab-content">
    <div class="chart-container">
        <div id="rolling-sharpe-chart" style="height: 350px;"></div>
    </div>
    <div class="chart-container">
        <div id="monthly-returns-chart" style="height: 350px;"></div>
    </div>
</div>

<script>
const STRATEGIES = {strategies_json};
const STRATEGY_NAMES = {json.dumps(strategy_names)};

function switchTab(tabId, el) {{
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    el.classList.add('active');
    // Resize Plotly charts when tab becomes visible
    setTimeout(() => {{
        const container = document.getElementById(tabId);
        container.querySelectorAll('[id$="-chart"]').forEach(chartDiv => {{
            if (chartDiv.data) Plotly.Plots.resize(chartDiv);
        }});
    }}, 50);
}}

const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff', '#79c0ff', '#56d364', '#e3b341'];
const PLOTLY_LAYOUT_BASE = {{
    paper_bgcolor: '#1c2128',
    plot_bgcolor: '#1c2128',
    font: {{ color: '#8b949e' }},
    legend: {{ bgcolor: 'transparent' }},
    margin: {{ t: 40, r: 20, b: 40, l: 60 }},
}};

// --- TAB 1: Equity Curves ---
function renderEquityCurves() {{
    const traces = [];
    STRATEGY_NAMES.forEach((name, i) => {{
        const ec = STRATEGIES[name].equity_curve;
        if (ec && ec.length > 0) {{
            traces.push({{
                x: ec.map(p => new Date(p[0] * 1000)),
                y: ec.map(p => p[1]),
                name: name,
                type: 'scatter',
                mode: 'lines',
                line: {{ color: COLORS[i % COLORS.length], width: 2 }},
            }});
        }}
    }});

    if (traces.length > 0) {{
        Plotly.newPlot('equity-chart', traces, {{
            ...PLOTLY_LAYOUT_BASE,
            title: {{ text: 'Equity Curves (Normalized to 100)', font: {{ color: '#e6edf3' }} }},
            xaxis: {{ gridcolor: '#30363d' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Value' }},
        }});
    }} else {{
        document.getElementById('equity-chart').innerHTML = '<div class="no-data">No equity curve data available</div>';
    }}
}}

// --- TAB 1: Drawdown Chart ---
function renderDrawdowns() {{
    const traces = [];
    STRATEGY_NAMES.forEach((name, i) => {{
        const dc = STRATEGIES[name].drawdown_curve;
        if (dc && dc.length > 0) {{
            traces.push({{
                x: dc.map(p => new Date(p[0] * 1000)),
                y: dc.map(p => p[1] * 100),
                name: name,
                type: 'scatter',
                fill: 'tozeroy',
                line: {{ color: COLORS[i % COLORS.length], width: 1 }},
            }});
        }}
    }});

    if (traces.length > 0) {{
        Plotly.newPlot('drawdown-chart', traces, {{
            ...PLOTLY_LAYOUT_BASE,
            title: {{ text: 'Drawdown Comparison', font: {{ color: '#e6edf3' }} }},
            xaxis: {{ gridcolor: '#30363d' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Drawdown %', autorange: true }},
        }});
    }} else {{
        document.getElementById('drawdown-chart').innerHTML = '<div class="no-data">No drawdown data available</div>';
    }}
}}

// --- TAB 3: Regime Sharpe Chart ---
function renderRegimeSharpe() {{
    const regimes = ['R0', 'R1', 'R2', 'R3'];
    const traces = [];
    let hasData = false;
    STRATEGY_NAMES.forEach((name, i) => {{
        const data = STRATEGIES[name].per_regime_sharpe || {{}};
        const yVals = regimes.map(r => data[r] || 0);
        if (yVals.some(v => v !== 0)) hasData = true;
        traces.push({{
            x: regimes,
            y: yVals,
            name: name,
            type: 'bar',
            marker: {{ color: COLORS[i % COLORS.length] }},
        }});
    }});

    if (hasData) {{
        Plotly.newPlot('regime-sharpe-chart', traces, {{
            ...PLOTLY_LAYOUT_BASE,
            title: {{ text: 'Sharpe by Regime', font: {{ color: '#e6edf3' }} }},
            barmode: 'group',
            xaxis: {{ gridcolor: '#30363d', title: 'Regime' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Sharpe Ratio' }},
        }});
    }} else {{
        document.getElementById('regime-sharpe-chart').innerHTML = '<div class="no-data">No per-regime Sharpe data available (need 200+ bars for SMA-based regime detection)</div>';
    }}
}}

// --- TAB 3: Regime Return Chart ---
function renderRegimeReturn() {{
    const regimes = ['R0', 'R1', 'R2', 'R3'];
    const traces = [];
    let hasData = false;
    STRATEGY_NAMES.forEach((name, i) => {{
        const data = STRATEGIES[name].per_regime_return || {{}};
        const yVals = regimes.map(r => (data[r] || 0) * 100);
        if (yVals.some(v => v !== 0)) hasData = true;
        traces.push({{
            x: regimes,
            y: yVals,
            name: name,
            type: 'bar',
            marker: {{ color: COLORS[i % COLORS.length] }},
        }});
    }});

    if (hasData) {{
        Plotly.newPlot('regime-return-chart', traces, {{
            ...PLOTLY_LAYOUT_BASE,
            title: {{ text: 'Total Return by Regime', font: {{ color: '#e6edf3' }} }},
            barmode: 'group',
            xaxis: {{ gridcolor: '#30363d', title: 'Regime' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Return %' }},
        }});
    }} else {{
        document.getElementById('regime-return-chart').innerHTML = '<div class="no-data">No per-regime return data available</div>';
    }}
}}

// --- TAB 4: Per-symbol Heatmap ---
function renderSymbolHeatmap() {{
    const symbols = {json.dumps(data.get('symbols', []))};
    if (symbols.length === 0) return;

    const z = [];
    const text = [];
    STRATEGY_NAMES.forEach(name => {{
        const row = [];
        const textRow = [];
        symbols.forEach(sym => {{
            const val = (STRATEGIES[name].per_symbol_sharpe || {{}})[sym] || 0;
            row.push(val);
            textRow.push(val.toFixed(2));
        }});
        z.push(row);
        text.push(textRow);
    }});

    Plotly.newPlot('symbol-heatmap', [{{
        z: z,
        x: symbols,
        y: STRATEGY_NAMES,
        type: 'heatmap',
        colorscale: [[0, '#f85149'], [0.5, '#d29922'], [1, '#3fb950']],
        text: text,
        texttemplate: '%{{text}}',
        hovertemplate: '%{{y}}<br>%{{x}}: %{{z:.2f}}<extra></extra>',
    }}], {{
        ...PLOTLY_LAYOUT_BASE,
        title: {{ text: 'Per-Symbol Sharpe Ratio', font: {{ color: '#e6edf3' }} }},
        margin: {{ t: 40, r: 20, b: 80, l: 120 }},
    }});
}}

// --- TAB 5: Rolling Sharpe ---
function renderRollingSharpe() {{
    const traces = [];
    let hasData = false;
    STRATEGY_NAMES.forEach((name, i) => {{
        const rs = STRATEGIES[name].rolling_sharpe;
        if (rs && rs.length > 0) {{
            hasData = true;
            traces.push({{
                x: rs.map(p => new Date(p[0] * 1000)),
                y: rs.map(p => p[1]),
                name: name,
                type: 'scatter',
                mode: 'lines',
                line: {{ color: COLORS[i % COLORS.length], width: 1.5 }},
            }});
        }}
    }});

    // Add zero reference line
    if (hasData) {{
        traces.push({{
            x: traces[0].x,
            y: traces[0].x.map(() => 0),
            name: 'Zero',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#30363d', width: 1, dash: 'dash' }},
            showlegend: false,
        }});
        Plotly.newPlot('rolling-sharpe-chart', traces, {{
            ...PLOTLY_LAYOUT_BASE,
            title: {{ text: 'Rolling 60-Day Sharpe Ratio', font: {{ color: '#e6edf3' }} }},
            xaxis: {{ gridcolor: '#30363d' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Sharpe Ratio' }},
        }});
    }} else {{
        document.getElementById('rolling-sharpe-chart').innerHTML = '<div class="no-data">No rolling Sharpe data available (need 60+ bars)</div>';
    }}
}}

// --- TAB 5: Monthly Returns Heatmap ---
function renderMonthlyReturns() {{
    // Collect all months across strategies
    const allMonths = new Set();
    STRATEGY_NAMES.forEach(name => {{
        const mr = STRATEGIES[name].monthly_returns || {{}};
        Object.keys(mr).forEach(m => allMonths.add(m));
    }});
    const months = Array.from(allMonths).sort();
    if (months.length === 0) {{
        document.getElementById('monthly-returns-chart').innerHTML = '<div class="no-data">No monthly return data available</div>';
        return;
    }}

    const z = [];
    const text = [];
    STRATEGY_NAMES.forEach(name => {{
        const mr = STRATEGIES[name].monthly_returns || {{}};
        const row = [];
        const textRow = [];
        months.forEach(m => {{
            const val = (mr[m] || 0) * 100;
            row.push(val);
            textRow.push(val.toFixed(1) + '%');
        }});
        z.push(row);
        text.push(textRow);
    }});

    // Symmetrical color scale centered at 0
    const maxAbs = Math.max(...z.flat().map(Math.abs), 1);
    Plotly.newPlot('monthly-returns-chart', [{{
        z: z,
        x: months,
        y: STRATEGY_NAMES,
        type: 'heatmap',
        colorscale: [[0, '#f85149'], [0.5, '#1c2128'], [1, '#3fb950']],
        zmin: -maxAbs,
        zmax: maxAbs,
        text: text,
        texttemplate: '%{{text}}',
        hovertemplate: '%{{y}}<br>%{{x}}: %{{z:.1f}}%<extra></extra>',
    }}], {{
        ...PLOTLY_LAYOUT_BASE,
        title: {{ text: 'Monthly Returns (%)', font: {{ color: '#e6edf3' }} }},
        margin: {{ t: 40, r: 20, b: 80, l: 120 }},
        xaxis: {{ tickangle: -45 }},
    }});
}}

// Initialize all charts
renderEquityCurves();
renderDrawdowns();
renderRegimeSharpe();
renderRegimeReturn();
renderSymbolHeatmap();
renderRollingSharpe();
renderMonthlyReturns();
</script>
</body>
</html>"""
