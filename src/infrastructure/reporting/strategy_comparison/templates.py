"""
HTML templates for strategy comparison dashboard.

Renders a self-contained interactive HTML file with Plotly charts.
Uses the same dark theme as the existing signal report package.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _classify_health(name: str, m: Dict[str, Any]) -> tuple[str, str]:
    """Return (css_class, label) for a strategy's health badge."""
    if name == "buy_and_hold":
        return "health-baseline", "BASELINE"
    tc = m.get("trade_count", 0)
    wr = m.get("win_rate", 0)
    tr = m.get("total_return", 0)
    mdd = m.get("max_drawdown", 0)
    sr = m.get("sharpe", 0)
    if tc == 0 or wr < 0.15 or tr < -0.05 or mdd < -0.50:
        return "health-red", "BROKEN"
    if wr < 0.30 or sr < 0.1 or tr < 0:
        return "health-yellow", "NEEDS WORK"
    return "health-green", "HEALTHY"


def _param_badge(m: Dict[str, Any], is_baseline: bool) -> tuple[str, str]:
    """Return (display_text, css_class) for the param budget badge."""
    eff_p = m.get("effective_params", 0)
    tot_p = m.get("total_params", 0)
    if is_baseline or tot_p == 0:
        return "–", ""
    if eff_p <= 5:
        return f"{eff_p}/{8}", "positive"
    if eff_p <= 8:
        return f"{eff_p}/{8} &#9888;", "warning-text"
    return f"{eff_p}/{8} &#10060;", "negative"


def _build_scorecards_html(data: Dict[str, Any]) -> str:
    """Build the scorecard div for each strategy."""
    scorecards = ""
    for name, m in data["strategies"].items():
        is_baseline = name == "buy_and_hold"
        tier_badge = (
            f'<span class="tier-badge">{m.get("tier", "")}</span>'
            if m.get("tier") and not is_baseline
            else ""
        )
        health_cls, health_txt = _classify_health(name, m)
        health_badge = f'<span class="health-badge {health_cls}">{health_txt}</span>'
        wr_display = "–" if is_baseline else f"{m['win_rate']:.0%}"
        trades_display = "–" if is_baseline else str(m["trade_count"])
        param_display, param_cls = _param_badge(m, is_baseline)
        scorecards += f"""
        <div class="scorecard">
            <h3>{name} {tier_badge} {health_badge}</h3>
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
                <span class="value">{wr_display}</span>
            </div>
            <div class="metric">
                <span class="label">Trades</span>
                <span class="value">{trades_display}</span>
            </div>
            <div class="metric">
                <span class="label">Params</span>
                <span class="value {param_cls}">{param_display}</span>
            </div>
        </div>"""
    return scorecards


def _build_metrics_rows_html(data: Dict[str, Any]) -> tuple[str, str]:
    """Build metrics table <tr> rows. Returns (rows_html, tier_b_header)."""
    has_tier_b = any(m.get("tier_b_sharpe") is not None for m in data["strategies"].values())
    metrics_rows = ""
    for name, m in data["strategies"].items():
        is_baseline = name == "buy_and_hold"
        h_cls, h_txt = _classify_health(name, m)
        wr_display = "–" if is_baseline else f"{m['win_rate']:.0%}"
        pf_display = "–" if is_baseline else f"{m['profit_factor']:.2f}"
        trades_display = "–" if is_baseline else str(m["trade_count"])
        m_param_display, m_param_cls = _param_badge(m, is_baseline)
        sr = m.get("sharpe", 0)

        tier_b_cell = ""
        if has_tier_b:
            tier_b_sharpe = m.get("tier_b_sharpe")
            if tier_b_sharpe is not None and sr != 0:
                ratio = tier_b_sharpe / sr if sr != 0 else 0.0
                if ratio > 0.50:
                    tb_cls = "positive"
                    tb_label = "PASS"
                else:
                    tb_cls = "negative"
                    tb_label = "FAIL"
                delta_pct = (ratio - 1.0) * 100
                tier_b_cell = (
                    f'<td class="{tb_cls}">'
                    f"{tier_b_sharpe:.2f} ({delta_pct:+.0f}%) "
                    f'<span class="health-badge health-{"green" if ratio > 0.50 else "red"}">'
                    f"{tb_label}</span></td>"
                )
            else:
                tier_b_cell = "<td>–</td>"

        metrics_rows += f"""
        <tr>
            <td class="strategy-name">{name} <span class="health-badge {h_cls}">{h_txt}</span></td>
            <td class="{'best' if m['sharpe'] == max(s['sharpe'] for s in data['strategies'].values()) else ''}">{m['sharpe']:.2f}</td>
            <td>{m['sortino']:.2f}</td>
            <td>{m['calmar']:.2f}</td>
            <td>{m['max_drawdown']:.1%}</td>
            <td>{wr_display}</td>
            <td>{pf_display}</td>
            <td>{trades_display}</td>
            <td class="{m_param_cls}">{m_param_display}</td>
            {tier_b_cell}
        </tr>"""

    tier_b_header = "<th>Tier B &#916;</th>" if has_tier_b else ""
    return metrics_rows, tier_b_header


def _build_stress_rows_html(data: Dict[str, Any]) -> str:
    """Build stress table rows for each stress window."""
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
            ret = stress.get("total_return", 0)
            val = f"{ret:+.1%}" if ret != 0 else "N/A"
            row += f'<td class="{"positive" if ret > 0 else "negative"}">{val}</td>'
        row += "</tr>"
        stress_rows += row
    return stress_rows


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
    sector_map_json = json.dumps(data.get("sector_map", {}), indent=2)

    scorecards = _build_scorecards_html(data)
    metrics_rows, tier_b_header = _build_metrics_rows_html(data)
    stress_rows = _build_stress_rows_html(data)

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
    flex-wrap: wrap;
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
.health-badge {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
    color: var(--bg-primary);
}}
.health-green {{ background: var(--positive); }}
.health-yellow {{ background: var(--warning); }}
.health-red {{ background: var(--negative); }}
.health-baseline {{ background: var(--accent); }}
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
td.warning-text {{ color: var(--warning); }}
.metric .value.warning-text {{ color: var(--warning); }}

.no-data {{
    color: var(--text-secondary);
    text-align: center;
    padding: 40px;
    font-size: 14px;
}}

.filter-bar {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 16px;
    padding: 12px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
}}
.filter-bar label {{ color: var(--text-secondary); font-size: 13px; font-weight: 600; }}
.filter-bar select {{
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
}}
.filter-bar .count {{ color: var(--text-secondary); font-size: 12px; margin-left: auto; }}

#stock-table th {{ cursor: pointer; user-select: none; white-space: nowrap; }}
#stock-table th:hover {{ color: var(--text-primary); }}
#stock-table th.sorted-asc::after {{ content: ' \\25B2'; font-size: 10px; }}
#stock-table th.sorted-desc::after {{ content: ' \\25BC'; font-size: 10px; }}
td.symbol {{ font-weight: 600; color: var(--text-primary); }}
.section-title {{ font-size: 15px; font-weight: 600; margin: 0 0 12px; }}
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
    <div class="tab" onclick="switchTab('perstock', this)">Per-Stock</div>
    <div class="tab" onclick="switchTab('sector', this)">Sector</div>
    <div class="tab" onclick="switchTab('regime', this)">Regime Performance</div>
    <div class="tab" onclick="switchTab('regime-analysis', this)">Regime Analysis</div>
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
                <th>Params</th>
                {tier_b_header}
            </tr>
        </thead>
        <tbody>{metrics_rows}</tbody>
    </table>
</div>

<!-- TAB 3: PER-STOCK -->
<div id="perstock" class="tab-content">
    <div class="filter-bar">
        <label>Strategy:</label>
        <select id="stock-strategy-filter" onchange="filterStockTable()">
            <option value="all">All Strategies</option>
        </select>
        <label style="margin-left:16px">Sort by:</label>
        <select id="stock-sort-select" onchange="sortStockTableBySelect()">
            <option value="sharpe-desc">Sharpe (High to Low)</option>
            <option value="sharpe-asc">Sharpe (Low to High)</option>
            <option value="return-desc">Return (High to Low)</option>
            <option value="return-asc">Return (Low to High)</option>
            <option value="trades-desc">Trades (Most)</option>
            <option value="drawdown-asc">MaxDD (Smallest)</option>
        </select>
        <span class="count" id="stock-count"></span>
    </div>
    <table id="stock-table">
        <thead>
            <tr>
                <th onclick="sortStockTable('symbol')">Symbol</th>
                <th onclick="sortStockTable('strategy')">Strategy</th>
                <th onclick="sortStockTable('sharpe')">Sharpe</th>
                <th onclick="sortStockTable('total_return')">Return</th>
                <th onclick="sortStockTable('max_drawdown')">MaxDD</th>
                <th onclick="sortStockTable('win_rate')">WinRate</th>
                <th onclick="sortStockTable('total_trades')">Trades</th>
                <th onclick="sortStockTable('sortino')">Sortino</th>
                <th onclick="sortStockTable('profit_factor')">PF</th>
            </tr>
        </thead>
        <tbody id="stock-table-body"></tbody>
    </table>
</div>

<!-- TAB 4: SECTOR -->
<div id="sector" class="tab-content">
    <div class="charts-row">
        <div class="chart-container">
            <div id="sector-sharpe-heatmap" style="height: 400px;"></div>
        </div>
        <div class="chart-container">
            <div id="sector-return-heatmap" style="height: 400px;"></div>
        </div>
    </div>
    <h3 class="section-title">Sector Detail</h3>
    <table id="sector-table">
        <thead id="sector-table-head"></thead>
        <tbody id="sector-table-body"></tbody>
    </table>
</div>

<!-- TAB 5: REGIME PERFORMANCE -->
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
    <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 8px;">
        N/A = data period does not cover this stress window. 3yr data typically covers Aug 2024 only.
        For full stress coverage, use 5yr+ data (2019+).
    </p>
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

<!-- TAB: REGIME ANALYSIS (Trade Decomposition) -->
<div id="regime-analysis" class="tab-content">
    <div class="charts-row">
        <div class="chart-container">
            <div id="regime-trades-chart" style="height: 400px;"></div>
        </div>
        <div class="chart-container">
            <div id="regime-wr-chart" style="height: 400px;"></div>
        </div>
    </div>
    <h3 class="section-title">Per-Regime Trade Breakdown</h3>
    <table id="regime-analysis-table">
        <thead id="regime-analysis-head"></thead>
        <tbody id="regime-analysis-body"></tbody>
    </table>
    <div id="regime-alerts" style="margin-top: 16px;"></div>
</div>

<!-- TAB 6: PER-SYMBOL HEATMAP -->
<div id="heatmap" class="tab-content">
    <div class="chart-container">
        <div id="symbol-heatmap" style="height: 500px;"></div>
    </div>
</div>

<!-- TAB 7: TRADE ANALYSIS -->
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
const SECTOR_MAP = {sector_map_json};

function switchTab(tabId, el) {{
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    el.classList.add('active');
    // Resize Plotly charts when tab becomes visible
    setTimeout(() => {{
        const container = document.getElementById(tabId);
        container.querySelectorAll('[id$="-chart"], [id$="-heatmap"]').forEach(chartDiv => {{
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

// --- TAB 3: Per-Stock Table ---
let stockRows = [];
let currentSort = {{ col: 'sharpe', asc: false }};

function buildStockRows() {{
    stockRows = [];
    STRATEGY_NAMES.forEach(name => {{
        const psm = STRATEGIES[name].per_symbol_metrics || {{}};
        Object.keys(psm).forEach(sym => {{
            const m = psm[sym];
            stockRows.push({{
                symbol: sym,
                strategy: name,
                sharpe: m.sharpe || 0,
                total_return: m.total_return || 0,
                max_drawdown: m.max_drawdown || 0,
                win_rate: m.win_rate || 0,
                total_trades: m.total_trades || 0,
                sortino: m.sortino || 0,
                calmar: m.calmar || 0,
                profit_factor: m.profit_factor || 0,
            }});
        }});
    }});
}}

function renderStockTable() {{
    const filter = document.getElementById('stock-strategy-filter').value;
    let rows = filter === 'all' ? stockRows.slice() : stockRows.filter(r => r.strategy === filter);

    rows.sort((a, b) => {{
        let va = a[currentSort.col], vb = b[currentSort.col];
        if (typeof va === 'string') {{ va = va.toLowerCase(); vb = vb.toLowerCase(); }}
        if (va < vb) return currentSort.asc ? -1 : 1;
        if (va > vb) return currentSort.asc ? 1 : -1;
        return 0;
    }});

    const tbody = document.getElementById('stock-table-body');
    let html = '';
    rows.forEach(r => {{
        const sc = r.sharpe > 0 ? 'positive' : 'negative';
        const rc = r.total_return > 0 ? 'positive' : 'negative';
        const soc = r.sortino > 0 ? 'positive' : 'negative';
        html += '<tr>';
        html += '<td class="symbol">' + r.symbol + '</td>';
        html += '<td class="strategy-name">' + r.strategy + '</td>';
        html += '<td class="' + sc + '">' + r.sharpe.toFixed(2) + '</td>';
        html += '<td class="' + rc + '">' + (r.total_return * 100).toFixed(1) + '%</td>';
        html += '<td class="negative">' + (r.max_drawdown * 100).toFixed(1) + '%</td>';
        html += '<td>' + (r.win_rate * 100).toFixed(0) + '%</td>';
        html += '<td>' + r.total_trades + '</td>';
        html += '<td class="' + soc + '">' + r.sortino.toFixed(2) + '</td>';
        html += '<td>' + r.profit_factor.toFixed(2) + '</td>';
        html += '</tr>';
    }});
    tbody.innerHTML = html;

    document.getElementById('stock-count').textContent = rows.length + ' rows';

    // Update header sort indicators
    document.querySelectorAll('#stock-table th').forEach(th => {{
        th.classList.remove('sorted-asc', 'sorted-desc');
    }});
}}

function sortStockTable(col) {{
    if (currentSort.col === col) {{
        currentSort.asc = !currentSort.asc;
    }} else {{
        currentSort = {{ col: col, asc: col === 'symbol' || col === 'strategy' }};
    }}
    renderStockTable();
}}

function sortStockTableBySelect() {{
    const val = document.getElementById('stock-sort-select').value;
    const parts = val.split('-');
    const col = parts[0];
    const dir = parts[1];
    const colMap = {{ sharpe: 'sharpe', return: 'total_return', trades: 'total_trades', drawdown: 'max_drawdown' }};
    currentSort = {{ col: colMap[col] || col, asc: dir === 'asc' }};
    renderStockTable();
}}

function filterStockTable() {{
    renderStockTable();
}}

function initStockFilter() {{
    const select = document.getElementById('stock-strategy-filter');
    STRATEGY_NAMES.forEach(name => {{
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    }});
}}

// --- TAB 4: Sector ---
function computeSectorMetrics() {{
    const result = {{}};
    STRATEGY_NAMES.forEach(name => {{
        result[name] = {{}};
        const psm = STRATEGIES[name].per_symbol_metrics || {{}};
        Object.keys(SECTOR_MAP).forEach(sector => {{
            const symbols = SECTOR_MAP[sector];
            const valid = symbols.filter(s => psm[s]).map(s => psm[s]);
            if (valid.length > 0) {{
                const avg = (arr, key) => arr.reduce((s, v) => s + v[key], 0) / arr.length;
                const sum = (arr, key) => arr.reduce((s, v) => s + v[key], 0);
                result[name][sector] = {{
                    sharpe: avg(valid, 'sharpe'),
                    total_return: avg(valid, 'total_return'),
                    max_drawdown: avg(valid, 'max_drawdown'),
                    win_rate: avg(valid, 'win_rate'),
                    total_trades: sum(valid, 'total_trades'),
                    count: valid.length,
                }};
            }}
        }});
    }});
    return result;
}}

function renderSectorSharpe() {{
    const sectorMetrics = computeSectorMetrics();
    const sectors = Object.keys(SECTOR_MAP);
    if (sectors.length === 0) {{
        document.getElementById('sector-sharpe-heatmap').innerHTML = '<div class="no-data">No sector data available</div>';
        return;
    }}

    const z = [];
    const text = [];
    STRATEGY_NAMES.forEach(name => {{
        const row = [];
        const textRow = [];
        sectors.forEach(sector => {{
            const sm = sectorMetrics[name] && sectorMetrics[name][sector];
            const val = sm ? sm.sharpe : 0;
            row.push(val);
            textRow.push(val.toFixed(2));
        }});
        z.push(row);
        text.push(textRow);
    }});

    Plotly.newPlot('sector-sharpe-heatmap', [{{
        z: z,
        x: sectors.map(s => s.replace(/_/g, ' ')),
        y: STRATEGY_NAMES,
        type: 'heatmap',
        colorscale: [[0, '#f85149'], [0.5, '#d29922'], [1, '#3fb950']],
        text: text,
        texttemplate: '%{{text}}',
        hovertemplate: '%{{y}}<br>%{{x}}: Sharpe %{{z:.2f}}<extra></extra>',
    }}], {{
        ...PLOTLY_LAYOUT_BASE,
        title: {{ text: 'Avg Sharpe by Sector', font: {{ color: '#e6edf3' }} }},
        margin: {{ t: 40, r: 20, b: 100, l: 120 }},
        xaxis: {{ tickangle: -30 }},
    }});
}}

function renderSectorReturn() {{
    const sectorMetrics = computeSectorMetrics();
    const sectors = Object.keys(SECTOR_MAP);
    if (sectors.length === 0) {{
        document.getElementById('sector-return-heatmap').innerHTML = '<div class="no-data">No sector data available</div>';
        return;
    }}

    const z = [];
    const text = [];
    STRATEGY_NAMES.forEach(name => {{
        const row = [];
        const textRow = [];
        sectors.forEach(sector => {{
            const sm = sectorMetrics[name] && sectorMetrics[name][sector];
            const val = (sm ? sm.total_return : 0) * 100;
            row.push(val);
            textRow.push(val.toFixed(1) + '%');
        }});
        z.push(row);
        text.push(textRow);
    }});

    const maxAbs = Math.max(...z.flat().map(Math.abs), 1);
    Plotly.newPlot('sector-return-heatmap', [{{
        z: z,
        x: sectors.map(s => s.replace(/_/g, ' ')),
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
        title: {{ text: 'Avg Return by Sector (%)', font: {{ color: '#e6edf3' }} }},
        margin: {{ t: 40, r: 20, b: 100, l: 120 }},
        xaxis: {{ tickangle: -30 }},
    }});
}}

function renderSectorTable() {{
    const sectorMetrics = computeSectorMetrics();
    const sectors = Object.keys(SECTOR_MAP);
    if (sectors.length === 0) return;

    const thead = document.getElementById('sector-table-head');
    let headHtml = '<tr><th>Strategy</th>';
    sectors.forEach(s => {{
        headHtml += '<th style="text-align:center" colspan="2">' + s.replace(/_/g, ' ') + '</th>';
    }});
    headHtml += '</tr><tr><th></th>';
    sectors.forEach(() => {{
        headHtml += '<th>Sharpe</th><th>Return</th>';
    }});
    headHtml += '</tr>';
    thead.innerHTML = headHtml;

    const tbody = document.getElementById('sector-table-body');
    let bodyHtml = '';
    STRATEGY_NAMES.forEach(name => {{
        let cells = '';
        sectors.forEach(sector => {{
            const m = sectorMetrics[name] && sectorMetrics[name][sector];
            if (!m) {{ cells += '<td>-</td><td>-</td>'; return; }}
            const sc = m.sharpe > 0 ? 'positive' : 'negative';
            const rc = m.total_return > 0 ? 'positive' : 'negative';
            cells += '<td class="' + sc + '">' + m.sharpe.toFixed(2) + '</td>';
            cells += '<td class="' + rc + '">' + (m.total_return * 100).toFixed(1) + '%</td>';
        }});
        bodyHtml += '<tr><td class="strategy-name">' + name + '</td>' + cells + '</tr>';
    }});
    tbody.innerHTML = bodyHtml;
}}

// --- TAB 5: Regime Sharpe Chart ---
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

// --- TAB 5: Regime Return Chart ---
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

// --- TAB 6: Per-symbol Heatmap ---
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

// --- TAB 7: Rolling Sharpe ---
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

// --- TAB 7: Monthly Returns Heatmap ---
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

// --- REGIME ANALYSIS: Trade Decomposition ---
const REGIME_LABELS = {{
    R0: 'R0 (Uptrend)',
    R1: 'R1 (Choppy)',
    R2: 'R2 (Risk-Off)',
    R3: 'R3 (Rebound)',
}};

function renderRegimeTradesChart() {{
    const regimes = ['R0', 'R1', 'R2', 'R3'];
    const traces = [];
    let hasData = false;
    STRATEGY_NAMES.forEach((name, i) => {{
        const data = STRATEGIES[name].per_regime_trades || {{}};
        const yVals = regimes.map(r => data[r] || 0);
        if (yVals.some(v => v > 0)) hasData = true;
        traces.push({{
            x: regimes.map(r => REGIME_LABELS[r]),
            y: yVals,
            name: name,
            type: 'bar',
            marker: {{ color: COLORS[i % COLORS.length] }},
        }});
    }});

    if (hasData) {{
        Plotly.newPlot('regime-trades-chart', traces, {{
            ...PLOTLY_LAYOUT_BASE,
            title: {{ text: 'Trade Count by Regime', font: {{ color: '#e6edf3' }} }},
            barmode: 'group',
            xaxis: {{ gridcolor: '#30363d' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Number of Trades' }},
        }});
    }} else {{
        document.getElementById('regime-trades-chart').innerHTML = '<div class="no-data">No per-regime trade data available</div>';
    }}
}}

function renderRegimeWRChart() {{
    const regimes = ['R0', 'R1', 'R2', 'R3'];
    const traces = [];
    let hasData = false;
    STRATEGY_NAMES.forEach((name, i) => {{
        const wrData = STRATEGIES[name].per_regime_wr || {{}};
        const trData = STRATEGIES[name].per_regime_trades || {{}};
        const yVals = regimes.map(r => (trData[r] || 0) > 0 ? (wrData[r] || 0) * 100 : 0);
        if (yVals.some(v => v > 0)) hasData = true;
        traces.push({{
            x: regimes.map(r => REGIME_LABELS[r]),
            y: yVals,
            name: name,
            type: 'bar',
            marker: {{ color: COLORS[i % COLORS.length] }},
        }});
    }});

    if (hasData) {{
        // Add 50% reference line
        traces.push({{
            x: Object.values(REGIME_LABELS),
            y: [50, 50, 50, 50],
            name: '50% WR',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#30363d', width: 1, dash: 'dash' }},
            showlegend: false,
        }});
        Plotly.newPlot('regime-wr-chart', traces, {{
            ...PLOTLY_LAYOUT_BASE,
            title: {{ text: 'Win Rate by Regime (%)', font: {{ color: '#e6edf3' }} }},
            barmode: 'group',
            xaxis: {{ gridcolor: '#30363d' }},
            yaxis: {{ gridcolor: '#30363d', title: 'Win Rate %', range: [0, 100] }},
        }});
    }} else {{
        document.getElementById('regime-wr-chart').innerHTML = '<div class="no-data">No per-regime win rate data available</div>';
    }}
}}

function renderRegimeAnalysisTable() {{
    const regimes = ['R0', 'R1', 'R2', 'R3'];
    const thead = document.getElementById('regime-analysis-head');
    let headHtml = '<tr><th>Strategy</th>';
    regimes.forEach(r => {{
        headHtml += '<th style="text-align:center" colspan="4">' + REGIME_LABELS[r] + '</th>';
    }});
    headHtml += '</tr><tr><th></th>';
    regimes.forEach(() => {{
        headHtml += '<th>Trades</th><th>WR</th><th>PF</th><th>Avg Hold</th>';
    }});
    headHtml += '</tr>';
    thead.innerHTML = headHtml;

    const tbody = document.getElementById('regime-analysis-body');
    let bodyHtml = '';
    STRATEGY_NAMES.forEach(name => {{
        const s = STRATEGIES[name];
        const isBaseline = name === 'buy_and_hold';
        let cells = '';
        regimes.forEach(r => {{
            const trades = (s.per_regime_trades || {{}})[r] || 0;
            const wr = (s.per_regime_wr || {{}})[r] || 0;
            const pf = (s.per_regime_pf || {{}})[r] || 0;
            const hold = (s.per_regime_avg_hold || {{}})[r] || 0;
            if (isBaseline || trades === 0) {{
                cells += '<td>-</td><td>-</td><td>-</td><td>-</td>';
            }} else {{
                const wrCls = wr >= 0.5 ? 'positive' : (wr < 0.3 ? 'negative' : '');
                const pfCls = pf >= 1.0 ? 'positive' : 'negative';
                cells += '<td>' + trades + '</td>';
                cells += '<td class="' + wrCls + '">' + (wr * 100).toFixed(0) + '%</td>';
                cells += '<td class="' + pfCls + '">' + pf.toFixed(2) + '</td>';
                cells += '<td>' + hold.toFixed(1) + 'd</td>';
            }}
        }});
        bodyHtml += '<tr><td class="strategy-name">' + name + '</td>' + cells + '</tr>';
    }});
    tbody.innerHTML = bodyHtml;
}}

function renderRegimeAlerts() {{
    const alerts = [];
    STRATEGY_NAMES.forEach(name => {{
        if (name === 'buy_and_hold') return;
        const s = STRATEGIES[name];
        const trades = s.per_regime_trades || {{}};
        const totalTrades = Object.values(trades).reduce((a, b) => a + b, 0);
        if (totalTrades === 0) return;

        // Concentration alert: >70% of trades in one regime
        Object.entries(trades).forEach(([r, n]) => {{
            const pct = n / totalTrades;
            if (pct > 0.7) {{
                alerts.push({{
                    level: 'warning',
                    msg: '<strong>' + name + '</strong>: ' + (pct * 100).toFixed(0) + '% of trades concentrated in ' + REGIME_LABELS[r] + '. Strategy may underperform in other market conditions.',
                }});
            }}
        }});

        // Loss concentration: check if >80% of total losses come from one regime
        const regimeReturns = s.per_regime_return || {{}};
        const negReturns = Object.entries(regimeReturns).filter(([, v]) => v < 0);
        if (negReturns.length > 0) {{
            const totalNeg = negReturns.reduce((a, [, v]) => a + Math.abs(v), 0);
            negReturns.forEach(([r, v]) => {{
                if (totalNeg > 0 && Math.abs(v) / totalNeg > 0.8) {{
                    alerts.push({{
                        level: 'danger',
                        msg: '<strong>' + name + '</strong>: ' + (Math.abs(v) / totalNeg * 100).toFixed(0) + '% of losses concentrated in ' + REGIME_LABELS[r] + '. Consider reducing exposure in this regime.',
                    }});
                }}
            }});
        }}

        // Poor performance alert: WR < 20% in any regime with >5 trades
        const wr = s.per_regime_wr || {{}};
        Object.entries(trades).forEach(([r, n]) => {{
            if (n > 5 && wr[r] !== undefined && wr[r] < 0.2) {{
                alerts.push({{
                    level: 'warning',
                    msg: '<strong>' + name + '</strong>: Win rate only ' + (wr[r] * 100).toFixed(0) + '% in ' + REGIME_LABELS[r] + ' (' + n + ' trades). Strategy may be ill-suited for this regime.',
                }});
            }}
        }});
    }});

    const container = document.getElementById('regime-alerts');
    if (alerts.length === 0) {{
        container.innerHTML = '';
        return;
    }}
    let html = '<h3 class="section-title" style="margin-bottom:8px">Concentration Alerts</h3>';
    alerts.forEach(a => {{
        const bgColor = a.level === 'danger' ? 'rgba(248,81,73,0.15)' : 'rgba(210,153,34,0.15)';
        const borderColor = a.level === 'danger' ? 'var(--negative)' : 'var(--warning)';
        html += '<div style="padding:10px 16px;margin-bottom:8px;border-left:3px solid ' + borderColor + ';background:' + bgColor + ';border-radius:4px;font-size:13px;">' + a.msg + '</div>';
    }});
    container.innerHTML = html;
}}

// Initialize all charts
renderEquityCurves();
renderDrawdowns();
buildStockRows();
initStockFilter();
renderStockTable();
renderSectorSharpe();
renderSectorReturn();
renderSectorTable();
renderRegimeSharpe();
renderRegimeReturn();
renderRegimeTradesChart();
renderRegimeWRChart();
renderRegimeAnalysisTable();
renderRegimeAlerts();
renderSymbolHeatmap();
renderRollingSharpe();
renderMonthlyReturns();
</script>
</body>
</html>"""
