"""
HTML templates for experiment tracking dashboard.

Renders a self-contained interactive HTML page showing Optuna optimization
results with per-strategy experiment cards, Sharpe trends, and trial details.
Uses the same dark theme as the APEX strategy comparison dashboard.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def render_experiment_html(data: Dict[str, Any]) -> str:
    """
    Render the full experiment tracking HTML page.

    Args:
        data: Experiment data with strategies, experiments, trials, etc.

    Returns:
        Complete HTML string.
    """
    experiments_json = json.dumps(data["experiments"], indent=2)
    strategy_names_json = json.dumps(data["strategy_names"])

    # Build strategy cards
    strategy_cards = ""
    for strategy_name, experiments in data["experiments"].items():
        if not experiments:
            continue

        latest = experiments[-1]
        best_score = latest["best_trial"]["score"]
        best_return = latest["best_trial"].get("user_attrs", {}).get("total_return", 0)
        best_mdd = latest["best_trial"].get("user_attrs", {}).get("max_drawdown", 0)
        n_versions = len(experiments)

        # Sharpe trend: scores across experiment versions
        scores = [e["best_trial"]["score"] for e in experiments]
        trend_direction = ""
        if len(scores) >= 2:
            delta = scores[-1] - scores[-2]
            if delta > 0.01:
                trend_direction = '<span style="color: var(--positive);">improving</span>'
            elif delta < -0.01:
                trend_direction = '<span style="color: var(--negative);">declining</span>'
            else:
                trend_direction = '<span style="color: var(--text-secondary);">stable</span>'

        # Version timeline dots
        version_dots = ""
        for i, exp in enumerate(experiments):
            is_latest = i == len(experiments) - 1
            dot_cls = "version-dot latest" if is_latest else "version-dot"
            version_dots += (
                f'<span class="{dot_cls}" '
                f'title="v{i + 1} ({exp["date"]}): score={exp["best_trial"]["score"]:.3f}">'
                f"v{i + 1}</span>"
            )

        score_cls = "positive" if best_score > 0 else "negative"
        ret_cls = "positive" if best_return > 0 else "negative"

        strategy_cards += f"""
        <div class="experiment-card" data-strategy="{strategy_name}">
            <div class="card-header">
                <h3>{strategy_name}</h3>
                <span class="version-count">{n_versions} run{"s" if n_versions != 1 else ""}</span>
            </div>
            <div class="version-timeline">{version_dots}</div>
            <div class="card-metrics">
                <div class="metric">
                    <span class="label">Best Score</span>
                    <span class="value {score_cls}">{best_score:.3f}</span>
                </div>
                <div class="metric">
                    <span class="label">Best Return</span>
                    <span class="value {ret_cls}">{best_return:+.1%}</span>
                </div>
                <div class="metric">
                    <span class="label">Max DD</span>
                    <span class="value negative">{best_mdd:.1%}</span>
                </div>
                <div class="metric">
                    <span class="label">Trials</span>
                    <span class="value">{latest["n_trials"]}</span>
                </div>
                <div class="metric">
                    <span class="label">Latest</span>
                    <span class="value">{latest["date"]}</span>
                </div>
                <div class="metric">
                    <span class="label">Trend</span>
                    <span class="value">{trend_direction if trend_direction else "N/A"}</span>
                </div>
            </div>
            <div class="sparkline-container">
                <div id="sparkline-{strategy_name}" class="sparkline"></div>
            </div>
            <div class="best-params-section">
                <h4>Best Parameters</h4>
                <div class="params-grid">
                    {"".join(
                        f'<div class="param-item"><span class="param-key">{k}</span>'
                        f'<span class="param-val">{_format_param(v)}</span></div>'
                        for k, v in latest["best_trial"]["params"].items()
                    )}
                </div>
            </div>
            <button class="toggle-trials" onclick="toggleTrials('{strategy_name}')">
                Show Trial Details
            </button>
            <div id="trials-{strategy_name}" class="trials-section" style="display:none;">
                <div id="trials-chart-{strategy_name}" class="trials-chart"></div>
                <table class="trials-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Score</th>
                            <th>Return</th>
                            <th>MaxDD</th>
                            <th>State</th>
                            <th>Params</th>
                        </tr>
                    </thead>
                    <tbody id="trials-body-{strategy_name}"></tbody>
                </table>
            </div>
        </div>"""

    comparison_link = ""
    if data.get("comparison_url"):
        comparison_link = (
            f'<a href="{data["comparison_url"]}" class="comparison-link">'
            f"View Strategy Comparison Dashboard</a>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APEX Experiment Tracker</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {{
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-card: #16213e;
    --border: #0f3460;
    --text-primary: #e5e5e5;
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
    flex-wrap: wrap;
    gap: 12px;
}}
.header h1 {{ font-size: 20px; font-weight: 600; }}
.header .meta {{ color: var(--text-secondary); font-size: 13px; }}

.comparison-link {{
    color: var(--accent);
    text-decoration: none;
    font-size: 13px;
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    transition: background 0.15s;
}}
.comparison-link:hover {{ background: var(--bg-card); }}

.filter-bar {{
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 12px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
}}
.filter-bar label {{ color: var(--text-secondary); font-size: 13px; font-weight: 600; }}
.filter-bar select, .filter-bar input {{
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
}}
.filter-bar .count {{ color: var(--text-secondary); font-size: 12px; margin-left: auto; }}

.content {{
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
}}

.cards-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 20px;
    margin-bottom: 24px;
}}

.experiment-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    transition: border-color 0.15s;
}}
.experiment-card:hover {{ border-color: var(--accent); }}

.card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}
.card-header h3 {{ font-size: 17px; font-weight: 600; color: var(--accent); }}
.version-count {{
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    padding: 2px 10px;
    border-radius: 10px;
}}

.version-timeline {{
    display: flex;
    gap: 6px;
    margin-bottom: 14px;
    flex-wrap: wrap;
}}
.version-dot {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--bg-primary);
    color: var(--text-secondary);
    cursor: default;
    border: 1px solid var(--border);
}}
.version-dot.latest {{
    background: var(--accent);
    color: var(--bg-primary);
    font-weight: 600;
    border-color: var(--accent);
}}

.card-metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}}
.metric {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 13px;
}}
.metric .label {{ color: var(--text-secondary); }}
.metric .value.positive {{ color: var(--positive); }}
.metric .value.negative {{ color: var(--negative); }}

.sparkline-container {{ margin-bottom: 12px; }}
.sparkline {{ height: 60px; }}

.best-params-section {{
    border-top: 1px solid var(--border);
    padding-top: 12px;
    margin-bottom: 12px;
}}
.best-params-section h4 {{
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 8px;
}}
.params-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 4px;
}}
.param-item {{
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 3px 8px;
    background: var(--bg-primary);
    border-radius: 4px;
}}
.param-key {{ color: var(--text-secondary); }}
.param-val {{ color: var(--text-primary); font-weight: 500; font-family: monospace; }}

.toggle-trials {{
    width: 100%;
    padding: 8px;
    background: var(--bg-primary);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
}}
.toggle-trials:hover {{ color: var(--text-primary); border-color: var(--accent); }}

.trials-section {{ margin-top: 12px; }}
.trials-chart {{ height: 200px; margin-bottom: 12px; }}

.trials-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 8px;
}}
.trials-table th, .trials-table td {{
    padding: 6px 10px;
    text-align: right;
    border-bottom: 1px solid var(--border);
}}
.trials-table th {{
    background: var(--bg-primary);
    color: var(--text-secondary);
    font-weight: 600;
    position: sticky;
    top: 0;
}}
.trials-table th:first-child, .trials-table td:first-child {{ text-align: center; width: 40px; }}
.trials-table td:last-child {{ text-align: left; font-family: monospace; font-size: 11px; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.trials-table tr:hover {{ background: rgba(88, 166, 255, 0.05); }}
.trials-table .best-row {{ background: rgba(63, 185, 80, 0.1); }}

td.positive {{ color: var(--positive); }}
td.negative {{ color: var(--negative); }}
td.state-complete {{ color: var(--positive); }}
td.state-pruned {{ color: var(--warning); }}
td.state-fail {{ color: var(--negative); }}

.overview-charts {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
}}
.chart-container {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}}

.no-data {{
    color: var(--text-secondary);
    text-align: center;
    padding: 40px;
    font-size: 14px;
}}

@media (max-width: 900px) {{
    .cards-grid {{ grid-template-columns: 1fr; }}
    .overview-charts {{ grid-template-columns: 1fr; }}
    .card-metrics {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>

<div class="header">
    <div>
        <h1>APEX Experiment Tracker</h1>
        <div class="meta">
            {data["strategy_count"]} strategies |
            {data["total_experiments"]} experiments |
            Generated: {data["generated_at"]}
        </div>
    </div>
    {comparison_link}
</div>

<div class="filter-bar">
    <label>Strategy:</label>
    <select id="strategy-filter" onchange="applyFilters()">
        <option value="all">All Strategies</option>
    </select>
    <label style="margin-left: 12px;">Date From:</label>
    <input type="date" id="date-from" onchange="applyFilters()">
    <label>To:</label>
    <input type="date" id="date-to" onchange="applyFilters()">
    <span class="count" id="visible-count"></span>
</div>

<div class="content">
    <!-- Overview charts -->
    <div class="overview-charts">
        <div class="chart-container">
            <div id="score-evolution-chart" style="height: 300px;"></div>
        </div>
        <div class="chart-container">
            <div id="best-scores-chart" style="height: 300px;"></div>
        </div>
    </div>

    <!-- Strategy cards -->
    <div class="cards-grid" id="cards-container">
        {strategy_cards}
    </div>
</div>

<script>
const EXPERIMENTS = {experiments_json};
const STRATEGY_NAMES = {strategy_names_json};

const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff', '#79c0ff', '#56d364', '#e3b341'];
const PLOTLY_LAYOUT = {{
    paper_bgcolor: '#16213e',
    plot_bgcolor: '#16213e',
    font: {{ color: '#8b949e' }},
    legend: {{ bgcolor: 'transparent' }},
    margin: {{ t: 40, r: 20, b: 40, l: 50 }},
}};

// --- Filters ---
function initFilters() {{
    const select = document.getElementById('strategy-filter');
    STRATEGY_NAMES.forEach(name => {{
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    }});

    // Set date range from data
    let minDate = null, maxDate = null;
    Object.values(EXPERIMENTS).forEach(exps => {{
        exps.forEach(e => {{
            if (!minDate || e.date < minDate) minDate = e.date;
            if (!maxDate || e.date > maxDate) maxDate = e.date;
        }});
    }});
    if (minDate) document.getElementById('date-from').value = minDate;
    if (maxDate) document.getElementById('date-to').value = maxDate;

    updateVisibleCount();
}}

function applyFilters() {{
    const strategy = document.getElementById('strategy-filter').value;
    const dateFrom = document.getElementById('date-from').value;
    const dateTo = document.getElementById('date-to').value;

    document.querySelectorAll('.experiment-card').forEach(card => {{
        const cardStrategy = card.getAttribute('data-strategy');
        let visible = true;

        if (strategy !== 'all' && cardStrategy !== strategy) {{
            visible = false;
        }}

        // Date filtering: hide card if ALL its experiments fall outside the range
        if (visible && (dateFrom || dateTo)) {{
            const exps = EXPERIMENTS[cardStrategy] || [];
            const anyInRange = exps.some(e => {{
                if (dateFrom && e.date < dateFrom) return false;
                if (dateTo && e.date > dateTo) return false;
                return true;
            }});
            if (!anyInRange) visible = false;
        }}

        card.style.display = visible ? '' : 'none';
    }});

    updateVisibleCount();
}}

function updateVisibleCount() {{
    const visible = document.querySelectorAll('.experiment-card:not([style*="display: none"])').length;
    const total = document.querySelectorAll('.experiment-card').length;
    document.getElementById('visible-count').textContent = visible + ' of ' + total + ' strategies';
}}

// --- Sparklines ---
function renderSparklines() {{
    STRATEGY_NAMES.forEach((name, i) => {{
        const exps = EXPERIMENTS[name];
        if (!exps || exps.length === 0) return;

        const scores = exps.map(e => e.best_trial.score);
        const dates = exps.map(e => e.date);
        const div = document.getElementById('sparkline-' + name);
        if (!div) return;

        Plotly.newPlot(div, [{{
            x: dates,
            y: scores,
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: COLORS[i % COLORS.length], width: 2 }},
            marker: {{ size: 5 }},
            hovertemplate: '%{{x}}<br>Score: %{{y:.3f}}<extra></extra>',
        }}], {{
            ...PLOTLY_LAYOUT,
            margin: {{ t: 5, r: 10, b: 20, l: 35 }},
            xaxis: {{ showgrid: false, showticklabels: scores.length > 1, tickfont: {{ size: 9 }} }},
            yaxis: {{ gridcolor: '#0f3460', tickfont: {{ size: 9 }}, title: '' }},
            showlegend: false,
        }}, {{ displayModeBar: false, responsive: true }});
    }});
}}

// --- Overview Charts ---
function renderScoreEvolution() {{
    const traces = [];
    STRATEGY_NAMES.forEach((name, i) => {{
        const exps = EXPERIMENTS[name];
        if (!exps || exps.length === 0) return;

        traces.push({{
            x: exps.map(e => e.date),
            y: exps.map(e => e.best_trial.score),
            name: name,
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: COLORS[i % COLORS.length], width: 2 }},
            marker: {{ size: 6 }},
        }});
    }});

    if (traces.length > 0) {{
        Plotly.newPlot('score-evolution-chart', traces, {{
            ...PLOTLY_LAYOUT,
            title: {{ text: 'Best Score Evolution Over Time', font: {{ color: '#e5e5e5' }} }},
            xaxis: {{ gridcolor: '#0f3460' }},
            yaxis: {{ gridcolor: '#0f3460', title: 'Score (Sharpe)' }},
        }});
    }} else {{
        document.getElementById('score-evolution-chart').innerHTML = '<div class="no-data">No experiment data</div>';
    }}
}}

function renderBestScores() {{
    const names = [];
    const scores = [];
    const colors = [];

    STRATEGY_NAMES.forEach((name, i) => {{
        const exps = EXPERIMENTS[name];
        if (!exps || exps.length === 0) return;
        const latest = exps[exps.length - 1];
        names.push(name);
        scores.push(latest.best_trial.score);
        colors.push(latest.best_trial.score > 0 ? '#3fb950' : '#f85149');
    }});

    if (names.length > 0) {{
        Plotly.newPlot('best-scores-chart', [{{
            x: names,
            y: scores,
            type: 'bar',
            marker: {{ color: colors }},
            hovertemplate: '%{{x}}<br>Score: %{{y:.3f}}<extra></extra>',
        }}], {{
            ...PLOTLY_LAYOUT,
            title: {{ text: 'Latest Best Score by Strategy', font: {{ color: '#e5e5e5' }} }},
            xaxis: {{ gridcolor: '#0f3460', tickangle: -30 }},
            yaxis: {{ gridcolor: '#0f3460', title: 'Score' }},
        }});
    }} else {{
        document.getElementById('best-scores-chart').innerHTML = '<div class="no-data">No experiment data</div>';
    }}
}}

// --- Trial Details ---
const expandedTrials = new Set();

function toggleTrials(strategy) {{
    const section = document.getElementById('trials-' + strategy);
    const btn = section.previousElementSibling;

    if (expandedTrials.has(strategy)) {{
        section.style.display = 'none';
        btn.textContent = 'Show Trial Details';
        expandedTrials.delete(strategy);
    }} else {{
        section.style.display = 'block';
        btn.textContent = 'Hide Trial Details';
        expandedTrials.add(strategy);
        renderTrialDetails(strategy);
    }}
}}

function renderTrialDetails(strategy) {{
    const exps = EXPERIMENTS[strategy];
    if (!exps || exps.length === 0) return;

    // Use latest experiment's trials
    const latest = exps[exps.length - 1];
    const trials = latest.all_trials || [];
    const bestNum = latest.best_trial.number;

    // Trial score scatter plot
    const completed = trials.filter(t => t.state === 'COMPLETE');
    const chartDiv = document.getElementById('trials-chart-' + strategy);
    if (completed.length > 0) {{
        const trialNums = completed.map(t => t.number);
        const trialScores = completed.map(t => t.score);
        const trialColors = completed.map(t => t.number === bestNum ? '#3fb950' : '#58a6ff');
        const trialSizes = completed.map(t => t.number === bestNum ? 10 : 5);

        Plotly.newPlot(chartDiv, [{{
            x: trialNums,
            y: trialScores,
            type: 'scatter',
            mode: 'markers',
            marker: {{ color: trialColors, size: trialSizes }},
            hovertemplate: 'Trial %{{x}}<br>Score: %{{y:.3f}}<extra></extra>',
        }}], {{
            ...PLOTLY_LAYOUT,
            margin: {{ t: 30, r: 10, b: 30, l: 45 }},
            title: {{ text: 'Trial Scores (latest run)', font: {{ color: '#e5e5e5', size: 13 }} }},
            xaxis: {{ gridcolor: '#0f3460', title: 'Trial #' }},
            yaxis: {{ gridcolor: '#0f3460', title: 'Score' }},
            showlegend: false,
        }}, {{ displayModeBar: false, responsive: true }});
    }}

    // Trial table (top 30 by score, descending)
    const sorted = completed.slice().sort((a, b) => b.score - a.score).slice(0, 30);
    const tbody = document.getElementById('trials-body-' + strategy);
    let html = '';
    sorted.forEach(t => {{
        const isBest = t.number === bestNum;
        const rowCls = isBest ? ' class="best-row"' : '';
        const ret = t.user_attrs?.total_return || 0;
        const mdd = t.user_attrs?.max_drawdown || 0;
        const state = t.state || 'UNKNOWN';
        const stateCls = state === 'COMPLETE' ? 'state-complete' : (state === 'PRUNED' ? 'state-pruned' : 'state-fail');
        const paramStr = Object.entries(t.params || {{}}).map(([k, v]) => {{
            if (typeof v === 'number') return k + '=' + (Number.isInteger(v) ? v : v.toFixed(3));
            return k + '=' + v;
        }}).join(', ');

        html += '<tr' + rowCls + '>';
        html += '<td>' + t.number + '</td>';
        html += '<td class="' + (t.score > 0 ? 'positive' : 'negative') + '">' + t.score.toFixed(3) + '</td>';
        html += '<td class="' + (ret > 0 ? 'positive' : 'negative') + '">' + (ret * 100).toFixed(1) + '%</td>';
        html += '<td class="negative">' + (mdd * 100).toFixed(1) + '%</td>';
        html += '<td class="' + stateCls + '">' + state + '</td>';
        html += '<td title="' + paramStr + '">' + paramStr + '</td>';
        html += '</tr>';
    }});
    tbody.innerHTML = html;
}}

// --- Init ---
initFilters();
renderScoreEvolution();
renderBestScores();
renderSparklines();
</script>
</body>
</html>"""


def _format_param(value: object) -> str:
    """Format a parameter value for display."""
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.4g}"
    return str(value)
