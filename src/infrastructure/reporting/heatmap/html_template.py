"""
HTML Template - Generates the complete heatmap HTML page.

Renders the HTML template with embedded Plotly data and ETF dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .model import ColorMetric, HeatmapModel, SizeMetric


def render_heatmap_template(
    model: HeatmapModel,
    plotly_data: Dict[str, Any],
    output_dir: Path,
    css_path: str,
    dashboard_html: str,
) -> str:
    """
    Render the HTML template with external CSS and ETF dashboard.

    Args:
        model: HeatmapModel with etf_dashboard populated
        plotly_data: Plotly treemap data structure
        output_dir: Output directory (for potential asset paths)
        css_path: Relative path to external CSS file
        dashboard_html: Pre-rendered ETF dashboard HTML

    Returns:
        Complete HTML page content
    """
    # Embedded model data for frontend
    model_json = json.dumps(model.to_dict(), indent=2)
    plotly_json = json.dumps(plotly_data)

    # Build regime distribution stats
    regime_stats = "".join(
        f'<div class="hm-stat"><span>{r}:</span><span class="hm-stat-value">{c}</span></div>'
        for r, c in sorted(model.regime_distribution.items())
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Heatmap - {model.generated_at.strftime('%Y-%m-%d') if model.generated_at else 'N/A'}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="stylesheet" href="{css_path}">
</head>
<body class="apex-hm">
    <div class="hm-header">
        <h1>Signal Heatmap</h1>
        <div class="meta">
            Generated: {model.generated_at.strftime('%Y-%m-%d %H:%M') if model.generated_at else 'N/A'}
        </div>
    </div>

    <!-- ETF Dashboard -->
    {dashboard_html}

    <!-- Controls -->
    <div class="hm-controls">
        <div class="hm-control-group">
            <label>Color by:</label>
            <select id="colorMetric">
                <option value="rule_frequency" selected>🔥 Trending (Activity)</option>
                <option value="rule_frequency_direction">📈 Trending (Direction)</option>
                <option value="regime">Regime (R0-R3)</option>
                <option value="daily_change">Daily Change</option>
                <option value="alignment">Alignment Score</option>
            </select>
        </div>
        <div class="hm-control-group">
            <label>Size by:</label>
            <select id="sizeMetric">
                <option value="market_cap" {'selected' if model.size_metric == SizeMetric.MARKET_CAP else ''}>Market Cap</option>
                <option value="volume" {'selected' if model.size_metric == SizeMetric.VOLUME else ''}>Volume</option>
                <option value="equal" {'selected' if model.size_metric == SizeMetric.EQUAL else ''}>Equal Weight</option>
            </select>
        </div>
        <!-- Regime Legend (shown when regime mode selected) -->
        <div class="hm-legend" id="legend-regime" style="display: none;">
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #22c55e;"></div>
                <span>R0 Healthy</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #f59e0b;"></div>
                <span>R1 Choppy</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #ef4444;"></div>
                <span>R2 Risk-Off</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #3b82f6;"></div>
                <span>R3 Rebound</span>
            </div>
        </div>
        <!-- Trending Legend (shown when rule_frequency mode selected) -->
        <div class="hm-legend" id="legend-trending">
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #ff4444;"></div>
                <span>Hot (8+)</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #ff8844;"></div>
                <span>Warm (5-7)</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #ffcc44;"></div>
                <span>Active (3-4)</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #88cc88;"></div>
                <span>Cool (1-2)</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #444444;"></div>
                <span>None (0)</span>
            </div>
        </div>
        <!-- Direction Legend (shown when rule_frequency_direction mode selected) -->
        <div class="hm-legend" id="legend-direction" style="display: none;">
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #22c55e;"></div>
                <span>Strong Bull</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #4ade80;"></div>
                <span>Bullish</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #6b7280;"></div>
                <span>Neutral</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #f87171;"></div>
                <span>Bearish</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #ef4444;"></div>
                <span>Strong Bear</span>
            </div>
        </div>
    </div>

    <!-- Stats -->
    <div class="hm-stats">
        <div class="hm-stat">
            <span>Symbols:</span>
            <span class="hm-stat-value">{model.symbol_count}</span>
        </div>
        <div class="hm-stat">
            <span>🔥 Signals (24h):</span>
            <span class="hm-stat-value">{model.total_signals}</span>
        </div>
        <div class="hm-stat">
            <span>Missing Caps:</span>
            <span class="hm-stat-value">{model.cap_missing_count}</span>
        </div>
        {regime_stats}
    </div>

    <!-- Treemap (Stocks Only) -->
    <div class="hm-treemap-container">
        <div id="heatmap"></div>
    </div>

    <script>
{get_heatmap_javascript(model_json, plotly_json, model.size_metric, model.color_metric)}
    </script>
</body>
</html>"""

    return html


def get_heatmap_javascript(
    model_json: str,
    plotly_json: str,
    size_metric: SizeMetric,
    color_metric: ColorMetric,
) -> str:
    """
    Generate the JavaScript code for heatmap interactivity.

    Args:
        model_json: JSON string of model data
        plotly_json: JSON string of Plotly data
        size_metric: Current size metric
        color_metric: Current color metric

    Returns:
        JavaScript code string
    """
    return f"""
        // Embedded model data
        const modelData = {model_json};
        const plotlyData = {plotly_json};

        // Report URL mapping for click navigation (from etf_dashboard and sectors)
        const reportUrls = {{}};

        // Build URLs from ETF dashboard (new structure)
        if (modelData.etf_dashboard) {{
            Object.values(modelData.etf_dashboard).forEach(cards => {{
                cards.forEach(card => {{
                    if (card.report_url) reportUrls[card.symbol] = card.report_url;
                }});
            }});
        }}

        // Build URLs from sectors (stocks in treemap)
        modelData.sectors.forEach(s => {{
            s.stocks.forEach(stock => {{
                if (stock.report_url) reportUrls[stock.symbol] = stock.report_url;
            }});
        }});

        // Debug: Check if Plotly loaded
        console.log('Plotly loaded:', typeof Plotly !== 'undefined');

        // Determine scale based on size metric
        const sizeMetric = modelData.size_metric;
        const getScale = (metric) => metric === 'volume' ? 1e6 : metric === 'equal' ? 1 : 1e9;
        let currentScale = getScale(sizeMetric);
        const normalizedValues = plotlyData.values.map(v => v / currentScale);
        console.log('Data loaded - ids:', plotlyData.ids.length, 'values:', normalizedValues, 'scale:', currentScale);

        // Build parent-children map once for parent recalculation
        function buildChildrenMap(ids, parents) {{
            const children = {{}};
            parents.forEach((parent, i) => {{
                if (parent) {{
                    if (!children[parent]) children[parent] = [];
                    children[parent].push(i);
                }}
            }});
            return children;
        }}

        const childrenMap = buildChildrenMap(plotlyData.ids, plotlyData.parents);

        // Recalculate parent values bottom-up
        function recalculateParents(values, children, ids) {{
            function sumChildren(nodeId) {{
                const idx = ids.indexOf(nodeId);
                if (idx === -1) return 0;
                if (!children[nodeId]) return values[idx];

                let sum = 0;
                for (const childIdx of children[nodeId]) {{
                    const childId = ids[childIdx];
                    sum += sumChildren(childId);
                }}
                values[idx] = sum;
                return sum;
            }}
            sumChildren('root');
        }}

        // Debug: Check container dimensions
        const container = document.getElementById('heatmap');
        console.log('Container dimensions:', container.offsetWidth, 'x', container.offsetHeight);

        // Color schemes
        function getRegimeColor(regime) {{
            const colors = {{'R0': '#22c55e', 'R1': '#eab308', 'R2': '#ef4444', 'R3': '#3b82f6'}};
            return colors[regime] || '#9ca3af';
        }}

        function getDailyChangeColor(pct) {{
            if (pct === null || pct === undefined) return '#9ca3af';
            const clamped = Math.max(-5, Math.min(5, pct));
            if (clamped >= 0) {{
                const intensity = Math.floor(200 - (clamped / 5) * 100);
                return `rgb(${{intensity}}, 197, 94)`;
            }} else {{
                const intensity = Math.floor(200 + (clamped / 5) * 100);
                return `rgb(244, ${{intensity}}, ${{intensity}})`;
            }}
        }}

        function getAlignmentColor(score) {{
            if (score === null || score === undefined) return '#9ca3af';
            const clamped = Math.max(-100, Math.min(100, score));
            if (clamped >= 0) {{
                const g = Math.floor(180 + (clamped / 100) * 40);
                return `rgb(34, ${{g}}, 94)`;
            }} else {{
                const r = Math.floor(200 + (clamped / 100) * 44);
                return `rgb(${{r}}, 68, 68)`;
            }}
        }}

        // Phase 3: Rule frequency color for trending mode (activity)
        function getRuleFrequencyColor(signalCount) {{
            if (signalCount === null || signalCount === undefined || signalCount === 0) return '#444444';
            if (signalCount >= 8) return '#ff4444';  // Hot red
            if (signalCount >= 5) return '#ff8844';  // Warm orange
            if (signalCount >= 3) return '#ffcc44';  // Yellow
            return '#88cc88';  // Cool green
        }}

        // Phase 3: Rule frequency direction color for trending mode (direction)
        function getRuleFrequencyDirectionColor(buyCount, sellCount) {{
            const total = (buyCount || 0) + (sellCount || 0);
            if (total === 0) return '#444444';  // Dark gray - no signals

            const net = (buyCount || 0) - (sellCount || 0);
            if (net === 0) return '#6b7280';  // Neutral gray - balanced

            // Calculate intensity based on net magnitude (capped at 8)
            const magnitude = Math.min(Math.abs(net), 8);
            const intensity = magnitude / 8.0;

            if (net > 0) {{
                // Bullish: interpolate from muted green to bright green
                const r = Math.floor(74 - intensity * 52);
                const g = Math.floor(222 - intensity * 25);
                const b = Math.floor(128 - intensity * 34);
                return `rgb(${{r}}, ${{g}}, ${{b}})`;
            }} else {{
                // Bearish: interpolate from muted red to bright red
                const r = Math.floor(248 - intensity * 9);
                const g = Math.floor(113 - intensity * 45);
                const b = Math.floor(113 - intensity * 45);
                return `rgb(${{r}}, ${{g}}, ${{b}})`;
            }}
        }}

        // Phase 3: Update legend visibility based on color mode
        function updateLegend(metric) {{
            const regimeLegend = document.getElementById('legend-regime');
            const trendingLegend = document.getElementById('legend-trending');
            const directionLegend = document.getElementById('legend-direction');
            if (metric === 'rule_frequency') {{
                regimeLegend.style.display = 'none';
                trendingLegend.style.display = 'flex';
                directionLegend.style.display = 'none';
            }} else if (metric === 'rule_frequency_direction') {{
                regimeLegend.style.display = 'none';
                trendingLegend.style.display = 'none';
                directionLegend.style.display = 'flex';
            }} else if (metric === 'regime') {{
                regimeLegend.style.display = 'flex';
                trendingLegend.style.display = 'none';
                directionLegend.style.display = 'none';
            }} else {{
                // Hide all for daily_change and alignment (they use gradients)
                regimeLegend.style.display = 'none';
                trendingLegend.style.display = 'none';
                directionLegend.style.display = 'none';
            }}
        }}

        function updateColors(metric) {{
            const newColors = plotlyData.customdata.map((d, i) => {{
                if (!d || !d.symbol) return plotlyData.colors[i];
                if (metric === 'regime') return getRegimeColor(d.regime);
                if (metric === 'daily_change') return getDailyChangeColor(d.daily_change_pct);
                if (metric === 'alignment') return getAlignmentColor(d.alignment_score);
                if (metric === 'rule_frequency') return getRuleFrequencyColor(d.signal_count);
                if (metric === 'rule_frequency_direction') return getRuleFrequencyDirectionColor(d.buy_signal_count, d.sell_signal_count);
                return plotlyData.colors[i];
            }});
            Plotly.restyle('heatmap', {{'marker.colors': [newColors]}});
            updateLegend(metric);
        }}

        function updateSizes(metric) {{
            const scale = getScale(metric);
            currentScale = scale;

            // Copy original values to avoid mutating
            const newValues = [...plotlyData.values];

            // Update leaf node values based on metric
            plotlyData.customdata.forEach((d, i) => {{
                if (d && d.symbol) {{
                    // Only update leaf nodes (nodes with a symbol)
                    if (metric === 'market_cap') {{
                        newValues[i] = (d.market_cap || 1) / scale;
                    }} else if (metric === 'volume') {{
                        newValues[i] = (d.volume || 1) / scale;
                    }} else {{
                        newValues[i] = 1;  // equal weight
                    }}
                }}
            }});

            // Recalculate parent values bottom-up
            recalculateParents(newValues, childrenMap, plotlyData.ids);

            // Update hovertemplate unit suffix
            const unitSuffix = metric === 'volume' ? 'M' : metric === 'equal' ? '' : 'B';
            const newTemplate = '<b>%{{label}}</b><br>' +
                'Signals: %{{customdata.signal_count}}<br>' +
                'Regime: %{{customdata.regime}}<br>' +
                'Close: $%{{customdata.close_price}}<br>' +
                'Daily: %{{customdata.daily_change_pct}}%<br>' +
                'Value: %{{value:,.2f}}' + unitSuffix + '<extra></extra>';

            Plotly.restyle('heatmap', {{'values': [newValues], 'hovertemplate': newTemplate}});
        }}

        // Phase 3: Compute initial colors for trending mode (default)
        const initialColors = plotlyData.customdata.map((d, i) => {{
            if (!d || !d.symbol) return plotlyData.colors[i];
            return getRuleFrequencyColor(d.signal_count);
        }});

        // Initial render - stocks-only treemap (Trending mode by default)
        const trace = {{
            type: 'treemap',
            ids: plotlyData.ids,
            labels: plotlyData.labels,
            parents: plotlyData.parents,
            values: normalizedValues,
            customdata: plotlyData.customdata,
            marker: {{
                colors: initialColors,
                line: {{ width: 1, color: '#0c0f14' }}
            }},
            textinfo: 'label',
            textfont: {{ size: 14, color: '#f0f4f8' }},
            hovertemplate: '<b>%{{label}}</b><br>' +
                'Signals: %{{customdata.signal_count}}<br>' +
                'Regime: %{{customdata.regime}}<br>' +
                'Close: $%{{customdata.close_price}}<br>' +
                'Daily: %{{customdata.daily_change_pct}}%<br>' +
                'Value: %{{value:,.2f}}' + (sizeMetric === 'volume' ? 'M' : sizeMetric === 'equal' ? '' : 'B') + '<extra></extra>',
            pathbar: {{ visible: true, textfont: {{ size: 12 }} }},
            branchvalues: 'total',
            maxdepth: 3
        }};

        // Calculate treemap height - account for ETF dashboard
        function calculateTreemapHeight() {{
            const dashboard = document.querySelector('.hm-dashboard');
            const dashboardHeight = dashboard ? dashboard.offsetHeight : 0;
            const reservedHeight = 60 + dashboardHeight + 50 + 40 + 20;
            return Math.max(500, window.innerHeight - reservedHeight);
        }}

        const layout = {{
            margin: {{ t: 30, l: 10, r: 10, b: 10 }},
            paper_bgcolor: '#0c0f14',
            font: {{ color: '#f0f4f8' }},
            autosize: true,
            height: calculateTreemapHeight()
        }};

        const config = {{
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
            responsive: true
        }};

        console.log('Calling Plotly.newPlot with', plotlyData.ids.length, 'nodes, root_value:', normalizedValues[0]);
        Plotly.newPlot('heatmap', [trace], layout, config).then(function(gd) {{
            console.log('Plotly render complete!');

            // Click handler for navigation
            gd.on('plotly_click', function(data) {{
                if (data && data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    const symbol = point.label;
                    const url = reportUrls[symbol];
                    console.log('Clicked:', symbol, 'URL:', url);
                    if (url) {{
                        window.location.href = url;
                    }}
                }}
            }});

        }}).catch(function(err) {{
            console.error('Plotly render error:', err);
        }});

        // Control handlers
        document.getElementById('colorMetric').addEventListener('change', function(e) {{
            updateColors(e.target.value);
        }});

        document.getElementById('sizeMetric').addEventListener('change', function(e) {{
            updateSizes(e.target.value);
        }});

        // Handle window resize
        window.addEventListener('resize', function() {{
            Plotly.relayout('heatmap', {{
                height: calculateTreemapHeight()
            }});
        }});
    """
