"""
Plotly Scripts - JavaScript code generation for chart rendering.

Generates the JavaScript code for Plotly charts with multi-subplot layout.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def get_scripts(
    chart_data: Dict[str, Any],
    symbols: List[str],
    timeframes: List[str],
    signal_history: Dict[str, List[Dict[str, Any]]],
    confluence_data: Dict[str, Dict[str, Any]],
    colors: Dict[str, str],
) -> str:
    """
    Generate JavaScript code for signal report interactivity.

    Args:
        chart_data: Dict mapping key to chart data
        symbols: List of symbol names
        timeframes: List of timeframe strings
        signal_history: Dict mapping key to list of signal dictionaries
        confluence_data: Dict mapping key to confluence score data
        colors: Theme color dictionary

    Returns:
        JavaScript code string
    """
    data_json = json.dumps(chart_data, default=str)
    symbols_json = json.dumps(symbols)
    timeframes_json = json.dumps(timeframes)
    colors_json = json.dumps(colors)
    signals_json = json.dumps(signal_history, default=str)
    confluence_json = json.dumps(confluence_data, default=str)

    return f"""
const chartData = {data_json};
const symbols = {symbols_json};
const timeframes = {timeframes_json};
const colors = {colors_json};
const signalHistory = {signals_json};
const confluenceData = {confluence_json};

let currentSymbol = symbols[0] || '';
let currentTimeframe = timeframes[0] || '1d';

function getDataKey() {{
    return `${{currentSymbol}}_${{currentTimeframe}}`;
}}

function selectTimeframe(tf, btn) {{
    currentTimeframe = tf;
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    updateChart();
}}

function updateChart() {{
    currentSymbol = document.getElementById('symbol-select').value;
    const key = getDataKey();
    const data = chartData[key];

    if (!data) {{
        console.warn('No data for', key);
        return;
    }}

    renderMainChart(data);
    updateSignalHistoryTable();
    updateConfluencePanel();
    updateRegimeSection();
}}

function updateRegimeSection() {{
    // Hide all regime symbol sections
    const sections = document.querySelectorAll('.regime-symbol-section');
    sections.forEach(section => {{
        section.style.display = 'none';
    }});

    // Show the selected symbol's section
    const selectedSection = document.getElementById('regime-' + currentSymbol);
    if (selectedSection) {{
        selectedSection.style.display = 'block';
    }}
}}

function renderMainChart(data) {{
    // Fixed 4-row layout: Price (55%), RSI (15%), MACD (14%), Volume (10%)
    const traces = [];
    const hasData = (values) => values && !values.every(v => v === null);

    // For intraday charts, use index-based x-axis to avoid gaps
    const isIntraday = ['1m', '5m', '15m', '30m', '1h', '4h'].includes(data.timeframe);
    const xValues = isIntraday
        ? data.timestamps.map((_, i) => i)
        : data.timestamps;

    // Row 1: Price candlesticks
    traces.push({{
        type: 'candlestick',
        x: xValues,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        name: 'Price',
        increasing: {{ line: {{ color: colors.candle_up }}, fillcolor: colors.candle_up }},
        decreasing: {{ line: {{ color: colors.candle_down }}, fillcolor: colors.candle_down }},
        xaxis: 'x',
        yaxis: 'y',
    }});

    // Overlay indicators: Bollinger Bands, SuperTrend
    const overlayConfig = {{
        'bollinger_bb_upper': {{ color: '#3b82f6', dash: 'dot' }},
        'bollinger_bb_middle': {{ color: '#6366f1', dash: 'solid' }},
        'bollinger_bb_lower': {{ color: '#3b82f6', dash: 'dot' }},
        'supertrend_supertrend': {{ color: '#f59e0b', dash: 'solid' }},
    }};
    for (const [name, config] of Object.entries(overlayConfig)) {{
        const values = data.overlays[name];
        if (!hasData(values)) continue;
        traces.push({{
            type: 'scatter',
            mode: 'lines',
            x: xValues,
            y: values,
            name: name.replace('bollinger_bb_', 'BB ').replace('supertrend_', 'ST '),
            line: {{ color: config.color, width: 1, dash: config.dash }},
            xaxis: 'x',
            yaxis: 'y',
        }});
    }}

    // Row 2: RSI with threshold lines
    const rsiValues = data.rsi['rsi_rsi'];
    if (hasData(rsiValues)) {{
        traces.push({{
            type: 'scatter',
            mode: 'lines',
            x: xValues,
            y: rsiValues,
            name: 'RSI',
            line: {{ color: '#8b5cf6', width: 1.5 }},
            xaxis: 'x',
            yaxis: 'y2',
        }});
        const boundsX = [xValues[0], xValues[xValues.length - 1]];
        const rsiLevels = [
            {{ value: 70, name: 'Overbought', color: colors.candle_down }},
            {{ value: 30, name: 'Oversold', color: colors.candle_up }},
        ];
        for (const level of rsiLevels) {{
            traces.push({{
                type: 'scatter',
                mode: 'lines',
                x: boundsX,
                y: [level.value, level.value],
                name: level.name,
                line: {{ color: level.color, width: 1, dash: 'dash' }},
                xaxis: 'x',
                yaxis: 'y2',
                showlegend: false,
            }});
        }}
    }}

    // Row 3: MACD subplot
    const macdHist = data.macd['macd_histogram'];
    if (hasData(macdHist)) {{
        const barColors = macdHist.map(v => v >= 0 ? colors.candle_up : colors.candle_down);
        traces.push({{
            type: 'bar',
            x: xValues,
            y: macdHist,
            name: 'MACD Hist',
            marker: {{ color: barColors }},
            xaxis: 'x',
            yaxis: 'y3',
        }});
    }}
    const macdLines = [
        {{ key: 'macd_macd', name: 'MACD', color: '#3b82f6' }},
        {{ key: 'macd_signal', name: 'Signal', color: '#f59e0b' }},
    ];
    for (const {{ key, name, color }} of macdLines) {{
        const values = data.macd[key];
        if (!hasData(values)) continue;
        traces.push({{
            type: 'scatter',
            mode: 'lines',
            x: xValues,
            y: values,
            name,
            line: {{ color, width: 1.5 }},
            xaxis: 'x',
            yaxis: 'y3',
        }});
    }}

    // Row 4: Volume bars
    if (data.volume && data.volume.length > 0) {{
        const volColors = data.close.map((c, i) => {{
            if (i === 0) return colors.text_muted;
            return c >= data.close[i-1] ? colors.candle_up : colors.candle_down;
        }});
        traces.push({{
            type: 'bar',
            x: xValues,
            y: data.volume,
            name: 'Volume',
            marker: {{ color: volColors, opacity: 0.5 }},
            xaxis: 'x',
            yaxis: 'y4',
        }});
    }}

    // Add signal markers on price chart
    const key = getDataKey();
    const signals = signalHistory[key] || [];
    const buySignals = signals.filter(s => s.direction === 'buy');
    const sellSignals = signals.filter(s => s.direction === 'sell');

    if (buySignals.length > 0) {{
        const buyData = buySignals.map(s => {{
            const idx = data.timestamps.findIndex(t => t === s.timestamp);
            if (idx < 0) return null;
            return {{
                x: isIntraday ? idx : s.timestamp,
                y: data.low[idx] * 0.995,
                rule: s.rule
            }};
        }}).filter(d => d !== null);

        if (buyData.length > 0) {{
            traces.push({{
                type: 'scatter',
                mode: 'markers',
                x: buyData.map(d => d.x),
                y: buyData.map(d => d.y),
                name: 'Buy Signal',
                marker: {{
                    symbol: 'triangle-up',
                    size: 12,
                    color: colors.candle_up,
                    line: {{ color: 'white', width: 1 }}
                }},
                hovertemplate: '%{{text}}<extra></extra>',
                text: buyData.map(d => d.rule),
                xaxis: 'x',
                yaxis: 'y',
            }});
        }}
    }}

    if (sellSignals.length > 0) {{
        const sellData = sellSignals.map(s => {{
            const idx = data.timestamps.findIndex(t => t === s.timestamp);
            if (idx < 0) return null;
            return {{
                x: isIntraday ? idx : s.timestamp,
                y: data.high[idx] * 1.005,
                rule: s.rule
            }};
        }}).filter(d => d !== null);

        if (sellData.length > 0) {{
            traces.push({{
                type: 'scatter',
                mode: 'markers',
                x: sellData.map(d => d.x),
                y: sellData.map(d => d.y),
                name: 'Sell Signal',
                marker: {{
                    symbol: 'triangle-down',
                    size: 12,
                    color: colors.candle_down,
                    line: {{ color: 'white', width: 1 }}
                }},
                hovertemplate: '%{{text}}<extra></extra>',
                text: sellData.map(d => d.rule),
                xaxis: 'x',
                yaxis: 'y',
            }});
        }}
    }}

    // Layout with 4 subplots
    const isDaily = ['1d', '1w', '1D', '1W'].includes(data.timeframe);
    let rangebreaks = [];
    if (isDaily) {{
        rangebreaks = [{{ bounds: ['sat', 'mon'] }}];
    }}

    // Custom tick labels for intraday
    let tickvals = null;
    let ticktext = null;
    if (isIntraday && data.timestamps.length > 0) {{
        const n = data.timestamps.length;
        const step = Math.max(1, Math.floor(n / 15));
        tickvals = [];
        ticktext = [];
        for (let i = 0; i < n; i += step) {{
            tickvals.push(i);
            const ts = new Date(data.timestamps[i]);
            const month = String(ts.getUTCMonth() + 1).padStart(2, '0');
            const day = String(ts.getUTCDate()).padStart(2, '0');
            const hour = String(ts.getUTCHours()).padStart(2, '0');
            const min = String(ts.getUTCMinutes()).padStart(2, '0');
            ticktext.push(`${{month}}/${{day}} ${{hour}}:${{min}}`);
        }}
        if (tickvals[tickvals.length - 1] !== n - 1) {{
            tickvals.push(n - 1);
            const ts = new Date(data.timestamps[n - 1]);
            const month = String(ts.getUTCMonth() + 1).padStart(2, '0');
            const day = String(ts.getUTCDate()).padStart(2, '0');
            const hour = String(ts.getUTCHours()).padStart(2, '0');
            const min = String(ts.getUTCMinutes()).padStart(2, '0');
            ticktext.push(`${{month}}/${{day}} ${{hour}}:${{min}}`);
        }}
    }}

    const layout = {{
        title: {{
            text: `${{data.symbol}} - ${{data.timeframe}} (${{data.bar_count}} bars)`,
            font: {{ color: colors.text, size: 18 }},
        }},
        showlegend: true,
        legend: {{
            orientation: 'h',
            y: -0.08,
            font: {{ color: colors.text, size: 10 }},
        }},
        paper_bgcolor: colors.card_bg,
        plot_bgcolor: colors.card_bg,
        font: {{ color: colors.text }},
        margin: {{ t: 50, r: 50, b: 80, l: 50 }},
        hovermode: 'x unified',
        bargap: 0.1,

        xaxis: {{
            title: {{ text: 'Time (UTC)', standoff: 10, font: {{ size: 11, color: colors.text_muted }} }},
            gridcolor: colors.border,
            showgrid: true,
            rangeslider: {{ visible: false }},
            tickangle: -45,
            domain: [0, 1],
            rangebreaks: rangebreaks,
            ...(isIntraday && tickvals ? {{ tickvals: tickvals, ticktext: ticktext }} : {{ nticks: 15 }}),
        }},

        annotations: [{{
            text: 'All times displayed in UTC',
            xref: 'paper',
            yref: 'paper',
            x: 1,
            y: 1.02,
            xanchor: 'right',
            yanchor: 'bottom',
            showarrow: false,
            font: {{ size: 10, color: colors.text_muted }},
        }}],

        yaxis: {{
            title: 'Price',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.52, 1.00],
            autorange: true,
        }},

        yaxis2: {{
            title: 'RSI',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.36, 0.48],
            range: [0, 100],
            dtick: 25,
        }},

        yaxis3: {{
            title: 'MACD',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.20, 0.32],
        }},

        yaxis4: {{
            title: 'Vol',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.00, 0.16],
        }},
    }};

    const config = {{
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    }};

    Plotly.newPlot('main-chart', traces, layout, config);
}}

function updateSignalHistoryTable() {{
    const key = getDataKey();
    const allSignals = signalHistory[key] || [];
    const container = document.getElementById('signal-history-table');
    const data = chartData[key] || {{}};
    const priceLevels = data.price_levels || {{}};
    const lastClose = data.close && data.close.length > 0 ? data.close[data.close.length - 1] : null;

    let html = '';

    // ========== SECTION 1: Active Signals ==========
    if (allSignals.length > 0) {{
        // Get currently active signals: most recent signal per indicator
        const activeByIndicator = {{}};
        const sortedByTime = [...allSignals].sort((a, b) =>
            new Date(a.timestamp) - new Date(b.timestamp)
        );

        for (const sig of sortedByTime) {{
            const indicator = sig.indicator || 'unknown';
            activeByIndicator[indicator] = sig;
        }}

        const signals = Object.values(activeByIndicator);
        const buySignals = signals.filter(s => s.direction === 'buy');
        const sellSignals = signals.filter(s => s.direction === 'sell');
        const alertSignals = signals.filter(s => s.direction !== 'buy' && s.direction !== 'sell');

        const sortedActiveSignals = [...signals].sort((a, b) =>
            (a.indicator || '').localeCompare(b.indicator || '')
        );

        html += `
            <div class="rule-frequency-summary">
                <h4 style="margin-bottom: 12px; color: ${{colors.text_muted}}; font-size: 12px; text-transform: uppercase;">
                    📊 Active Signals - ${{currentTimeframe.toUpperCase()}} (${{signals.length}} indicators)
                </h4>
                <div style="display: flex; gap: 16px; margin-bottom: 12px;">
                    <span style="color: ${{colors.candle_up}};">▲ ${{buySignals.length}} Bullish</span>
                    <span style="color: ${{colors.candle_down}};">▼ ${{sellSignals.length}} Bearish</span>
                    <span style="color: ${{colors.text_muted}};">● ${{alertSignals.length}} Neutral</span>
                </div>
                <div class="rule-freq-bars">
        `;

        for (const sig of sortedActiveSignals.slice(0, 15)) {{
            const direction = sig.direction || 'alert';
            const barColor = direction === 'buy' ? colors.candle_up : direction === 'sell' ? colors.candle_down : colors.text_muted;
            html += `
                <div class="rule-freq-item">
                    <div class="rule-freq-label">
                        <span class="rule-name">${{sig.rule}}</span>
                        <span class="rule-count ${{direction}}">${{sig.indicator}}</span>
                    </div>
                    <div class="rule-freq-bar-bg">
                        <div class="rule-freq-bar ${{direction}}" style="width: 100%; background: ${{barColor}}40;"></div>
                    </div>
                </div>
            `;
        }}
        if (signals.length > 15) {{
            html += `<div style="color: ${{colors.text_muted}}; font-size: 11px; margin-top: 8px;">... and ${{signals.length - 15}} more</div>`;
        }}
        html += '</div></div>';
    }} else {{
        html += '<div class="no-signals">No active signals for this symbol/timeframe</div>';
    }}

    // ========== SECTION 2: Indicator Values (Price Levels) ==========
    const hasLevels = Object.keys(priceLevels).length > 0;
    if (hasLevels && lastClose) {{
        html += `
            <div class="price-levels-section" style="margin-top: 24px;">
                <h4 style="margin-bottom: 12px; color: ${{colors.text_muted}}; font-size: 12px; text-transform: uppercase;">
                    📏 Key Price Levels (Current: $${{lastClose.toFixed(2)}})
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
        `;

        // Group levels by indicator
        const levelGroups = {{}};
        for (const [col, values] of Object.entries(priceLevels)) {{
            if (!values || values.length === 0) continue;
            const lastVal = values[values.length - 1];
            if (lastVal === null || lastVal === undefined) continue;

            const parts = col.split('_');
            const ind = parts[0].toUpperCase();
            const level = parts.slice(1).join('_').toUpperCase();

            if (!levelGroups[ind]) levelGroups[ind] = [];
            levelGroups[ind].push({{ level, value: lastVal }});
        }}

        for (const [ind, levels] of Object.entries(levelGroups)) {{
            levels.sort((a, b) => b.value - a.value);

            let indName = ind;
            if (ind === 'FIB') indName = '📐 Fibonacci';
            else if (ind === 'SR') indName = '📊 Support/Resistance';
            else if (ind === 'PIVOT') indName = '🎯 Pivot Points';

            html += `
                <div style="background: ${{colors.card_bg}}; padding: 12px; border-radius: 8px; border: 1px solid ${{colors.border}};">
                    <div style="font-weight: 600; margin-bottom: 8px; color: ${{colors.text}};">${{indName}}</div>
            `;

            for (const {{ level, value }} of levels) {{
                const diff = ((value - lastClose) / lastClose * 100).toFixed(2);
                const isAbove = value > lastClose;
                const levelColor = isAbove ? colors.candle_down : colors.candle_up;
                const arrow = isAbove ? '↑' : '↓';

                html += `
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px;">
                        <span style="color: ${{colors.text_muted}};">${{level.replace(/_/g, ' ')}}</span>
                        <span>
                            <span style="color: ${{colors.text}}; font-weight: 500;">$${{value.toFixed(2)}}</span>
                            <span style="color: ${{levelColor}}; margin-left: 8px;">${{arrow}}${{Math.abs(diff)}}%</span>
                        </span>
                    </div>
                `;
            }}
            html += '</div>';
        }}
        html += '</div></div>';
    }}

    // ========== SECTION 3: Full Signal History (Collapsible) ==========
    if (allSignals.length > 0) {{
        const sortedAll = [...allSignals].sort((a, b) =>
            new Date(b.timestamp) - new Date(a.timestamp)
        );

        html += `
            <div style="margin-top: 24px;">
                <h4 class="collapsible-header" onclick="toggleHistoryTable()" style="cursor: pointer; margin-bottom: 12px; color: ${{colors.text_muted}}; font-size: 12px; text-transform: uppercase;">
                    <span id="history-toggle-icon">▶</span> Full Signal History (${{allSignals.length}} signals) - Click to expand
                </h4>
                <div id="full-history-table" style="display: none;">
                    <table class="signal-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Signal</th>
                                <th>Direction</th>
                                <th>Indicator</th>
                                <th>Message</th>
                            </tr>
                        </thead>
                        <tbody>
        `;

        for (const sig of sortedAll.slice(0, 100)) {{
            const time = new Date(sig.timestamp).toLocaleString();
            const direction = sig.direction || 'alert';
            html += `
                <tr>
                    <td>${{time}}</td>
                    <td>${{sig.rule}}</td>
                    <td><span class="signal-badge ${{direction}}">${{direction}}</span></td>
                    <td>${{sig.indicator}}</td>
                    <td>${{sig.message || '-'}}</td>
                </tr>
            `;
        }}
        if (allSignals.length > 100) {{
            html += `<tr><td colspan="5" style="text-align: center; color: ${{colors.text_muted}};">... and ${{allSignals.length - 100}} more signals</td></tr>`;
        }}

        html += '</tbody></table></div></div>';
    }}

    container.innerHTML = html;
}}

// Toggle full history table visibility
function toggleHistoryTable() {{
    const table = document.getElementById('full-history-table');
    const icon = document.getElementById('history-toggle-icon');
    if (table && icon) {{
        if (table.style.display === 'none') {{
            table.style.display = 'block';
            icon.textContent = '▼';
        }} else {{
            table.style.display = 'none';
            icon.textContent = '▶';
        }}
    }}
}}

function updateConfluencePanel() {{
    const key = getDataKey();
    const confluence = confluenceData[key];
    const container = document.getElementById('confluence-panel');

    if (!confluence) {{
        container.innerHTML = '<div class="no-divergences">No confluence data available for this symbol/timeframe</div>';
        return;
    }}

    const alignmentPct = (confluence.alignment_score + 100) / 2;
    const alignmentClass = confluence.alignment_score > 20 ? 'bullish' : confluence.alignment_score < -20 ? 'bearish' : 'neutral';
    const strongestClass = confluence.strongest_signal === 'bullish' ? 'bullish' : confluence.strongest_signal === 'bearish' ? 'bearish' : 'neutral';

    let divergenceHtml = '';
    if (confluence.diverging_pairs && confluence.diverging_pairs.length > 0) {{
        divergenceHtml = confluence.diverging_pairs.slice(0, 5).map(p => `
            <div class="divergence-item">
                <div class="indicators">${{p.ind1}} - ${{p.ind2}}</div>
                <div class="reason">${{p.reason}}</div>
            </div>
        `).join('');
    }} else {{
        divergenceHtml = '<div class="no-divergences">No divergences detected - indicators are aligned</div>';
    }}

    container.innerHTML = `
        <div class="confluence-panel">
            <div class="confluence-score">
                <div class="alignment-meter">
                    <div class="alignment-bar">
                        <div class="alignment-indicator" style="left: ${{alignmentPct}}%"></div>
                    </div>
                </div>
                <div class="alignment-value ${{alignmentClass}}">${{confluence.alignment_score > 0 ? '+' : ''}}${{confluence.alignment_score}}</div>
                <div class="signal-counts">
                    <div class="count-item">
                        <div class="count-value bullish">&#9650; ${{confluence.bullish_count}}</div>
                        <div class="count-label">Bullish</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value neutral">&#9679; ${{confluence.neutral_count}}</div>
                        <div class="count-label">Neutral</div>
                    </div>
                    <div class="count-item">
                        <div class="count-value bearish">&#9660; ${{confluence.bearish_count}}</div>
                        <div class="count-label">Bearish</div>
                    </div>
                </div>
                <div class="strongest-signal">
                    <div class="label">Strongest Signal</div>
                    <div class="value ${{strongestClass}}">${{confluence.strongest_signal ? confluence.strongest_signal.toUpperCase() : 'NONE'}}</div>
                </div>
            </div>
            <div class="divergence-list">
                <h4 style="margin-bottom: 12px; color: ${{colors.text_muted}}; font-size: 12px; text-transform: uppercase;">Diverging Indicators</h4>
                ${{divergenceHtml}}
            </div>
        </div>
    `;
}}

function toggleSection(arg) {{
    if (typeof arg === 'string') {{
        const content = document.getElementById(arg);
        if (!content) return;
        const header = content.previousElementSibling;
        const icon = header ? header.querySelector('.toggle-icon') : null;

        content.classList.toggle('collapsed');
        if (icon) {{
            icon.style.transform = content.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
        }}
    }} else {{
        const header = arg;
        const section = header.parentElement;
        if (!section || !section.classList.contains('report-section')) return;

        section.classList.toggle('collapsed');
        const indicator = header.querySelector('.collapse-indicator');
        if (indicator) {{
            indicator.style.transform = section.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
        }}
    }}
}}

document.addEventListener('DOMContentLoaded', () => {{
    updateChart();
    updateSignalHistoryTable();
    updateConfluencePanel();
    updateRegimeSection();
}});
"""
