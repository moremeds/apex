"""
Coverage Visualizer - Generates HTML report with line charts of historical data.

Creates a self-contained HTML file with:
- Per-symbol collapsible sections
- Line chart per timeframe showing close prices
- Source labels (IB, Yahoo, live) color-coded
- Validation status badges (PASS/WARN/CAUTION/FAIL)
- Actual data from Parquet (not unreliable DuckDB metadata)
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..domain.services.data_validator import DataValidator
from ..infrastructure.stores.parquet_historical_store import ParquetHistoricalStore
from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


TIMEFRAME_DISPLAY_ORDER = ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]


@dataclass
class TimeframeData:
    """Data for one symbol/timeframe combination."""

    symbol: str
    timeframe: str
    timestamps: List[str]  # ISO format for JS
    closes: List[float]
    sources: List[str]  # Source per bar for color coding
    bar_count: int
    earliest: Optional[datetime]
    latest: Optional[datetime]
    source_breakdown: Dict[str, int]  # source -> count
    # Validation fields
    validation_status: str = "UNKNOWN"  # PASS, WARN, CAUTION, FAIL
    coverage_pct: float = 0.0
    expected_bars: int = 0
    gap_count: int = 0


class CoverageVisualizer:
    """
    Generates HTML visualization with line charts of historical data.

    Reads actual bar data from Parquet storage (not metadata).
    Shows close prices per timeframe with source color coding.
    """

    # Source colors for chart segments
    SOURCE_COLORS = {
        "ib": "#3b82f6",  # Blue
        "yahoo": "#f97316",  # Orange
        "live": "#22c55e",  # Green
        "unknown": "#6b7280",  # Gray
    }

    # Validation status colors
    STATUS_COLORS = {
        "PASS": "#22c55e",  # Green
        "WARN": "#eab308",  # Yellow
        "CAUTION": "#f97316",  # Orange
        "FAIL": "#ef4444",  # Red
        "UNKNOWN": "#6b7280",  # Gray
    }

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = base_dir or Path("data/historical")
        self._store = ParquetHistoricalStore(base_dir=self._base_dir)
        self._validator = DataValidator(bar_store=self._store)

    def generate_html(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        max_points_per_chart: int = 500,
    ) -> str:
        """
        Generate coverage report HTML with line charts.

        Args:
            symbols: Symbols to include (default: all with data).
            timeframes: Timeframes to show (default: standard set).
            output_path: Optional path to write HTML file.
            max_points_per_chart: Downsample if more bars (for performance).

        Returns:
            HTML string.
        """
        timeframes = timeframes or TIMEFRAME_DISPLAY_ORDER

        # Discover symbols if not provided
        if symbols:
            display_symbols = sorted(symbols)
        else:
            display_symbols = sorted(self._store.list_symbols())

        # Collect data for all charts
        all_data: Dict[str, Dict[str, TimeframeData]] = {}
        total_bars = 0

        for symbol in display_symbols:
            all_data[symbol] = {}
            for tf in timeframes:
                data = self._load_timeframe_data(symbol, tf, max_points_per_chart)
                if data and data.bar_count > 0:
                    all_data[symbol][tf] = data
                    total_bars += data.bar_count

        # Generate HTML
        html_content = self._render_html(all_data, display_symbols, timeframes, total_bars)

        # Write to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content)
            logger.info(f"Coverage report written to {output_path}")

        return html_content

    def _load_timeframe_data(
        self,
        symbol: str,
        timeframe: str,
        max_points: int,
    ) -> Optional[TimeframeData]:
        """Load actual bar data from Parquet and run validation."""
        bars = self._store.read_bars(symbol, timeframe)
        if not bars:
            return None

        # Run validation to get coverage stats
        validation_result = self._validator.validate(symbol, timeframe)

        # Downsample if too many points (for chart rendering only)
        display_bars = bars
        if len(bars) > max_points:
            step = len(bars) // max_points
            display_bars = bars[::step]

        timestamps = []
        closes = []
        sources = []
        source_counts: Dict[str, int] = {}

        for bar in display_bars:
            ts = bar.bar_start or bar.timestamp
            if ts:
                timestamps.append(ts.isoformat())
            else:
                timestamps.append("")

            closes.append(bar.close if bar.close is not None else 0)

            source = (bar.source or "unknown").lower()
            sources.append(source)
            source_counts[source] = source_counts.get(source, 0) + 1

        earliest = bars[0].bar_start or bars[0].timestamp if bars else None
        latest = bars[-1].bar_start or bars[-1].timestamp if bars else None

        return TimeframeData(
            symbol=symbol,
            timeframe=timeframe,
            timestamps=timestamps,
            closes=closes,
            sources=sources,
            bar_count=len(bars),  # Use full bar count, not downsampled
            earliest=earliest,
            latest=latest,
            source_breakdown=source_counts,
            validation_status=validation_result.status.name,
            coverage_pct=validation_result.coverage_pct,
            expected_bars=validation_result.expected_bars,
            gap_count=len(validation_result.gaps),
        )

    def _render_html(
        self,
        all_data: Dict[str, Dict[str, TimeframeData]],
        symbols: List[str],
        timeframes: List[str],
        total_bars: int,
    ) -> str:
        """Render complete HTML document."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build chart data JSON
        chart_data: Dict[str, Dict[str, Dict[str, object]]] = {}
        for symbol, tf_data in all_data.items():
            chart_data[symbol] = {}
            for tf, data in tf_data.items():
                chart_data[symbol][tf] = {
                    "timestamps": data.timestamps,
                    "closes": data.closes,
                    "sources": data.sources,
                    "bar_count": data.bar_count,
                    "earliest": data.earliest.strftime("%Y-%m-%d") if data.earliest else "N/A",
                    "latest": data.latest.strftime("%Y-%m-%d") if data.latest else "N/A",
                    "source_breakdown": data.source_breakdown,
                    "validation_status": data.validation_status,
                    "coverage_pct": data.coverage_pct,
                    "expected_bars": data.expected_bars,
                    "gap_count": data.gap_count,
                }

        symbols_with_data = sum(1 for s in symbols if all_data.get(s))

        # Count validation statuses
        validation_counts: Dict[str, int] = {"PASS": 0, "WARN": 0, "CAUTION": 0, "FAIL": 0}
        for symbol_data in all_data.values():
            for timeframe_data in symbol_data.values():
                status = timeframe_data.validation_status
                if status in validation_counts:
                    validation_counts[status] += 1

        chart_data_json = json.dumps(chart_data)
        source_colors_json = json.dumps(self.SOURCE_COLORS)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEX Historical Data Coverage Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>{self._get_css()}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>APEX Historical Data Coverage</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section class="summary">
            <div class="summary-card">
                <div class="summary-value">{symbols_with_data}</div>
                <div class="summary-label">Symbols with Data</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{len(timeframes)}</div>
                <div class="summary-label">Timeframes</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{total_bars:,}</div>
                <div class="summary-label">Total Bars</div>
            </div>
            <div class="summary-card validation-summary">
                <div class="validation-counts">
                    <span class="v-count pass">{validation_counts['PASS']}</span>
                    <span class="v-count warn">{validation_counts['WARN']}</span>
                    <span class="v-count caution">{validation_counts['CAUTION']}</span>
                    <span class="v-count fail">{validation_counts['FAIL']}</span>
                </div>
                <div class="summary-label">Validation (PASS/WARN/CAUTION/FAIL)</div>
            </div>
        </section>

        <section class="legend">
            <h3>Data Sources:</h3>
            <span class="legend-item"><span class="legend-color" style="background: #3b82f6;"></span> IB</span>
            <span class="legend-item"><span class="legend-color" style="background: #f97316;"></span> Yahoo</span>
            <span class="legend-item"><span class="legend-color" style="background: #22c55e;"></span> Live</span>
            <span class="legend-item"><span class="legend-color" style="background: #6b7280;"></span> Unknown</span>
            <span class="legend-divider">|</span>
            <h3>Validation:</h3>
            <span class="legend-item"><span class="legend-color" style="background: #22c55e;"></span> PASS (&ge;98%)</span>
            <span class="legend-item"><span class="legend-color" style="background: #eab308;"></span> WARN (95-98%)</span>
            <span class="legend-item"><span class="legend-color" style="background: #f97316;"></span> CAUTION (90-95%)</span>
            <span class="legend-item"><span class="legend-color" style="background: #ef4444;"></span> FAIL (&lt;90%)</span>
        </section>

        <div id="symbols-container">
            {self._render_symbol_sections(symbols, timeframes, all_data)}
        </div>
    </div>

    <script>
        const chartData = {chart_data_json};
        const sourceColors = {source_colors_json};

        function toggleSymbol(symbol) {{
            const content = document.getElementById('content-' + symbol);
            const chevron = document.getElementById('chevron-' + symbol);
            const isExpanded = content.style.display !== 'none';

            content.style.display = isExpanded ? 'none' : 'block';
            chevron.textContent = isExpanded ? '▶' : '▼';

            if (!isExpanded) {{
                renderChartsForSymbol(symbol);
            }}
        }}

        function renderChartsForSymbol(symbol) {{
            const symbolData = chartData[symbol];
            if (!symbolData) return;

            for (const [tf, data] of Object.entries(symbolData)) {{
                const chartId = 'chart-' + symbol + '-' + tf;
                const chartDiv = document.getElementById(chartId);
                if (!chartDiv || chartDiv.dataset.rendered === 'true') continue;

                // Create trace with source-based coloring
                const trace = {{
                    x: data.timestamps,
                    y: data.closes,
                    type: 'scatter',
                    mode: 'lines',
                    name: tf,
                    line: {{ width: 1.5 }},
                    hovertemplate: '%{{x}}<br>Close: $%{{y:.2f}}<extra></extra>'
                }};

                // Determine dominant source for line color
                let dominantSource = 'unknown';
                let maxCount = 0;
                for (const [src, count] of Object.entries(data.source_breakdown)) {{
                    if (count > maxCount) {{
                        maxCount = count;
                        dominantSource = src;
                    }}
                }}
                trace.line.color = sourceColors[dominantSource] || sourceColors.unknown;

                const layout = {{
                    margin: {{ l: 60, r: 20, t: 30, b: 40 }},
                    height: 200,
                    paper_bgcolor: '#1e293b',
                    plot_bgcolor: '#1e293b',
                    font: {{ color: '#e2e8f0', size: 11 }},
                    xaxis: {{
                        gridcolor: '#334155',
                        tickformat: '%Y-%m-%d',
                    }},
                    yaxis: {{
                        gridcolor: '#334155',
                        tickprefix: '$',
                    }},
                    showlegend: false,
                }};

                Plotly.newPlot(chartId, [trace], layout, {{ responsive: true, displayModeBar: false }});
                chartDiv.dataset.rendered = 'true';
            }}
        }}

        // Expand first symbol by default
        document.addEventListener('DOMContentLoaded', function() {{
            const symbols = Object.keys(chartData);
            if (symbols.length > 0) {{
                toggleSymbol(symbols[0]);
            }}
        }});
    </script>
</body>
</html>"""

    def _render_symbol_sections(
        self,
        symbols: List[str],
        timeframes: List[str],
        all_data: Dict[str, Dict[str, TimeframeData]],
    ) -> str:
        """Render collapsible sections for each symbol."""
        sections = []

        for symbol in symbols:
            symbol_data = all_data.get(symbol, {})
            tf_count = len(symbol_data)
            total_bars = sum(d.bar_count for d in symbol_data.values())

            # Summary sources
            all_sources: Dict[str, int] = {}
            for d in symbol_data.values():
                for src, cnt in d.source_breakdown.items():
                    all_sources[src] = all_sources.get(src, 0) + cnt

            source_tags = " ".join(
                f'<span class="source-tag" style="background: {self.SOURCE_COLORS.get(src, "#6b7280")}">{src}: {cnt:,}</span>'
                for src, cnt in sorted(all_sources.items(), key=lambda x: -x[1])
            )

            # Build charts for each timeframe
            charts_html = []
            for tf in timeframes:
                if tf in symbol_data:
                    data = symbol_data[tf]
                    status_color = self.STATUS_COLORS.get(data.validation_status, "#6b7280")
                    gap_info = f" | {data.gap_count} gaps" if data.gap_count > 0 else ""
                    chart_html = f"""
                    <div class="chart-container">
                        <div class="chart-header">
                            <span class="chart-title">{tf}</span>
                            <span class="validation-badge" style="background: {status_color}">{data.validation_status} {data.coverage_pct:.1f}%</span>
                            <span class="chart-info">{data.bar_count:,}/{data.expected_bars:,} bars{gap_info} | {data.earliest.strftime("%Y-%m-%d") if data.earliest else "N/A"} to {data.latest.strftime("%Y-%m-%d") if data.latest else "N/A"}</span>
                        </div>
                        <div id="chart-{html.escape(symbol)}-{tf}" class="chart"></div>
                    </div>
                    """
                    charts_html.append(chart_html)
                else:
                    charts_html.append(f"""
                    <div class="chart-container no-data">
                        <div class="chart-header">
                            <span class="chart-title">{tf}</span>
                            <span class="validation-badge" style="background: #6b7280">NO DATA</span>
                            <span class="chart-info">No data</span>
                        </div>
                    </div>
                    """)

            section = f"""
            <div class="symbol-section">
                <div class="symbol-header" onclick="toggleSymbol('{html.escape(symbol)}')">
                    <span id="chevron-{html.escape(symbol)}" class="chevron">▶</span>
                    <span class="symbol-name">{html.escape(symbol)}</span>
                    <span class="symbol-summary">{tf_count} timeframes | {total_bars:,} bars</span>
                    <div class="source-tags">{source_tags}</div>
                </div>
                <div id="content-{html.escape(symbol)}" class="symbol-content" style="display: none;">
                    {"".join(charts_html)}
                </div>
            </div>
            """
            sections.append(section)

        return "\n".join(sections)

    def _get_css(self) -> str:
        """Return embedded CSS styles."""
        return """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    padding: 20px;
    line-height: 1.5;
}
.container { max-width: 1400px; margin: 0 auto; }
header { text-align: center; margin-bottom: 30px; }
h1 { font-size: 24px; margin-bottom: 8px; }
.timestamp { color: #64748b; font-size: 14px; }

.summary {
    display: flex;
    justify-content: center;
    gap: 24px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.summary-card {
    background: #1e293b;
    padding: 16px 24px;
    border-radius: 8px;
    text-align: center;
    min-width: 150px;
}
.summary-value { font-size: 28px; font-weight: bold; margin-bottom: 4px; }
.summary-label { color: #94a3b8; font-size: 14px; }
.validation-summary { min-width: 200px; }
.validation-counts { display: flex; gap: 8px; justify-content: center; margin-bottom: 4px; }
.v-count { font-size: 18px; font-weight: bold; padding: 4px 10px; border-radius: 4px; }
.v-count.pass { background: #22c55e; }
.v-count.warn { background: #eab308; }
.v-count.caution { background: #f97316; }
.v-count.fail { background: #ef4444; }

.legend {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 24px;
    margin-bottom: 30px;
    padding: 12px;
    background: #1e293b;
    border-radius: 8px;
}
.legend h3 { color: #94a3b8; font-size: 14px; font-weight: 500; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.legend-color { width: 14px; height: 14px; border-radius: 3px; }
.legend-divider { color: #475569; font-size: 20px; margin: 0 8px; }

.symbol-section {
    background: #1e293b;
    border-radius: 8px;
    margin-bottom: 12px;
    overflow: hidden;
}
.symbol-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    cursor: pointer;
    transition: background 0.2s;
}
.symbol-header:hover { background: #334155; }
.chevron { color: #64748b; font-size: 12px; width: 16px; }
.symbol-name { font-size: 16px; font-weight: 600; }
.symbol-summary { color: #94a3b8; font-size: 13px; margin-left: 8px; }
.source-tags { margin-left: auto; display: flex; gap: 8px; }
.source-tag {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    color: white;
    font-weight: 500;
}

.symbol-content {
    padding: 0 18px 18px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 16px;
}

.chart-container {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
}
.chart-container.no-data {
    opacity: 0.5;
    min-height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chart-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.chart-title {
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
}
.validation-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    color: white;
    font-weight: 600;
}
.chart-info {
    font-size: 12px;
    color: #64748b;
    margin-left: auto;
}
.chart {
    width: 100%;
    min-height: 200px;
}
"""
