"""
HTML Assets - CSS and HTML template generation.

Generates CSS stylesheets and HTML page templates for the signal package.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from src.domain.signals.indicators.regime import RegimeOutput

from .constants import THEMES


def build_css(theme: str = "dark") -> str:
    """
    Build CSS content for styles.css.

    Uses shared HEATMAP_CSS for consistent theming between heatmap (index.html)
    and signal report (report.html). Report-specific styles are already included
    in HEATMAP_CSS under the .apex-hm scope.

    Adds regime styles for 1:1 feature parity with SignalReportGenerator.

    Args:
        theme: Color theme ("dark" or "light")

    Returns:
        Complete CSS content string
    """
    from ..heatmap.css import HEATMAP_CSS
    from ..regime import generate_regime_styles

    # Get regime styles (function doesn't take theme arg - uses CSS vars)
    regime_css = generate_regime_styles()

    # Combine shared CSS with regime styles
    return f"""{HEATMAP_CSS}

/* Regime Report Styles (1:1 Feature Parity) */
{regime_css}
"""


def build_index_html(
    symbols: List[str],
    timeframes: List[str],
    colors: Dict[str, str],
    regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
    validation_url: Optional[str] = None,
) -> str:
    """
    Build the index.html shell with full feature parity sections.

    Args:
        symbols: List of symbol names
        timeframes: List of timeframe strings
        colors: Theme color dictionary
        regime_outputs: Optional dict mapping symbol to RegimeOutput
        validation_url: Optional URL to validation results page

    Returns:
        Complete HTML content string
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    symbol_options = "\n".join(
        f'                    <option value="{s}">{s}</option>' for s in symbols
    )

    timeframe_buttons = "\n".join(
        f'                    <button class="tf-btn" data-tf="{tf}" onclick="setTimeframe(\'{tf}\')">{tf}</button>'
        for tf in timeframes
    )

    # Validation link if URL provided
    validation_link = ""
    if validation_url:
        validation_link = (
            f'<a href="{validation_url}" class="validation-link">:bar_chart: Validation Results</a>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="stylesheet" href="assets/styles.css">
</head>
<body class="apex-hm">
    <div class="container">
        <header class="header">
            <div class="header-top">
                <a href="index.html" class="back-link">\u2190 Heatmap</a>
                <h1>Signal Analysis Report</h1>
                <a href="strategies.html" class="strategies-link" style="color:#58a6ff;font-size:13px;text-decoration:none;">Strategy Comparison</a>
                <a href="pead.html" style="color:#58a6ff;font-size:13px;text-decoration:none;">PEAD Screen</a>
                {validation_link}
            </div>
            <div class="meta">
                <span><strong>Symbols:</strong> {len(symbols)}</span>
                <span><strong>Timeframes:</strong> {', '.join(timeframes)}</span>
                <span><strong>Generated:</strong> {generated_at}</span>
            </div>
        </header>

        <div class="controls">
            <div class="control-group">
                <label>Symbol</label>
                <select id="symbol-select" onchange="updateChart()">
{symbol_options}
                </select>
            </div>
            <div class="control-group">
                <label>Timeframe</label>
                <div class="timeframe-buttons">
{timeframe_buttons}
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div id="main-chart" class="loading">
                <div class="loading-spinner">Loading chart data...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('confluence-content')">
                <span class="toggle-icon">&#9660;</span> Confluence Analysis
            </h2>
            <div id="confluence-content" class="section-content">
                <div class="no-confluence">Loading confluence data...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('dual-macd-history-content')">
                <span class="toggle-icon">&#9660;</span> DualMACD
            </h2>
            <div id="dual-macd-history-content" class="section-content">
                <div id="dual-macd-history-table">
                    <div style="color: #94a3b8; padding: 16px;">Loading DualMACD state...</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('trend-pulse-history-content')">
                <span class="toggle-icon">&#9660;</span> TrendPulse
            </h2>
            <div id="trend-pulse-history-content" class="section-content">
                <div id="trend-pulse-history-table">
                    <div style="color: #94a3b8; padding: 16px;">Loading TrendPulse state...</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('regime-content')">
                <span class="toggle-icon">&#9660;</span> Regime Analysis
            </h2>
            <div id="regime-content" class="section-content">
                <div class="no-regime">Loading regime data...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('signals-content')">
                <span class="toggle-icon">&#9660;</span> Signal History
            </h2>
            <div id="signals-content" class="section-content">
                <div class="no-signals">Loading signal history...</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('pulse-dip-content')">
                <span class="toggle-icon">&#9660;</span> PulseDip Strategy
            </h2>
            <div id="pulse-dip-content" class="section-content">
                <div id="pulse-dip-table">
                    <div style="color: #94a3b8; padding: 16px;">Loading PulseDip signals...</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('squeeze-play-content')">
                <span class="toggle-icon">&#9660;</span> SqueezePlay Strategy
            </h2>
            <div id="squeeze-play-content" class="section-content">
                <div id="squeeze-play-table">
                    <div style="color: #94a3b8; padding: 16px;">Loading SqueezePlay signals...</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('regime-flex-content')">
                <span class="toggle-icon">&#9660;</span> RegimeFlex Strategy
            </h2>
            <div id="regime-flex-content" class="section-content">
                <div id="regime-flex-table">
                    <div style="color: #94a3b8; padding: 16px;">Loading RegimeFlex signals...</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('sector-pulse-content')">
                <span class="toggle-icon">&#9660;</span> SectorPulse Strategy
            </h2>
            <div id="sector-pulse-content" class="section-content">
                <div id="sector-pulse-table">
                    <div style="color: #94a3b8; padding: 16px;">Loading SectorPulse signals...</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-header" onclick="toggleSection('indicators-content')">
                <span class="toggle-icon">&#9660;</span> Indicators
            </h2>
            <div id="indicators-content" class="section-content collapsed">
                <div class="no-indicators">Loading indicators...</div>
            </div>
        </div>
    </div>

    <script src="assets/app.js"></script>
</body>
</html>"""


def get_theme_colors(theme: str) -> Dict[str, str]:
    """
    Get color scheme for a theme.

    Args:
        theme: Theme name ("dark" or "light")

    Returns:
        Dict of color values
    """
    return THEMES.get(theme, THEMES["dark"])


def timeframe_seconds(tf: str) -> int:
    """
    Convert timeframe string to seconds for sorting.

    Args:
        tf: Timeframe string (e.g., "1d", "5m")

    Returns:
        Number of seconds in the timeframe
    """
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
        "1w": 604800,
    }
    return mapping.get(tf, 0)
