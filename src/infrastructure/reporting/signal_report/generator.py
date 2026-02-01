"""
Signal Report Generator - Interactive HTML reports for signal analysis.

Generates self-contained HTML reports with:
- Symbol selector dropdown
- Timeframe toggle buttons
- Candlestick price charts with overlay indicators
- Separate subplots for oscillators (RSI, MACD, etc.)
- Auto-generated descriptions for indicators and rules
- Signal history showing when rules would have triggered
- Collapsible indicator sections for better readability
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.timezone import DisplayTimezone, now_utc

from .confluence_analyzer import calculate_confluence
from .constants import (
    BOUNDED_OSCILLATORS,
    OVERLAY_INDICATORS,
    TIMEFRAME_SECONDS,
    UNBOUNDED_OSCILLATORS,
    VOLUME_INDICATORS,
)
from .dual_macd_section import compute_dual_macd_history, render_dual_macd_history_html
from .html_renderer import (
    render_indicator_cards,
    render_symbol_options,
    render_timeframe_buttons,
)
from .plotly_scripts import get_scripts
from .regime_renderer import (
    compute_param_analysis,
    compute_regime_outputs,
    render_regime_sections,
)
from .signal_detection import detect_historical_signals
from .theme_styles import get_styles, get_theme_colors
from .trend_pulse_section import compute_trend_pulse_history, render_trend_pulse_history_html

if TYPE_CHECKING:
    from src.domain.signals.indicators.base import Indicator
    from src.domain.signals.indicators.regime import RegimeOutput
    from src.domain.signals.models import SignalRule

logger = get_logger(__name__)


class SignalReportGenerator:
    """
    Generate interactive HTML reports for signal analysis.

    Uses Plotly for charts with proper subplots:
    - Price chart with overlay indicators (Bollinger, SuperTrend, etc.)
    - RSI subplot (0-100 scale)
    - MACD subplot (unbounded scale)
    - Volume subplot
    """

    def __init__(self, theme: str = "dark") -> None:
        self.theme = theme
        self._colors = get_theme_colors(theme)

    def generate(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
        rules: List["SignalRule"],
        output_path: Path,
        regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
        display_timezone: str = "US/Eastern",
    ) -> Path:
        """
        Generate combined HTML report with symbol selector.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame with OHLCV + indicator columns
            indicators: List of computed indicators
            rules: List of signal rules
            output_path: Where to save HTML
            regime_outputs: Optional dict mapping symbol to RegimeOutput for regime sections

        Returns:
            Path to generated HTML file
        """
        symbols = sorted(set(sym for sym, tf in data.keys()))
        timeframes = sorted(
            set(tf for sym, tf in data.keys()),
            key=lambda x: TIMEFRAME_SECONDS.get(x, 0),
        )

        # Build chart data for JavaScript
        chart_data = self._build_chart_data(data)

        # Calculate confluence scores for each symbol/timeframe
        confluence_scores = calculate_confluence(data)
        confluence_data = {
            key: {
                "symbol": score.symbol,
                "timeframe": score.timeframe,
                "bullish_count": score.bullish_count,
                "bearish_count": score.bearish_count,
                "neutral_count": score.neutral_count,
                "alignment_score": score.alignment_score,
                "diverging_pairs": [
                    {"ind1": p[0], "ind2": p[1], "reason": p[2]} for p in score.diverging_pairs
                ],
                "strongest_signal": score.strongest_signal,
            }
            for key, score in confluence_scores.items()
        }

        # Detect historical signals for each symbol/timeframe
        signal_history: Dict[str, List[Dict[str, Any]]] = {}
        for (symbol, timeframe), df in data.items():
            key = f"{symbol}_{timeframe}"
            signals = detect_historical_signals(df, rules, symbol, timeframe)
            signal_history[key] = signals
            if signals:
                logger.info(f"Detected {len(signals)} signals for {key}")

        # Build indicator and rule descriptions
        indicator_info = self._build_indicator_info(indicators, rules)

        # Compute regime outputs if not provided
        if regime_outputs is None:
            regime_outputs = compute_regime_outputs(data, indicators)

        # Compute parameter provenance and recommendations
        provenance_dict, recommendations_dict = compute_param_analysis(data, indicators)

        # Compute DualMACD historical state for verification table
        dual_macd_history = compute_dual_macd_history(data, display_timezone=display_timezone)

        # Compute TrendPulse historical state for verification table
        trend_pulse_history = compute_trend_pulse_history(data)
        logger.info(
            f"TrendPulse history: {len(trend_pulse_history)} symbol/tf pairs, "
            f"input bar counts: {{{', '.join(f'{k}: {len(v)}' for k, v in data.items())}}}"
        )

        # Render HTML
        html = self._render_html(
            symbols=symbols,
            timeframes=timeframes,
            chart_data=chart_data,
            indicator_info=indicator_info,
            signal_history=signal_history,
            confluence_data=confluence_data,
            regime_outputs=regime_outputs,
            provenance_dict=provenance_dict,
            recommendations_dict=recommendations_dict,
            dual_macd_history=dual_macd_history,
            trend_pulse_history=trend_pulse_history,
            display_timezone=display_timezone,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.info(f"Signal report generated: {output_path}")
        return output_path

    def _build_chart_data(self, data: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[str, Any]:
        """Build chart data structure for JavaScript with indicator grouping."""
        chart_data = {}
        oscillator_names = BOUNDED_OSCILLATORS | UNBOUNDED_OSCILLATORS

        for (symbol, timeframe), df in data.items():
            key = f"{symbol}_{timeframe}"

            # Convert timestamps to ISO strings
            if hasattr(df.index, "strftime"):
                timestamps = df.index.strftime("%Y-%m-%dT%H:%M:%S").tolist()
            else:
                timestamps = [str(t) for t in df.index]

            # Base OHLCV data
            ohlcv = {
                name: df[name].tolist() if name in df.columns else []
                for name in ("open", "high", "low", "close", "volume")
            }
            chart_data[key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "bar_count": len(df),
                "timestamps": timestamps,
                **ohlcv,
                "overlays": {},
                "rsi": {},
                "macd": {},
                "dual_macd": {},  # DualMACD (55/89 + 13/21) overlapping histograms
                "oscillators": {},
                "volume_ind": {},
            }

            # Categorize indicator columns
            ohlcv_cols = {"open", "high", "low", "close", "volume", "timestamp"}
            for col in df.columns:
                if col.lower() in ohlcv_cols:
                    continue

                values = df[col].tolist()
                values = [None if pd.isna(v) else v for v in values]

                # Parse indicator name from prefixed column (e.g., "macd_histogram" -> "macd")
                parts = col.split("_")
                ind_name = parts[0].lower() if parts else col.lower()

                # Route to appropriate subplot bucket
                if ind_name in OVERLAY_INDICATORS:
                    bucket = "overlays"
                elif ind_name == "rsi":
                    bucket = "rsi"
                elif col.startswith("trend_pulse_"):
                    # TrendPulse columns: used for history table only, not charted
                    continue
                elif col.startswith("dual_macd_"):
                    # DualMACD indicator (dual_macd_slow_histogram, dual_macd_fast_histogram, etc.)
                    bucket = "dual_macd"
                elif ind_name == "macd":
                    bucket = "macd"
                elif ind_name in oscillator_names:
                    bucket = "oscillators"
                elif ind_name in VOLUME_INDICATORS:
                    bucket = "volume_ind"
                else:
                    bucket = "oscillators"
                chart_data[key][bucket][col] = values

        return chart_data

    def _build_indicator_info(
        self,
        indicators: List["Indicator"],
        rules: List["SignalRule"],
    ) -> List[Dict[str, Any]]:
        """Build indicator information with descriptions and linked rules."""
        from ..description_generator import (
            generate_indicator_description,
            generate_rule_description,
        )

        rules_by_indicator: Dict[str, List[Dict[str, Any]]] = {}
        for rule in rules:
            rules_by_indicator.setdefault(rule.indicator, []).append(
                {
                    "name": rule.name,
                    "description": generate_rule_description(rule),
                    "direction": rule.direction.value,
                    "timeframes": list(rule.timeframes),
                }
            )

        info_list = [
            {
                "name": ind.name,
                "category": ind.category.value,
                "description": generate_indicator_description(ind),
                "warmup_periods": ind.warmup_periods,
                "rules": rules_by_indicator.get(ind.name, []),
            }
            for ind in indicators
        ]
        info_list.sort(key=lambda x: (x["category"], x["name"]))
        return info_list

    def _render_html(
        self,
        symbols: List[str],
        timeframes: List[str],
        chart_data: Dict[str, Any],
        indicator_info: List[Dict[str, Any]],
        signal_history: Dict[str, List[Dict[str, Any]]],
        confluence_data: Dict[str, Dict[str, Any]],
        regime_outputs: Optional[Dict[str, "RegimeOutput"]] = None,
        provenance_dict: Optional[Dict[str, Any]] = None,
        recommendations_dict: Optional[Dict[str, Any]] = None,
        dual_macd_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        trend_pulse_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        display_timezone: str = "US/Eastern",
    ) -> str:
        _tz = DisplayTimezone(display_timezone)
        generated_at = _tz.format_with_tz(now_utc(), "%Y-%m-%d %H:%M %Z")
        regime_outputs = regime_outputs or {}
        provenance_dict = provenance_dict or {}
        recommendations_dict = recommendations_dict or {}
        dual_macd_history = dual_macd_history or {}
        trend_pulse_history = trend_pulse_history or {}

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
{get_styles(self._colors)}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Signal Analysis Report</h1>
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
                    {render_symbol_options(symbols)}
                </select>
            </div>
            <div class="control-group">
                <label>Timeframe</label>
                <div class="timeframe-buttons">
                    {render_timeframe_buttons(timeframes)}
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div id="main-chart"></div>
        </div>

        <div class="confluence-section">
            <h2 class="section-header" onclick="toggleSection('confluence-content')">
                <span class="toggle-icon">\u25bc</span> Confluence Analysis
            </h2>
            <div id="confluence-content" class="section-content">
                <div id="confluence-panel"></div>
            </div>
        </div>

        {render_regime_sections(regime_outputs, provenance_dict, recommendations_dict, self.theme)}

        <div class="signal-history-section">
            <h2 class="section-header" onclick="toggleSection('signal-history-content')">
                <span class="toggle-icon">\u25bc</span> Signal History
            </h2>
            <div id="signal-history-content" class="section-content">
                <div id="signal-history-table"></div>
            </div>
        </div>

        {render_dual_macd_history_html(dual_macd_history, self.theme)}

        {render_trend_pulse_history_html(trend_pulse_history, self.theme)}

        <div class="indicators-section">
            <h2 class="section-header" onclick="toggleSection('indicators-content')">
                <span class="toggle-icon">\u25bc</span> Indicators
            </h2>
            <div id="indicators-content" class="section-content collapsed">
                {render_indicator_cards(indicator_info)}
            </div>
        </div>
    </div>

    <script>
{get_scripts(chart_data, symbols, timeframes, signal_history, confluence_data, self._colors)}
    </script>
</body>
</html>"""
