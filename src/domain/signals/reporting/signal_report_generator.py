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

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

from .description_generator import generate_indicator_description, generate_rule_description

if TYPE_CHECKING:
    from ..indicators.base import Indicator
    from ..models import SignalRule

logger = get_logger(__name__)


# Timeframe ordering for consistent display
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}

# Indicator grouping for chart layout
# Overlays: Same Y-axis as price
OVERLAY_INDICATORS = {"bollinger", "supertrend", "sma", "ema", "vwap", "keltner", "donchian", "ichimoku"}
# Bounded oscillators (0-100 or similar fixed range)
BOUNDED_OSCILLATORS = {"rsi", "stochastic", "kdj", "williams_r", "mfi", "cci", "adx"}
# Unbounded oscillators (MACD-style, centered around 0)
UNBOUNDED_OSCILLATORS = {"macd", "momentum", "roc", "cmf", "pvo", "force_index"}
# Volume indicators
VOLUME_INDICATORS = {"obv", "volume_profile", "vwma", "ease_of_movement", "chaikin_volatility"}


def detect_historical_signals(
    df: pd.DataFrame,
    rules: List["SignalRule"],
    symbol: str,
    timeframe: str,
) -> List[Dict[str, Any]]:
    """
    Detect where signal rules would have triggered in historical data.

    Scans the DataFrame for cross events, threshold breaches, and state changes
    that match the rule conditions.

    Args:
        df: DataFrame with OHLCV and indicator columns
        rules: List of signal rules to check
        symbol: Trading symbol
        timeframe: Timeframe string

    Returns:
        List of detected signal events with timestamps and details
    """
    from ..models import ConditionType

    signals = []
    timestamps = df.index.tolist()

    for rule in rules:
        if not rule.enabled or timeframe not in rule.timeframes:
            continue

        indicator = rule.indicator.lower()
        cond = rule.condition_config

        # Match columns by indicator prefix
        prefix = f"{indicator}_"
        ind_cols = [c for c in df.columns if c.lower().startswith(prefix)]

        if rule.condition_type == ConditionType.CROSS_UP:
            line_a = cond.get("line_a", "")
            line_b = cond.get("line_b", "")
            col_a = next((c for c in ind_cols if line_a in c.lower()), None)
            col_b = next((c for c in ind_cols if line_b in c.lower()), None)

            if col_a and col_b:
                a_vals = df[col_a].values
                b_vals = df[col_b].values
                for i in range(1, len(df)):
                    if pd.notna(a_vals[i]) and pd.notna(b_vals[i]) and pd.notna(a_vals[i-1]) and pd.notna(b_vals[i-1]):
                        if a_vals[i-1] <= b_vals[i-1] and a_vals[i] > b_vals[i]:
                            signals.append({
                                "timestamp": timestamps[i],
                                "rule": rule.name,
                                "direction": rule.direction.value,
                                "indicator": rule.indicator,
                                "message": rule.message_template.format(symbol=symbol),
                                "value": float(a_vals[i]),
                            })

        elif rule.condition_type == ConditionType.CROSS_DOWN:
            line_a = cond.get("line_a", "")
            line_b = cond.get("line_b", "")
            col_a = next((c for c in ind_cols if line_a in c.lower()), None)
            col_b = next((c for c in ind_cols if line_b in c.lower()), None)

            if col_a and col_b:
                a_vals = df[col_a].values
                b_vals = df[col_b].values
                for i in range(1, len(df)):
                    if pd.notna(a_vals[i]) and pd.notna(b_vals[i]) and pd.notna(a_vals[i-1]) and pd.notna(b_vals[i-1]):
                        if a_vals[i-1] >= b_vals[i-1] and a_vals[i] < b_vals[i]:
                            signals.append({
                                "timestamp": timestamps[i],
                                "rule": rule.name,
                                "direction": rule.direction.value,
                                "indicator": rule.indicator,
                                "message": rule.message_template.format(symbol=symbol),
                                "value": float(a_vals[i]),
                            })

        elif rule.condition_type == ConditionType.THRESHOLD_CROSS_UP:
            field = cond.get("field", "value")
            threshold = cond.get("threshold")
            col = next((c for c in ind_cols if field in c.lower()), None)

            if col and threshold is not None:
                vals = df[col].values
                for i in range(1, len(df)):
                    if pd.notna(vals[i]) and pd.notna(vals[i-1]):
                        if vals[i-1] <= threshold < vals[i]:
                            signals.append({
                                "timestamp": timestamps[i],
                                "rule": rule.name,
                                "direction": rule.direction.value,
                                "indicator": rule.indicator,
                                "message": rule.message_template.format(symbol=symbol, value=vals[i], threshold=threshold),
                                "value": float(vals[i]),
                                "threshold": threshold,
                            })

        elif rule.condition_type == ConditionType.THRESHOLD_CROSS_DOWN:
            field = cond.get("field", "value")
            threshold = cond.get("threshold")
            col = next((c for c in ind_cols if field in c.lower()), None)

            if col and threshold is not None:
                vals = df[col].values
                for i in range(1, len(df)):
                    if pd.notna(vals[i]) and pd.notna(vals[i-1]):
                        if vals[i-1] >= threshold > vals[i]:
                            signals.append({
                                "timestamp": timestamps[i],
                                "rule": rule.name,
                                "direction": rule.direction.value,
                                "indicator": rule.indicator,
                                "message": rule.message_template.format(symbol=symbol, value=vals[i], threshold=threshold),
                                "value": float(vals[i]),
                                "threshold": threshold,
                            })

        # MACD signal/zero line cross detection
        if indicator == "macd":
            macd_col = next((c for c in df.columns if "macd_macd" in c.lower()), None)
            signal_col = next((c for c in df.columns if "macd_signal" in c.lower()), None)

            if macd_col and signal_col and "cross" not in rule.name.lower():
                macd_vals = df[macd_col].values
                signal_vals = df[signal_col].values
                for i in range(1, len(df)):
                    if all(pd.notna(v) for v in [macd_vals[i], macd_vals[i-1], signal_vals[i], signal_vals[i-1]]):
                        # Bullish cross
                        if macd_vals[i-1] <= signal_vals[i-1] and macd_vals[i] > signal_vals[i]:
                            if not any(s["timestamp"] == timestamps[i] and "macd" in s["rule"].lower() for s in signals):
                                signals.append({
                                    "timestamp": timestamps[i],
                                    "rule": "macd_bullish_cross",
                                    "direction": "buy",
                                    "indicator": "macd",
                                    "message": f"{symbol} MACD crossed above signal line",
                                    "value": float(macd_vals[i]),
                                })
                        # Bearish cross
                        elif macd_vals[i-1] >= signal_vals[i-1] and macd_vals[i] < signal_vals[i]:
                            if not any(s["timestamp"] == timestamps[i] and "macd" in s["rule"].lower() for s in signals):
                                signals.append({
                                    "timestamp": timestamps[i],
                                    "rule": "macd_bearish_cross",
                                    "direction": "sell",
                                    "indicator": "macd",
                                    "message": f"{symbol} MACD crossed below signal line",
                                    "value": float(macd_vals[i]),
                                })

    # Sort by timestamp
    signals.sort(key=lambda x: x["timestamp"])
    return signals


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
        self._colors = self._get_theme_colors(theme)

    def generate(
        self,
        data: Dict[Tuple[str, str], pd.DataFrame],
        indicators: List["Indicator"],
        rules: List["SignalRule"],
        output_path: Path,
    ) -> Path:
        """
        Generate combined HTML report with symbol selector.

        Args:
            data: Dict mapping (symbol, timeframe) to DataFrame with OHLCV + indicator columns
            indicators: List of computed indicators
            rules: List of signal rules
            output_path: Where to save HTML

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

        # Render HTML
        html = self._render_html(
            symbols=symbols,
            timeframes=timeframes,
            chart_data=chart_data,
            indicator_info=indicator_info,
            signal_history=signal_history,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.info(f"Signal report generated: {output_path}")
        return output_path

    def _get_theme_colors(self, theme: str) -> Dict[str, str]:
        if theme == "dark":
            return {
                "bg": "#0f172a",
                "card_bg": "#1e293b",
                "text": "#e2e8f0",
                "text_muted": "#94a3b8",
                "border": "#334155",
                "profit": "#22c55e",
                "loss": "#ef4444",
                "primary": "#3b82f6",
                "candle_up": "#22c55e",
                "candle_down": "#ef4444",
            }
        return {
            "bg": "#f8fafc",
            "card_bg": "#ffffff",
            "text": "#1e293b",
            "text_muted": "#64748b",
            "border": "#e2e8f0",
            "profit": "#16a34a",
            "loss": "#dc2626",
            "primary": "#2563eb",
            "candle_up": "#16a34a",
            "candle_down": "#dc2626",
        }

    def _build_chart_data(
        self, data: Dict[Tuple[str, str], pd.DataFrame]
    ) -> Dict[str, Any]:
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

                # Parse indicator name from prefixed column (e.g., "macd_histogram" → "macd")
                parts = col.split("_")
                ind_name = parts[0].lower() if parts else col.lower()

                # Route to appropriate subplot bucket
                if ind_name in OVERLAY_INDICATORS:
                    bucket = "overlays"
                elif ind_name == "rsi":
                    bucket = "rsi"
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
        rules_by_indicator: Dict[str, List[Dict[str, str]]] = {}
        for rule in rules:
            rules_by_indicator.setdefault(rule.indicator, []).append({
                "name": rule.name,
                "description": generate_rule_description(rule),
                "direction": rule.direction.value,
                "timeframes": list(rule.timeframes),
            })

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
    ) -> str:
        c = self._colors
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
{self._get_styles()}
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
                    {self._render_symbol_options(symbols)}
                </select>
            </div>
            <div class="control-group">
                <label>Timeframe</label>
                <div class="timeframe-buttons">
                    {self._render_timeframe_buttons(timeframes)}
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div id="main-chart"></div>
        </div>

        <div class="signal-history-section">
            <h2 class="section-header" onclick="toggleSection('signal-history-content')">
                <span class="toggle-icon">▼</span> Signal History
            </h2>
            <div id="signal-history-content" class="section-content">
                <div id="signal-history-table"></div>
            </div>
        </div>

        <div class="indicators-section">
            <h2 class="section-header" onclick="toggleSection('indicators-content')">
                <span class="toggle-icon">▼</span> Indicators
            </h2>
            <div id="indicators-content" class="section-content collapsed">
                {self._render_indicator_cards(indicator_info)}
            </div>
        </div>
    </div>

    <script>
{self._get_scripts(chart_data, symbols, timeframes, signal_history)}
    </script>
</body>
</html>"""

    def _get_styles(self) -> str:
        c = self._colors
        return f"""
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: {c['bg']};
    color: {c['text']};
    line-height: 1.6;
}}

.container {{
    max-width: 1600px;
    margin: 0 auto;
    padding: 20px;
}}

.header {{
    text-align: center;
    padding: 24px;
    margin-bottom: 24px;
    background: linear-gradient(135deg, #1e40af 0%, {c['primary']} 100%);
    border-radius: 12px;
    color: white;
}}

.header h1 {{
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 8px;
}}

.header .meta {{
    display: flex;
    justify-content: center;
    gap: 24px;
    font-size: 14px;
    opacity: 0.9;
}}

.controls {{
    display: flex;
    gap: 24px;
    align-items: end;
    margin-bottom: 24px;
    padding: 16px;
    background: {c['card_bg']};
    border-radius: 12px;
    border: 1px solid {c['border']};
}}

.control-group {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}

.control-group label {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    color: {c['text_muted']};
}}

.control-group select {{
    padding: 10px 16px;
    font-size: 14px;
    border: 1px solid {c['border']};
    border-radius: 8px;
    background: {c['bg']};
    color: {c['text']};
    cursor: pointer;
    min-width: 150px;
}}

.timeframe-buttons {{
    display: flex;
    gap: 4px;
}}

.tf-btn {{
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 500;
    border: 1px solid {c['border']};
    border-radius: 8px;
    background: {c['bg']};
    color: {c['text']};
    cursor: pointer;
    transition: all 0.2s;
}}

.tf-btn:hover {{
    border-color: {c['primary']};
}}

.tf-btn.active {{
    background: {c['primary']};
    border-color: {c['primary']};
    color: white;
}}

.chart-container {{
    background: {c['card_bg']};
    border-radius: 12px;
    border: 1px solid {c['border']};
    padding: 16px;
    margin-bottom: 24px;
}}

#main-chart {{
    height: 900px;
}}

.signal-history-section,
.indicators-section {{
    background: {c['card_bg']};
    border-radius: 12px;
    border: 1px solid {c['border']};
    padding: 24px;
    margin-bottom: 24px;
}}

.section-header {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid {c['border']};
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-header:hover {{
    color: {c['primary']};
}}

.toggle-icon {{
    font-size: 12px;
    transition: transform 0.2s ease;
}}

.section-content.collapsed {{
    display: none;
}}

.section-content.collapsed + .section-header .toggle-icon {{
    transform: rotate(-90deg);
}}

.signal-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

.signal-table th {{
    text-align: left;
    padding: 12px 8px;
    border-bottom: 2px solid {c['border']};
    color: {c['text_muted']};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
}}

.signal-table td {{
    padding: 10px 8px;
    border-bottom: 1px solid {c['border']};
}}

.signal-table tr:hover {{
    background: {c['bg']};
}}

.signal-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}}

.signal-badge.buy {{
    background: rgba(34, 197, 94, 0.2);
    color: {c['profit']};
}}

.signal-badge.sell {{
    background: rgba(239, 68, 68, 0.2);
    color: {c['loss']};
}}

.signal-badge.alert {{
    background: rgba(59, 130, 246, 0.2);
    color: {c['primary']};
}}

.no-signals {{
    text-align: center;
    color: {c['text_muted']};
    padding: 24px;
    font-style: italic;
}}

.category-group {{
    margin-bottom: 24px;
}}

.category-title {{
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: {c['text_muted']};
    margin-bottom: 12px;
    padding: 8px 12px;
    background: {c['bg']};
    border-radius: 6px;
}}

.indicator-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 16px;
}}

.indicator-card {{
    padding: 16px;
    background: {c['bg']};
    border-radius: 8px;
    border: 1px solid {c['border']};
}}

.indicator-card h3 {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
    color: {c['primary']};
}}

.indicator-card .description {{
    font-size: 14px;
    color: {c['text_muted']};
    margin-bottom: 12px;
}}

.indicator-card .rules {{
    font-size: 13px;
}}

.indicator-card .rules h4 {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    color: {c['text_muted']};
    margin-bottom: 8px;
}}

.rule-item {{
    padding: 8px;
    background: {c['card_bg']};
    border-radius: 4px;
    margin-bottom: 4px;
}}

.rule-item .rule-name {{
    font-weight: 500;
}}

.rule-item .rule-desc {{
    font-size: 12px;
    color: {c['text_muted']};
}}

.direction-buy {{ color: {c['profit']}; }}
.direction-sell {{ color: {c['loss']}; }}
.direction-alert {{ color: {c['primary']}; }}

@media (max-width: 768px) {{
    .controls {{
        flex-direction: column;
        align-items: stretch;
    }}
    .timeframe-buttons {{
        flex-wrap: wrap;
    }}
    .indicator-cards {{
        grid-template-columns: 1fr;
    }}
}}
"""

    def _render_symbol_options(self, symbols: List[str]) -> str:
        return "\n".join(
            f'<option value="{s}">{s}</option>' for s in symbols
        )

    def _render_timeframe_buttons(self, timeframes: List[str]) -> str:
        return "\n".join(
            f'<button class="tf-btn{" active" if i == 0 else ""}" data-tf="{tf}" '
            f"onclick=\"selectTimeframe('{tf}', this)\">{tf}</button>"
            for i, tf in enumerate(timeframes)
        )

    def _render_rules(self, rules: List[Dict[str, Any]]) -> str:
        """Render rules section for an indicator card."""
        if not rules:
            return ""
        rule_items = "\n".join(
            f"""<div class="rule-item">
                <span class="rule-name direction-{rule['direction']}">{rule['name']}</span>
                <div class="rule-desc">{rule['description']}</div>
            </div>"""
            for rule in rules
        )
        return f"""<div class="rules"><h4>Rules</h4>{rule_items}</div>"""

    def _render_indicator_cards(self, indicator_info: List[Dict[str, Any]]) -> str:
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for info in indicator_info:
            categories.setdefault(info["category"], []).append(info)

        category_order = ["momentum", "trend", "volatility", "volume", "pattern"]
        category_labels = {
            "momentum": "Momentum Indicators",
            "trend": "Trend Indicators",
            "volatility": "Volatility Indicators",
            "volume": "Volume Indicators",
            "pattern": "Pattern Indicators",
        }

        html_parts = []
        for cat in category_order:
            if cat not in categories:
                continue

            cards_html = []
            for ind in categories[cat]:
                rules_html = self._render_rules(ind["rules"])
                cards_html.append(f"""
                    <div class="indicator-card">
                        <h3>{ind['name'].upper()}</h3>
                        <div class="description">{ind['description']}</div>
                        {rules_html}
                    </div>
                """)

            html_parts.append(f"""
                <div class="category-group">
                    <div class="category-title">{category_labels.get(cat, cat.title())}</div>
                    <div class="indicator-cards">
                        {''.join(cards_html)}
                    </div>
                </div>
            """)

        return "\n".join(html_parts)

    def _get_scripts(
        self,
        chart_data: Dict[str, Any],
        symbols: List[str],
        timeframes: List[str],
        signal_history: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        data_json = json.dumps(chart_data, default=str)
        symbols_json = json.dumps(symbols)
        timeframes_json = json.dumps(timeframes)
        colors_json = json.dumps(self._colors)
        signals_json = json.dumps(signal_history, default=str)

        return f"""
const chartData = {data_json};
const symbols = {symbols_json};
const timeframes = {timeframes_json};
const colors = {colors_json};
const signalHistory = {signals_json};

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
}}

function renderMainChart(data) {{
    // Fixed 4-row layout: Price (55%), RSI (15%), MACD (14%), Volume (10%)
    const traces = [];
    const hasData = (values) => values && !values.every(v => v === null);

    // Row 1: Price candlesticks
    traces.push({{
        type: 'candlestick',
        x: data.timestamps,
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
            x: data.timestamps,
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
            x: data.timestamps,
            y: rsiValues,
            name: 'RSI',
            line: {{ color: '#8b5cf6', width: 1.5 }},
            xaxis: 'x',
            yaxis: 'y2',
        }});
        const boundsX = [data.timestamps[0], data.timestamps[data.timestamps.length - 1]];
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
            x: data.timestamps,
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
            x: data.timestamps,
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
            x: data.timestamps,
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
        const buyTimestamps = buySignals.map(s => s.timestamp);
        const buyPrices = buyTimestamps.map(ts => {{
            const idx = data.timestamps.findIndex(t => t === ts);
            return idx >= 0 ? data.low[idx] * 0.995 : null;
        }}).filter(p => p !== null);
        const validBuyTs = buyTimestamps.filter((ts, i) => {{
            const idx = data.timestamps.findIndex(t => t === ts);
            return idx >= 0;
        }});

        if (validBuyTs.length > 0) {{
            traces.push({{
                type: 'scatter',
                mode: 'markers',
                x: validBuyTs,
                y: buyPrices,
                name: 'Buy Signal',
                marker: {{
                    symbol: 'triangle-up',
                    size: 12,
                    color: colors.candle_up,
                    line: {{ color: 'white', width: 1 }}
                }},
                hovertemplate: '%{{text}}<extra></extra>',
                text: buySignals.filter((s, i) => {{
                    const idx = data.timestamps.findIndex(t => t === s.timestamp);
                    return idx >= 0;
                }}).map(s => s.rule),
                xaxis: 'x',
                yaxis: 'y',
            }});
        }}
    }}

    if (sellSignals.length > 0) {{
        const sellTimestamps = sellSignals.map(s => s.timestamp);
        const sellPrices = sellTimestamps.map(ts => {{
            const idx = data.timestamps.findIndex(t => t === ts);
            return idx >= 0 ? data.high[idx] * 1.005 : null;
        }}).filter(p => p !== null);
        const validSellTs = sellTimestamps.filter((ts, i) => {{
            const idx = data.timestamps.findIndex(t => t === ts);
            return idx >= 0;
        }});

        if (validSellTs.length > 0) {{
            traces.push({{
                type: 'scatter',
                mode: 'markers',
                x: validSellTs,
                y: sellPrices,
                name: 'Sell Signal',
                marker: {{
                    symbol: 'triangle-down',
                    size: 12,
                    color: colors.candle_down,
                    line: {{ color: 'white', width: 1 }}
                }},
                hovertemplate: '%{{text}}<extra></extra>',
                text: sellSignals.filter((s, i) => {{
                    const idx = data.timestamps.findIndex(t => t === s.timestamp);
                    return idx >= 0;
                }}).map(s => s.rule),
                xaxis: 'x',
                yaxis: 'y',
            }});
        }}
    }}

    // Layout with 4 subplots
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

        // Shared X-axis
        xaxis: {{
            gridcolor: colors.border,
            showgrid: true,
            rangeslider: {{ visible: false }},
            tickangle: -45,
            nticks: 15,
            domain: [0, 1],
        }},

        // Y1: Price (top 55%)
        yaxis: {{
            title: 'Price',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.45, 1],
        }},

        // Y2: RSI (15%)
        yaxis2: {{
            title: 'RSI',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.28, 0.43],
            range: [0, 100],
            dtick: 25,
        }},

        // Y3: MACD (15%)
        yaxis3: {{
            title: 'MACD',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0.12, 0.26],
        }},

        // Y4: Volume (10%)
        yaxis4: {{
            title: 'Vol',
            side: 'right',
            gridcolor: colors.border,
            showgrid: true,
            domain: [0, 0.10],
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
    const signals = signalHistory[key] || [];
    const container = document.getElementById('signal-history-table');

    if (signals.length === 0) {{
        container.innerHTML = '<div class="no-signals">No signals detected for this symbol/timeframe</div>';
        return;
    }}

    let html = `
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

    // Show most recent first
    const sortedSignals = [...signals].reverse();
    for (const sig of sortedSignals) {{
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

    html += '</tbody></table>';
    container.innerHTML = html;
}}

function toggleSection(contentId) {{
    const content = document.getElementById(contentId);
    const header = content.previousElementSibling;
    const icon = header.querySelector('.toggle-icon');

    content.classList.toggle('collapsed');
    icon.style.transform = content.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
}}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {{
    updateChart();
    updateSignalHistoryTable();
}});
"""
