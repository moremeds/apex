"""
HTML Report Generator for Backtest Results.

Generates self-contained interactive HTML reports with:
- Tabbed interface (Summary, Per-Asset, Parameters, Validation, Trades, Raw Data)
- Plotly charts for visualizations
- Per-ticker metric breakdown with sub-tabs by category
- No external dependencies (everything inline)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Backtest Report"
    theme: str = "light"  # "light" or "dark"

    # Colors
    profit_color: str = "#22c55e"  # Green
    loss_color: str = "#ef4444"  # Red
    neutral_color: str = "#6b7280"  # Gray
    primary_color: str = "#3b82f6"  # Blue

    # Chart settings
    chart_height: int = 400
    show_watermark: bool = True

    # Data inclusion
    include_trade_log: bool = True
    max_trades_in_report: int = 500
    include_equity_curves: bool = True


@dataclass
class ReportData:
    """Data container for report generation."""

    # Identification
    experiment_id: str = ""
    strategy_name: str = ""
    code_version: str = ""
    data_version: str = ""

    # Time range
    start_date: str = ""
    end_date: str = ""

    # Universe
    symbols: List[str] = field(default_factory=list)

    # Configuration
    n_folds: int = 0
    train_days: int = 0
    test_days: int = 0
    total_trials: int = 0

    # Best trial info
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_trial_score: float = 0.0

    # Aggregate metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Validation metrics
    validation: Dict[str, float] = field(default_factory=dict)

    # Per-symbol metrics
    per_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-window performance
    per_window: List[Dict[str, Any]] = field(default_factory=list)

    # Equity curve data
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    # Trade log
    trades: List[Dict[str, Any]] = field(default_factory=list)

    # Top trials
    top_trials: List[Dict[str, Any]] = field(default_factory=list)

    # Parameter definitions
    param_definitions: Dict[str, Any] = field(default_factory=dict)


class HTMLReportGenerator:
    """
    Generate self-contained HTML backtest reports.

    Example:
        generator = HTMLReportGenerator()
        data = ReportData(
            experiment_id="exp_123",
            strategy_name="ma_cross",
            symbols=["AAPL", "MSFT"],
            metrics={"sharpe": 1.42, "return": 0.243},
            ...
        )
        generator.generate(data, Path("report.html"))
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()

    def generate(self, data: ReportData, output_path: Path) -> Path:
        """
        Generate HTML report from data.

        Args:
            data: ReportData with all metrics and results
            output_path: Where to save the HTML file

        Returns:
            Path to generated HTML file
        """
        html = self._build_html(data)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.info(f"Report generated: {output_path}")
        return output_path

    def _build_html(self, data: ReportData) -> str:
        """Build complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title} - {data.experiment_id}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    {self._get_styles()}
</head>
<body class="theme-{self.config.theme}">
    <div class="container">
        {self._build_header(data)}
        {self._build_tabs()}
        {self._build_summary_tab(data)}
        {self._build_per_asset_tab(data)}
        {self._build_parameters_tab(data)}
        {self._build_validation_tab(data)}
        {self._build_trades_tab(data)}
        {self._build_raw_data_tab(data)}
    </div>
    {self._get_scripts(data)}
</body>
</html>"""

    def _get_styles(self) -> str:
        """Return inline CSS styles."""
        c = self.config
        return f"""<style>
/* Reset and base */
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #1f2937;
    background: #f9fafb;
}}

/* Theme variants */
.theme-dark {{
    color: #f3f4f6;
    background: #111827;
}}
.theme-dark .card {{ background: #1f2937; border-color: #374151; }}
.theme-dark .tab-btn {{ color: #9ca3af; }}
.theme-dark .tab-btn.active {{ color: #f3f4f6; border-color: {c.primary_color}; }}
.theme-dark .data-table th {{ background: #374151; }}
.theme-dark .data-table td {{ border-color: #374151; }}
.theme-dark .metric-card {{ background: #374151; }}

/* Layout */
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{
    background: linear-gradient(135deg, #1e40af 0%, {c.primary_color} 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 24px;
}}
.header h1 {{ font-size: 24px; font-weight: 600; }}
.header .meta {{ opacity: 0.9; margin-top: 8px; font-size: 14px; }}

/* Cards */
.card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}}
.card-title {{
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 12px;
}}

/* Metrics grid */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 16px;
}}
.metric-card {{
    text-align: center;
    padding: 16px;
    background: #f9fafb;
    border-radius: 8px;
}}
.metric-value {{ font-size: 28px; font-weight: 700; }}
.metric-label {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
.metric-status {{ font-size: 16px; margin-top: 4px; }}
.positive {{ color: {c.profit_color}; }}
.negative {{ color: {c.loss_color}; }}
.neutral {{ color: {c.neutral_color}; }}

/* Tabs */
.tabs {{
    display: flex;
    border-bottom: 2px solid #e5e7eb;
    margin-bottom: 24px;
    overflow-x: auto;
}}
.tab-btn {{
    padding: 12px 24px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: #6b7280;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    white-space: nowrap;
}}
.tab-btn:hover {{ color: {c.primary_color}; }}
.tab-btn.active {{ color: {c.primary_color}; border-color: {c.primary_color}; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

/* Sub-tabs (for metrics categories) */
.metric-tabs {{
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    background: #f3f4f6;
    padding: 4px;
    border-radius: 8px;
    width: fit-content;
    flex-wrap: wrap;
}}
.metric-tab-btn {{
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    background: transparent;
    cursor: pointer;
    font-size: 13px;
}}
.metric-tab-btn:hover {{ background: #e5e7eb; }}
.metric-tab-btn.active {{ background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }}
.metric-content {{ display: none; }}
.metric-content.active {{ display: block; }}

/* Asset tabs */
.asset-tabs {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}}
.asset-btn {{
    padding: 8px 16px;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    background: white;
    cursor: pointer;
    font-size: 13px;
}}
.asset-btn:hover {{ border-color: {c.primary_color}; }}
.asset-btn.active {{ background: {c.primary_color}; color: white; border-color: {c.primary_color}; }}

/* Tables */
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}}
.data-table th {{
    text-align: left;
    padding: 12px;
    background: #f9fafb;
    border-bottom: 2px solid #e5e7eb;
    font-weight: 600;
}}
.data-table td {{
    padding: 12px;
    border-bottom: 1px solid #e5e7eb;
}}
.data-table tr:hover {{ background: #f9fafb; }}

/* Status badges */
.badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}}
.badge-pass {{ background: #dcfce7; color: #166534; }}
.badge-fail {{ background: #fee2e2; color: #991b1b; }}
.badge-warn {{ background: #fef3c7; color: #92400e; }}

/* Export buttons */
.export-btn {{
    padding: 12px 20px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    background: white;
    cursor: pointer;
    font-size: 14px;
    margin-right: 8px;
    margin-bottom: 8px;
}}
.export-btn:hover {{ background: #f9fafb; }}

/* Responsive */
@media (max-width: 768px) {{
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .tab-btn {{ padding: 10px 16px; }}
}}
</style>"""

    def _build_header(self, data: ReportData) -> str:
        """Build report header."""
        return f"""
        <div class="header">
            <h1>{self.config.title}</h1>
            <div class="meta">
                <strong>Strategy:</strong> {data.strategy_name} |
                <strong>Universe:</strong> {len(data.symbols)} symbols |
                <strong>Period:</strong> {data.start_date} to {data.end_date} |
                <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
"""

    def _build_tabs(self) -> str:
        """Build main tab navigation."""
        return """
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('summary')">Summary</button>
            <button class="tab-btn" onclick="showTab('per-asset')">Per-Asset</button>
            <button class="tab-btn" onclick="showTab('parameters')">Parameters</button>
            <button class="tab-btn" onclick="showTab('validation')">Validation</button>
            <button class="tab-btn" onclick="showTab('trades')">Trades</button>
            <button class="tab-btn" onclick="showTab('raw-data')">Raw Data</button>
        </div>
"""

    def _build_summary_tab(self, data: ReportData) -> str:
        """Build summary tab content."""
        m = data.metrics
        v = data.validation

        # Status indicators
        sharpe_class = "positive" if m.get("sharpe", 0) > 0 else "negative"
        return_class = "positive" if m.get("total_return", 0) > 0 else "negative"
        pbo_pass = v.get("pbo", 1) < 0.3
        dsr_pass = v.get("dsr", 0) > 0.95

        # Build symbol rows
        symbol_rows = ""
        for sym, sm in data.per_symbol.items():
            sharpe = sm.get("sharpe", 0)
            ret = sm.get("total_return", 0)
            status = "pass" if sharpe > 0.5 else "warn" if sharpe > 0 else "fail"
            status_label = "Pass" if status == "pass" else "Marginal" if status == "warn" else "Fail"
            symbol_rows += f"""
                <tr>
                    <td><strong>{sym}</strong></td>
                    <td>{sharpe:.2f}</td>
                    <td class="{'positive' if ret > 0 else 'negative'}">{ret*100:.1f}%</td>
                    <td>{sm.get('max_drawdown', 0)*100:.1f}%</td>
                    <td>{sm.get('total_trades', 0)}</td>
                    <td>{sm.get('win_rate', 0)*100:.0f}%</td>
                    <td><span class="badge badge-{status}">{status_label}</span></td>
                </tr>
"""

        return f"""
        <div id="summary" class="tab-content active">
            <div class="card">
                <div class="card-title">Key Metrics (Best Parameters)</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value {sharpe_class}">{m.get('sharpe', 0):.2f}</div>
                        <div class="metric-label">Median Sharpe</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {return_class}">{m.get('total_return', 0)*100:.1f}%</div>
                        <div class="metric-label">Median Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{m.get('max_drawdown', 0)*100:.1f}%</div>
                        <div class="metric-label">Median Max DD</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{m.get('win_rate', 0)*100:.0f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{v.get('pbo', 0)*100:.0f}%</div>
                        <div class="metric-label">PBO</div>
                        <div class="metric-status {'positive' if pbo_pass else 'negative'}">{'Pass' if pbo_pass else 'Fail'}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{v.get('dsr', 0):.2f}</div>
                        <div class="metric-label">DSR</div>
                        <div class="metric-status {'positive' if dsr_pass else 'negative'}">{'Pass' if dsr_pass else 'Fail'}</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Portfolio Equity Curve</div>
                <div id="equity-chart" style="height: {self.config.chart_height}px;"></div>
            </div>

            <div class="card">
                <div class="card-title">Universe Performance</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Sharpe</th>
                            <th>Return</th>
                            <th>Max DD</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {symbol_rows}
                    </tbody>
                </table>
            </div>
        </div>
"""

    def _build_per_asset_tab(self, data: ReportData) -> str:
        """Build per-asset deep dive tab."""
        # Asset buttons
        asset_buttons = ""
        for i, sym in enumerate(data.symbols):
            active = " active" if i == 0 else ""
            asset_buttons += f'<button class="asset-btn{active}" onclick="selectAsset(\'{sym}\')">{sym}</button>'

        return f"""
        <div id="per-asset" class="tab-content">
            <div class="card">
                <div class="card-title">Select Asset</div>
                <div class="asset-tabs">
                    {asset_buttons}
                </div>
            </div>

            <div id="asset-detail">
                <div class="card">
                    <div class="card-title">Metrics</div>
                    <div class="metric-tabs">
                        <button class="metric-tab-btn active" onclick="showMetricTab('returns')">Returns</button>
                        <button class="metric-tab-btn" onclick="showMetricTab('risk')">Risk</button>
                        <button class="metric-tab-btn" onclick="showMetricTab('tail-risk')">Tail Risk</button>
                        <button class="metric-tab-btn" onclick="showMetricTab('stability')">Stability</button>
                        <button class="metric-tab-btn" onclick="showMetricTab('statistical')">Statistical</button>
                        <button class="metric-tab-btn" onclick="showMetricTab('trading')">Trading</button>
                    </div>

                    <div id="metric-returns" class="metric-content active">
                        <div class="metrics-grid" id="returns-metrics"></div>
                    </div>
                    <div id="metric-risk" class="metric-content">
                        <div class="metrics-grid" id="risk-metrics"></div>
                    </div>
                    <div id="metric-tail-risk" class="metric-content">
                        <div class="metrics-grid" id="tail-risk-metrics"></div>
                    </div>
                    <div id="metric-stability" class="metric-content">
                        <div class="metrics-grid" id="stability-metrics"></div>
                    </div>
                    <div id="metric-statistical" class="metric-content">
                        <div class="metrics-grid" id="statistical-metrics"></div>
                    </div>
                    <div id="metric-trading" class="metric-content">
                        <div class="metrics-grid" id="trading-metrics"></div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">Per-Window Performance</div>
                    <table class="data-table" id="window-table">
                        <thead>
                            <tr>
                                <th>Window</th>
                                <th>Train Period</th>
                                <th>Test Period</th>
                                <th>IS Sharpe</th>
                                <th>OOS Sharpe</th>
                                <th>Degradation</th>
                            </tr>
                        </thead>
                        <tbody id="window-table-body"></tbody>
                    </table>
                </div>

                <div class="card">
                    <div class="card-title">Asset Equity Curve</div>
                    <div id="asset-equity-chart" style="height: {self.config.chart_height}px;"></div>
                </div>
            </div>
        </div>
"""

    def _build_parameters_tab(self, data: ReportData) -> str:
        """Build parameters optimization tab."""
        params_display = " | ".join([f"{k}: {v}" for k, v in data.best_params.items()])

        # Top trials table
        trial_rows = ""
        for i, trial in enumerate(data.top_trials[:10], 1):
            params = trial.get("params", {})
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            trial_rows += f"""
                <tr>
                    <td>{i}</td>
                    <td>{params_str}</td>
                    <td>{trial.get('sharpe', 0):.2f}</td>
                    <td>{trial.get('return', 0)*100:.1f}%</td>
                    <td>{trial.get('max_dd', 0)*100:.1f}%</td>
                    <td>{trial.get('stability', 0):.2f}</td>
                    <td>{trial.get('score', 0):.3f}</td>
                </tr>
"""

        return f"""
        <div id="parameters" class="tab-content">
            <div class="card">
                <div class="card-title">Best Parameters</div>
                <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">
                    {params_display}
                </div>
                <div style="color: #6b7280;">
                    Trial Score: {data.best_trial_score:.4f} | Rank: 1 of {data.total_trials}
                </div>
            </div>

            <div class="card">
                <div class="card-title">Parameter Sensitivity</div>
                <div id="param-heatmap" style="height: 350px;"></div>
            </div>

            <div class="card">
                <div class="card-title">Top 10 Parameter Combinations</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Parameters</th>
                            <th>Sharpe</th>
                            <th>Return</th>
                            <th>Max DD</th>
                            <th>Stability</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trial_rows}
                    </tbody>
                </table>
            </div>
        </div>
"""

    def _build_validation_tab(self, data: ReportData) -> str:
        """Build statistical validation tab."""
        v = data.validation
        pbo = v.get("pbo", 0)
        dsr = v.get("dsr", 0)
        degradation = v.get("degradation", 0)

        pbo_status = "pass" if pbo < 0.3 else "fail"
        dsr_status = "pass" if dsr > 0.95 else "fail"
        deg_status = "pass" if abs(degradation) < 0.4 else "fail"

        return f"""
        <div id="validation" class="tab-content">
            <div class="card">
                <div class="card-title">Overfitting Diagnostics</div>

                <div style="margin-bottom: 24px;">
                    <h3 style="margin-bottom: 12px;">PBO Analysis</h3>
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div>
                            <span style="font-size: 24px; font-weight: 700;">{pbo*100:.0f}%</span>
                            <span class="badge badge-{pbo_status}">{'PASS' if pbo_status == 'pass' else 'FAIL'}</span>
                        </div>
                        <div style="color: #6b7280;">
                            Probability of Backtest Overfit (threshold: &lt;30%)
                        </div>
                    </div>
                </div>

                <div style="margin-bottom: 24px;">
                    <h3 style="margin-bottom: 12px;">DSR Analysis</h3>
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div>
                            <span style="font-size: 24px; font-weight: 700;">{dsr:.2f}</span>
                            <span class="badge badge-{dsr_status}">{'PASS' if dsr_status == 'pass' else 'FAIL'}</span>
                        </div>
                        <div style="color: #6b7280;">
                            Deflated Sharpe Ratio (threshold: &gt;0.95)
                        </div>
                    </div>
                </div>

                <div>
                    <h3 style="margin-bottom: 12px;">IS/OOS Degradation</h3>
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div>
                            <span style="font-size: 24px; font-weight: 700;">{degradation*100:.1f}%</span>
                            <span class="badge badge-{deg_status}">{'ACCEPTABLE' if deg_status == 'pass' else 'HIGH'}</span>
                        </div>
                        <div style="color: #6b7280;">
                            Performance drop from In-Sample to Out-of-Sample (threshold: &lt;40%)
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Monte Carlo Confidence Bands</div>
                <div id="monte-carlo-chart" style="height: 350px;"></div>
                <div style="margin-top: 12px; display: flex; gap: 24px; color: #6b7280; font-size: 14px;">
                    <div>P5 Final: ${v.get('mc_p5', 0):,.0f}</div>
                    <div>P50 Final: ${v.get('mc_p50', 0):,.0f}</div>
                    <div>P95 Final: ${v.get('mc_p95', 0):,.0f}</div>
                </div>
            </div>
        </div>
"""

    def _build_trades_tab(self, data: ReportData) -> str:
        """Build trades analysis tab."""
        m = data.metrics

        # Trade rows (limited)
        trade_rows = ""
        for i, t in enumerate(data.trades[:100], 1):
            ret = t.get("return_pct", 0)
            status = "Win" if ret > 0 else "Loss"
            status_class = "positive" if ret > 0 else "negative"
            trade_rows += f"""
                <tr>
                    <td>{i}</td>
                    <td>{t.get('symbol', '')}</td>
                    <td>{t.get('entry_date', '')}</td>
                    <td>{t.get('exit_date', '')}</td>
                    <td>{t.get('side', '')}</td>
                    <td class="{status_class}">{ret*100:.2f}%</td>
                    <td>{t.get('bars', 0)}</td>
                    <td>{status}</td>
                </tr>
"""

        return f"""
        <div id="trades" class="tab-content">
            <div class="card">
                <div class="card-title">Trade Summary</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{m.get('total_trades', 0)}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{m.get('win_rate', 0)*100:.0f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{m.get('profit_factor', 0):.2f}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{m.get('expectancy', 0)*100:.2f}%</div>
                        <div class="metric-label">Expectancy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{m.get('max_consecutive_wins', 0)}</div>
                        <div class="metric-label">Max Consec. Wins</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{m.get('max_consecutive_losses', 0)}</div>
                        <div class="metric-label">Max Consec. Losses</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Trade Distributions</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div id="trade-return-hist" style="height: 250px;"></div>
                    <div id="trade-duration-hist" style="height: 250px;"></div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Trade Log (showing first 100)</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Symbol</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>Side</th>
                            <th>Return</th>
                            <th>Bars</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows}
                    </tbody>
                </table>
            </div>
        </div>
"""

    def _build_raw_data_tab(self, data: ReportData) -> str:
        """Build raw data export tab."""
        params_lines = ""
        for k, v in data.param_definitions.items():
            params_lines += f"  {k}: {v}<br>"

        return f"""
        <div id="raw-data" class="tab-content">
            <div class="card">
                <div class="card-title">Quick Export</div>
                <div>
                    <button class="export-btn" onclick="exportCSV('summary')">Export Summary CSV</button>
                    <button class="export-btn" onclick="exportCSV('equity')">Export Equity CSV</button>
                    <button class="export-btn" onclick="exportCSV('trades')">Export Trades CSV</button>
                    <button class="export-btn" onclick="copyDuckDBQuery()">Copy DuckDB Query</button>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Run Configuration</div>
                <div style="font-family: monospace; font-size: 13px; line-height: 1.8;">
                    <div><strong>Experiment ID:</strong> {data.experiment_id}</div>
                    <div><strong>Strategy:</strong> {data.strategy_name}</div>
                    <div><strong>Code Version:</strong> {data.code_version}</div>
                    <div><strong>Data Version:</strong> {data.data_version}</div>
                    <div style="margin-top: 12px;"><strong>Parameters:</strong></div>
                    <div style="padding-left: 16px;">{params_lines}</div>
                    <div style="margin-top: 12px;"><strong>Universe:</strong> {', '.join(data.symbols)}</div>
                    <div><strong>Date Range:</strong> {data.start_date} to {data.end_date}</div>
                    <div><strong>Walk-Forward:</strong> {data.n_folds} folds, {data.train_days} train / {data.test_days} test days</div>
                </div>
            </div>
        </div>
"""

    def _get_scripts(self, data: ReportData) -> str:
        """Return JavaScript for interactivity."""
        # Serialize data for JavaScript
        data_json = json.dumps({
            "symbols": data.symbols,
            "per_symbol": data.per_symbol,
            "equity_curve": data.equity_curve,
            "trades": data.trades,
            "per_window": data.per_window,
            "experiment_id": data.experiment_id,
        }, default=str)

        return f"""
<script>
const reportData = {data_json};

// Tab switching
function showTab(tabId) {{
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.target.classList.add('active');
}}

// Metric category tab switching
function showMetricTab(tabId) {{
    document.querySelectorAll('.metric-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.metric-tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('metric-' + tabId).classList.add('active');
    event.target.classList.add('active');
}}

// Asset selection
let currentAsset = reportData.symbols[0] || '';

function selectAsset(symbol) {{
    currentAsset = symbol;
    document.querySelectorAll('.asset-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    updateAssetMetrics(symbol);
}}

function updateAssetMetrics(symbol) {{
    const data = reportData.per_symbol[symbol] || {{}};

    // Returns metrics
    document.getElementById('returns-metrics').innerHTML = buildMetricCards([
        {{ label: 'Total Return', value: ((data.total_return || 0) * 100).toFixed(1) + '%' }},
        {{ label: 'CAGR', value: ((data.cagr || 0) * 100).toFixed(1) + '%' }},
        {{ label: 'Best Month', value: ((data.best_month_return || 0) * 100).toFixed(1) + '%' }},
        {{ label: 'Worst Month', value: ((data.worst_month_return || 0) * 100).toFixed(1) + '%' }},
    ]);

    // Risk metrics
    document.getElementById('risk-metrics').innerHTML = buildMetricCards([
        {{ label: 'Sharpe', value: (data.sharpe || 0).toFixed(2) }},
        {{ label: 'Sortino', value: (data.sortino || 0).toFixed(2) }},
        {{ label: 'Calmar', value: (data.calmar || 0).toFixed(2) }},
        {{ label: 'Max Drawdown', value: ((data.max_drawdown || 0) * 100).toFixed(1) + '%' }},
    ]);

    // Tail risk metrics
    document.getElementById('tail-risk-metrics').innerHTML = buildMetricCards([
        {{ label: 'VaR 95%', value: ((data.var_95 || 0) * 100).toFixed(2) + '%' }},
        {{ label: 'CVaR 95%', value: ((data.cvar_95 || 0) * 100).toFixed(2) + '%' }},
        {{ label: 'Skewness', value: (data.skewness || 0).toFixed(2) }},
        {{ label: 'Kurtosis', value: (data.kurtosis || 0).toFixed(2) }},
    ]);

    // Stability metrics
    document.getElementById('stability-metrics').innerHTML = buildMetricCards([
        {{ label: 'Ulcer Index', value: (data.ulcer_index || 0).toFixed(4) }},
        {{ label: 'Pain Index', value: (data.pain_index || 0).toFixed(4) }},
        {{ label: 'Recovery Factor', value: (data.recovery_factor || 0).toFixed(2) }},
        {{ label: 'Serenity Index', value: (data.serenity_index || 0).toFixed(2) }},
    ]);

    // Statistical metrics
    document.getElementById('statistical-metrics').innerHTML = buildMetricCards([
        {{ label: 't-Statistic', value: (data.returns_tstat || 0).toFixed(2) }},
        {{ label: 'p-Value', value: (data.returns_pvalue || 1).toFixed(4) }},
        {{ label: 'Jarque-Bera', value: (data.jarque_bera_stat || 0).toFixed(2) }},
        {{ label: 'Autocorr(1)', value: (data.autocorr_lag1 || 0).toFixed(3) }},
    ]);

    // Trading metrics
    document.getElementById('trading-metrics').innerHTML = buildMetricCards([
        {{ label: 'Total Trades', value: data.total_trades || 0 }},
        {{ label: 'Win Rate', value: ((data.win_rate || 0) * 100).toFixed(0) + '%' }},
        {{ label: 'Profit Factor', value: (data.profit_factor || 0).toFixed(2) }},
        {{ label: 'Edge Ratio', value: (data.edge_ratio || 0).toFixed(2) }},
    ]);
}}

function buildMetricCards(metrics) {{
    return metrics.map(m => `
        <div class="metric-card">
            <div class="metric-value">${{m.value}}</div>
            <div class="metric-label">${{m.label}}</div>
        </div>
    `).join('');
}}

// Initialize charts
document.addEventListener('DOMContentLoaded', () => {{
    initializeCharts();
    if (currentAsset) updateAssetMetrics(currentAsset);
}});

function initializeCharts() {{
    // Main equity curve
    if (reportData.equity_curve && reportData.equity_curve.length > 0) {{
        Plotly.newPlot('equity-chart', [{{
            x: reportData.equity_curve.map(d => d.date),
            y: reportData.equity_curve.map(d => d.equity),
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#3b82f6', width: 2 }},
            name: 'Portfolio'
        }}], {{
            margin: {{ t: 20, r: 20, b: 40, l: 60 }},
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Equity ($)' }},
            hovermode: 'x unified'
        }}, {{ responsive: true }});
    }}

    // Trade return histogram
    if (reportData.trades && reportData.trades.length > 0) {{
        const returns = reportData.trades.map(t => (t.return_pct || 0) * 100);
        Plotly.newPlot('trade-return-hist', [{{
            x: returns,
            type: 'histogram',
            marker: {{ color: '#3b82f6' }},
            name: 'Trade Returns'
        }}], {{
            margin: {{ t: 20, r: 20, b: 40, l: 40 }},
            xaxis: {{ title: 'Return (%)' }},
            yaxis: {{ title: 'Count' }}
        }}, {{ responsive: true }});

        const durations = reportData.trades.map(t => t.bars || 0);
        Plotly.newPlot('trade-duration-hist', [{{
            x: durations,
            type: 'histogram',
            marker: {{ color: '#22c55e' }},
            name: 'Holding Period'
        }}], {{
            margin: {{ t: 20, r: 20, b: 40, l: 40 }},
            xaxis: {{ title: 'Bars Held' }},
            yaxis: {{ title: 'Count' }}
        }}, {{ responsive: true }});
    }}
}}

// Export functions
function exportCSV(type) {{
    let csv = '';
    let filename = '';

    if (type === 'summary') {{
        csv = 'symbol,sharpe,return,max_dd,trades,win_rate\\n';
        for (const [sym, m] of Object.entries(reportData.per_symbol)) {{
            csv += `${{sym}},${{m.sharpe || 0}},${{m.total_return || 0}},${{m.max_drawdown || 0}},${{m.total_trades || 0}},${{m.win_rate || 0}}\\n`;
        }}
        filename = 'summary.csv';
    }} else if (type === 'equity') {{
        csv = 'date,equity\\n';
        reportData.equity_curve.forEach(d => {{
            csv += `${{d.date}},${{d.equity}}\\n`;
        }});
        filename = 'equity.csv';
    }} else if (type === 'trades') {{
        csv = 'symbol,entry,exit,side,return,bars\\n';
        reportData.trades.forEach(t => {{
            csv += `${{t.symbol}},${{t.entry_date}},${{t.exit_date}},${{t.side}},${{t.return_pct}},${{t.bars}}\\n`;
        }});
        filename = 'trades.csv';
    }}

    const blob = new Blob([csv], {{ type: 'text/csv' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
}}

function copyDuckDBQuery() {{
    const query = `SELECT * FROM trials WHERE experiment_id = '${{reportData.experiment_id}}' ORDER BY trial_score DESC LIMIT 10;`;
    navigator.clipboard.writeText(query);
    alert('DuckDB query copied to clipboard!');
}}
</script>
"""
