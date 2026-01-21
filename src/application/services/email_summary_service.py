"""
Email Summary Service - Generate email summaries from summary.json.

M3 PR-06 Deliverable: Creates HTML and plain text email summaries
for the signal report, reading from summary.json (no re-computation).

Features:
- 4 timeframe columns for multi-TF display
- HTML and plain text output
- No re-computation (reads existing data only)
- Template-based rendering with Jinja2
"""

from __future__ import annotations

import html
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _escape(text: Any) -> str:
    """Escape HTML special characters to prevent XSS."""
    return html.escape(str(text)) if text is not None else ""

# Default template directory
DEFAULT_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "domain" / "signals" / "reporting" / "templates"


class EmailSummaryService:
    """
    Generate email summary from summary.json (no re-computation).

    Reads the pre-computed summary.json file and renders it into
    HTML and plain text email formats suitable for distribution.
    """

    def __init__(self, template_dir: Optional[Path] = None) -> None:
        """
        Initialize email summary service.

        Args:
            template_dir: Directory containing email templates
        """
        self.template_dir = template_dir or DEFAULT_TEMPLATE_DIR

    def render(self, summary_path: Path) -> Tuple[str, str]:
        """
        Render HTML and plain text email from summary.json.

        Args:
            summary_path: Path to summary.json file

        Returns:
            Tuple of (html_content, text_content)

        Raises:
            FileNotFoundError: If summary file doesn't exist
            json.JSONDecodeError: If summary file is malformed
        """
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except FileNotFoundError:
            logger.error(f"Summary file not found: {summary_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in summary file: {summary_path} - {e}")
            raise

        html_content = self._render_html(summary)
        text_content = self._render_text(summary)

        return html_content, text_content

    def render_to_files(
        self,
        summary_path: Path,
        output_dir: Path,
    ) -> Tuple[Path, Path]:
        """
        Render email to files.

        Args:
            summary_path: Path to summary.json
            output_dir: Directory to write output files

        Returns:
            Tuple of (html_path, text_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        html, text = self.render(summary_path)

        html_path = output_dir / "email_summary.html"
        text_path = output_dir / "email_summary.txt"

        html_path.write_text(html, encoding="utf-8")
        text_path.write_text(text, encoding="utf-8")

        return html_path, text_path

    def preview(self, summary_path: Path) -> None:
        """
        Open HTML preview in browser.

        Args:
            summary_path: Path to summary.json
        """
        import tempfile

        html, _ = self.render(summary_path)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html)
            temp_path = f.name

        webbrowser.open(f"file://{temp_path}")
        logger.info(f"Opened email preview: {temp_path}")

    def _render_html(self, summary: Dict[str, Any]) -> str:
        """Render HTML email from summary data."""
        generated_at = summary.get("generated_at", datetime.now().isoformat())
        symbols = summary.get("symbols", [])
        timeframes = summary.get("timeframes", ["1d"])
        tickers = summary.get("tickers", [])
        market = summary.get("market", {})
        confluence = summary.get("confluence", {})

        # Build ticker rows with 4 TF columns
        ticker_rows = self._build_ticker_rows_html(tickers, timeframes, confluence)

        # Build market overview
        market_html = self._build_market_html(market)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Summary Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            color: white;
            padding: 24px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
        }}
        .header .meta {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .content {{
            padding: 24px;
        }}
        .section {{
            margin-bottom: 24px;
        }}
        .section h2 {{
            font-size: 18px;
            margin: 0 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }}
        .market-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }}
        .market-card {{
            background: #f8fafc;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .market-card .symbol {{
            font-weight: 600;
            font-size: 14px;
        }}
        .market-card .regime {{
            font-size: 20px;
            font-weight: 700;
            margin: 4px 0;
        }}
        .regime-r0 {{ color: #22c55e; }}
        .regime-r1 {{ color: #eab308; }}
        .regime-r2 {{ color: #ef4444; }}
        .regime-r3 {{ color: #3b82f6; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            text-align: left;
            padding: 10px 8px;
            background: #f8fafc;
            border-bottom: 2px solid #e2e8f0;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            color: #64748b;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .symbol-name {{
            font-weight: 600;
        }}
        .tf-cell {{
            text-align: center;
        }}
        .bullish {{ color: #22c55e; }}
        .bearish {{ color: #ef4444; }}
        .neutral {{ color: #64748b; }}
        .footer {{
            background: #f8fafc;
            padding: 16px 24px;
            text-align: center;
            font-size: 12px;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Signal Summary Report</h1>
            <div class="meta">
                {len(symbols)} symbols | {', '.join(timeframes)} | Generated: {generated_at[:16].replace('T', ' ')}
            </div>
        </div>

        <div class="content">
            <div class="section">
                <h2>Market Overview</h2>
                {market_html}
            </div>

            <div class="section">
                <h2>Ticker Summary ({len(tickers)} symbols)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Regime</th>
                            {''.join(f'<th class="tf-cell">{tf}</th>' for tf in timeframes[:4])}
                        </tr>
                    </thead>
                    <tbody>
                        {ticker_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            Generated by APEX Signal Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
        </div>
    </div>
</body>
</html>"""

    def _render_text(self, summary: Dict[str, Any]) -> str:
        """Render plain text email from summary data."""
        generated_at = summary.get("generated_at", datetime.now().isoformat())
        symbols = summary.get("symbols", [])
        timeframes = summary.get("timeframes", ["1d"])
        tickers = summary.get("tickers", [])
        market = summary.get("market", {})

        # Build header
        lines = [
            "=" * 60,
            "SIGNAL SUMMARY REPORT",
            "=" * 60,
            f"Symbols: {len(symbols)}",
            f"Timeframes: {', '.join(timeframes)}",
            f"Generated: {generated_at[:16].replace('T', ' ')}",
            "",
            "-" * 60,
            "MARKET OVERVIEW",
            "-" * 60,
        ]

        # Market overview
        benchmarks = market.get("benchmarks", {})
        for symbol, data in benchmarks.items():
            regime = data.get("regime", "N/A")
            confidence = data.get("confidence", 0)
            lines.append(f"  {symbol}: {regime} (confidence: {confidence}%)")

        lines.extend([
            "",
            "-" * 60,
            "TICKER SUMMARY",
            "-" * 60,
        ])

        # Ticker table header
        tf_headers = timeframes[:4]
        header = f"{'Symbol':<8} {'Regime':<6} " + " ".join(f"{tf:>6}" for tf in tf_headers)
        lines.append(header)
        lines.append("-" * len(header))

        # Ticker rows
        confluence = summary.get("confluence", {})
        for ticker in tickers[:30]:  # Limit for email
            symbol = ticker.get("symbol", "")
            regime = ticker.get("regime", "N/A")

            tf_scores = []
            for tf in tf_headers:
                key = f"{symbol}_{tf}"
                conf = confluence.get(key, {})
                score = conf.get("alignment_score", 0)
                if score > 20:
                    tf_scores.append(f"{'+':>6}")
                elif score < -20:
                    tf_scores.append(f"{'-':>6}")
                else:
                    tf_scores.append(f"{'~':>6}")

            row = f"{symbol:<8} {regime:<6} " + " ".join(tf_scores)
            lines.append(row)

        lines.extend([
            "",
            "-" * 60,
            f"Generated by APEX Signal Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        ])

        return "\n".join(lines)

    def _build_ticker_rows_html(
        self,
        tickers: List[Dict[str, Any]],
        timeframes: List[str],
        confluence: Dict[str, Any],
    ) -> str:
        """Build HTML rows for ticker table with 4 TF columns."""
        rows = []
        tf_headers = timeframes[:4]  # Max 4 timeframes

        for ticker in tickers[:50]:  # Limit for email
            symbol = _escape(ticker.get("symbol", ""))
            regime = _escape(ticker.get("regime", "N/A"))
            regime_class = f"regime-{regime.lower()}" if regime else ""

            # Build timeframe cells
            tf_cells = []
            for tf in tf_headers:
                key = f"{ticker.get('symbol', '')}_{tf}"
                conf = confluence.get(key, {})
                score = conf.get("alignment_score", 0)

                if score > 20:
                    cell_class = "bullish"
                    cell_text = "+"
                elif score < -20:
                    cell_class = "bearish"
                    cell_text = "-"
                else:
                    cell_class = "neutral"
                    cell_text = "~"

                tf_cells.append(
                    f'<td class="tf-cell {cell_class}">{cell_text}</td>'
                )

            row = f"""
                <tr>
                    <td class="symbol-name">{symbol}</td>
                    <td class="{regime_class}">{regime}</td>
                    {''.join(tf_cells)}
                </tr>
            """
            rows.append(row)

        return "\n".join(rows)

    def _build_market_html(self, market: Dict[str, Any]) -> str:
        """Build HTML for market overview section."""
        benchmarks = market.get("benchmarks", {})
        if not benchmarks:
            return "<p>No market data available</p>"

        cards = []
        for symbol in ["SPY", "QQQ", "IWM", "DIA"]:
            data = benchmarks.get(symbol, {})
            regime = _escape(data.get("regime", "N/A"))
            confidence = int(data.get("confidence", 0))
            regime_class = f"regime-{regime.lower()}" if regime else ""
            safe_symbol = _escape(symbol)

            card = f"""
                <div class="market-card">
                    <div class="symbol">{safe_symbol}</div>
                    <div class="regime {regime_class}">{regime}</div>
                    <div style="font-size: 12px; color: #64748b;">{confidence}% conf</div>
                </div>
            """
            cards.append(card)

        return f'<div class="market-grid">{"".join(cards)}</div>'


def create_email_preview(summary_path: Path) -> None:
    """
    Create and open email preview.

    Args:
        summary_path: Path to summary.json
    """
    service = EmailSummaryService()
    service.preview(summary_path)
