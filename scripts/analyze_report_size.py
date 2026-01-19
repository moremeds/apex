#!/usr/bin/env python3
"""
Report Size Analyzer - Measure current signal report size breakdown.

PR-00 Deliverable: Establishes size baseline for Signal Service v2.

Usage:
    # Analyze existing report
    python scripts/analyze_report_size.py --report results/signals/signal_report.html

    # Generate fresh report and analyze (requires IB connection or cached data)
    python scripts/analyze_report_size.py --generate --symbols AAPL SPY QQQ

    # Analyze with detailed section breakdown
    python scripts/analyze_report_size.py --report results/signals/signal_report.html --detailed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SectionSize:
    """Size metrics for a report section."""

    name: str
    bytes: int
    lines: int
    pct_of_total: float = 0.0

    @property
    def kb(self) -> float:
        return self.bytes / 1024

    @property
    def mb(self) -> float:
        return self.bytes / (1024 * 1024)


@dataclass
class ReportSizeAnalysis:
    """Complete size analysis of a signal report."""

    file_path: Path
    total_bytes: int
    total_lines: int

    # Major sections
    html_structure: SectionSize
    css_styles: SectionSize
    javascript: SectionSize
    chart_data_json: SectionSize
    signal_history_json: SectionSize
    confluence_json: SectionSize
    regime_sections: SectionSize
    indicator_cards: SectionSize

    # JSON payload details (extracted from JavaScript)
    chart_data_details: Dict[str, int]  # symbol_tf -> bytes
    symbols_count: int
    timeframes_count: int

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON export."""
        return {
            "file_path": str(self.file_path),
            "total_bytes": self.total_bytes,
            "total_kb": round(self.total_bytes / 1024, 2),
            "total_mb": round(self.total_bytes / (1024 * 1024), 3),
            "total_lines": self.total_lines,
            "symbols_count": self.symbols_count,
            "timeframes_count": self.timeframes_count,
            "sections": {
                "html_structure": {
                    "bytes": self.html_structure.bytes,
                    "kb": round(self.html_structure.kb, 2),
                    "pct": round(self.html_structure.pct_of_total, 1),
                },
                "css_styles": {
                    "bytes": self.css_styles.bytes,
                    "kb": round(self.css_styles.kb, 2),
                    "pct": round(self.css_styles.pct_of_total, 1),
                },
                "javascript": {
                    "bytes": self.javascript.bytes,
                    "kb": round(self.javascript.kb, 2),
                    "pct": round(self.javascript.pct_of_total, 1),
                },
                "chart_data_json": {
                    "bytes": self.chart_data_json.bytes,
                    "kb": round(self.chart_data_json.kb, 2),
                    "pct": round(self.chart_data_json.pct_of_total, 1),
                },
                "signal_history_json": {
                    "bytes": self.signal_history_json.bytes,
                    "kb": round(self.signal_history_json.kb, 2),
                    "pct": round(self.signal_history_json.pct_of_total, 1),
                },
                "confluence_json": {
                    "bytes": self.confluence_json.bytes,
                    "kb": round(self.confluence_json.kb, 2),
                    "pct": round(self.confluence_json.pct_of_total, 1),
                },
                "regime_sections": {
                    "bytes": self.regime_sections.bytes,
                    "kb": round(self.regime_sections.kb, 2),
                    "pct": round(self.regime_sections.pct_of_total, 1),
                },
                "indicator_cards": {
                    "bytes": self.indicator_cards.bytes,
                    "kb": round(self.indicator_cards.kb, 2),
                    "pct": round(self.indicator_cards.pct_of_total, 1),
                },
            },
            "chart_data_per_symbol": {
                k: {"bytes": v, "kb": round(v / 1024, 2)}
                for k, v in self.chart_data_details.items()
            },
        }

    def print_summary(self, detailed: bool = False) -> None:
        """Print human-readable summary."""
        print("=" * 70)
        print("SIGNAL REPORT SIZE ANALYSIS")
        print("=" * 70)
        print(f"File: {self.file_path}")
        print(f"Total Size: {self.total_bytes:,} bytes ({self.total_bytes / 1024:.1f} KB)")
        print(f"Total Lines: {self.total_lines:,}")
        print(f"Symbols: {self.symbols_count}")
        print(f"Timeframes: {self.timeframes_count}")
        print()

        print("-" * 70)
        print("SECTION BREAKDOWN")
        print("-" * 70)
        print(f"{'Section':<25} {'Size (KB)':<12} {'% of Total':<12} {'Lines':<10}")
        print("-" * 70)

        sections = [
            self.html_structure,
            self.css_styles,
            self.javascript,
            self.chart_data_json,
            self.signal_history_json,
            self.confluence_json,
            self.regime_sections,
            self.indicator_cards,
        ]

        for section in sorted(sections, key=lambda s: s.bytes, reverse=True):
            print(
                f"{section.name:<25} {section.kb:>10.1f}  {section.pct_of_total:>10.1f}%  {section.lines:>8}"
            )

        print("-" * 70)

        if detailed and self.chart_data_details:
            print()
            print("-" * 70)
            print("CHART DATA PER SYMBOL/TIMEFRAME")
            print("-" * 70)
            print(f"{'Symbol_TF':<25} {'Size (KB)':<12}")
            print("-" * 70)

            for key, size in sorted(
                self.chart_data_details.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"{key:<25} {size / 1024:>10.1f}")

            print("-" * 70)

        # Size budget comparison
        print()
        print("-" * 70)
        print("SIZE BUDGET COMPARISON (PR-03 Target: 200KB for summary.json)")
        print("-" * 70)
        json_total = (
            self.chart_data_json.bytes
            + self.signal_history_json.bytes
            + self.confluence_json.bytes
        )
        print(f"Total JSON payload: {json_total / 1024:.1f} KB")
        print(f"Target summary.json: 200 KB")
        print(
            f"Status: {'UNDER BUDGET' if json_total < 200 * 1024 else 'OVER BUDGET - needs optimization'}"
        )
        print()


def extract_section_sizes(html_content: str) -> Dict[str, SectionSize]:
    """Extract size of major sections from HTML content."""
    total_bytes = len(html_content.encode("utf-8"))
    total_lines = html_content.count("\n") + 1

    sections = {}

    # Extract <style> section
    style_match = re.search(r"<style[^>]*>(.*?)</style>", html_content, re.DOTALL)
    if style_match:
        style_content = style_match.group(1)
        sections["css_styles"] = SectionSize(
            name="CSS Styles",
            bytes=len(style_content.encode("utf-8")),
            lines=style_content.count("\n") + 1,
        )
    else:
        sections["css_styles"] = SectionSize(name="CSS Styles", bytes=0, lines=0)

    # Extract <script> section
    script_match = re.search(r"<script[^>]*>(.*?)</script>", html_content, re.DOTALL)
    if script_match:
        script_content = script_match.group(1)
        sections["javascript"] = SectionSize(
            name="JavaScript",
            bytes=len(script_content.encode("utf-8")),
            lines=script_content.count("\n") + 1,
        )

        # Extract JSON data blocks from JavaScript
        chart_data_match = re.search(r"const chartData = ({.*?});", script_content, re.DOTALL)
        if chart_data_match:
            chart_json = chart_data_match.group(1)
            sections["chart_data_json"] = SectionSize(
                name="Chart Data JSON",
                bytes=len(chart_json.encode("utf-8")),
                lines=chart_json.count("\n") + 1,
            )
        else:
            sections["chart_data_json"] = SectionSize(name="Chart Data JSON", bytes=0, lines=0)

        signal_history_match = re.search(
            r"const signalHistory = ({.*?});", script_content, re.DOTALL
        )
        if signal_history_match:
            signal_json = signal_history_match.group(1)
            sections["signal_history_json"] = SectionSize(
                name="Signal History JSON",
                bytes=len(signal_json.encode("utf-8")),
                lines=signal_json.count("\n") + 1,
            )
        else:
            sections["signal_history_json"] = SectionSize(
                name="Signal History JSON", bytes=0, lines=0
            )

        confluence_match = re.search(
            r"const confluenceData = ({.*?});", script_content, re.DOTALL
        )
        if confluence_match:
            confluence_json = confluence_match.group(1)
            sections["confluence_json"] = SectionSize(
                name="Confluence JSON",
                bytes=len(confluence_json.encode("utf-8")),
                lines=confluence_json.count("\n") + 1,
            )
        else:
            sections["confluence_json"] = SectionSize(name="Confluence JSON", bytes=0, lines=0)
    else:
        sections["javascript"] = SectionSize(name="JavaScript", bytes=0, lines=0)
        sections["chart_data_json"] = SectionSize(name="Chart Data JSON", bytes=0, lines=0)
        sections["signal_history_json"] = SectionSize(name="Signal History JSON", bytes=0, lines=0)
        sections["confluence_json"] = SectionSize(name="Confluence JSON", bytes=0, lines=0)

    # Extract regime analysis sections
    regime_match = re.search(
        r'<div class="regime-analysis-section">(.*?)</div>\s*<div class="signal-history-section">',
        html_content,
        re.DOTALL,
    )
    if regime_match:
        regime_content = regime_match.group(1)
        sections["regime_sections"] = SectionSize(
            name="Regime Analysis",
            bytes=len(regime_content.encode("utf-8")),
            lines=regime_content.count("\n") + 1,
        )
    else:
        sections["regime_sections"] = SectionSize(name="Regime Analysis", bytes=0, lines=0)

    # Extract indicator cards section
    indicators_match = re.search(
        r'<div class="indicators-section">(.*?)</div>\s*</div>\s*<script>',
        html_content,
        re.DOTALL,
    )
    if indicators_match:
        indicators_content = indicators_match.group(1)
        sections["indicator_cards"] = SectionSize(
            name="Indicator Cards",
            bytes=len(indicators_content.encode("utf-8")),
            lines=indicators_content.count("\n") + 1,
        )
    else:
        sections["indicator_cards"] = SectionSize(name="Indicator Cards", bytes=0, lines=0)

    # Calculate HTML structure (everything else)
    accounted_bytes = sum(s.bytes for s in sections.values())
    html_bytes = total_bytes - accounted_bytes
    html_lines = total_lines - sum(s.lines for s in sections.values())
    sections["html_structure"] = SectionSize(
        name="HTML Structure",
        bytes=max(0, html_bytes),
        lines=max(0, html_lines),
    )

    # Calculate percentages
    for section in sections.values():
        section.pct_of_total = (section.bytes / total_bytes * 100) if total_bytes > 0 else 0

    return sections


def extract_chart_data_details(html_content: str) -> Dict[str, int]:
    """Extract per-symbol chart data sizes from the chartData JSON."""
    details = {}

    script_match = re.search(r"const chartData = ({.*?});", html_content, re.DOTALL)
    if not script_match:
        return details

    try:
        # Try to parse the JSON (may fail if it contains JS-specific syntax)
        chart_json_str = script_match.group(1)
        # Handle JavaScript object syntax (unquoted keys)
        # This is a simplified approach - just look for symbol_tf patterns
        symbol_tf_pattern = re.compile(r'"([A-Z]+_\d+[mhdw])":\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}')
        for match in symbol_tf_pattern.finditer(chart_json_str):
            key = match.group(1)
            value_str = match.group(0)
            details[key] = len(value_str.encode("utf-8"))
    except Exception:
        pass

    return details


def count_symbols_timeframes(html_content: str) -> tuple[int, int]:
    """Count symbols and timeframes from the HTML."""
    symbols = set()
    timeframes = set()

    # Look for symbol options
    symbol_matches = re.findall(r'<option value="([A-Z]+)">', html_content)
    symbols.update(symbol_matches)

    # Look for timeframe buttons
    tf_matches = re.findall(r'data-tf="(\d+[mhdw])"', html_content)
    timeframes.update(tf_matches)

    return len(symbols), len(timeframes)


def analyze_report(report_path: Path) -> ReportSizeAnalysis:
    """Analyze a signal report file and return size metrics."""
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    html_content = report_path.read_text(encoding="utf-8")
    total_bytes = len(html_content.encode("utf-8"))
    total_lines = html_content.count("\n") + 1

    sections = extract_section_sizes(html_content)
    chart_details = extract_chart_data_details(html_content)
    symbols_count, timeframes_count = count_symbols_timeframes(html_content)

    return ReportSizeAnalysis(
        file_path=report_path,
        total_bytes=total_bytes,
        total_lines=total_lines,
        html_structure=sections["html_structure"],
        css_styles=sections["css_styles"],
        javascript=sections["javascript"],
        chart_data_json=sections["chart_data_json"],
        signal_history_json=sections["signal_history_json"],
        confluence_json=sections["confluence_json"],
        regime_sections=sections["regime_sections"],
        indicator_cards=sections["indicator_cards"],
        chart_data_details=chart_details,
        symbols_count=symbols_count,
        timeframes_count=timeframes_count,
    )


def generate_and_analyze(
    symbols: List[str],
    output_path: Optional[Path] = None,
) -> ReportSizeAnalysis:
    """Generate a fresh report and analyze it."""
    import asyncio
    import tempfile

    # Default output path
    if output_path is None:
        output_path = Path(tempfile.mkdtemp()) / "signal_report.html"

    print(f"Generating report for symbols: {', '.join(symbols)}")
    print(f"Output: {output_path}")
    print()

    # Run signal runner to generate report
    from src.runners.signal_runner import SignalRunner, SignalRunnerConfig

    config = SignalRunnerConfig(
        symbols=symbols,
        timeframes=["1d"],
        live=True,
        html_output=str(output_path),
    )
    runner = SignalRunner(config)
    asyncio.run(runner.run())

    return analyze_report(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze signal report size breakdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to existing HTML report to analyze",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate fresh report before analyzing",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "SPY", "QQQ"],
        help="Symbols for report generation (default: AAPL SPY QQQ)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for generated report",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-symbol breakdown",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis as JSON",
    )

    args = parser.parse_args()

    try:
        if args.generate:
            analysis = generate_and_analyze(args.symbols, args.output)
        elif args.report:
            analysis = analyze_report(args.report)
        else:
            # Default: look for existing report
            default_path = Path("results/signals/signal_report.html")
            if default_path.exists():
                analysis = analyze_report(default_path)
            else:
                print(
                    "Error: No report found. Use --report <path> or --generate to create one.",
                    file=sys.stderr,
                )
                return 1

        if args.json:
            print(json.dumps(analysis.to_dict(), indent=2))
        else:
            analysis.print_summary(detailed=args.detailed)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
