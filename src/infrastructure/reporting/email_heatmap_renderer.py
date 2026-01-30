"""
Email renderer — plain-text TUI format for mobile reading.

Renders summary.json as a monospace plain-text report (≤40 chars wide)
optimized for phone email clients. Uses Unicode box-drawing characters.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infrastructure.reporting.heatmap.etf_dashboard import REGIME_NAMES
from src.infrastructure.reporting.heatmap.extractors import extract_regime
from src.infrastructure.reporting.heatmap.model import (
    ETF_CONFIG,
    MARKET_ETFS,
)
from src.infrastructure.reporting.package.score_history import ScoreHistoryManager


def render_email_text(
    summary_path: str | Path,
    history_path: Optional[str | Path] = None,
    report_label: Optional[str] = None,
) -> str:
    """
    Render summary.json as plain-text email body.

    Args:
        summary_path: Path to summary.json from signal pipeline.
        history_path: Path to score_history.json for delta computation.
        report_label: Time-of-day label (e.g. "Intraday 1H", "End of Day").

    Returns:
        Plain-text string (≤40 chars wide) suitable for email body.
    """
    summary_data = json.loads(Path(summary_path).read_text())
    tickers = {t["symbol"]: t for t in summary_data.get("tickers", [])}

    # Load history for change detection
    history_mgr = ScoreHistoryManager()
    if history_path:
        history_mgr.load(Path(history_path))

    # Timestamp
    timestamp_str = _format_timestamp(summary_data.get("generated_at"))

    sector_symbols = list(ETF_CONFIG["sectors"]["symbols"])

    lines: List[str] = []
    title = f"APEX {report_label}" if report_label else "APEX Signal Report"
    lines.append(title)
    lines.append(timestamp_str)
    lines.append("═" * 36)
    lines.append("")

    # Market section
    lines.append("MARKET")
    lines.append("─" * 36)
    for symbol in MARKET_ETFS:
        lines.append(_format_market_line(tickers.get(symbol, {}), symbol))
    lines.append("")

    # Sectors — two columns
    lines.append("SECTORS")
    lines.append("─" * 36)
    for i in range(0, len(sector_symbols), 2):
        left = _format_sector_cell(tickers.get(sector_symbols[i], {}), sector_symbols[i])
        if i + 1 < len(sector_symbols):
            right = _format_sector_cell(
                tickers.get(sector_symbols[i + 1], {}), sector_symbols[i + 1]
            )
            lines.append(f"{left} │ {right}")
        else:
            lines.append(left)
    lines.append("")

    # Regime distribution
    lines.append("REGIME DISTRIBUTION")
    lines.append("─" * 36)
    regime_counts = _count_regimes(summary_data)
    total = sum(regime_counts.values()) or 1
    for regime, count in regime_counts.items():
        name = REGIME_NAMES.get(regime, regime)
        pct = (count / total) * 100
        # Truncate name to fit
        short_name = name[:15].ljust(15)
        lines.append(f"■ {regime} {short_name} {count:>3} stocks ({pct:.0f}%)")
    lines.append("")

    # Changes section
    _append_changes_section(lines, history_mgr)

    # Footer
    lines.append("─" * 36)
    lines.append("Full report → moremeds.github.io/apex")

    return "\n".join(lines)


def _format_timestamp(generated_at: Optional[str]) -> str:
    if generated_at:
        try:
            dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            return dt.strftime("%b %d, %Y %I:%M %p ET")
        except (ValueError, AttributeError):
            return generated_at
    return datetime.now(tz=timezone.utc).strftime("%b %d, %Y %I:%M %p")


def _format_market_line(ticker: Dict[str, Any], symbol: str) -> str:
    """Format: SPY  R0 Healthy    +0.8%  Score: 72"""
    regime = extract_regime(ticker) or "—"
    name = REGIME_NAMES.get(regime, "")
    # Truncate regime name to 10 chars
    short_name = name[:10].ljust(10) if name else "          "
    change = ticker.get("daily_change_pct")
    change_str = f"{change:+.1f}%" if change is not None else "     "
    score = ticker.get("composite_score_avg")
    score_str = f"Score: {score:.0f}" if score is not None else ""
    return f"{symbol:<4} {regime:<2} {short_name} {change_str:>6}  {score_str}"


def _format_sector_cell(ticker: Dict[str, Any], symbol: str) -> str:
    """Format: XLK  R0  +1.1%"""
    regime = extract_regime(ticker) or "—"
    change = ticker.get("daily_change_pct")
    change_str = f"{change:+.1f}%" if change is not None else "     "
    return f"{symbol:<4} {regime:<2} {change_str:>6}"


def _count_regimes(summary: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {"R0": 0, "R1": 0, "R2": 0, "R3": 0}
    for ticker in summary.get("tickers", []):
        regime = extract_regime(ticker)
        if regime in counts:
            counts[regime] += 1
    return counts


def _append_changes_section(lines: List[str], history_mgr: ScoreHistoryManager) -> None:
    """Append the CHANGES SINCE LAST REPORT section if deltas exist."""
    score_changes = history_mgr.get_score_changes(min_delta=5.0)
    trend_changes = history_mgr.get_trend_state_changes()
    momentum_changes = history_mgr.get_momentum_changes(min_delta=0.005)

    if not score_changes and not trend_changes and not momentum_changes:
        return

    lines.append("▼ CHANGES SINCE LAST REPORT ▼")
    lines.append("═" * 36)
    lines.append("")

    if score_changes:
        lines.append("COMPOSITE SCORE CHANGES (|Δ| ≥ 5)")
        lines.append("─" * 36)
        for c in score_changes[:10]:
            arrow = "▲" if c["delta"] > 0 else "▼"
            warn = "  ⚠" if abs(c["delta"]) >= 7 else ""
            lines.append(
                f"{c['symbol']:<5} {c['prev']:.0f} → {c['curr']:.0f}  "
                f"{arrow} {c['delta']:+.0f}{warn}"
            )
        lines.append("")

    if trend_changes:
        lines.append("TREND STATE SHIFTS")
        lines.append("─" * 36)
        # States that warrant warning
        warn_states = {"bearish", "deteriorating", "risk_off"}
        for c in trend_changes[:10]:
            warn = (
                "  ⚠"
                if c["curr"].lower() in warn_states or c["prev"].lower() in warn_states
                else ""
            )
            lines.append(f"{c['symbol']:<5} {c['prev']} → {c['curr']}{warn}")
        lines.append("")

    if momentum_changes:
        lines.append("MOMENTUM CHANGES")
        lines.append("─" * 36)
        for c in momentum_changes[:10]:
            arrow = "▲" if c["delta"] > 0 else "▼"
            lines.append(f"{c['symbol']:<5} slope: {c['prev']:.4f} → {c['curr']:.4f}  {arrow}")
        lines.append("")
