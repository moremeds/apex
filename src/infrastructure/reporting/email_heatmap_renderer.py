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

from src.infrastructure.reporting.heatmap.model import (
    ETF_CONFIG,
    MARKET_ETFS,
)
from src.infrastructure.reporting.package.score_history import ScoreHistoryManager
from src.utils.timezone import DisplayTimezone

# Report label → timeframe mapping for Dual MACD
_LABEL_TO_TF: Dict[str, str] = {
    "Intraday 1H": "1h",
    "Intraday 4H": "4h",
    "End of Day": "1d",
}


def render_email_text(
    summary_path: str | Path,
    history_path: Optional[str | Path] = None,
    report_label: Optional[str] = None,
    display_timezone: str = "Asia/Hong_Kong",
) -> str:
    """
    Render summary.json as plain-text email body.

    Args:
        summary_path: Path to summary.json from signal pipeline.
        history_path: Path to score_history.json for delta computation.
        report_label: Time-of-day label (e.g. "Intraday 1H", "End of Day").
        display_timezone: IANA timezone for timestamp display.

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
    timestamp_str = _format_timestamp(summary_data.get("generated_at"), display_timezone)

    sector_symbols = list(ETF_CONFIG["sectors"]["symbols"])

    lines: List[str] = []
    title = f"📊 APEX {report_label}" if report_label else "📊 APEX Signal Report"
    lines.append(title)
    lines.append(timestamp_str)
    lines.append("═" * 36)
    lines.append("")

    # TrendPulse signals (at TOP, before alerts)
    _append_trend_pulse_section(lines, summary_data, report_label)

    # Dual MACD alerts section (at TOP, before market)
    _append_dual_macd_alerts(lines, summary_data, report_label)

    # Market section
    lines.append("📈 MARKET (Δ Price | Score)")
    lines.append("─" * 36)
    for symbol in MARKET_ETFS:
        lines.append(_format_market_line(tickers.get(symbol, {}), symbol))
    lines.append("")

    # Sectors — two columns
    lines.append("🏢 SECTORS (Δ Price)")
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

    # Score distribution (replaces regime distribution)
    lines.append("📊 SCORE DISTRIBUTION")
    lines.append("─" * 36)
    bands = _count_score_bands(summary_data)
    total = sum(bands.values()) or 1
    for emoji, label, count in [
        ("🟢", "Strong (70-100)", bands["strong"]),
        ("🟡", "Neutral (30-70)", bands["neutral"]),
        ("🔴", "Weak (0-30)", bands["weak"]),
        ("⚪", "No Data", bands["no_data"]),
    ]:
        pct = (count / total) * 100
        lines.append(f"{emoji} {label:<18} {count:>3} ({pct:.0f}%)")
    lines.append("")

    # Changes section
    _append_changes_section(lines, history_mgr)

    # Dual MACD signal changes
    _append_dual_macd_signal_changes(lines, history_mgr)

    # Footer
    lines.append("─" * 36)
    lines.append("Full report → moremeds.github.io/apex")

    return "\n".join(lines)


def _format_timestamp(generated_at: Optional[str], display_timezone: str = "Asia/Hong_Kong") -> str:
    if generated_at:
        try:
            # Parse ISO string, handle Z suffix
            dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            # If naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            # Convert to display timezone
            disp = DisplayTimezone(display_timezone)
            return disp.format_with_tz(dt, fmt="%b %d, %Y %I:%M %p %Z")
        except (ValueError, AttributeError):
            return generated_at
    return datetime.now(tz=timezone.utc).strftime("%b %d, %Y %I:%M %p UTC")


def _format_market_line(ticker: Dict[str, Any], symbol: str) -> str:
    """Format: 🟢 SPY  +0.3%  Score: 46"""
    change = ticker.get("daily_change_pct")
    change_str = f"{change:+.1f}%" if change is not None else "     "
    emoji = "🟢" if change is not None and change >= 0 else "🔴"
    score = ticker.get("composite_score_avg")
    score_str = f"Score: {score:.0f}" if score is not None else ""
    return f"{emoji} {symbol:<4} {change_str:>6}  {score_str}"


def _format_sector_cell(ticker: Dict[str, Any], symbol: str) -> str:
    """Format: XLK  +1.1%"""
    change = ticker.get("daily_change_pct")
    change_str = f"{change:+.1f}%" if change is not None else "     "
    return f"{symbol:<4} {change_str:>6}"


def _count_score_bands(summary: Dict[str, Any]) -> Dict[str, int]:
    """Count tickers by composite score band."""
    bands = {"strong": 0, "neutral": 0, "weak": 0, "no_data": 0}
    for ticker in summary.get("tickers", []):
        score = ticker.get("composite_score_avg")
        if score is None:
            bands["no_data"] += 1
        elif score >= 70:
            bands["strong"] += 1
        elif score >= 30:
            bands["neutral"] += 1
        else:
            bands["weak"] += 1
    return bands


def _append_trend_pulse_section(
    lines: List[str],
    summary_data: Dict[str, Any],
    report_label: Optional[str],
) -> None:
    """Append TrendPulse entry/exit signals section at top of email."""
    trend_pulse = summary_data.get("trend_pulse")
    if not trend_pulse:
        return

    tf = _LABEL_TO_TF.get(report_label or "", "1d")
    tf_data = trend_pulse.get(tf) or trend_pulse.get("1d") or {}
    entries = tf_data.get("entries", [])
    exits = tf_data.get("exits", [])

    if not entries and not exits:
        return

    lines.append("⚡ TRENDPULSE SIGNALS")
    lines.append("═" * 36)

    if entries:
        lines.append("🟢 ENTRY")
        for e in entries:
            sym = e.get("symbol", "")
            dm = e.get("dm_state", "")[:4]
            conf = int(e.get("confidence_4f", 0) * 100)
            stop = e.get("atr_stop_level", 0)
            lines.append(f"  {sym:<5} {dm:<4}  Conf:{conf:>2}  Stop:${stop:.0f}")
        lines.append("")

    if exits:
        lines.append("🔴 EXIT")
        for x in exits:
            sym = x.get("symbol", "")
            reason = x.get("exit_reason", "")
            conf = int(x.get("confidence_4f", 0) * 100)
            lines.append(f"  {sym:<5} {reason:<10} Conf:{conf:>2}")
        lines.append("")

    if not exits and entries:
        # entries already appended blank line
        pass

    # Show potential daily signals in intraday emails
    if tf != "1d":
        daily_data = trend_pulse.get("1d", {})
        daily_entries = daily_data.get("entries", [])
        daily_exits = daily_data.get("exits", [])
        if daily_entries or daily_exits:
            lines.append("📅 POTENTIAL DAILY (spot price)")
            lines.append("─" * 36)
            for e in daily_entries:
                sym = e.get("symbol", "")
                dm = e.get("dm_state", "")[:4]
                conf = int(e.get("confidence_4f", 0) * 100)
                stop = e.get("atr_stop_level", 0)
                lines.append(f"  🟢 {sym:<5} {dm:<4}  Conf:{conf:>2}  Stop:${stop:.0f}")
            for x in daily_exits:
                sym = x.get("symbol", "")
                reason = x.get("exit_reason", "")
                conf = int(x.get("confidence_4f", 0) * 100)
                lines.append(f"  🔴 {sym:<5} {reason:<10} Conf:{conf:>2}")
            lines.append("")

    lines.append("")


def _append_dual_macd_alerts(
    lines: List[str],
    summary_data: Dict[str, Any],
    report_label: Optional[str],
) -> None:
    """Append ALERTS section from dual MACD data, at top of email."""
    dual_macd = summary_data.get("dual_macd")
    if not dual_macd:
        return

    # Pick timeframe based on report label
    tf = _LABEL_TO_TF.get(report_label or "", "1d")
    tf_data = dual_macd.get(tf) or dual_macd.get("1d") or {}
    alerts = tf_data.get("alerts", {})
    trends = tf_data.get("trends", [])

    dip_buy = alerts.get("dip_buy", [])
    rally_sell = alerts.get("rally_sell", [])

    if not dip_buy and not rally_sell and not trends:
        return

    lines.append("🚨 ALERTS")
    lines.append("═" * 36)

    if dip_buy:
        lines.append(f"💚 DIP BUY:   {', '.join(dip_buy)}")
    if rally_sell:
        lines.append(f"❤️ RALLY SELL: {', '.join(rally_sell)}")

    if trends:
        lines.append("")
        lines.append("📉 TREND DELTAS (top movers)")
        lines.append("─" * 36)
        for t in trends[:8]:
            sym = t.get("symbol", "")
            sd = t.get("slow_hist_delta", 0)
            fd = t.get("fast_hist_delta", 0)
            state = t.get("trend_state", "")
            lines.append(f"{sym:<5} ΔSlow:{sd:+.2f} ΔFast:{fd:+.2f} {state}")

    lines.append("")


def _append_changes_section(lines: List[str], history_mgr: ScoreHistoryManager) -> None:
    """Append the CHANGES SINCE LAST REPORT section if deltas exist."""
    score_changes = history_mgr.get_score_changes(min_delta=5.0)
    trend_changes = history_mgr.get_trend_state_changes()
    momentum_changes = history_mgr.get_momentum_changes(min_delta=0.005)

    if not score_changes and not trend_changes and not momentum_changes:
        return

    lines.append("🔄 CHANGES SINCE LAST REPORT")
    lines.append("═" * 36)
    lines.append("")

    if score_changes:
        lines.append("COMPOSITE SCORE CHANGES (|Δ| ≥ 5)")
        lines.append("─" * 36)
        for c in score_changes[:10]:
            arrow = "🔺" if c["delta"] > 0 else "🔻"
            warn = "  🚨" if abs(c["delta"]) >= 7 else ""
            lines.append(
                f"{c['symbol']:<5} {c['prev']:.0f} → {c['curr']:.0f}  "
                f"{arrow} {c['delta']:+.0f}{warn}"
            )
        lines.append("")

    if trend_changes:
        lines.append("🔀 TREND STATE SHIFTS")
        lines.append("─" * 36)
        warn_states = {"trend_down", "deteriorating"}
        for c in trend_changes[:10]:
            warn = (
                "  🚨"
                if c["curr"].lower() in warn_states or c["prev"].lower() in warn_states
                else ""
            )
            lines.append(f"{c['symbol']:<5} {c['prev']} → {c['curr']}{warn}")
        lines.append("")

    if momentum_changes:
        lines.append("⚡ MOMENTUM CHANGES")
        lines.append("─" * 36)
        for c in momentum_changes[:10]:
            arrow = "🔺" if c["delta"] > 0 else "🔻"
            lines.append(f"{c['symbol']:<5} slope: {c['prev']:.4f} → {c['curr']:.4f}  {arrow}")
        lines.append("")


def _append_dual_macd_signal_changes(lines: List[str], history_mgr: ScoreHistoryManager) -> None:
    """Append DUAL MACD SIGNAL CHANGES if tactical signals changed."""
    changes = history_mgr.get_tactical_signal_changes()
    if not changes:
        return

    lines.append("⚡ DUAL MACD SIGNAL CHANGES")
    lines.append("─" * 36)
    for c in changes[:10]:
        prev_emoji = (
            "💚" if c["prev"] == "DIP_BUY" else ("❤️" if c["prev"] == "RALLY_SELL" else "⬜")
        )
        curr_emoji = (
            "💚" if c["curr"] == "DIP_BUY" else ("❤️" if c["curr"] == "RALLY_SELL" else "⬜")
        )
        lines.append(f"{c['symbol']:<5} {prev_emoji} {c['prev']} → {curr_emoji} {c['curr']}")
    lines.append("")
