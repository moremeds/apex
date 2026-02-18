"""Email renderer — plain-text momentum screener report.

Renders momentum_watchlist.json as monospace plain-text for mobile email clients.
Mirrors the format of email_pead_renderer.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.regime_display import regime_label
from src.utils.timezone import DisplayTimezone


def render_momentum_email_text(
    watchlist_path: str | Path,
    display_timezone: str = "America/New_York",
) -> str:
    """Render momentum watchlist as plain-text email body.

    Args:
        watchlist_path: Path to momentum_watchlist.json.
        display_timezone: IANA timezone for timestamp display.

    Returns:
        Plain-text string suitable for email body.
    """
    path = Path(watchlist_path)
    if not path.exists():
        return "Momentum Screen: No data available."

    data = json.loads(path.read_text())
    candidates = data.get("candidates", [])
    regime = data.get("regime", "?")
    universe_size = data.get("universe_size", 0)
    passed_filters = data.get("passed_filters", 0)
    generated_at = data.get("generated_at", "")
    data_as_of = data.get("data_as_of", "")
    errors = data.get("errors", [])

    timestamp = _format_timestamp(generated_at, display_timezone)

    lines: list[str] = []
    lines.append("Momentum Screen")
    lines.append(timestamp)
    if data_as_of:
        lines.append(f"Data as of: {data_as_of}")
    lines.append(f"Regime: {regime} ({regime_label(regime)})")
    lines.append("\u2550" * 42)
    lines.append("")

    if candidates:
        count = len(candidates)
        lines.append(
            f"CANDIDATES ({count} found | universe: {universe_size} | passed: {passed_filters})"
        )
        lines.append("\u2500" * 42)

        for c in candidates:
            rank = c.get("rank", 0)
            symbol = c.get("symbol", "?")
            tier = c.get("liquidity_tier", "?").upper()
            mom = c.get("momentum_12_1", 0)
            fip = c.get("fip", 0)
            comp = c.get("composite_rank", 0)
            quality = c.get("quality_label", "?")
            close = c.get("last_close", 0)
            mktcap = c.get("market_cap")
            addv = c.get("avg_daily_dollar_volume", 0)
            slip = c.get("estimated_slippage_bps", 0)
            size = c.get("position_size_factor", 1)

            mktcap_str = _format_market_cap(mktcap) if mktcap else "N/A"

            lines.append(f"#{rank} {symbol} [{tier}]")
            lines.append(f"   Mom 12-1: {mom:+.1%}  FIP: {fip:.2f}")
            lines.append(f"   Composite: {comp:.2f}  Quality: {quality}")
            lines.append(f"   Close: ${close:.2f}  MktCap: {mktcap_str}")
            lines.append(f"   ADDV: ${addv:,.0f}  Slippage: ~{slip}bps")
            lines.append(f"   Size: {size:.0%}")
            lines.append("\u2500" * 42)
    else:
        lines.append(f"0 momentum candidates.")
        lines.append(f"(universe: {universe_size}, passed: {passed_filters}, regime: {regime})")

    if errors:
        lines.append("")
        lines.append(f"Errors ({len(errors)}):")
        for err in errors[:5]:
            lines.append(f"  - {err}")

    lines.append("")
    lines.append("\u2500" * 42)
    lines.append("Full report \u2192 moremeds.github.io/apex/momentum/report.html")

    return "\n".join(lines)


def _format_timestamp(generated_at: str, display_timezone: str) -> str:
    if generated_at:
        try:
            dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            disp = DisplayTimezone(display_timezone)
            return disp.format_with_tz(dt, fmt="%b %d, %Y %I:%M %p %Z")
        except (ValueError, AttributeError):
            return generated_at
    return datetime.now(tz=timezone.utc).strftime("%b %d, %Y %I:%M %p UTC")


def _format_market_cap(mktcap: float | int | None) -> str:
    if mktcap is None:
        return "N/A"
    if mktcap >= 1e12:
        return f"${mktcap / 1e12:.1f}T"
    if mktcap >= 1e9:
        return f"${mktcap / 1e9:.1f}B"
    if mktcap >= 1e6:
        return f"${mktcap / 1e6:.0f}M"
    return f"${mktcap:,.0f}"
