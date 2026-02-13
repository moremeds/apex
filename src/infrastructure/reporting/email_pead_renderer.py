"""Email renderer — plain-text PEAD screener report.

Renders pead_candidates.json as monospace plain-text for mobile email clients.
Mirrors the format of email_heatmap_renderer.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.timezone import DisplayTimezone


def render_pead_email_text(
    candidates_path: str | Path,
    display_timezone: str = "America/New_York",
) -> str:
    """Render PEAD candidates as plain-text email body.

    Args:
        candidates_path: Path to pead_candidates.json.
        display_timezone: IANA timezone for timestamp display.

    Returns:
        Plain-text string suitable for email body.
    """
    path = Path(candidates_path)
    if not path.exists():
        return "PEAD Screen: No data available."

    data = json.loads(path.read_text())
    candidates = data.get("candidates", [])
    regime = data.get("regime", "?")
    screened = data.get("screened_count", 0)
    skipped = data.get("skipped_count", 0)
    generated_at = data.get("generated_at", "")

    # Format timestamp
    timestamp = _format_timestamp(generated_at, display_timezone)

    lines: list[str] = []
    lines.append("PEAD Earnings Drift Screen")
    lines.append(timestamp)
    lines.append(f"Regime: {regime} ({_regime_label(regime)})")
    lines.append("\u2550" * 36)
    lines.append("")

    if candidates:
        count = len(candidates)
        lines.append(f"CANDIDATES ({count} found, {screened} screened)")
        if skipped:
            lines.append(f"({skipped} symbols skipped: FMP tier limit)")
        lines.append("\u2500" * 36)

        for i, c in enumerate(candidates, 1):
            tier = c.get("liquidity_tier", "?").upper()
            sue = c.get("sue_score", 0)
            gap = c.get("earnings_day_gap", 0) * 100
            vol = c.get("earnings_day_volume_ratio", 0)
            score = c.get("quality_score", 0)
            label = c.get("quality_label", "?")
            rev = "Y" if c.get("revenue_beat") else "N"
            target = c.get("profit_target_pct", 0) * 100
            stop = c.get("stop_loss_pct", 0) * 100
            trail = c.get("trailing_stop_atr", 0)
            trail_act = c.get("trailing_activation_pct", 0) * 100
            size = c.get("position_size_factor", 1) * 100
            hold = c.get("max_hold_days", 0)
            slip = c.get("estimated_slippage_bps", 0)
            gap_held = "Y" if c.get("gap_held") else "N"

            mq_sue = c.get("multi_quarter_sue")
            mq_str = f" MQ:{mq_sue:.1f}" if mq_sue is not None else ""

            lines.append(f"#{i} {c['symbol']} [{tier}]")
            lines.append(f"   SUE:{sue:.1f}{mq_str} Gap:{gap:+.1f}% Vol:{vol:.1f}x")
            lines.append(f"   Quality: {score:.0f} {label}  Rev: {rev}")
            lines.append(f"   Gap Held: {gap_held}")
            lines.append(f"   Target: +{target:.1f}%  Stop: {stop:.1f}%")
            lines.append(f"   Trail: {trail:.1f} ATR after +{trail_act:.0f}%")
            lines.append(f"   Size: {size:.0f}%  Hold: {hold}d")
            lines.append(f"   Slippage: ~{slip}bps")
            lines.append("\u2500" * 36)
    else:
        lines.append(f"0 PEAD candidates.")
        lines.append(f"({screened} screened, regime: {regime})")
        if skipped:
            lines.append(f"({skipped} symbols skipped: FMP tier limit)")

    lines.append("")
    lines.append("\u2500" * 36)
    lines.append("Full report \u2192 moremeds.github.io/apex/pead.html")

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


def _regime_label(regime: str) -> str:
    labels = {
        "R0": "Healthy Uptrend",
        "R1": "Choppy/Extended",
        "R2": "Risk-Off",
        "R3": "Rebound Window",
    }
    return labels.get(regime, "Unknown")
