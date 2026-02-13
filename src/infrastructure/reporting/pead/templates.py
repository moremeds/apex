"""HTML templates for PEAD screener dashboard.

Renders a standalone dark-theme HTML page matching the strategy comparison style.
No Plotly dependency — pure HTML/CSS tables with expandable detail rows.
"""

from __future__ import annotations

from typing import Any


def render_pead_html(data: dict[str, Any]) -> str:
    """Render complete PEAD HTML dashboard.

    Args:
        data: Serialized PEADScreenResult dict from builder.

    Returns:
        Complete HTML page content.
    """
    candidates = data.get("candidates", [])
    regime = data.get("regime", "?")
    screened = data.get("screened_count", 0)
    passed = data.get("passed_filters", 0)
    skipped = data.get("skipped_count", 0)
    generated_at = data.get("generated_at", "N/A")

    # Summary stats
    avg_quality = sum(c["quality_score"] for c in candidates) / len(candidates) if candidates else 0
    strong = sum(1 for c in candidates if c["quality_label"] == "STRONG")
    moderate = sum(1 for c in candidates if c["quality_label"] == "MODERATE")
    marginal = sum(1 for c in candidates if c["quality_label"] == "MARGINAL")

    # Group by liquidity tier
    by_tier: dict[str, list[dict[str, Any]]] = {
        "large_cap": [],
        "mid_cap": [],
        "small_cap": [],
    }
    for c in candidates:
        tier = c.get("liquidity_tier", "small_cap")
        by_tier.setdefault(tier, []).append(c)

    summary_cards = _build_summary_cards(
        passed,
        screened,
        skipped,
        regime,
        avg_quality,
        strong,
        moderate,
        marginal,
    )
    tier_tables = _build_tier_tables(by_tier)
    methodology = _build_methodology()

    timestamp = generated_at[:19] if len(generated_at) > 19 else generated_at

    tracker_section = _build_tracker_section(data.get("tracker_stats"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PEAD Earnings Drift Screen</title>
<style>
{_CSS}
</style>
<script>
{_JS}
</script>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="header-top">
            <a href="index.html" class="back-link">&larr; Heatmap</a>
            <h1>PEAD Earnings Drift Screen</h1>
            <span class="regime-badge regime-{regime.lower()}">{regime}</span>
        </div>
        <div class="meta">Generated: {timestamp}</div>
    </div>

    {summary_cards}
    {tier_tables}
    {tracker_section}
    {methodology}

    <div class="footer">
        <a href="index.html">&larr; Back to Heatmap</a>
        &nbsp;|&nbsp;
        <a href="strategies.html">Strategy Comparison</a>
    </div>
</div>
</body>
</html>"""


def _build_summary_cards(
    passed: int,
    screened: int,
    skipped: int,
    regime: str,
    avg_quality: float,
    strong: int,
    moderate: int,
    marginal: int,
) -> str:
    regime_label = {
        "R0": "Healthy Uptrend",
        "R1": "Choppy/Extended",
        "R2": "Risk-Off",
        "R3": "Rebound Window",
    }.get(regime, "Unknown")

    skip_note = f" ({skipped} skipped)" if skipped else ""

    return f"""
    <div class="cards">
        <div class="card">
            <div class="card-label">Candidates</div>
            <div class="card-value">{passed}</div>
            <div class="card-sub">{screened} screened{skip_note}</div>
        </div>
        <div class="card">
            <div class="card-label">Avg Quality</div>
            <div class="card-value">{avg_quality:.0f}</div>
            <div class="card-sub">{strong}S / {moderate}M / {marginal}m</div>
        </div>
        <div class="card">
            <div class="card-label">Regime</div>
            <div class="card-value regime-{regime.lower()}">{regime}</div>
            <div class="card-sub">{regime_label}</div>
        </div>
    </div>"""


def _build_tier_tables(by_tier: dict[str, list[dict[str, Any]]]) -> str:
    tier_labels = {
        "large_cap": ("Large Cap", "&gt;$50B", "10bps"),
        "mid_cap": ("Mid Cap", "$2B-$50B", "25bps"),
        "small_cap": ("Small Cap", "&lt;$2B", "50bps"),
    }

    sections = []
    for tier_key in ["large_cap", "mid_cap", "small_cap"]:
        candidates = by_tier.get(tier_key, [])
        if not candidates:
            continue

        label, cap_range, slip = tier_labels[tier_key]
        rows = _build_candidate_rows(candidates)
        net_alpha = _build_net_alpha(candidates)

        sections.append(f"""
    <div class="tier-section">
        <h2>{label} <span class="tier-meta">{cap_range} &middot; est. slippage ~{slip}</span></h2>
        <table class="candidates-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Symbol</th>
                    <th>SUE</th>
                    <th>MQ-SUE</th>
                    <th>Gap</th>
                    <th>Vol</th>
                    <th>Rev</th>
                    <th>Quality</th>
                    <th>Gap Held</th>
                    <th>Size</th>
                    <th>Target</th>
                    <th>Stop</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        {net_alpha}
    </div>""")

    if not sections:
        return '<div class="no-candidates">No candidates passed filters.</div>'

    return "\n".join(sections)


def _build_candidate_rows(candidates: list[dict[str, Any]]) -> str:
    rows = ""
    for i, c in enumerate(candidates, 1):
        q_class = c["quality_label"].lower()
        gap_pct = c["earnings_day_gap"] * 100
        gap_cls = "positive" if gap_pct >= 0 else "negative"
        rev = "&#10003;" if c["revenue_beat"] else "&ndash;"
        gap_held = "&#10003;" if c["gap_held"] else "&#10007;"
        gap_held_cls = "positive" if c["gap_held"] else "negative"
        size_pct = c["position_size_factor"] * 100
        target_pct = c["profit_target_pct"] * 100
        stop_pct = c["stop_loss_pct"] * 100
        mq_sue = c.get("multi_quarter_sue")
        mq_str = f"{mq_sue:.1f}" if mq_sue is not None else "&ndash;"
        mq_cls = (
            "positive"
            if mq_sue is not None and mq_sue > 0
            else "negative" if mq_sue is not None and mq_sue < 0 else ""
        )

        # Detail row content
        detail = _build_detail(c)

        rows += f"""
                <tr class="candidate-row" onclick="toggleDetail(this)">
                    <td>{i}</td>
                    <td class="symbol">{c['symbol']}</td>
                    <td>{c['sue_score']:.1f}</td>
                    <td class="{mq_cls}">{mq_str}</td>
                    <td class="{gap_cls}">{gap_pct:+.1f}%</td>
                    <td>{c['earnings_day_volume_ratio']:.1f}x</td>
                    <td>{rev}</td>
                    <td><span class="quality-badge {q_class}">{c['quality_score']:.0f} {c['quality_label']}</span></td>
                    <td class="{gap_held_cls}">{gap_held}</td>
                    <td>{size_pct:.0f}%</td>
                    <td class="positive">+{target_pct:.1f}%</td>
                    <td class="negative">{stop_pct:.1f}%</td>
                </tr>
                <tr class="detail-row" style="display:none;">
                    <td colspan="12">{detail}</td>
                </tr>"""
    return rows


def _build_detail(c: dict[str, Any]) -> str:
    trail_act = c["trailing_activation_pct"] * 100
    fwd_pe = f"{c['forward_pe']:.1f}" if c.get("forward_pe") is not None else "N/A"

    return f"""
        <div class="detail-grid">
            <div class="detail-col">
                <h4>Earnings</h4>
                <p>Report: {c['report_date']}</p>
                <p>EPS: {c['actual_eps']:.2f} vs {c['consensus_eps']:.2f} ({c['surprise_pct']:+.1f}%)</p>
                <p>Day Return: {c['earnings_day_return']*100:+.1f}%</p>
                <p>Forward PE: {fwd_pe}</p>
            </div>
            <div class="detail-col">
                <h4>Trade Plan</h4>
                <p>Entry: {c['entry_date']} @ ${c['entry_price']:.2f}</p>
                <p>Target: +{c['profit_target_pct']*100:.1f}% &middot; Stop: {c['stop_loss_pct']*100:.1f}%</p>
                <p>Trail: {c['trailing_stop_atr']:.1f} ATR after +{trail_act:.0f}%</p>
                <p>Max Hold: {c['max_hold_days']} trading days</p>
            </div>
            <div class="detail-col">
                <h4>Quality Breakdown</h4>
                <p>SUE: {c['sue_score']:.1f} (earnings surprise magnitude)</p>
                <p>Multi-Q SUE: {f"{c['multi_quarter_sue']:.1f}" if c.get('multi_quarter_sue') is not None else "N/A"} (historical trajectory)</p>
                <p>Gap: {c['earnings_day_gap']*100:+.1f}% (post-report gap)</p>
                <p>Volume: {c['earnings_day_volume_ratio']:.1f}x avg (institutional confirmation)</p>
                <p>Revenue Beat: {'Yes' if c['revenue_beat'] else 'No'}</p>
            </div>
        </div>"""


def _build_net_alpha(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""

    avg_target = sum(c["profit_target_pct"] for c in candidates) / len(candidates) * 100
    avg_slip = sum(c["estimated_slippage_bps"] for c in candidates) / len(candidates)
    net_alpha = avg_target - (avg_slip * 2 / 100)  # round-trip slippage in pct

    return f"""
        <div class="net-alpha">
            Avg gross target: {avg_target:.1f}% &middot;
            Est. round-trip cost: ~{avg_slip*2:.0f}bps &middot;
            <strong>Net alpha: ~{net_alpha:.1f}%</strong>
        </div>"""


def _build_tracker_section(tracker_stats: dict[str, Any] | None) -> str:
    """Build historical performance section from tracker stats."""
    if not tracker_stats:
        return ""

    total = tracker_stats.get("total", 0)
    if total == 0:
        return ""

    won = tracker_stats.get("won", 0)
    lost = tracker_stats.get("lost", 0)
    timeout = tracker_stats.get("timeout", 0)
    open_count = tracker_stats.get("open", 0)
    win_rate = tracker_stats.get("win_rate")
    avg_pnl = tracker_stats.get("avg_pnl_pct")
    avg_hold = tracker_stats.get("avg_hold_days")

    wr_str = f"{win_rate:.1%}" if win_rate is not None else "N/A"
    pnl_str = f"{avg_pnl:+.2%}" if avg_pnl is not None else "N/A"
    pnl_cls = (
        "positive" if avg_pnl and avg_pnl > 0 else "negative" if avg_pnl and avg_pnl < 0 else ""
    )
    hold_str = f"{avg_hold:.1f}d" if avg_hold is not None else "N/A"

    # Quality tier breakdown
    by_quality = tracker_stats.get("by_quality", {})
    tier_rows = ""
    for label in ["STRONG", "MODERATE", "MARGINAL"]:
        data = by_quality.get(label)
        if not data:
            continue
        t_wr = data.get("win_rate")
        t_pnl = data.get("avg_pnl_pct")
        tier_rows += f"""
            <tr>
                <td><span class="quality-badge {label.lower()}">{label}</span></td>
                <td>{data['total']}</td>
                <td>{f'{t_wr:.1%}' if t_wr is not None else 'N/A'}</td>
                <td class="{'positive' if t_pnl and t_pnl > 0 else 'negative' if t_pnl and t_pnl < 0 else ''}">{f'{t_pnl:+.2%}' if t_pnl is not None else 'N/A'}</td>
            </tr>"""

    return f"""
    <div class="tier-section">
        <h2>Historical Performance</h2>
        <div class="cards">
            <div class="card">
                <div class="card-label">Win Rate</div>
                <div class="card-value">{wr_str}</div>
                <div class="card-sub">{won}W / {lost}L / {timeout}T</div>
            </div>
            <div class="card">
                <div class="card-label">Avg P&amp;L</div>
                <div class="card-value {pnl_cls}">{pnl_str}</div>
                <div class="card-sub">{total} total tracked</div>
            </div>
            <div class="card">
                <div class="card-label">Avg Hold</div>
                <div class="card-value">{hold_str}</div>
                <div class="card-sub">{open_count} still open</div>
            </div>
        </div>
        {"<table class='candidates-table'><thead><tr><th>Tier</th><th>Count</th><th>Win Rate</th><th>Avg P&L</th></tr></thead><tbody>" + tier_rows + "</tbody></table>" if tier_rows else ""}
    </div>"""


def _build_methodology() -> str:
    return """
    <details class="methodology">
        <summary>Methodology</summary>
        <div class="methodology-content">
            <p><strong>PEAD (Post-Earnings Announcement Drift)</strong> identifies stocks where
            the market underreacts to earnings surprises. Academic evidence spans 60 years
            (Ball &amp; Brown 1968 &rarr; Chordia 2009).</p>

            <h4>Filter Pipeline</h4>
            <ol>
                <li>Regime gate: R2 blocks all new positions</li>
                <li>Entry window: 1-5 NYSE trading days since report</li>
                <li>SUE &ge; 2.0 (Standardized Unexpected Earnings)</li>
                <li>Earnings gap &ge; 2%</li>
                <li>Volume &ge; 2x 20-day average</li>
                <li>Not within 5% of 52-week high (hard exclude)</li>
                <li>No analyst downgrade within 3 days (hard exclude)</li>
                <li>Forward PE &le; 40</li>
            </ol>

            <h4>Quality Score (0-100)</h4>
            <ul>
                <li>SUE magnitude: 0-30 pts</li>
                <li>Gap quality: 0-30 pts</li>
                <li>Volume confirmation: 0-25 pts</li>
                <li>Revenue beat bonus: 0-15 pts</li>
            </ul>

            <h4>Position Sizing</h4>
            <p>R0: 100% &middot; R1: 50% &middot; R2: blocked &middot;
            Trailing stop: 2 ATR after +3% gain</p>

            <h4>Data Sources</h4>
            <p>FMP (earnings calendar, history, analyst grades) + yfinance (OHLCV, 52w high)</p>
        </div>
    </details>"""


# ── CSS ──────────────────────────────────────────────────────────────────

_CSS = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
}

.container { max-width: 1200px; margin: 0 auto; padding: 20px; }

.header { margin-bottom: 24px; }
.header-top { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.header h1 { font-size: 1.5rem; }
.back-link { color: var(--accent); text-decoration: none; font-size: 13px; }
.meta { color: var(--text-secondary); font-size: 13px; margin-top: 4px; }

.regime-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
}
.regime-r0 { background: var(--green); color: var(--bg-primary); }
.regime-r1 { background: var(--yellow); color: var(--bg-primary); }
.regime-r2 { background: var(--red); color: var(--bg-primary); }
.regime-r3 { background: var(--accent); color: var(--bg-primary); }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; text-align: center;
}
.card-label { color: var(--text-secondary); font-size: 12px; text-transform: uppercase; }
.card-value { font-size: 2rem; font-weight: 700; margin: 4px 0; }
.card-sub { color: var(--text-secondary); font-size: 12px; }

.tier-section { margin-bottom: 32px; }
.tier-section h2 { font-size: 1.1rem; margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.tier-meta { color: var(--text-secondary); font-size: 12px; font-weight: 400; }

.candidates-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.candidates-table th {
    text-align: left; padding: 8px 12px; background: var(--bg-secondary);
    color: var(--text-secondary); font-size: 11px; text-transform: uppercase;
    border-bottom: 1px solid var(--border);
}
.candidates-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
.candidate-row { cursor: pointer; transition: background 0.15s; }
.candidate-row:hover { background: var(--bg-card); }
.symbol { font-weight: 600; color: var(--accent); }

.positive { color: var(--green); }
.negative { color: var(--red); }

.quality-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600;
}
.strong { background: rgba(63, 185, 80, 0.2); color: var(--green); }
.moderate { background: rgba(210, 153, 34, 0.2); color: var(--yellow); }
.marginal { background: rgba(248, 81, 73, 0.15); color: var(--text-secondary); }

.detail-row td { padding: 16px 12px; background: var(--bg-card); }
.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
.detail-col h4 { color: var(--accent); font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }
.detail-col p { color: var(--text-secondary); font-size: 12px; margin-bottom: 4px; }

.net-alpha {
    padding: 8px 12px; margin-top: 8px; background: var(--bg-secondary);
    border-radius: 4px; font-size: 12px; color: var(--text-secondary);
}
.net-alpha strong { color: var(--green); }

.no-candidates {
    text-align: center; padding: 40px; color: var(--text-secondary);
    font-size: 1.1rem;
}

.methodology {
    margin-top: 32px; background: var(--bg-secondary);
    border: 1px solid var(--border); border-radius: 8px;
}
.methodology summary {
    padding: 12px 16px; cursor: pointer; color: var(--text-secondary);
    font-size: 13px;
}
.methodology-content { padding: 0 16px 16px; font-size: 13px; color: var(--text-secondary); }
.methodology-content h4 { color: var(--accent); margin: 12px 0 4px; font-size: 12px; }
.methodology-content ol, .methodology-content ul { padding-left: 20px; margin: 4px 0; }
.methodology-content li { margin-bottom: 2px; }

.footer {
    margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border);
    text-align: center; font-size: 13px;
}
.footer a { color: var(--accent); text-decoration: none; }
"""

_JS = """
function toggleDetail(row) {
    const detail = row.nextElementSibling;
    detail.style.display = detail.style.display === 'none' ? 'table-row' : 'none';
}
"""
