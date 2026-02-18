"""HTML templates for momentum screener dashboard.

Renders standalone dark-theme HTML pages:
- ``render_momentum_html``: live watchlist table + filter funnel
- ``render_backtest_html``: walk-forward results + ablation comparison
"""

from __future__ import annotations

from typing import Any


def render_momentum_html(data: dict[str, Any]) -> str:
    """Render complete momentum screener HTML dashboard.

    Args:
        data: Serialized MomentumScreenResult dict from builder.

    Returns:
        Complete HTML page content.
    """
    candidates = data.get("candidates", [])
    regime = data.get("regime", "?")
    universe_size = data.get("universe_size", 0)
    passed = data.get("passed_filters", 0)
    generated_at = data.get("generated_at", "N/A")

    # Summary stats
    strong = sum(1 for c in candidates if c["quality_label"] == "STRONG")
    moderate = sum(1 for c in candidates if c["quality_label"] == "MODERATE")
    marginal = sum(1 for c in candidates if c["quality_label"] == "MARGINAL")

    avg_momentum = (
        sum(c["momentum_12_1"] for c in candidates) / len(candidates) if candidates else 0
    )
    avg_fip = sum(c["fip"] for c in candidates) / len(candidates) if candidates else 0

    # Tier distribution
    by_tier: dict[str, int] = {"large_cap": 0, "mid_cap": 0, "small_cap": 0}
    for c in candidates:
        tier = c.get("liquidity_tier", "small_cap")
        by_tier[tier] = by_tier.get(tier, 0) + 1

    timestamp = generated_at[:19] if len(generated_at) > 19 else generated_at
    table_rows = _build_table_rows(candidates)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Momentum Screener</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Quantitative Momentum Screen</h1>
        <span class="regime-badge regime-{regime.lower()}">{regime}</span>
    </div>
    <div class="meta">Generated: {timestamp}</div>

    <div class="cards">
        <div class="card">
            <div class="card-value">{universe_size:,}</div>
            <div class="card-label">Universe</div>
        </div>
        <div class="card">
            <div class="card-value">{passed:,}</div>
            <div class="card-label">Passed Filters</div>
        </div>
        <div class="card">
            <div class="card-value">{len(candidates)}</div>
            <div class="card-label">Top-N Picks</div>
        </div>
        <div class="card">
            <div class="card-value">{avg_momentum:+.1%}</div>
            <div class="card-label">Avg Momentum</div>
        </div>
        <div class="card">
            <div class="card-value">{avg_fip:.2f}</div>
            <div class="card-label">Avg FIP</div>
        </div>
    </div>

    <div class="quality-summary">
        <span class="quality-badge strong">{strong} STRONG</span>
        <span class="quality-badge moderate">{moderate} MODERATE</span>
        <span class="quality-badge marginal">{marginal} MARGINAL</span>
        <span class="tier-info">
            Large: {by_tier['large_cap']} | Mid: {by_tier['mid_cap']} | Small: {by_tier['small_cap']}
        </span>
    </div>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Symbol</th>
                <th>Mom 12-1</th>
                <th>FIP</th>
                <th>Composite</th>
                <th>Quality</th>
                <th>Tier</th>
                <th>Close</th>
                <th>Mkt Cap</th>
                <th>Slippage</th>
                <th>Size</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <div class="methodology">
        <h3>Methodology</h3>
        <p><strong>12-1 Momentum</strong>: Cumulative return over months 2-12
        (skipping most recent month to avoid short-term reversal).
        Based on Jegadeesh &amp; Titman (1993).</p>
        <p><strong>FIP (Frog-In-Pan)</strong>: Fraction of positive vs negative
        return days in the momentum window. Measures return path smoothness.
        Higher FIP = more gradual price appreciation = stronger momentum signal.
        Based on Da, Gurun &amp; Warachka (2014).</p>
        <p><strong>Composite</strong>: Weighted average of momentum and FIP
        percentile ranks (50/50 default). Regime-dependent threshold applied.</p>
    </div>
</div>
</body>
</html>"""


def _build_table_rows(candidates: list[dict[str, Any]]) -> str:
    """Build HTML table rows for candidate watchlist."""
    rows: list[str] = []
    for c in candidates:
        mom = c["momentum_12_1"]
        fip = c["fip"]
        mom_color = "color: #3fb950" if mom > 0 else "color: #f85149"
        fip_color = "color: #3fb950" if fip > 0.5 else "color: #f85149"
        quality_cls = c["quality_label"].lower()

        # Format market cap
        cap = c.get("market_cap", 0)
        if cap >= 1e12:
            cap_str = f"${cap / 1e12:.1f}T"
        elif cap >= 1e9:
            cap_str = f"${cap / 1e9:.1f}B"
        elif cap >= 1e6:
            cap_str = f"${cap / 1e6:.0f}M"
        else:
            cap_str = f"${cap:,.0f}"

        tier_display = c["liquidity_tier"].replace("_", " ").title()

        rows.append(f"""<tr>
                <td>{c['rank']}</td>
                <td class="symbol">{c['symbol']}</td>
                <td style="{mom_color}">{mom:+.1%}</td>
                <td style="{fip_color}">{fip:.3f}</td>
                <td>{c['composite_rank']:.3f}</td>
                <td><span class="quality-badge {quality_cls}">{c['quality_label']}</span></td>
                <td>{tier_display}</td>
                <td>${c['last_close']:.2f}</td>
                <td>{cap_str}</td>
                <td>{c['estimated_slippage_bps']}bp</td>
                <td>{c['position_size_factor']:.0%}</td>
            </tr>""")

    return "\n".join(rows)


_CSS = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border: #30363d;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-blue: #58a6ff;
    --accent-yellow: #d29922;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
}

.container { max-width: 1400px; margin: 0 auto; padding: 24px; }

.header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 8px;
}
.header h1 { font-size: 24px; font-weight: 600; }

.meta { color: var(--text-secondary); font-size: 13px; margin-bottom: 20px; }

.regime-badge {
    padding: 4px 12px; border-radius: 12px;
    font-weight: 600; font-size: 13px;
}
.regime-r0 { background: #1a3a1a; color: var(--accent-green); }
.regime-r1 { background: #3a2a0a; color: var(--accent-yellow); }
.regime-r2 { background: #3a1a1a; color: var(--accent-red); }
.regime-r3 { background: #1a2a3a; color: var(--accent-blue); }

.cards {
    display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap;
}
.card {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 24px; flex: 1; min-width: 140px;
    text-align: center;
}
.card-value { font-size: 28px; font-weight: 700; }
.card-label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

.quality-summary {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; flex-wrap: wrap;
}
.quality-badge {
    padding: 3px 10px; border-radius: 10px; font-size: 12px; font-weight: 600;
}
.quality-badge.strong { background: #1a3a1a; color: var(--accent-green); }
.quality-badge.moderate { background: #3a2a0a; color: var(--accent-yellow); }
.quality-badge.marginal { background: var(--bg-tertiary); color: var(--text-secondary); }
.tier-info { color: var(--text-secondary); font-size: 13px; margin-left: auto; }

table {
    width: 100%; border-collapse: collapse;
    background: var(--bg-secondary); border-radius: 8px; overflow: hidden;
    margin-bottom: 24px;
}
thead { background: var(--bg-tertiary); }
th {
    padding: 10px 12px; text-align: left; font-size: 12px;
    color: var(--text-secondary); font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
}
td { padding: 10px 12px; border-top: 1px solid var(--border); font-size: 14px; }
tr:hover { background: var(--bg-tertiary); }
.symbol { font-weight: 600; color: var(--accent-blue); }

.methodology {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; margin-top: 16px;
}
.methodology h3 { margin-bottom: 12px; font-size: 16px; }
.methodology p { color: var(--text-secondary); font-size: 13px; margin-bottom: 8px; }
"""


# ── Backtest HTML ─────────────────────────────────────────────────────


def render_backtest_html(data: dict[str, Any]) -> str:
    """Render momentum backtest + ablation HTML report.

    Args:
        data: Dict with keys ``backtest`` (walk-forward results) and
              ``ablation`` (3-config comparison).  Both produced by
              ``cmd_backtest`` in the momentum runner.

    Returns:
        Complete standalone HTML page.
    """
    bt = data.get("backtest", {})
    abl = data.get("ablation", {})

    start = bt.get("start", "?")
    end = bt.get("end", "?")
    top_n = abl.get("top_n", bt.get("top_n", "?"))
    hold_days = abl.get("hold_days", bt.get("hold_days", "?"))

    cum_ret = bt.get("cumulative_return", 0.0)
    sharpe = bt.get("sharpe_approx", 0.0)
    max_dd = bt.get("max_drawdown", 0.0)
    avg_weekly = bt.get("avg_weekly_return", 0.0)
    n_periods = len(bt.get("periods", []))

    # Summary card colors
    cum_color = "var(--accent-green)" if cum_ret >= 0 else "var(--accent-red)"
    dd_color = "var(--accent-red)" if max_dd < -0.1 else "var(--accent-yellow)"

    # Ablation table rows
    ablation_rows = _build_ablation_rows(abl.get("configs", []))

    # Period returns table rows
    period_rows = _build_period_rows(bt.get("periods", []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Momentum Backtest</title>
<style>
{_BACKTEST_CSS}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Momentum Backtest</h1>
        <span class="date-range">{start} &rarr; {end}</span>
    </div>
    <div class="meta">Top-{top_n} &middot; {hold_days}-day hold &middot; weekly rebalance</div>

    <div class="cards">
        <div class="card">
            <div class="card-value" style="color:{cum_color}">{cum_ret:+.1%}</div>
            <div class="card-label">Cumulative Return</div>
        </div>
        <div class="card">
            <div class="card-value">{sharpe:.2f}</div>
            <div class="card-label">Sharpe (approx)</div>
        </div>
        <div class="card">
            <div class="card-value" style="color:{dd_color}">{max_dd:+.1%}</div>
            <div class="card-label">Max Drawdown</div>
        </div>
        <div class="card">
            <div class="card-value">{avg_weekly:+.3%}</div>
            <div class="card-label">Avg Weekly</div>
        </div>
        <div class="card">
            <div class="card-value">{n_periods}</div>
            <div class="card-label"># Periods</div>
        </div>
    </div>

    <h2 class="section-title">Ablation Comparison</h2>
    <table class="ablation">
        <thead>
            <tr>
                <th>Configuration</th>
                <th>Cum Return</th>
                <th>Sharpe</th>
                <th>Max DD</th>
                <th>Avg Weekly</th>
            </tr>
        </thead>
        <tbody>
            {ablation_rows}
        </tbody>
    </table>

    <h2 class="section-title">Period Returns</h2>
    <div class="table-scroll">
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th># Picks</th>
                <th>Avg Return</th>
                <th>Top 5 Picks</th>
            </tr>
        </thead>
        <tbody>
            {period_rows}
        </tbody>
    </table>
    </div>
</div>
</body>
</html>"""


def _build_ablation_rows(configs: list[dict[str, Any]]) -> str:
    """Build HTML rows for the ablation comparison table."""
    rows: list[str] = []
    for cfg in configs:
        label = cfg.get("label", "?")
        cr = cfg.get("cumulative_return", 0.0)
        sh = cfg.get("sharpe_approx", 0.0)
        dd = cfg.get("max_drawdown", 0.0)
        aw = cfg.get("avg_weekly_return", 0.0)
        cr_color = "color: #3fb950" if cr >= 0 else "color: #f85149"
        dd_color = "color: #f85149" if dd < -0.1 else "color: #d29922"
        rows.append(f"""<tr>
                <td class="config-label">{label}</td>
                <td style="{cr_color}">{cr:+.2%}</td>
                <td>{sh:.2f}</td>
                <td style="{dd_color}">{dd:+.2%}</td>
                <td>{aw:+.4%}</td>
            </tr>""")
    return "\n".join(rows)


def _build_period_rows(periods: list[dict[str, Any]]) -> str:
    """Build HTML rows for the period returns table."""
    rows: list[str] = []
    for p in periods:
        ret = p.get("avg_return", 0.0)
        ret_color = "color: #3fb950" if ret >= 0 else "color: #f85149"
        picks = ", ".join(p.get("picks", [])[:5])
        rows.append(f"""<tr>
                <td>{p.get('date', '?')}</td>
                <td>{p.get('n_picks', 0)}</td>
                <td style="{ret_color}">{ret:+.2%}</td>
                <td class="picks">{picks}</td>
            </tr>""")
    return "\n".join(rows)


_BACKTEST_CSS = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border: #30363d;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-blue: #58a6ff;
    --accent-yellow: #d29922;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
}

.container { max-width: 1200px; margin: 0 auto; padding: 24px; }

.header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 8px;
}
.header h1 { font-size: 24px; font-weight: 600; }
.date-range {
    padding: 4px 12px; border-radius: 12px;
    background: var(--bg-tertiary); color: var(--accent-blue);
    font-size: 13px; font-weight: 600;
}

.meta { color: var(--text-secondary); font-size: 13px; margin-bottom: 20px; }

.cards {
    display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap;
}
.card {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 24px; flex: 1; min-width: 140px;
    text-align: center;
}
.card-value { font-size: 28px; font-weight: 700; }
.card-label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

.section-title {
    font-size: 18px; font-weight: 600; margin-bottom: 12px;
}

table {
    width: 100%; border-collapse: collapse;
    background: var(--bg-secondary); border-radius: 8px; overflow: hidden;
    margin-bottom: 24px;
}
table.ablation { margin-bottom: 32px; }
thead { background: var(--bg-tertiary); }
th {
    padding: 10px 12px; text-align: left; font-size: 12px;
    color: var(--text-secondary); font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px;
}
td { padding: 10px 12px; border-top: 1px solid var(--border); font-size: 14px; }
tr:hover { background: var(--bg-tertiary); }

.config-label { font-weight: 600; color: var(--accent-blue); }
.picks { color: var(--text-secondary); font-size: 13px; }

.table-scroll { max-height: 600px; overflow-y: auto; border-radius: 8px; }
.table-scroll thead { position: sticky; top: 0; z-index: 1; }
"""
