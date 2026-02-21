let momentumTable = null;
let peadTable = null;
let screenersData = null;
function qualityBadge(val) {
  const cls = (val || "").toLowerCase();
  if (cls === "strong") return `<span class="quality-badge strong">${val}</span>`;
  if (cls === "moderate") return `<span class="quality-badge moderate">${val}</span>`;
  return `<span class="quality-badge marginal">${val || "\u2014"}</span>`;
}
function regimeBadge(val) {
  if (!val) return "\u2014";
  const rc = val.toLowerCase();
  return `<span class="regime-badge ${rc}">${val}</span>`;
}
function qualityFormatter(cell) {
  return qualityBadge(cell.getValue());
}
function regimeFormatter(cell) {
  return regimeBadge(cell.getValue());
}
function symbolFormatter(cell) {
  const sym = cell.getValue();
  return `<a href="#/signals?symbol=${sym}&tf=1d" style="color:var(--accent-primary);text-decoration:none;font-weight:600">${sym}</a>`;
}
function pctFormatter(cell) {
  const val = cell.getValue();
  if (val == null) return "\u2014";
  return `${(val * 100).toFixed(1)}%`;
}
function dollarFormatter(cell) {
  const val = cell.getValue();
  if (val == null) return "\u2014";
  return `$${val.toFixed(2)}`;
}
function bigNumberFormatter(cell) {
  const val = cell.getValue();
  if (val == null) return "\u2014";
  if (val >= 1e9) return `$${(val / 1e9).toFixed(1)}B`;
  if (val >= 1e6) return `$${(val / 1e6).toFixed(0)}M`;
  return `$${val.toLocaleString()}`;
}
function floatFormatter(decimals) {
  return (cell) => {
    const v = cell.getValue();
    return v != null ? v.toFixed(decimals) : "\u2014";
  };
}
function earnRetFormatter(cell) {
  const v = cell.getValue();
  if (v == null) return "\u2014";
  const cls = v >= 0 ? "positive" : "negative";
  return `<span style="color:var(--${cls})">${v >= 0 ? "+" : ""}${v.toFixed(2)}%</span>`;
}
const MOMENTUM_COLUMNS = [
  { title: "#", field: "rank", width: 55, hozAlign: "center" },
  { title: "Symbol", field: "symbol", formatter: symbolFormatter, formatterParams: { html: true }, width: 80 },
  { title: "Mom 12-1", field: "momentum_12_1", formatter: pctFormatter, hozAlign: "right", width: 90 },
  { title: "FIP", field: "fip", hozAlign: "right", width: 70, formatter: floatFormatter(3) },
  { title: "Composite", field: "composite_rank", formatter: pctFormatter, hozAlign: "right", width: 95 },
  { title: "Quality", field: "quality_label", formatter: qualityFormatter, formatterParams: { html: true }, width: 90 },
  { title: "Regime", field: "regime", formatter: regimeFormatter, formatterParams: { html: true }, width: 75 },
  { title: "Price", field: "last_close", formatter: dollarFormatter, hozAlign: "right", width: 80 },
  { title: "Mkt Cap", field: "market_cap", formatter: bigNumberFormatter, hozAlign: "right", width: 90 },
  { title: "ADDV", field: "avg_daily_dollar_volume", formatter: bigNumberFormatter, hozAlign: "right", width: 90 },
  { title: "Liquidity", field: "liquidity_tier", width: 85 },
  { title: "Size", field: "position_size_factor", hozAlign: "right", width: 70, formatter: floatFormatter(2) }
];
const PEAD_COLUMNS = [
  { title: "Symbol", field: "symbol", formatter: symbolFormatter, formatterParams: { html: true }, width: 80 },
  { title: "Report", field: "report_date", width: 100 },
  { title: "SUE", field: "sue_score", hozAlign: "right", width: 75, formatter: floatFormatter(2) },
  { title: "MQ-SUE", field: "mq_sue", hozAlign: "right", width: 80, formatter: floatFormatter(2) },
  { title: "Earn Ret", field: "earnings_day_return", hozAlign: "right", width: 90, formatter: earnRetFormatter, formatterParams: { html: true } },
  { title: "Vol", field: "earnings_day_volume_ratio", hozAlign: "right", width: 70, formatter: (c) => {
    const v = c.getValue();
    return v != null ? v.toFixed(1) + "x" : "\u2014";
  } },
  { title: "Rev", field: "revenue_surprise_pct", hozAlign: "right", width: 70, formatter: pctFormatter },
  { title: "Quality", field: "quality_label", formatter: qualityFormatter, formatterParams: { html: true }, width: 90 },
  { title: "Gap Held", field: "gap_held", hozAlign: "center", width: 80, formatter: (c) => {
    const v = c.getValue();
    return v ? '<span style="color:var(--positive)">Yes</span>' : '<span style="color:var(--negative)">No</span>';
  }, formatterParams: { html: true } },
  { title: "Entry", field: "entry_price", formatter: dollarFormatter, hozAlign: "right", width: 80 },
  { title: "Stop %", field: "stop_loss_pct", hozAlign: "right", width: 75, formatter: (c) => {
    const v = c.getValue();
    return v != null ? v.toFixed(1) + "%" : "\u2014";
  } },
  { title: "Target %", field: "profit_target_pct", hozAlign: "right", width: 80, formatter: (c) => {
    const v = c.getValue();
    return v != null ? v.toFixed(1) + "%" : "\u2014";
  } },
  { title: "Max Days", field: "max_hold_days", hozAlign: "center", width: 80 }
];
function setupTabs() {
  document.querySelectorAll("#page-screeners .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll("#page-screeners .tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
      document.querySelectorAll("#page-screeners .screener-panel").forEach((p) => p.classList.toggle("active", p.id === `screener-${tab}`));
    });
  });
}
function addExportButton(containerId, table, label) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const btn = document.createElement("button");
  btn.className = "tf-btn";
  btn.style.marginBottom = "8px";
  btn.textContent = `Export ${label} CSV`;
  btn.addEventListener("click", () => table.download("csv", `${label.toLowerCase()}_screener.csv`));
  container.parentElement?.insertBefore(btn, container);
}
function renderMomentumSummary(data) {
  const container = document.getElementById("momentum-summary");
  if (!container) return;
  const candidates = data?.candidates || [];
  const total = candidates.length;
  const universe = data?.universe_size || data?.metadata?.universe_size || "\u2014";
  const topN = data?.top_n || data?.metadata?.top_n || total;
  const avgMom = total > 0 ? (candidates.reduce((s, c) => s + (c.momentum_12_1 || 0), 0) / total * 100).toFixed(1) + "%" : "\u2014";
  const avgFip = total > 0 ? (candidates.reduce((s, c) => s + (c.fip || 0), 0) / total).toFixed(3) : "\u2014";
  container.innerHTML = `
    <div class="metric-card"><div class="metric-label">Universe</div><div class="metric-value">${universe}</div></div>
    <div class="metric-card"><div class="metric-label">Passed Filters</div><div class="metric-value">${total}</div></div>
    <div class="metric-card"><div class="metric-label">Top-N Picks</div><div class="metric-value">${topN}</div></div>
    <div class="metric-card"><div class="metric-label">Avg Momentum</div><div class="metric-value">${avgMom}</div></div>
    <div class="metric-card"><div class="metric-label">Avg FIP</div><div class="metric-value">${avgFip}</div></div>
  `;
}
function renderPeadSummary(data) {
  const container = document.getElementById("pead-summary");
  if (!container) return;
  const candidates = data?.candidates || [];
  const total = candidates.length;
  const qualityCounts = { STRONG: 0, MODERATE: 0, MARGINAL: 0 };
  candidates.forEach((c) => {
    const q = (c.quality_label || "").toUpperCase();
    if (q in qualityCounts) qualityCounts[q]++;
  });
  const avgQuality = total > 0 ? `${qualityCounts.STRONG}S / ${qualityCounts.MODERATE}M / ${qualityCounts.MARGINAL}W` : "\u2014";
  const regimeCounts = {};
  candidates.forEach((c) => {
    const r = c.regime || "N/A";
    regimeCounts[r] = (regimeCounts[r] || 0) + 1;
  });
  const dominantRegime = Object.entries(regimeCounts).sort((a, b) => b[1] - a[1])[0];
  const regimeStr = dominantRegime ? `${dominantRegime[0]} (${dominantRegime[1]})` : "\u2014";
  container.innerHTML = `
    <div class="metric-card"><div class="metric-label">Candidates</div><div class="metric-value">${total}</div></div>
    <div class="metric-card"><div class="metric-label">Quality Mix</div><div class="metric-value" style="font-size:14px">${avgQuality}</div></div>
    <div class="metric-card"><div class="metric-label">Dominant Regime</div><div class="metric-value">${regimeStr}</div></div>
  `;
}
function renderMomentumTable(data) {
  const candidates = data?.candidates || [];
  const emptyEl = document.getElementById("momentum-empty");
  const tableEl = document.getElementById("momentum-table");
  if (candidates.length === 0) {
    if (emptyEl) emptyEl.style.display = "block";
    if (tableEl) tableEl.style.display = "none";
    return;
  }
  if (emptyEl) emptyEl.style.display = "none";
  if (tableEl) tableEl.style.display = "block";
  const ranked = candidates.map((c, i) => ({ rank: i + 1, ...c }));
  momentumTable = new window.Tabulator("#momentum-table", {
    data: ranked,
    columns: MOMENTUM_COLUMNS,
    layout: "fitColumns",
    height: "500px",
    placeholder: "No momentum candidates",
    headerSort: true,
    initialSort: [{ column: "rank", dir: "asc" }]
  });
  addExportButton("momentum-table", momentumTable, "Momentum");
}
function renderPeadTable(data) {
  const candidates = data?.candidates || [];
  const emptyEl = document.getElementById("pead-empty");
  const tableEl = document.getElementById("pead-table");
  if (candidates.length === 0) {
    if (emptyEl) emptyEl.style.display = "block";
    if (tableEl) tableEl.style.display = "none";
    return;
  }
  if (emptyEl) emptyEl.style.display = "none";
  if (tableEl) tableEl.style.display = "block";
  peadTable = new window.Tabulator("#pead-table", {
    data: candidates,
    columns: PEAD_COLUMNS,
    layout: "fitColumns",
    height: "500px",
    placeholder: "No PEAD candidates",
    headerSort: true
  });
  addExportButton("pead-table", peadTable, "PEAD");
}
function renderPeadMethodology() {
  const container = document.getElementById("pead-methodology");
  if (!container) return;
  container.innerHTML = `
    <details class="section-header">
      <summary style="cursor:pointer;padding:12px 16px;font-weight:600;color:var(--text-secondary)">
        Methodology &amp; Definitions
      </summary>
      <div style="padding:8px 16px 16px;color:var(--text-secondary);font-size:13px;line-height:1.6">
        <p><strong>Post-Earnings Announcement Drift (PEAD)</strong> exploits the tendency for stock prices
        to continue drifting in the direction of an earnings surprise for weeks after the announcement.</p>
        <p><strong>SUE Score</strong>: Standardized Unexpected Earnings \u2014 measures how far actual EPS
        deviated from consensus, normalized by historical forecast error.</p>
        <p><strong>MQ-SUE</strong>: Multi-Quarter SUE \u2014 weighted average of recent quarters' surprises.</p>
        <p><strong>Quality Label</strong>: STRONG (high SUE + high volume + gap held), MODERATE (partial signals),
        MARGINAL (weak confirmation).</p>
        <p><strong>Gap Held</strong>: Whether the post-earnings gap held through the first few trading sessions.</p>
      </div>
    </details>`;
}
async function init(_apex) {
  setupTabs();
  try {
    const resp = await fetch("data/screeners.json");
    if (resp.ok) {
      screenersData = await resp.json();
    }
  } catch {
  }
  if (screenersData) {
    if (screenersData.momentum) {
      renderMomentumSummary(screenersData.momentum);
      renderMomentumTable(screenersData.momentum);
    }
    if (screenersData.pead) {
      renderPeadSummary(screenersData.pead);
      renderPeadTable(screenersData.pead);
    }
  }
  if (!screenersData?.momentum) {
    const el = document.getElementById("momentum-empty");
    if (el) el.style.display = "block";
  }
  if (!screenersData?.pead) {
    const el = document.getElementById("pead-empty");
    if (el) el.style.display = "block";
  }
  renderPeadMethodology();
}
function update() {
}
export {
  init,
  update
};
