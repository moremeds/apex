import { createCandlestickChart } from "../charts.js";
let monitorTable = null;
let qualityData = null;
let universeData = null;
function statusIcon(status) {
  const map = {
    PASS: '<span class="monitor-status pass" title="PASS">\u25CF</span>',
    WARN: '<span class="monitor-status warn" title="WARN">\u25D0</span>',
    CAUTION: '<span class="monitor-status caution" title="CAUTION">\u25D1</span>',
    FAIL: '<span class="monitor-status fail" title="FAIL">\u25C9</span>',
    EMPTY: '<span class="monitor-status empty" title="EMPTY">\u25CB</span>'
  };
  return map[status] || status;
}
function statusFormatter(cell) {
  return statusIcon(cell.getValue());
}
function coverageFormatter(cell) {
  const v = cell.getValue();
  if (v == null) return "\u2014";
  const cls = v >= 98 ? "pass" : v >= 95 ? "warn" : v >= 90 ? "caution" : "fail";
  return `<span class="monitor-cov ${cls}">${v.toFixed(1)}%</span>`;
}
function tierBadge(val) {
  const cls = val === "sp500" ? "sp500" : val === "nq100" ? "nq100" : "screener";
  return `<span class="tier-badge ${cls}">${val}</span>`;
}
function tierFormatter(cell) {
  return tierBadge(cell.getValue());
}
function symbolLinkFormatter(cell) {
  const sym = cell.getValue();
  return `<a href="#" class="monitor-sym-link" data-sym="${sym}" style="color:var(--accent-primary);text-decoration:none;font-weight:600">${sym} \u25B8</a>`;
}
function numberFormatter(cell) {
  const v = cell.getValue();
  if (v == null) return "\u2014";
  return v.toLocaleString();
}
const MONITOR_COLUMNS = [
  { title: "#", formatter: "rownum", width: 50, hozAlign: "center", headerSort: false },
  { title: "Symbol", field: "symbol", formatter: symbolLinkFormatter, formatterParams: { html: true }, width: 100 },
  { title: "Tier", field: "tier", formatter: tierFormatter, formatterParams: { html: true }, width: 90 },
  { title: "Sector", field: "sector", width: 110 },
  { title: "Bars", field: "bars", hozAlign: "right", width: 80, formatter: numberFormatter },
  { title: "Coverage", field: "coverage_pct", hozAlign: "right", width: 95, formatter: coverageFormatter, formatterParams: { html: true } },
  { title: "Gaps", field: "gaps", hozAlign: "right", width: 65 },
  { title: "Last Bar", field: "last_bar", width: 120 },
  { title: "Status", field: "status", formatter: statusFormatter, formatterParams: { html: true }, width: 70, hozAlign: "center" }
];
function renderSummary(entries) {
  const total = entries.length;
  const counts = { PASS: 0, WARN: 0, CAUTION: 0, FAIL: 0, EMPTY: 0 };
  for (const e of entries) counts[e.status] = (counts[e.status] || 0) + 1;
  return `<div class="monitor-summary">
    <span class="monitor-summary-total">${total.toLocaleString()} Entries</span>
    <span class="monitor-summary-item">${statusIcon("PASS")} ${counts.PASS}</span>
    <span class="monitor-summary-item">${statusIcon("WARN")} ${counts.WARN}</span>
    <span class="monitor-summary-item">${statusIcon("CAUTION")} ${counts.CAUTION}</span>
    <span class="monitor-summary-item">${statusIcon("FAIL")} ${counts.FAIL}</span>
    <span class="monitor-summary-item">${statusIcon("EMPTY")} ${counts.EMPTY}</span>
  </div>`;
}
function renderControls(entries) {
  const sectors = [...new Set(entries.map((e) => e.sector).filter(Boolean))].sort();
  return `<div class="monitor-controls">
    <input type="text" id="monitor-search" class="dash-input" placeholder="Search symbol...">
    <select id="monitor-tf-filter" class="dash-select">
      <option value="">All Timeframes</option>
      <option value="1d">1d</option>
      <option value="1w">1w</option>
      <option value="1h">1h</option>
      <option value="4h">4h</option>
    </select>
    <select id="monitor-tier-filter" class="dash-select">
      <option value="">All Tiers</option>
      <option value="sp500">SP500</option>
      <option value="nq100">NQ100</option>
      <option value="screener">Screener</option>
    </select>
    <select id="monitor-status-filter" class="dash-select">
      <option value="">All Status</option>
      <option value="PASS">PASS</option>
      <option value="WARN">WARN</option>
      <option value="CAUTION">CAUTION</option>
      <option value="FAIL">FAIL</option>
      <option value="EMPTY">EMPTY</option>
    </select>
    <select id="monitor-sector-filter" class="dash-select">
      <option value="">All Sectors</option>
      ${sectors.map((s) => `<option value="${s}">${s}</option>`).join("")}
    </select>
  </div>`;
}
function enrichEntries(entries) {
  if (!universeData?.tickers) return entries;
  const universeMap = {};
  for (const t of universeData.tickers) {
    universeMap[t.symbol] = t;
  }
  return entries.map((e) => {
    const u = universeMap[e.symbol];
    return {
      ...e,
      tier: u?.tier || "",
      sector: u?.sector || "",
      name: u?.name || "",
      industry: u?.industry || "",
      marketCap: u?.marketCap || 0,
      last_bar: e.last_bar ? e.last_bar.split(/[T ]/)[0] : ""
    };
  });
}
function renderOverview(container, entries) {
  const enriched = enrichEntries(entries);
  container.innerHTML = renderSummary(enriched) + renderControls(enriched) + '<div id="monitor-table"></div>';
  const tableEl = document.getElementById("monitor-table");
  if (!tableEl || typeof window.Tabulator === "undefined") return;
  monitorTable = new window.Tabulator(tableEl, {
    data: enriched,
    columns: MONITOR_COLUMNS,
    layout: "fitDataStretch",
    height: "calc(100vh - 220px)",
    pagination: true,
    paginationSize: 50,
    paginationSizeSelector: [25, 50, 100, 200],
    initialSort: [{ column: "coverage_pct", dir: "asc" }],
    placeholder: "No data available"
  });
  const search = document.getElementById("monitor-search");
  const tfFilter = document.getElementById("monitor-tf-filter");
  const tierFilter = document.getElementById("monitor-tier-filter");
  const statusFilter = document.getElementById("monitor-status-filter");
  const sectorFilter = document.getElementById("monitor-sector-filter");
  function applyFilters() {
    const filters = [];
    if (search?.value) filters.push({ field: "symbol", type: "like", value: search.value.toUpperCase() });
    if (tfFilter?.value) filters.push({ field: "timeframe", type: "=", value: tfFilter.value });
    if (tierFilter?.value) filters.push({ field: "tier", type: "=", value: tierFilter.value });
    if (statusFilter?.value) filters.push({ field: "status", type: "=", value: statusFilter.value });
    if (sectorFilter?.value) filters.push({ field: "sector", type: "=", value: sectorFilter.value });
    monitorTable?.setFilter(filters);
  }
  search?.addEventListener("input", applyFilters);
  tfFilter?.addEventListener("change", applyFilters);
  tierFilter?.addEventListener("change", applyFilters);
  statusFilter?.addEventListener("change", applyFilters);
  sectorFilter?.addEventListener("change", applyFilters);
  tableEl.addEventListener("click", (e) => {
    const link = e.target.closest(".monitor-sym-link");
    if (link) {
      e.preventDefault();
      const sym = link.dataset.sym;
      const tf = tfFilter?.value || "1d";
      window.APEX.navigateTo("monitor", { symbol: sym, tf });
    }
  });
}
async function renderDrillDown(container, symbol, tf) {
  const entries = qualityData?.entries || [];
  const entry = entries.find((e) => e.symbol === symbol && e.timeframe === tf) || entries.find((e) => e.symbol === symbol);
  const uTicker = universeData?.tickers?.find((t) => t.symbol === symbol);
  const symEntries = entries.filter((e) => e.symbol === symbol);
  const availTFs = [...new Set(symEntries.map((e) => e.timeframe))];
  const tierStr = uTicker?.tiers?.join("+") || uTicker?.tier || "";
  const capVal = Number(uTicker?.marketCap) || 0;
  const capStr = capVal >= 1e12 ? `$${(capVal / 1e12).toFixed(2)}T` : capVal >= 1e9 ? `$${(capVal / 1e9).toFixed(1)}B` : capVal >= 1e6 ? `$${(capVal / 1e6).toFixed(0)}M` : capVal > 0 ? `$${capVal.toLocaleString()}` : "";
  container.innerHTML = `
    <div class="monitor-detail-header">
      <button class="monitor-back-btn" id="monitor-back">\u2190 Back</button>
      <div class="monitor-symbol-title">${symbol} \u2014 ${uTicker?.name || ""}</div>
      <div class="monitor-symbol-meta">${tierStr} | ${uTicker?.sector || ""} | ${capStr}</div>
    </div>
    <div class="monitor-tf-tabs" id="monitor-detail-tfs">
      ${availTFs.map((t) => `<button class="tf-btn${t === tf ? " active" : ""}" data-tf="${t}">${t}</button>`).join("")}
    </div>
    <div class="monitor-cards" id="monitor-detail-cards"></div>
    <div class="monitor-chart-container" id="monitor-detail-chart"></div>
    <div class="monitor-detail-sections" id="monitor-detail-sections"></div>
  `;
  document.getElementById("monitor-back")?.addEventListener("click", () => {
    window.APEX.navigateTo("monitor", {});
  });
  document.getElementById("monitor-detail-tfs")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".tf-btn");
    if (btn) {
      window.APEX.navigateTo("monitor", { symbol, tf: btn.dataset.tf });
    }
  });
  const cardsEl = document.getElementById("monitor-detail-cards");
  if (cardsEl && entry) {
    cardsEl.innerHTML = `
      <div class="metric-card"><div class="metric-value">${statusIcon(entry.status)}</div><div class="metric-label">Status</div></div>
      <div class="metric-card"><div class="metric-value">${(entry.bars || 0).toLocaleString()}</div><div class="metric-label">Bars</div></div>
      <div class="metric-card"><div class="metric-value">${entry.gaps ?? 0}</div><div class="metric-label">Gaps</div></div>
      <div class="metric-card"><div class="metric-value">${entry.coverage_pct?.toFixed(1) ?? "0.0"}%</div><div class="metric-label">Coverage</div></div>
    `;
  }
  const chartEl = document.getElementById("monitor-detail-chart");
  if (chartEl) {
    try {
      const resp = await fetch(`data/${symbol}_${tf}.json`);
      if (resp.ok) {
        const symbolData = await resp.json();
        if (symbolData?.chart_data?.timestamps?.length) {
          createCandlestickChart(chartEl, symbolData.chart_data);
        } else {
          chartEl.innerHTML = '<div class="empty-state">No chart data available for this symbol/timeframe.</div>';
        }
      } else {
        chartEl.innerHTML = '<div class="empty-state">No signal data available. Coverage-only view.</div>';
      }
    } catch {
      chartEl.innerHTML = '<div class="empty-state">Chart data not available.</div>';
    }
  }
  const sectionsEl = document.getElementById("monitor-detail-sections");
  if (sectionsEl && entry) {
    let html = "";
    html += `<details class="collapsible-section">
      <summary>Bar Statistics</summary>
      <div class="section-content">
        <table class="detail-table">
          <tr><td>First Bar</td><td>${entry.first_bar || "\u2014"}</td></tr>
          <tr><td>Last Bar</td><td>${entry.last_bar || "\u2014"}</td></tr>
          <tr><td>Total Bars</td><td>${(entry.bars || 0).toLocaleString()}</td></tr>
          <tr><td>Coverage</td><td>${entry.coverage_pct?.toFixed(1) ?? "0.0"}%</td></tr>
          <tr><td>Anomalies</td><td>${entry.anomalies ?? 0}</td></tr>
        </table>
      </div>
    </details>`;
    if (entry.gaps > 0) {
      html += `<details class="collapsible-section">
        <summary>Gap Analysis (${entry.gaps} gaps)</summary>
        <div class="section-content">
          <p>Gap details available in full data_quality report.</p>
        </div>
      </details>`;
    }
    sectionsEl.innerHTML = html;
  }
}
async function init(apex, params) {
  const [qd, ud] = await Promise.all([
    fetch("data/data_quality.json").then((r) => r.ok ? r.json() : null).catch(() => null),
    fetch("data/universe.json").then((r) => r.ok ? r.json() : null).catch(() => null)
  ]);
  qualityData = qd;
  universeData = ud;
  const container = document.getElementById("monitor-overview");
  const detailContainer = document.getElementById("monitor-detail");
  if (!qualityData) {
    container.innerHTML = '<div class="empty-state">No data quality report available. Run the R2 pipeline first.</div>';
    return;
  }
  if (params.symbol) {
    container.style.display = "none";
    detailContainer.style.display = "block";
    await renderDrillDown(detailContainer, params.symbol, params.tf || "1d");
  } else {
    container.style.display = "block";
    detailContainer.style.display = "none";
    renderOverview(container, qualityData.entries || []);
  }
}
async function update(apex, params) {
  const container = document.getElementById("monitor-overview");
  const detailContainer = document.getElementById("monitor-detail");
  if (params.symbol) {
    container.style.display = "none";
    detailContainer.style.display = "block";
    await renderDrillDown(detailContainer, params.symbol, params.tf || "1d");
  } else {
    container.style.display = "block";
    detailContainer.style.display = "none";
    if (qualityData && !monitorTable) {
      renderOverview(container, qualityData.entries || []);
    }
  }
}
export {
  init,
  update
};
