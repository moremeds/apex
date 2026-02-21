import { createLineChart } from "../charts.js";
const REGIME_NAMES = {
  R0: "Healthy Uptrend",
  R1: "Choppy / Extended",
  R2: "Risk-Off",
  R3: "Rebound Window"
};
const REGIME_COLORS = {
  R0: "#22c55e",
  R1: "#f59e0b",
  R2: "#ef4444",
  R3: "#3b82f6"
};
const COMPONENT_LABELS = {
  trend_state: { label: "Trend", valueKey: "ma50_slope" },
  vol_state: { label: "Volatility", valueKey: "atr_pct_63" },
  chop_state: { label: "Choppiness", valueKey: "chop_pct_252" },
  ext_state: { label: "Extension", valueKey: "ext" },
  iv_state: { label: "IV", valueKey: "iv_pct_63" }
};
let timelineChart = null;
let activeFilter = "all";
function findBenchmark(tickers, preferred = ["SPY", "QQQ"]) {
  for (const sym of preferred) {
    const t = tickers.find((tt) => tt.symbol === sym);
    if (t) return t;
  }
  return tickers[0] || null;
}
function renderBadge(ticker) {
  const container = document.getElementById("regime-badge");
  if (!container || !ticker) return;
  const regime = ticker.regime || "R0";
  const rc = regime.toLowerCase();
  const name = REGIME_NAMES[regime] || regime;
  const confidence = ticker.confidence != null ? ticker.confidence : "\u2014";
  const score = ticker.composite_score_avg != null ? ticker.composite_score_avg.toFixed(1) : "\u2014";
  container.innerHTML = `
    <div class="badge-regime ${rc}">${regime} \u2014 ${name}</div>
    <div class="badge-details">
      ${ticker.symbol} | Confidence: ${typeof confidence === "number" ? confidence.toFixed(0) + "%" : confidence}
      | Composite Score: ${score}
    </div>`;
}
function renderComponents(ticker) {
  const container = document.getElementById("regime-components");
  if (!container || !ticker) return;
  const states = ticker.component_states || {};
  const values = ticker.component_values || {};
  container.innerHTML = Object.entries(COMPONENT_LABELS).filter(([key]) => states[key] !== void 0).map(([key, meta]) => {
    const state = states[key] || "\u2014";
    const val = values[meta.valueKey];
    const valStr = val != null ? Number(val).toFixed(1) : "\u2014";
    let stateColor = "var(--text-secondary)";
    const s = state.toLowerCase();
    if (s.includes("trend_up") || s.includes("trending") || s.includes("vol_normal") || s.includes("neutral"))
      stateColor = "var(--positive)";
    else if (s.includes("trend_down") || s.includes("choppy") || s.includes("vol_high") || s.includes("overbought"))
      stateColor = "var(--negative)";
    return `
        <div class="card">
          <div class="card-label">${meta.label}</div>
          <div class="card-value" style="font-size:16px;color:${stateColor}">${state}</div>
          <div class="card-sub">Value: ${valStr}</div>
        </div>`;
  }).join("");
}
async function renderTimeline(symbol) {
  const container = document.getElementById("regime-timeline");
  if (!container) return;
  let history = null;
  try {
    const resp = await fetch(`data/${symbol}_1d.json`);
    if (resp.ok) {
      const data = await resp.json();
      history = data.regime_flex_history;
    }
  } catch {
  }
  if (!history || history.length === 0) {
    container.innerHTML = '<div class="empty-state">No regime history data</div>';
    return;
  }
  const reversed = [...history].reverse();
  const lineData = reversed.map((row) => ({
    time: Math.floor(new Date(row.date).getTime() / 1e3),
    value: row.target_exposure
  }));
  if (timelineChart) {
    timelineChart.chart.remove();
    timelineChart = null;
  }
  timelineChart = createLineChart(container, lineData, {
    color: "#e07a3b",
    lineWidth: 2
  });
  const { series } = timelineChart;
  series.createPriceLine({ price: 80, color: "rgba(34,197,94,0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
  series.createPriceLine({ price: 40, color: "rgba(245,158,11,0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
  series.createPriceLine({ price: 20, color: "rgba(239,68,68,0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
}
function renderGrid(tickers) {
  const container = document.getElementById("regime-grid");
  if (!container) return;
  const filtered = activeFilter === "all" ? tickers : tickers.filter((t) => t.regime === activeFilter);
  container.innerHTML = filtered.map((t) => {
    const regime = t.regime || "R0";
    const rc = regime.toLowerCase();
    return `
      <div class="hm-cell ${rc}" data-symbol="${t.symbol}" title="${t.symbol} \u2014 ${REGIME_NAMES[regime] || regime}">
        <div class="cell-symbol">${t.symbol}</div>
        <div class="cell-change">${regime}</div>
      </div>`;
  }).join("");
  container.querySelectorAll(".hm-cell").forEach((cell) => {
    cell.addEventListener("click", () => {
      const sym = cell.dataset.symbol;
      if (sym) window.APEX.navigateTo("signals", { symbol: sym, tf: "1d" });
    });
  });
}
function renderFilters(tickers) {
  const container = document.getElementById("regime-filters");
  if (!container) return;
  const counts = { all: tickers.length, R0: 0, R1: 0, R2: 0, R3: 0 };
  tickers.forEach((t) => {
    const r = t.regime || "R0";
    if (r in counts) counts[r]++;
  });
  container.innerHTML = Object.entries(counts).map(([filter, count]) => `
    <button class="filter-btn ${filter === activeFilter ? "active" : ""}" data-filter="${filter}">
      ${filter === "all" ? "All" : filter} (${count})
    </button>`).join("");
  container.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeFilter = btn.dataset.filter || "all";
      container.querySelectorAll(".filter-btn").forEach((b) => b.classList.toggle("active", b.dataset.filter === activeFilter));
      renderGrid(tickers);
    });
  });
}
async function init(apex) {
  if (!apex.summary) return;
  const tickers = apex.summary.tickers || [];
  const benchmark = findBenchmark(tickers);
  if (benchmark) {
    renderBadge(benchmark);
    renderComponents(benchmark);
    await renderTimeline(benchmark.symbol);
  }
  renderFilters(tickers);
  renderGrid(tickers);
}
async function update(apex) {
  if (apex.summary?.tickers) {
    renderGrid(apex.summary.tickers);
  }
}
export {
  init,
  update
};
