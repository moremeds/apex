/**
 * Backtest Page — 8-tab strategy comparison interface.
 *
 * Tabs: Overview | Metrics | Per-Stock | Sector | Regime Perf | Regime Analysis | Symbol Map | Trades
 *
 * LC v5 charts imported from '../charts.js'; Plotly loaded eagerly via index.html script tag.
 * Data: strategies.json (comparison bundle or legacy flat format).
 */

import { createMultiLineChart, createAreaChart, createLineChart } from '../charts.js';

// ─── Types ───────────────────────────────────────────────────────────────────

interface PerSymbolMetric {
  sharpe: number;
  total_return: number;
  max_drawdown: number;
  win_rate: number;
  trade_count: number;
}

interface StrategyMetrics {
  name: string;
  tier: string;
  sharpe: number;
  sortino: number;
  calmar: number;
  total_return: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  trade_count: number;
  avg_trade_pnl: number;
  equity_curve: number[][];
  drawdown_curve: number[][];
  monthly_returns: Record<string, number>;
  per_symbol_sharpe: Record<string, number>;
  per_symbol_metrics: Record<string, PerSymbolMetric>;
  per_regime_sharpe: Record<string, number>;
  per_regime_return: Record<string, number>;
  per_regime_trades: Record<string, number>;
  per_regime_wr: Record<string, number>;
  rolling_sharpe: number[][];
  effective_params: number;
  total_params: number;
}

interface ComparisonBundle {
  title?: string;
  generated_at?: string;
  universe_name?: string;
  period?: string;
  strategy_count?: number;
  symbols?: string[];
  strategies?: Record<string, StrategyMetrics>;
  sector_map?: Record<string, string[]>;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const STRATEGY_COLORS = [
  '#58a6ff', '#3fb950', '#d29922', '#f85149',
  '#bc8cff', '#79c0ff', '#56d364', '#e3b341',
];

const TAB_NAMES = [
  'Overview', 'Metrics', 'Per-Stock', 'Sector',
  'Regime Perf', 'Regime Analysis', 'Symbol Map', 'Trades',
];

const PLOTLY_LAYOUT_BASE: Record<string, any> = {
  paper_bgcolor: '#1c2128',
  plot_bgcolor: '#1c2128',
  font: { color: '#8b949e' },
  legend: { bgcolor: 'transparent' },
  margin: { t: 40, r: 20, b: 40, l: 60 },
};

const REGIMES = ['R0', 'R1', 'R2', 'R3'];

// ─── Module State ────────────────────────────────────────────────────────────

let strategiesData: ComparisonBundle | null = null;
let strategies: Record<string, StrategyMetrics> = {};
let symbols: string[] = [];
let sectorMap: Record<string, string[]> = {};
let activeTab = 0;
let lcCharts: any[] = []; // track LC chart instances for cleanup

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtNum(val: number | null | undefined, decimals = 2): string {
  return val != null ? val.toFixed(decimals) : '\u2014';
}

function fmtPct(val: number | null | undefined): string {
  if (val == null) return '\u2014';
  return `${(val * 100).toFixed(1)}%`;
}

function colorForValue(val: number): string {
  return val >= 0 ? 'var(--positive, #3fb950)' : 'var(--negative, #f85149)';
}

function strategyColor(idx: number): string {
  return STRATEGY_COLORS[idx % STRATEGY_COLORS.length];
}

function strategyNames(): string[] {
  return Object.keys(strategies);
}

function getHealthBadge(m: StrategyMetrics): { label: string; cls: string } {
  const wr = m.win_rate ?? 0;
  const ret = m.total_return ?? 0;
  const mdd = m.max_drawdown ?? 0;
  const sharpe = m.sharpe ?? 0;
  const trades = m.trade_count ?? 0;
  const name = (m.name || '').toLowerCase();

  if (trades === 0 || wr < 0.15 || ret < -0.05 || mdd < -0.50) {
    return { label: 'BROKEN', cls: 'health-broken' };
  }
  if (wr < 0.30 || sharpe < 0.1 || ret < 0) {
    return { label: 'NEEDS WORK', cls: 'health-needs-work' };
  }
  if (name === 'buy_and_hold') {
    return { label: 'BASELINE', cls: 'health-baseline' };
  }
  return { label: 'HEALTHY', cls: 'health-healthy' };
}

function tierBadgeHTML(tier: string): string {
  if (!tier) return '';
  return `<span class="tier-badge">${tier}</span>`;
}

function destroyLCCharts(): void {
  for (const c of lcCharts) {
    try { c.chart?.remove(); } catch { /* already disposed */ }
  }
  lcCharts = [];
}

// ─── Tab Bar ─────────────────────────────────────────────────────────────────

function renderTabBar(container: HTMLElement): void {
  const tabBar = document.createElement('div');
  tabBar.className = 'comparison-tabs';

  TAB_NAMES.forEach((name, idx) => {
    const btn = document.createElement('button');
    btn.className = `tab-btn${idx === activeTab ? ' active' : ''}`;
    btn.textContent = name;
    btn.dataset.tab = String(idx);
    tabBar.appendChild(btn);
  });

  // ES module event binding -- no inline onclick
  tabBar.addEventListener('click', (e: Event) => {
    const target = e.target as HTMLElement;
    if (!target.classList.contains('tab-btn')) return;
    const tabIdx = parseInt(target.dataset.tab || '0', 10);
    if (tabIdx === activeTab) return;

    activeTab = tabIdx;
    tabBar.querySelectorAll('.tab-btn').forEach((btn, i) => {
      btn.classList.toggle('active', i === tabIdx);
    });

    renderActiveTab(container);
  });

  container.prepend(tabBar);
}

function renderActiveTab(container: HTMLElement): void {
  // Content area below tab bar
  let content = container.querySelector('.tab-content') as HTMLElement;
  if (!content) {
    content = document.createElement('div');
    content.className = 'tab-content';
    container.appendChild(content);
  }
  content.innerHTML = '';

  // Destroy any previous LC charts
  destroyLCCharts();

  switch (activeTab) {
    case 0: renderOverviewTab(content); break;
    case 1: renderMetricsTab(content); break;
    case 2: renderPerStockTab(content); break;
    case 3: renderSectorTab(content); break;
    case 4: renderRegimePerfTab(content); break;
    case 5: renderRegimeAnalysisTab(content); break;
    case 6: renderSymbolMapTab(content); break;
    case 7: renderTradesTab(content); break;
  }
}

// ─── Tab 1: Overview ─────────────────────────────────────────────────────────

function renderOverviewTab(container: HTMLElement): void {
  // Header info
  const bundle = strategiesData;
  const headerHTML = bundle?.title
    ? `<div style="margin-bottom:16px;color:var(--text-secondary, #8b949e);font-size:13px">
         ${bundle.title || ''} | ${bundle.universe_name || ''} | ${bundle.period || ''}
         ${bundle.generated_at ? ` | Generated: ${bundle.generated_at}` : ''}
       </div>`
    : '';

  // Scorecards
  const names = strategyNames();
  const scorecardHTML = names.map((name, idx) => {
    const m = strategies[name];
    const health = getHealthBadge(m);
    const color = strategyColor(idx);

    return `
      <div class="strategy-scorecard" style="border-left:3px solid ${color}">
        <div class="scorecard-header">
          <span class="strategy-name" style="color:${color}">${name}</span>
          <span class="health-badge ${health.cls}">${health.label}</span>
          ${tierBadgeHTML(m.tier)}
        </div>
        <div class="scorecard-metrics">
          <div class="metric"><span class="metric-label">Sharpe</span><span class="metric-value">${fmtNum(m.sharpe, 3)}</span></div>
          <div class="metric"><span class="metric-label">Return</span><span class="metric-value" style="color:${colorForValue(m.total_return || 0)}">${fmtPct(m.total_return)}</span></div>
          <div class="metric"><span class="metric-label">Max DD</span><span class="metric-value" style="color:var(--negative, #f85149)">${fmtPct(m.max_drawdown)}</span></div>
          <div class="metric"><span class="metric-label">Win Rate</span><span class="metric-value">${fmtPct(m.win_rate)}</span></div>
          <div class="metric"><span class="metric-label">Trades</span><span class="metric-value">${m.trade_count ?? '\u2014'}</span></div>
        </div>
      </div>`;
  }).join('');

  // Chart containers
  container.innerHTML = `
    ${headerHTML}
    <div class="scorecards-grid">${scorecardHTML}</div>
    <h4 style="margin-top:24px">Equity Curves</h4>
    <div id="bt-equity-chart" style="height:350px;margin-bottom:24px"></div>
    <h4>Drawdown</h4>
    <div id="bt-drawdown-chart" style="height:250px"></div>
  `;

  // Render equity curves with LC v5 multi-line
  renderEquityCurves();
  renderDrawdownCurves();
}

function renderEquityCurves(): void {
  const el = document.getElementById('bt-equity-chart');
  if (!el) return;

  const names = strategyNames();
  const seriesList: { name: string; data: any[]; color: string }[] = [];

  names.forEach((name, idx) => {
    const curve = strategies[name].equity_curve;
    if (!curve || curve.length === 0) return;

    seriesList.push({
      name,
      color: strategyColor(idx),
      data: curve.map(pt => ({
        time: pt[0] as number,
        value: pt[1] as number,
      })),
    });
  });

  if (seriesList.length === 0) {
    el.innerHTML = '<div class="empty-state">No equity curve data</div>';
    return;
  }

  const result = createMultiLineChart(el, seriesList);
  lcCharts.push(result);
}

function renderDrawdownCurves(): void {
  const el = document.getElementById('bt-drawdown-chart');
  if (!el) return;

  const names = strategyNames();
  const seriesList: { name: string; data: any[]; color: string }[] = [];

  names.forEach((name, idx) => {
    const curve = strategies[name].drawdown_curve;
    if (!curve || curve.length === 0) return;

    seriesList.push({
      name,
      color: strategyColor(idx),
      data: curve.map(pt => ({
        time: pt[0] as number,
        value: pt[1] as number,
      })),
    });
  });

  if (seriesList.length === 0) {
    el.innerHTML = '<div class="empty-state">No drawdown data</div>';
    return;
  }

  const result = createAreaChart(el, seriesList);
  lcCharts.push(result);
}

// ─── Tab 2: Metrics Table ────────────────────────────────────────────────────

function renderMetricsTab(container: HTMLElement): void {
  const names = strategyNames();
  if (names.length === 0) {
    container.innerHTML = '<div class="empty-state">No strategy data</div>';
    return;
  }

  // Find best Sharpe
  let bestSharpe = -Infinity;
  let bestSharpeName = '';
  for (const name of names) {
    const s = strategies[name].sharpe ?? -Infinity;
    if (s > bestSharpe) {
      bestSharpe = s;
      bestSharpeName = name;
    }
  }

  const metricKeys: { key: string; label: string; fmt: (v: any) => string; colorize?: boolean }[] = [
    { key: 'sharpe', label: 'Sharpe', fmt: (v) => fmtNum(v, 3) },
    { key: 'sortino', label: 'Sortino', fmt: (v) => fmtNum(v, 3) },
    { key: 'calmar', label: 'Calmar', fmt: (v) => fmtNum(v, 3) },
    { key: 'total_return', label: 'Total Return', fmt: fmtPct, colorize: true },
    { key: 'max_drawdown', label: 'Max Drawdown', fmt: fmtPct },
    { key: 'win_rate', label: 'Win Rate', fmt: fmtPct },
    { key: 'profit_factor', label: 'Profit Factor', fmt: (v) => fmtNum(v, 2) },
    { key: 'trade_count', label: 'Trades', fmt: (v) => v != null ? String(v) : '\u2014' },
    { key: 'avg_trade_pnl', label: 'Avg Trade P&L', fmt: fmtPct, colorize: true },
    { key: 'effective_params', label: 'Eff Params', fmt: (v) => v != null ? String(v) : '\u2014' },
  ];

  const headerCells = names.map((name, idx) => {
    const color = strategyColor(idx);
    return `<th style="color:${color}">${name}</th>`;
  }).join('');

  const bodyRows = metricKeys.map(mk => {
    const cells = names.map(name => {
      const m = strategies[name] as any;
      const val = m[mk.key];
      const formatted = mk.fmt(val);
      let style = '';

      if (mk.key === 'sharpe' && name === bestSharpeName) {
        style = 'background:rgba(63,185,80,0.15);font-weight:700;color:#3fb950';
      } else if (mk.key === 'max_drawdown') {
        style = 'color:var(--negative, #f85149)';
      } else if (mk.colorize && val != null) {
        style = `color:${colorForValue(val)}`;
      }

      return `<td style="${style}">${formatted}</td>`;
    }).join('');

    return `<tr><td style="font-weight:600;white-space:nowrap">${mk.label}</td>${cells}</tr>`;
  }).join('');

  container.innerHTML = `
    <table class="signal-table" style="width:100%">
      <thead><tr><th>Metric</th>${headerCells}</tr></thead>
      <tbody>${bodyRows}</tbody>
    </table>`;
}

// ─── Tab 3: Per-Stock ────────────────────────────────────────────────────────

function renderPerStockTab(container: HTMLElement): void {
  const names = strategyNames();

  // Gather all symbols across strategies
  const allSymbols = new Set<string>();
  for (const name of names) {
    const psm = strategies[name].per_symbol_metrics || {};
    for (const sym of Object.keys(psm)) allSymbols.add(sym);
  }

  if (allSymbols.size === 0) {
    container.innerHTML = '<div class="empty-state">No per-symbol metrics data</div>';
    return;
  }

  const sortedSymbols = [...allSymbols].sort();

  // One table per strategy
  const tablesHTML = names.map((name, idx) => {
    const color = strategyColor(idx);
    const psm = strategies[name].per_symbol_metrics || {};

    const rows = sortedSymbols.map(sym => {
      const sm = psm[sym];
      if (!sm) return `<tr><td>${sym}</td><td colspan="5" style="color:var(--text-muted, #484f58)">\u2014</td></tr>`;

      return `
        <tr>
          <td style="font-weight:600">${sym}</td>
          <td>${fmtNum(sm.sharpe, 3)}</td>
          <td style="color:${colorForValue(sm.total_return || 0)}">${fmtPct(sm.total_return)}</td>
          <td style="color:var(--negative, #f85149)">${fmtPct(sm.max_drawdown)}</td>
          <td>${fmtPct(sm.win_rate)}</td>
          <td>${sm.trade_count ?? '\u2014'}</td>
        </tr>`;
    }).join('');

    return `
      <h4 style="color:${color};margin-top:20px">${name}</h4>
      <table class="signal-table" style="width:100%">
        <thead>
          <tr>
            <th>Symbol</th><th>Sharpe</th><th>Return</th>
            <th>Max DD</th><th>Win Rate</th><th>Trades</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
  }).join('');

  container.innerHTML = tablesHTML;
}

// ─── Tab 4: Sector ───────────────────────────────────────────────────────────

function renderSectorTab(container: HTMLElement): void {
  if (!window.Plotly) {
    container.innerHTML = '<div class="empty-state">Plotly not loaded</div>';
    return;
  }

  const sectorNames = Object.keys(sectorMap);
  if (sectorNames.length === 0) {
    container.innerHTML = '<div class="empty-state">No sector data available</div>';
    return;
  }

  container.innerHTML = `
    <h4>Sector Sharpe Heatmap</h4>
    <div id="bt-sector-sharpe" style="height:400px;margin-bottom:24px"></div>
    <h4>Sector Return Heatmap</h4>
    <div id="bt-sector-return" style="height:400px"></div>
  `;

  const names = strategyNames();

  // Compute sector-level metrics by averaging per-symbol data
  const sharpeZ: number[][] = [];
  const returnZ: number[][] = [];

  for (const name of names) {
    const sharpeRow: number[] = [];
    const returnRow: number[] = [];

    for (const sector of sectorNames) {
      const sectorSymbols = sectorMap[sector] || [];
      const psm = strategies[name].per_symbol_metrics || {};
      const pss = strategies[name].per_symbol_sharpe || {};

      let sharpeSum = 0, sharpeCount = 0;
      let returnSum = 0, returnCount = 0;

      for (const sym of sectorSymbols) {
        if (psm[sym]) {
          if (psm[sym].sharpe != null) { sharpeSum += psm[sym].sharpe; sharpeCount++; }
          if (psm[sym].total_return != null) { returnSum += psm[sym].total_return; returnCount++; }
        } else if (pss[sym] != null) {
          sharpeSum += pss[sym]; sharpeCount++;
        }
      }

      sharpeRow.push(sharpeCount > 0 ? sharpeSum / sharpeCount : 0);
      returnRow.push(returnCount > 0 ? (returnSum / returnCount) * 100 : 0);
    }

    sharpeZ.push(sharpeRow);
    returnZ.push(returnRow);
  }

  // Sharpe heatmap
  const sharpeEl = document.getElementById('bt-sector-sharpe');
  if (sharpeEl) {
    window.Plotly.newPlot(sharpeEl, [{
      z: sharpeZ,
      x: sectorNames,
      y: names,
      type: 'heatmap',
      colorscale: [[0, '#f85149'], [0.5, '#d29922'], [1, '#3fb950']],
      hoverongaps: false,
      text: sharpeZ.map(row => row.map(v => v.toFixed(2))),
      texttemplate: '%{text}',
    }], {
      ...PLOTLY_LAYOUT_BASE,
      title: { text: 'Avg Sharpe by Sector', font: { color: '#8b949e', size: 14 } },
      xaxis: { tickangle: -45 },
    }, { responsive: true });
  }

  // Return heatmap
  const returnEl = document.getElementById('bt-sector-return');
  if (returnEl) {
    window.Plotly.newPlot(returnEl, [{
      z: returnZ,
      x: sectorNames,
      y: names,
      type: 'heatmap',
      colorscale: [[0, '#f85149'], [0.5, '#1c2128'], [1, '#3fb950']],
      zmid: 0,
      hoverongaps: false,
      text: returnZ.map(row => row.map(v => `${v.toFixed(1)}%`)),
      texttemplate: '%{text}',
    }], {
      ...PLOTLY_LAYOUT_BASE,
      title: { text: 'Avg Return by Sector (%)', font: { color: '#8b949e', size: 14 } },
      xaxis: { tickangle: -45 },
    }, { responsive: true });
  }
}

// ─── Tab 5: Regime Performance ───────────────────────────────────────────────

function renderRegimePerfTab(container: HTMLElement): void {
  if (!window.Plotly) {
    container.innerHTML = '<div class="empty-state">Plotly not loaded</div>';
    return;
  }

  container.innerHTML = `
    <h4>Sharpe by Regime</h4>
    <div id="bt-regime-sharpe" style="height:350px;margin-bottom:24px"></div>
    <h4>Return by Regime</h4>
    <div id="bt-regime-return" style="height:350px"></div>
  `;

  const names = strategyNames();

  // Sharpe grouped bar
  const sharpeEl = document.getElementById('bt-regime-sharpe');
  if (sharpeEl) {
    const traces = names.map((name, idx) => ({
      x: REGIMES,
      y: REGIMES.map(r => strategies[name].per_regime_sharpe?.[r] ?? 0),
      name,
      type: 'bar' as const,
      marker: { color: strategyColor(idx) },
    }));

    window.Plotly.newPlot(sharpeEl, traces, {
      ...PLOTLY_LAYOUT_BASE,
      barmode: 'group',
      title: { text: 'Sharpe Ratio by Regime', font: { color: '#8b949e', size: 14 } },
      xaxis: { title: 'Regime' },
      yaxis: { title: 'Sharpe' },
    }, { responsive: true });
  }

  // Return grouped bar
  const returnEl = document.getElementById('bt-regime-return');
  if (returnEl) {
    const traces = names.map((name, idx) => ({
      x: REGIMES,
      y: REGIMES.map(r => {
        const v = strategies[name].per_regime_return?.[r];
        return v != null ? v * 100 : 0;
      }),
      name,
      type: 'bar' as const,
      marker: { color: strategyColor(idx) },
    }));

    window.Plotly.newPlot(returnEl, traces, {
      ...PLOTLY_LAYOUT_BASE,
      barmode: 'group',
      title: { text: 'Return by Regime (%)', font: { color: '#8b949e', size: 14 } },
      xaxis: { title: 'Regime' },
      yaxis: { title: 'Return (%)' },
    }, { responsive: true });
  }
}

// ─── Tab 6: Regime Analysis ──────────────────────────────────────────────────

function renderRegimeAnalysisTab(container: HTMLElement): void {
  if (!window.Plotly) {
    container.innerHTML = '<div class="empty-state">Plotly not loaded</div>';
    return;
  }

  container.innerHTML = `
    <h4>Trade Count by Regime</h4>
    <div id="bt-regime-trades" style="height:350px;margin-bottom:24px"></div>
    <h4>Win Rate by Regime</h4>
    <div id="bt-regime-wr" style="height:350px;margin-bottom:24px"></div>
    <div id="bt-regime-alerts"></div>
  `;

  const names = strategyNames();

  // Trade count grouped bar
  const tradesEl = document.getElementById('bt-regime-trades');
  if (tradesEl) {
    const traces = names.map((name, idx) => ({
      x: REGIMES,
      y: REGIMES.map(r => strategies[name].per_regime_trades?.[r] ?? 0),
      name,
      type: 'bar' as const,
      marker: { color: strategyColor(idx) },
    }));

    window.Plotly.newPlot(tradesEl, traces, {
      ...PLOTLY_LAYOUT_BASE,
      barmode: 'group',
      title: { text: 'Trade Count by Regime', font: { color: '#8b949e', size: 14 } },
      xaxis: { title: 'Regime' },
      yaxis: { title: 'Trades' },
    }, { responsive: true });
  }

  // Win rate grouped bar with 50% reference line
  const wrEl = document.getElementById('bt-regime-wr');
  if (wrEl) {
    const traces: any[] = names.map((name, idx) => ({
      x: REGIMES,
      y: REGIMES.map(r => {
        const v = strategies[name].per_regime_wr?.[r];
        return v != null ? v * 100 : 0;
      }),
      name,
      type: 'bar' as const,
      marker: { color: strategyColor(idx) },
    }));

    const shapes = [{
      type: 'line',
      x0: -0.5,
      x1: REGIMES.length - 0.5,
      y0: 50,
      y1: 50,
      line: { color: '#8b949e', width: 1, dash: 'dash' },
    }];

    window.Plotly.newPlot(wrEl, traces, {
      ...PLOTLY_LAYOUT_BASE,
      barmode: 'group',
      title: { text: 'Win Rate by Regime (%)', font: { color: '#8b949e', size: 14 } },
      xaxis: { title: 'Regime' },
      yaxis: { title: 'Win Rate (%)' },
      shapes,
    }, { responsive: true });
  }

  // Concentration alerts (>70% trades in one regime)
  const alertsEl = document.getElementById('bt-regime-alerts');
  if (alertsEl) {
    const alerts: string[] = [];

    for (const name of names) {
      const regimeTrades = strategies[name].per_regime_trades || {};
      const total = Object.values(regimeTrades).reduce((a, b) => a + b, 0);
      if (total === 0) continue;

      for (const r of REGIMES) {
        const count = regimeTrades[r] || 0;
        const pct = (count / total) * 100;
        if (pct > 70) {
          alerts.push(
            `<div class="alert-item alert-sell" style="margin-bottom:8px">
               <strong>${name}</strong>: ${pct.toFixed(0)}% of trades concentrated in ${r}
             </div>`
          );
        }
      }
    }

    if (alerts.length > 0) {
      alertsEl.innerHTML = `
        <h4 style="color:#d29922">Concentration Alerts</h4>
        ${alerts.join('')}`;
    }
  }
}

// ─── Tab 7: Symbol Map ───────────────────────────────────────────────────────

function renderSymbolMapTab(container: HTMLElement): void {
  if (!window.Plotly) {
    container.innerHTML = '<div class="empty-state">Plotly not loaded</div>';
    return;
  }

  const names = strategyNames();

  // Gather all symbols
  const allSymbols = new Set<string>();
  for (const name of names) {
    const pss = strategies[name].per_symbol_sharpe || {};
    const psm = strategies[name].per_symbol_metrics || {};
    for (const sym of Object.keys(pss)) allSymbols.add(sym);
    for (const sym of Object.keys(psm)) allSymbols.add(sym);
  }
  // Also add from bundle symbols
  for (const sym of symbols) allSymbols.add(sym);

  const sortedSymbols = [...allSymbols].sort();
  if (sortedSymbols.length === 0) {
    container.innerHTML = '<div class="empty-state">No per-symbol data</div>';
    return;
  }

  container.innerHTML = `
    <h4>Symbol Sharpe Heatmap</h4>
    <div id="bt-symbol-heatmap" style="height:${Math.max(400, names.length * 40 + 100)}px"></div>
  `;

  const z: number[][] = [];
  const textArr: string[][] = [];

  for (const name of names) {
    const row: number[] = [];
    const textRow: string[] = [];
    const pss = strategies[name].per_symbol_sharpe || {};
    const psm = strategies[name].per_symbol_metrics || {};

    for (const sym of sortedSymbols) {
      const val = psm[sym]?.sharpe ?? pss[sym] ?? null;
      row.push(val ?? 0);
      textRow.push(val != null ? val.toFixed(2) : '\u2014');
    }

    z.push(row);
    textArr.push(textRow);
  }

  const el = document.getElementById('bt-symbol-heatmap');
  if (el) {
    window.Plotly.newPlot(el, [{
      z,
      x: sortedSymbols,
      y: names,
      text: textArr,
      texttemplate: '%{text}',
      type: 'heatmap',
      colorscale: [[0, '#f85149'], [0.5, '#d29922'], [1, '#3fb950']],
      hoverongaps: false,
    }], {
      ...PLOTLY_LAYOUT_BASE,
      title: { text: 'Per-Symbol Sharpe Ratio', font: { color: '#8b949e', size: 14 } },
      xaxis: { tickangle: -45 },
    }, { responsive: true });
  }
}

// ─── Tab 8: Trade Analysis ───────────────────────────────────────────────────

function renderTradesTab(container: HTMLElement): void {
  container.innerHTML = `
    <h4>Rolling 60-Day Sharpe</h4>
    <div id="bt-rolling-sharpe" style="height:350px;margin-bottom:24px"></div>
    <h4>Monthly Returns Heatmap</h4>
    <div id="bt-monthly-heatmap" style="height:400px"></div>
  `;

  renderRollingSharpe();
  renderMonthlyReturnsHeatmap();
}

function renderRollingSharpe(): void {
  const el = document.getElementById('bt-rolling-sharpe');
  if (!el) return;

  const names = strategyNames();
  const seriesList: { name: string; data: any[]; color: string }[] = [];

  names.forEach((name, idx) => {
    const rs = strategies[name].rolling_sharpe;
    if (!rs || rs.length === 0) return;

    seriesList.push({
      name,
      color: strategyColor(idx),
      data: rs.map(pt => ({
        time: pt[0] as number,
        value: pt[1] as number,
      })),
    });
  });

  if (seriesList.length === 0) {
    el.innerHTML = '<div class="empty-state">No rolling Sharpe data</div>';
    return;
  }

  if (seriesList.length === 1) {
    // Single strategy: use createLineChart with zero reference line
    const result = createLineChart(el, seriesList[0].data, {
      color: seriesList[0].color,
      lineWidth: 2,
    });
    result.series.createPriceLine({
      price: 0,
      color: 'rgba(139,148,158,0.5)',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: false,
    });
    lcCharts.push(result);
  } else {
    // Multiple strategies: use createMultiLineChart
    // Add a transparent "zero" series for the reference line
    const zeroData = seriesList[0].data.map((pt: any) => ({
      time: pt.time,
      value: 0,
    }));
    const allSeries = [
      ...seriesList,
      { name: 'Zero', color: 'rgba(139,148,158,0.5)', data: zeroData },
    ];
    const result = createMultiLineChart(el, allSeries);
    lcCharts.push(result);
  }
}

function renderMonthlyReturnsHeatmap(): void {
  const el = document.getElementById('bt-monthly-heatmap');
  if (!el || !window.Plotly) {
    if (el) el.innerHTML = '<div class="empty-state">Plotly not loaded</div>';
    return;
  }

  const names = strategyNames();
  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  // Collect all months across all strategies
  const allMonths = new Set<string>();
  for (const name of names) {
    const mr = strategies[name].monthly_returns || {};
    for (const key of Object.keys(mr)) allMonths.add(key);
  }

  if (allMonths.size === 0) {
    el.innerHTML = '<div class="empty-state">No monthly returns data</div>';
    return;
  }

  const sortedMonths = [...allMonths].sort();

  // Build x-axis labels as "Mon YYYY"
  const xLabels = sortedMonths.map(m => {
    const [year, mon] = m.split('-');
    const monIdx = parseInt(mon, 10) - 1;
    return `${monthNames[monIdx] || mon} ${year}`;
  });

  // Build z-matrix: rows = strategies, cols = months
  const z: (number | null)[][] = [];
  const textArr: string[][] = [];

  for (const name of names) {
    const mr = strategies[name].monthly_returns || {};
    const row: (number | null)[] = [];
    const textRow: string[] = [];

    for (const month of sortedMonths) {
      const val = mr[month];
      if (val != null) {
        row.push(val * 100);
        textRow.push(`${(val * 100).toFixed(1)}%`);
      } else {
        row.push(null);
        textRow.push('');
      }
    }

    z.push(row);
    textArr.push(textRow);
  }

  window.Plotly.newPlot(el, [{
    z,
    x: xLabels,
    y: names,
    text: textArr,
    texttemplate: '%{text}',
    type: 'heatmap',
    colorscale: [
      [0, '#f85149'],
      [0.5, '#1c2128'],
      [1, '#3fb950'],
    ],
    zmid: 0,
    hoverongaps: false,
  }], {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: 'Monthly Returns (%)', font: { color: '#8b949e', size: 14 } },
    xaxis: { tickangle: -45 },
  }, { responsive: true });
}

// ─── Page Lifecycle ──────────────────────────────────────────────────────────

export async function init(apex: any): Promise<void> {
  // Load strategies data
  try {
    const resp = await fetch('data/strategies.json');
    if (resp.ok) {
      strategiesData = await resp.json();
    }
  } catch {
    // Graceful fallback
  }

  // Parse bundle (handle both new and old format)
  if (strategiesData) {
    const bundle = strategiesData as any;
    strategies = bundle.strategies || bundle;
    symbols = bundle.symbols || [];
    sectorMap = bundle.sector_map || {};
  }

  if (!strategies || Object.keys(strategies).length === 0) {
    const emptyEl = document.getElementById('backtest-empty');
    if (emptyEl) emptyEl.style.display = 'block';
    return;
  }

  // Hide empty state
  const emptyEl = document.getElementById('backtest-empty');
  if (emptyEl) emptyEl.style.display = 'none';

  // Render into #backtest-content
  const container = document.getElementById('backtest-content');
  if (!container) return;
  container.innerHTML = '';

  activeTab = 0;
  renderTabBar(container);
  renderActiveTab(container);
}

export function update(): void {
  // Charts persist; re-render only if tab changed
}
