/**
 * Overview Page — 5-category ETF dashboard, controls bar, stats bar,
 * Plotly treemap heatmap with dynamic color/size switching.
 *
 * Ported 1:1 from the original gh-pages heatmap:
 *   - etf_dashboard.py (card renderers, sparklines)
 *   - html_template.py (controls, stats, JS functions)
 *   - plotly_data.py   (treemap trace config, hover template)
 *   - model.py         (score gradient, ETF_CONFIG, color schemes)
 */

// ─── ETF Configuration (Single Source of Truth — mirrors model.py) ──────────

const ETF_CONFIG = {
  market_indices: { name: 'Market Indices', symbols: ['SPY', 'QQQ', 'IWM', 'DIA'], card_style: 'large' },
  commodities:    { name: 'Commodities & Safe Haven', symbols: ['GLD', 'SLV'], card_style: 'compact' },
  fixed_income:   { name: 'Fixed Income', symbols: ['TLT'], card_style: 'compact' },
  volatility:     { name: 'Volatility', symbols: ['UVXY'], card_style: 'compact' },
  sectors:        { name: 'Sector ETFs', symbols: ['XLK','XLF','XLV','XLP','XLE','XLI','XLB','XLRE','XLU','XLC','XLY','SMH'], card_style: 'mini' },
};

const ALL_DASHBOARD_ETFS = new Set(
  Object.values(ETF_CONFIG).flatMap(c => c.symbols)
);

const MARKET_ETF_NAMES = { SPY: 'S&P 500', QQQ: 'NASDAQ 100', IWM: 'Russell 2000', DIA: 'Dow Jones' };
const SECTOR_ETF_NAMES = { XLK:'Technology', XLC:'Communication', XLY:'Cons. Disc.', XLF:'Financials', XLV:'Healthcare', XLP:'Cons. Staples', XLE:'Energy', XLI:'Industrials', XLB:'Materials', XLRE:'Real Estate', XLU:'Utilities', SMH:'Semiconductors' };
const OTHER_ETF_NAMES  = { GLD:'Gold', SLV:'Silver', TLT:'Long Treasury', UVXY:'VIX Short' };

const REGIME_NAMES = { R0: 'Healthy Uptrend', R1: 'Choppy / Extended', R2: 'Risk-Off', R3: 'Rebound Window' };
const REGIME_COLORS = { R0: '#22c55e', R1: '#f59e0b', R2: '#ef4444', R3: '#3b82f6' };
const CATEGORY_ORDER = ['market_indices', 'commodities', 'fixed_income', 'volatility', 'sectors'];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtPct(val) {
  if (val == null) return '—';
  return `${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
}

function fmtPrice(val) {
  if (val == null) return '—';
  return `$${val.toFixed(2)}`;
}

function getEtfDisplayName(sym) {
  return MARKET_ETF_NAMES[sym] || SECTOR_ETF_NAMES[sym] || OTHER_ETF_NAMES[sym] || sym;
}

/** HSL-based score gradient: 0=red → 50=amber → 100=green (mirrors model.py) */
function getScoreGradientColor(score) {
  score = Math.max(0, Math.min(100, score));
  let hue;
  if (score <= 50) {
    hue = (score / 50) * 45;
  } else {
    hue = 45 + ((score - 50) / 50) * 75;
  }
  // Convert HSL to RGB (S=0.7, L=0.5)
  const s = 0.7, l = 0.5;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = l - c / 2;
  let r1, g1, b1;
  if (hue < 60)      { r1 = c; g1 = x; b1 = 0; }
  else if (hue < 120) { r1 = x; g1 = c; b1 = 0; }
  else                { r1 = 0; g1 = c; b1 = x; }
  const r = Math.round((r1 + m) * 255);
  const g = Math.round((g1 + m) * 255);
  const b = Math.round((b1 + m) * 255);
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

/** Daily change color: ±5% clamped, RGB interpolation (mirrors model.py) */
function getDailyChangeColor(pct) {
  if (pct == null) return '#9ca3af';
  const clamped = Math.max(-5, Math.min(5, pct));
  if (clamped >= 0) {
    const t = clamped / 5;
    const r = Math.round(200 - t * (200 - 34));
    const g = Math.round(220 - t * (220 - 197));
    const b = Math.round(180 - t * (180 - 94));
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
  } else {
    const t = Math.abs(clamped) / 5;
    const r = Math.round(200 + t * (239 - 200));
    const g = Math.round(200 - t * (200 - 68));
    const b = Math.round(200 - t * (200 - 68));
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
  }
}

/** Alignment score color: ±100 clamped (mirrors model.py) */
function getAlignmentColor(score) {
  if (score == null) return '#9ca3af';
  const clamped = Math.max(-100, Math.min(100, score));
  if (clamped >= 0) {
    const g = Math.round(180 + (clamped / 100) * 40);
    return `#22${g.toString(16).padStart(2,'0')}5e`;
  } else {
    const r = Math.round(200 + (Math.abs(clamped) / 100) * 44);
    return `#${r.toString(16).padStart(2,'0')}4444`;
  }
}

// ─── Sparkline SVG (mirrors etf_dashboard.py:_render_sparkline_svg) ─────────

function renderSparklineSvg(points, width = 60, height = 20) {
  if (!points || points.length < 2) return '';
  const delta = points[points.length - 1] - points[0];
  let color;
  if (delta > 3) color = '#10b981';
  else if (delta < -3) color = '#ef4444';
  else color = '#94a3b8';

  const minVal = Math.max(0, Math.min(...points) - 5);
  const maxVal = Math.min(100, Math.max(...points) + 5);
  const valRange = maxVal - minVal || 1;
  const n = points.length;
  const coords = points.map((v, i) => {
    const x = (i / (n - 1) * width).toFixed(1);
    const y = (height - ((v - minVal) / valRange) * height).toFixed(1);
    return `${x},${y}`;
  });
  const lastCoord = coords[coords.length - 1].split(',');
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="display:inline-block;vertical-align:middle;margin-left:4px;" xmlns="http://www.w3.org/2000/svg"><polyline points="${coords.join(' ')}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${lastCoord[0]}" cy="${lastCoord[1]}" r="2" fill="${color}"/></svg>`;
}

// ─── ETF Dashboard Cards ─────────────────────────────────────────────────────

function renderEtfDashboard(tickers, scoreHistory) {
  const container = document.getElementById('overview-etf-dashboard');
  if (!container) return;

  const tickerMap = {};
  (tickers || []).forEach(t => { if (t.symbol) tickerMap[t.symbol] = t; });

  // Build sparkline points per-symbol from snapshots (score_history.json format)
  const sparklinePoints: Record<string, number[]> = {};
  if (scoreHistory?.snapshots?.length > 1) {
    const snapshots = scoreHistory.snapshots;
    const allSyms = new Set<string>();
    for (const snap of snapshots) {
      for (const sym of Object.keys(snap.scores || {})) allSyms.add(sym);
    }
    for (const sym of allSyms) {
      const pts = snapshots.map(s => s.scores?.[sym]).filter(v => v != null);
      if (pts.length >= 2) sparklinePoints[sym] = pts;
    }
  }

  let html = '<div class="hm-dashboard-title">Market Overview</div>';

  for (const catKey of CATEGORY_ORDER) {
    const config = ETF_CONFIG[catKey];
    if (!config) continue;

    const gridClass = config.card_style === 'large' ? 'hm-cards-large' :
                      config.card_style === 'mini' ? 'hm-cards-mini' : 'hm-cards-compact';

    let cardsHtml = '';
    for (const sym of config.symbols) {
      const t = tickerMap[sym] || {};
      const displayName = getEtfDisplayName(sym);
      const regime = t.regime || null;
      const compositeScore = t.composite_score_avg != null ? t.composite_score_avg : null;
      const regimeClass = regime ? `hm-regime-${regime.toLowerCase()}` : 'hm-regime-unknown';
      const regimeText = compositeScore != null ? Math.round(compositeScore).toString() : (regime || '—');
      const sparkline = renderSparklineSvg(sparklinePoints[sym] || []);
      const priceStr = fmtPrice(t.close);
      const changeStr = fmtPct(t.daily_change_pct);
      const changeClass = (t.daily_change_pct || 0) >= 0 ? 'hm-positive' : 'hm-negative';

      // Direction class based on buy/sell signal counts
      const buys = t.buy_signal_count || 0;
      const sells = t.sell_signal_count || 0;
      let dirClass = 'hm-direction-neutral';
      if (buys > sells) dirClass = 'hm-direction-bullish';
      else if (sells > buys) dirClass = 'hm-direction-bearish';

      const onClick = `window.APEX.navigateTo('signals', {symbol:'${sym}',tf:'1d'})`;

      if (config.card_style === 'large') {
        cardsHtml += `
          <div class="hm-card hm-card-large ${dirClass}" onclick="${onClick}">
            <div class="hm-card-header">
              <span class="hm-card-symbol">${sym}</span>
              <span class="hm-card-name">${displayName}</span>
              <span class="hm-regime ${regimeClass}">${regimeText}</span>${sparkline}
            </div>
            <div class="hm-card-price">${priceStr}</div>
            <div class="hm-card-change ${changeClass}">${changeStr}</div>
          </div>`;
      } else if (config.card_style === 'mini') {
        cardsHtml += `
          <div class="hm-card hm-card-mini ${dirClass}" onclick="${onClick}">
            <div class="hm-card-symbol">${sym}</div>
            <span class="hm-regime ${regimeClass}">${regimeText}</span>${sparkline}
            <div class="hm-card-name">${displayName}</div>
          </div>`;
      } else { // compact
        cardsHtml += `
          <div class="hm-card hm-card-compact ${dirClass}" onclick="${onClick}">
            <span class="hm-regime ${regimeClass}">${regimeText}</span>${sparkline}
            <span class="hm-card-symbol">${sym}</span>
            <span class="hm-card-price">${priceStr}</span>
            <span class="hm-card-change ${changeClass}">${changeStr}</span>
          </div>`;
      }
    }

    html += `
      <div class="hm-etf-category">
        <div class="hm-category-label">${config.name}</div>
        <div class="${gridClass}">${cardsHtml}</div>
      </div>`;
  }

  container.innerHTML = html;
}

// ─── Controls Bar ────────────────────────────────────────────────────────────

function renderControls() {
  const container = document.getElementById('overview-controls');
  if (!container) return;
  container.style.display = '';
  container.innerHTML = `
    <label>Color By:
      <select id="hm-color-by" class="dash-select" style="min-width:120px">
        <option value="trending">Trending</option>
        <option value="daily_change">Daily Change</option>
        <option value="alignment">Alignment Score</option>
      </select>
    </label>
    <label>Size By:
      <select id="hm-size-by" class="dash-select" style="min-width:120px">
        <option value="market_cap">Market Cap</option>
        <option value="volume">Volume</option>
        <option value="equal">Equal Weight</option>
      </select>
    </label>
    <div class="legend" style="margin-left:auto;display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text-muted)">
      <span>0</span>
      <div class="score-gradient"></div>
      <span>50</span>
      <div class="score-gradient"></div>
      <span>100</span>
      <div style="width:16px"></div>
      <span class="legend-swatch" style="background:var(--regime-r0);width:12px;height:12px;border-radius:2px;display:inline-block"></span><span>R0</span>
      <span class="legend-swatch" style="background:var(--regime-r1);width:12px;height:12px;border-radius:2px;display:inline-block"></span><span>R1</span>
      <span class="legend-swatch" style="background:var(--regime-r2);width:12px;height:12px;border-radius:2px;display:inline-block"></span><span>R2</span>
      <span class="legend-swatch" style="background:var(--regime-r3);width:12px;height:12px;border-radius:2px;display:inline-block"></span><span>R3</span>
    </div>`;

  document.getElementById('hm-color-by')?.addEventListener('change', () => updateColors());
  document.getElementById('hm-size-by')?.addEventListener('change', () => updateSizes());
}

// ─── Stats Bar ───────────────────────────────────────────────────────────────

function renderStats(tickers) {
  const container = document.getElementById('overview-stats');
  if (!container) return;
  container.style.display = '';

  const counts = { R0: 0, R1: 0, R2: 0, R3: 0 };
  let totalSignals = 0;
  let missingCaps = 0;

  (tickers || []).forEach(t => {
    const r = t.regime || 'R0';
    if (r in counts) counts[r]++;
    totalSignals += (t.signal_count || 0);
    if (!t.market_cap && !ALL_DASHBOARD_ETFS.has(t.symbol)) missingCaps++;
  });

  const nonEtfCount = (tickers || []).filter(t => !ALL_DASHBOARD_ETFS.has(t.symbol)).length;

  container.innerHTML = `
    <span>Symbols: <strong>${nonEtfCount}</strong></span>
    <span>Signals (24h): <strong>${totalSignals}</strong> \ud83d\udd25</span>
    <span>Missing Caps: <strong>${missingCaps}</strong></span>
    <span style="color:var(--regime-r0)">R0: <strong>${counts.R0}</strong></span>
    <span style="color:var(--regime-r1)">R1: <strong>${counts.R1}</strong></span>
    <span style="color:var(--regime-r2)">R2: <strong>${counts.R2}</strong></span>
    <span style="color:var(--regime-r3)">R3: <strong>${counts.R3}</strong></span>`;
}

// ─── Treemap ─────────────────────────────────────────────────────────────────

// Store treemap data globally for dynamic color/size switching
let treemapData: any = null;
let treemapContainer: HTMLElement | null = null;

async function renderHeatmap(tickers) {
  treemapContainer = document.getElementById('overview-heatmap');
  if (!treemapContainer || !tickers || tickers.length === 0) return;

  // Load market cap + sector data
  let caps = {};
  let sectors = {};
  try {
    const resp = await fetch('data/market_caps.json');
    if (resp.ok) {
      const data = await resp.json();
      caps = data.caps || {};
      sectors = data.sectors || {};
    }
  } catch { /* graceful fallback */ }

  if (!window.Plotly) {
    treemapContainer.innerHTML = '<div class="empty-state">Plotly not loaded</div>';
    return;
  }

  // Filter out ETFs from treemap stocks
  const stockTickers = tickers.filter(t => !ALL_DASHBOARD_ETFS.has(t.symbol));

  // Build treemap data: root → sectors → stocks
  const ids = ['root'];
  const labels = ['Stock Universe'];
  const parents = [''];
  const values = [0];  // Will be recalculated
  const colors = ['#0c0f14'];
  const customdata: any[] = [{ type: 'root' }];
  const textArr = [''];

  // Group tickers by sector
  const sectorGroups = {};
  const noSector = [];

  for (const t of stockTickers) {
    const sector = sectors[t.symbol];
    if (sector) {
      if (!sectorGroups[sector]) sectorGroups[sector] = [];
      sectorGroups[sector].push(t);
    } else {
      noSector.push(t);
    }
  }

  // Add sector + stock nodes
  const addSectorStocks = (sectorName, sectorId, stocks) => {
    ids.push(sectorId);
    labels.push(sectorName);
    parents.push('root');
    values.push(0); // placeholder — recalculated below
    colors.push('#1c2230');
    customdata.push({ type: 'sector' });
    textArr.push('');

    let sectorTotal = 0;
    for (const t of stocks) {
      const cap = caps[t.symbol] || (t.volume && t.close ? t.volume * t.close : 1);
      const leafVal = cap > 0 ? cap : 1;
      sectorTotal += leafVal;
      const score = t.composite_score_avg != null ? t.composite_score_avg : 50;

      ids.push(`stock_${t.symbol}`);
      labels.push(t.symbol);
      parents.push(sectorId);
      values.push(leafVal);
      colors.push(getScoreGradientColor(score));
      customdata.push({
        type: 'stock',
        symbol: t.symbol,
        regime: t.regime || 'R0',
        change: t.daily_change_pct,
        close: t.close,
        score,
        signal_count: t.signal_count || 0,
        volume: t.volume,
        market_cap: caps[t.symbol] || null,
        alignment_score: t.alignment_score,
      });
      const changeTxt = t.daily_change_pct != null ? fmtPct(t.daily_change_pct) : '';
      textArr.push(`${t.symbol}<br>${changeTxt}`);
    }

    // Set parent value = sum of children (required for branchvalues: 'total')
    values[ids.indexOf(sectorId)] = sectorTotal;
    return sectorTotal;
  };

  let rootTotal = 0;
  for (const [sector, stocks] of Object.entries(sectorGroups).sort((a, b) => a[0].localeCompare(b[0]))) {
    rootTotal += addSectorStocks(sector, `sector_${sector}`, stocks);
  }
  if (noSector.length > 0) {
    rootTotal += addSectorStocks('Other', 'sector_Other', noSector);
  }

  // Set root value = sum of all sectors
  values[0] = rootTotal;

  // Store globally for dynamic switching
  treemapData = { ids, labels, parents, values, colors, customdata, textArr, stockTickers, caps, sectors };

  // Compute dynamic height
  const reservedHeight = 48 + 300 + 120; // nav + ETF dashboard approx + controls/stats
  const treemapHeight = Math.max(500, window.innerHeight - reservedHeight);

  // Rich hover template
  const hovertemplate =
    '<b>%{label}</b><br>' +
    'Signals: %{customdata.signal_count}<br>' +
    'Regime: %{customdata.regime}<br>' +
    'Close: $%{customdata.close:.2f}<br>' +
    'Daily: %{customdata.change:+.2f}%<br>' +
    'Value: %{value:,.0f}' +
    '<extra></extra>';

  window.Plotly.newPlot(treemapContainer, [{
    type: 'treemap',
    ids,
    labels,
    parents,
    values,
    text: textArr,
    textinfo: 'text',
    marker: { colors },
    branchvalues: 'total',
    hovertemplate,
    pathbar: { visible: true, edgeshape: '>' },
    tiling: { pad: 2 },
    customdata,
  }], {
    paper_bgcolor: '#0c0f14',
    plot_bgcolor: '#0c0f14',
    font: { color: '#f0f4f8', size: 12 },
    margin: { l: 4, r: 4, t: 28, b: 4 },
    height: treemapHeight,
  }, { responsive: true, displayModeBar: false });

  // Click to navigate to signals page
  treemapContainer.on('plotly_click', (data) => {
    const pt = data.points?.[0];
    if (pt?.customdata?.symbol) {
      window.APEX.navigateTo('signals', { symbol: pt.customdata.symbol, tf: '1d' });
    }
  });

  // Dynamic height on resize
  window.addEventListener('resize', () => {
    if (!treemapContainer) return;
    const newHeight = Math.max(500, window.innerHeight - reservedHeight);
    window.Plotly.relayout(treemapContainer, { height: newHeight });
  });
}

// ─── Dynamic Color Switching ─────────────────────────────────────────────────

function updateColors() {
  if (!treemapData || !treemapContainer) return;
  const mode = (document.getElementById('hm-color-by') as HTMLSelectElement)?.value || 'trending';

  const newColors = treemapData.colors.map((c, i) => {
    const cd = treemapData.customdata[i];
    if (!cd || cd.type !== 'stock') return c;

    switch (mode) {
      case 'trending':
        return getScoreGradientColor(cd.score ?? 50);
      case 'daily_change':
        return getDailyChangeColor(cd.change);
      case 'alignment':
        return getAlignmentColor(cd.alignment_score);
      default:
        return c;
    }
  });

  window.Plotly.restyle(treemapContainer, { 'marker.colors': [newColors] }, [0]);
}

// ─── Dynamic Size Switching ──────────────────────────────────────────────────

function updateSizes() {
  if (!treemapData || !treemapContainer) return;
  const mode = (document.getElementById('hm-size-by') as HTMLSelectElement)?.value || 'market_cap';

  const { ids, parents, customdata } = treemapData;
  const newValues = [...treemapData.values];

  // Recalculate leaf values
  for (let i = 0; i < ids.length; i++) {
    const cd = customdata[i];
    if (!cd || cd.type !== 'stock') continue;

    switch (mode) {
      case 'market_cap':
        newValues[i] = cd.market_cap || (cd.volume && cd.close ? cd.volume * cd.close : 1);
        break;
      case 'volume':
        newValues[i] = cd.volume || 1;
        break;
      case 'equal':
        newValues[i] = 1;
        break;
    }
    if (newValues[i] <= 0) newValues[i] = 1;
  }

  // Recalculate parent values (bottom-up)
  const parentChildren = {};
  for (let i = 0; i < ids.length; i++) {
    const p = parents[i];
    if (p) {
      if (!parentChildren[p]) parentChildren[p] = [];
      parentChildren[p].push(i);
    }
  }

  function recalcParent(id) {
    const idx = ids.indexOf(id);
    if (idx < 0) return 0;
    const children = parentChildren[id];
    if (!children || children.length === 0) return newValues[idx];
    let sum = 0;
    for (const ci of children) {
      sum += recalcParent(ids[ci]);
    }
    newValues[idx] = sum;
    return sum;
  }

  recalcParent('root');

  treemapData.values = newValues;
  window.Plotly.restyle(treemapContainer, { values: [newValues] }, [0]);
}

// ─── Page Lifecycle ──────────────────────────────────────────────────────────

export async function init(apex) {
  if (!apex.summary) return;

  const tickers = apex.summary.tickers || [];
  renderEtfDashboard(tickers, apex.scoreHistory);
  renderControls();
  renderStats(tickers);
  await renderHeatmap(tickers);
}

export async function update(apex) {
  await init(apex);
}
