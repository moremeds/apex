/**
 * Signals Page — Per-symbol Plotly multi-subplot chart with 8 collapsible sections.
 *
 * Ported 1:1 from the original gh-pages signal report:
 *   - javascript.py (renderMainChart, updateConfluencePanel, updateDualMACDHistory,
 *     updateTrendPulseHistory, updateRegimeFlexHistory, updateSectorPulseHistory,
 *     updateSignalHistoryTable, loadIndicatorsSection)
 */

let currentSymbol = null;
let currentTf = null;
const dataCache = {};
const regimeHtmlCache = {};

// Theme colors (match original javascript.py)
const C = {
  success: '#10b981', danger: '#ef4444', warning: '#f59e0b', primary: '#3b82f6',
  text: '#f0f4f8', text_muted: '#94a3b8', border: '#334155',
  card_bg: '#1e293b', bg: '#1e293b',
};

// ─── Data Loading ────────────────────────────────────────────────────────────

async function loadSymbolData(symbol, tf) {
  const key = `${symbol}_${tf}`;
  if (dataCache[key]) return dataCache[key];
  try {
    const resp = await fetch(`data/${key}.json`);
    if (!resp.ok) return null;
    const data = await resp.json();
    dataCache[key] = data;
    return data;
  } catch { return null; }
}

// ─── Controls ────────────────────────────────────────────────────────────────

function populateSymbolSelect(apex) {
  const select = document.getElementById('signals-symbol-select') as HTMLSelectElement;
  if (!select || select.options.length > 1) return;
  const symbols = apex.manifest?.symbols || apex.summary?.symbols || [];
  symbols.forEach(sym => {
    const opt = document.createElement('option');
    opt.value = sym;
    opt.textContent = sym;
    select.appendChild(opt);
  });
  select.addEventListener('change', () => {
    if (select.value) window.APEX.navigateTo('signals', { symbol: select.value, tf: currentTf || '1d' });
  });
}

function populateTimeframeTabs(apex) {
  const container = document.getElementById('signals-tf-tabs');
  if (!container || container.children.length > 0) return;
  const timeframes = apex.manifest?.timeframes || ['1h', '4h', '1d'];
  timeframes.forEach(tf => {
    const btn = document.createElement('button');
    btn.className = 'tf-btn' + (tf === '1d' ? ' active' : '');
    btn.textContent = tf;
    btn.dataset.tf = tf;
    btn.addEventListener('click', () => {
      if (currentSymbol) window.APEX.navigateTo('signals', { symbol: currentSymbol, tf });
    });
    container.appendChild(btn);
  });
}

function updateActiveTimeframe(tf) {
  document.querySelectorAll('#signals-tf-tabs .tf-btn').forEach(btn => {
    (btn as HTMLElement).classList.toggle('active', (btn as HTMLElement).dataset.tf === tf);
  });
}

// ─── Plotly Multi-Subplot Chart (4 rows) ─────────────────────────────────────

function renderChart(data) {
  const chartEl = document.getElementById('signals-chart');
  if (!chartEl || !data?.chart_data || !window.Plotly) return;

  const cd = data.chart_data;
  const traces: any[] = [];
  const hasData = (v) => v && v.length > 0 && !v.every(x => x === null);

  // Intraday: use index-based x-axis to avoid gaps
  const isIntraday = ['1m','5m','15m','30m','1h','4h'].includes(data.timeframe);
  const xValues = isIntraday ? cd.timestamps.map((_, i) => i) : cd.timestamps;

  // Row 1: Candlestick
  traces.push({
    type: 'candlestick', x: xValues, open: cd.open, high: cd.high, low: cd.low, close: cd.close,
    name: 'Price',
    increasing: { line: { color: C.success }, fillcolor: C.success },
    decreasing: { line: { color: C.danger }, fillcolor: C.danger },
    xaxis: 'x', yaxis: 'y',
  });

  // Overlays: Bollinger Bands, SuperTrend
  const overlayConfig = {
    bollinger_bb_upper: { color: '#3b82f6', dash: 'dot' },
    bollinger_bb_middle: { color: '#6366f1', dash: 'solid' },
    bollinger_bb_lower: { color: '#3b82f6', dash: 'dot' },
    supertrend_supertrend: { color: '#f59e0b', dash: 'solid' },
  };
  for (const [name, cfg] of Object.entries(overlayConfig)) {
    const vals = cd.overlays?.[name];
    if (!hasData(vals)) continue;
    traces.push({
      type: 'scatter', mode: 'lines', x: xValues, y: vals,
      name: name.replace('bollinger_bb_', 'BB ').replace('supertrend_', 'ST '),
      line: { color: cfg.color, width: 1, dash: cfg.dash },
      xaxis: 'x', yaxis: 'y',
    });
  }

  // Row 2: RSI
  const rsiValues = cd.rsi?.['rsi_rsi'];
  if (hasData(rsiValues)) {
    traces.push({ type: 'scatter', mode: 'lines', x: xValues, y: rsiValues, name: 'RSI',
      line: { color: '#8b5cf6', width: 1.5 }, xaxis: 'x', yaxis: 'y2' });
    const boundsX = [xValues[0], xValues[xValues.length - 1]];
    traces.push({ type: 'scatter', mode: 'lines', x: boundsX, y: [70, 70], name: 'Overbought',
      line: { color: C.danger, width: 1, dash: 'dash' }, xaxis: 'x', yaxis: 'y2', showlegend: false });
    traces.push({ type: 'scatter', mode: 'lines', x: boundsX, y: [30, 30], name: 'Oversold',
      line: { color: C.success, width: 1, dash: 'dash' }, xaxis: 'x', yaxis: 'y2', showlegend: false });
  }

  // Row 3: MACD (DualMACD if available)
  const hasDualMACD = cd.dual_macd && (hasData(cd.dual_macd['dual_macd_slow_histogram']) || hasData(cd.dual_macd['dual_macd_fast_histogram']));
  if (hasDualMACD) {
    const slowHist = cd.dual_macd['dual_macd_slow_histogram'];
    const fastHist = cd.dual_macd['dual_macd_fast_histogram'];
    if (hasData(slowHist)) {
      traces.push({ type: 'bar', x: xValues, y: slowHist, name: 'Slow MACD (55/89)',
        marker: { color: slowHist.map(v => v >= 0 ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)') },
        xaxis: 'x', yaxis: 'y3', width: isIntraday ? 0.8 : 86400000 * 0.8 });
    }
    if (hasData(fastHist)) {
      traces.push({ type: 'bar', x: xValues, y: fastHist, name: 'Fast MACD (13/21)',
        marker: { color: fastHist.map(v => v >= 0 ? 'rgba(59,130,246,0.8)' : 'rgba(249,115,22,0.8)') },
        xaxis: 'x', yaxis: 'y3', width: isIntraday ? 0.4 : 86400000 * 0.4 });
    }
    traces.push({ type: 'scatter', mode: 'lines', x: [xValues[0], xValues[xValues.length-1]], y: [0,0],
      line: { color: C.text_muted, width: 1, dash: 'dot' }, xaxis: 'x', yaxis: 'y3', showlegend: false });
  } else {
    const macdHist = cd.macd?.['macd_histogram'];
    if (hasData(macdHist)) {
      traces.push({ type: 'bar', x: xValues, y: macdHist, name: 'MACD Hist',
        marker: { color: macdHist.map(v => v >= 0 ? C.success : C.danger) }, xaxis: 'x', yaxis: 'y3' });
    }
  }

  // Row 4: Volume
  if (cd.volume?.length > 0) {
    const volColors = cd.close.map((c, i) => i === 0 ? C.text_muted : (c >= cd.close[i-1] ? C.success : C.danger));
    traces.push({ type: 'bar', x: xValues, y: cd.volume, name: 'Volume',
      marker: { color: volColors, opacity: 0.5 }, xaxis: 'x', yaxis: 'y4' });
  }

  // Signal markers
  const signals = data.signals || [];
  for (const dir of ['buy', 'sell']) {
    const sigs = signals.filter(s => s.direction === dir);
    if (sigs.length === 0) continue;
    const pts = sigs.map(s => {
      const idx = cd.timestamps.findIndex(t => t === s.timestamp);
      if (idx < 0) return null;
      return { x: isIntraday ? idx : s.timestamp, y: dir === 'buy' ? cd.low[idx] * 0.995 : cd.high[idx] * 1.005, rule: s.rule };
    }).filter(Boolean);
    if (pts.length > 0) {
      traces.push({
        type: 'scatter', mode: 'markers', x: pts.map(p => p.x), y: pts.map(p => p.y),
        name: dir === 'buy' ? 'Buy Signal' : 'Sell Signal',
        marker: { symbol: dir === 'buy' ? 'triangle-up' : 'triangle-down', size: 12,
          color: dir === 'buy' ? C.success : C.danger, line: { color: 'white', width: 1 } },
        hovertemplate: '%{text}<extra></extra>', text: pts.map(p => p.rule),
        xaxis: 'x', yaxis: 'y',
      });
    }
  }

  // Intraday tick labels
  let tickConfig: any = { nticks: 15 };
  if (isIntraday && cd.timestamps.length > 0) {
    const n = cd.timestamps.length;
    const step = Math.max(1, Math.floor(n / 15));
    const tickvals = [], ticktext = [];
    for (let i = 0; i < n; i += step) {
      tickvals.push(i);
      const ts = new Date(cd.timestamps[i]);
      ticktext.push(`${String(ts.getUTCMonth()+1).padStart(2,'0')}/${String(ts.getUTCDate()).padStart(2,'0')} ${String(ts.getUTCHours()).padStart(2,'0')}:${String(ts.getUTCMinutes()).padStart(2,'0')}`);
    }
    tickConfig = { tickvals, ticktext };
  }

  const isDaily = ['1d','1w','1D','1W'].includes(data.timeframe);
  const layout = {
    title: { text: `${data.symbol} - ${data.timeframe} (${data.bar_count} bars)`, font: { color: C.text, size: 18 } },
    showlegend: true, legend: { orientation: 'h', y: -0.08, font: { color: C.text, size: 10 } },
    paper_bgcolor: C.card_bg, plot_bgcolor: C.card_bg, font: { color: C.text },
    margin: { t: 50, r: 50, b: 80, l: 50 }, hovermode: 'x unified', bargap: 0.1, barmode: 'overlay',
    xaxis: {
      gridcolor: C.border, showgrid: true, rangeslider: { visible: false }, tickangle: -45,
      domain: [0, 1], rangebreaks: isDaily ? [{ bounds: ['sat', 'mon'] }] : [],
      ...tickConfig,
    },
    yaxis:  { title: 'Price', side: 'right', gridcolor: C.border, showgrid: true, domain: [0.52, 1.00], autorange: true },
    yaxis2: { title: 'RSI',   side: 'right', gridcolor: C.border, showgrid: true, domain: [0.36, 0.48], range: [0, 100], dtick: 25 },
    yaxis3: { title: hasDualMACD ? 'DualMACD (55/89 + 13/21)' : 'MACD', side: 'right', gridcolor: C.border, showgrid: true, domain: [0.20, 0.32] },
    yaxis4: { title: 'Vol', side: 'right', gridcolor: C.border, showgrid: true, domain: [0.00, 0.16] },
  };

  window.Plotly.newPlot(chartEl, traces, layout, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
}

// ─── Collapsible Section Helper ──────────────────────────────────────────────

function makeSection(id, title, content, expanded = true) {
  const chevron = expanded ? '\u25bc' : '\u25b6';
  const expandedCls = expanded ? ' expanded' : '';
  return `
    <div class="section-header${expandedCls}" data-section="${id}">
      <span class="chevron">${chevron}</span> ${title}
    </div>
    <div class="section-content${expandedCls}" id="${id}">${content}</div>`;
}

function wireCollapsibles() {
  document.querySelectorAll('#signals-sections .section-header').forEach(header => {
    if (header.dataset.wired) return;
    header.dataset.wired = 'true';
    header.addEventListener('click', () => {
      header.classList.toggle('expanded');
      const content = header.nextElementSibling as HTMLElement;
      if (content) content.classList.toggle('expanded');
      const chevron = header.querySelector('.chevron');
      if (chevron) chevron.textContent = header.classList.contains('expanded') ? '\u25bc' : '\u25b6';
    });
  });
}

// ─── Section 1: Confluence Analysis ──────────────────────────────────────────

function renderConfluence(data, summary) {
  const key = `${data.symbol}_${data.timeframe}`;
  const confluence = summary?.confluence?.[key];
  const signals = data.signals || [];
  const cd = data.chart_data || {};
  const priceLevels = cd.price_levels || {};
  const lastClose = cd.close?.length > 0 ? cd.close[cd.close.length - 1] : null;

  if (!confluence) return makeSection('sec-confluence', 'Confluence Analysis', '<div style="color:var(--text-muted);padding:16px">No confluence data available</div>');

  const alignPct = (confluence.alignment_score + 100) / 2;
  const alignCls = confluence.alignment_score > 20 ? 'bullish' : confluence.alignment_score < -20 ? 'bearish' : 'neutral';
  const scoreColor = confluence.alignment_score > 20 ? C.success : confluence.alignment_score < -20 ? C.danger : C.text_muted;

  // Active signals
  let activeHtml = '';
  if (signals.length > 0) {
    const byInd = {};
    [...signals].sort((a,b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()).forEach(s => { byInd[s.indicator||'unknown'] = s; });
    const active = Object.values(byInd) as any[];
    const buys = active.filter(s => s.direction === 'buy').length;
    const sells = active.filter(s => s.direction === 'sell').length;
    const neutrals = active.length - buys - sells;
    const rows = active.sort((a:any,b:any) => (a.indicator||'').localeCompare(b.indicator||'')).slice(0, 8)
      .map((s:any) => `<div style="display:flex;justify-content:space-between;padding:3px 0;font-size:11px"><span style="color:${C.text_muted}">${s.indicator}</span><span style="color:${s.direction==='buy'?C.success:s.direction==='sell'?C.danger:C.text_muted}">${s.rule}</span></div>`).join('');
    const more = active.length > 8 ? `<div style="color:${C.text_muted};font-size:10px;text-align:center;margin-top:4px">+${active.length-8} more</div>` : '';
    activeHtml = `
      <div style="background:${C.card_bg};padding:12px;border-radius:8px;border:1px solid ${C.border}">
        <h4 style="margin:0 0 8px;color:${C.text};font-size:12px;font-weight:600">Active Signals - ${data.timeframe.toUpperCase()}</h4>
        <div style="display:flex;gap:12px;margin-bottom:8px;font-size:12px">
          <span style="color:${C.success}">\u25b2 ${buys} Bullish</span>
          <span style="color:${C.danger}">\u25bc ${sells} Bearish</span>
          <span style="color:${C.text_muted}">\u25cf ${neutrals} Neutral</span>
        </div>${rows}${more}
      </div>`;
  }

  // Price levels
  let levelsHtml = '';
  if (Object.keys(priceLevels).length > 0 && lastClose) {
    const groups = {};
    for (const [col, vals] of Object.entries(priceLevels) as any) {
      if (!vals?.length) continue;
      const lastVal = vals[vals.length - 1];
      if (lastVal == null) continue;
      const parts = col.split('_');
      const ind = parts[0].toUpperCase();
      const level = parts.slice(1).join('_').toUpperCase();
      if (!groups[ind]) groups[ind] = [];
      groups[ind].push({ level, value: lastVal });
    }
    let cards = '';
    for (const [ind, levels] of Object.entries(groups) as any) {
      levels.sort((a,b) => b.value - a.value);
      const name = ind === 'FIB' ? 'Fibonacci' : ind === 'SR' ? 'Support/Res' : ind === 'PIVOT' ? 'Pivot Points' : ind;
      const rows = levels.slice(0, 3).map(l => {
        const diff = ((l.value - lastClose) / lastClose * 100).toFixed(2);
        const above = l.value > lastClose;
        return `<div style="display:flex;justify-content:space-between;font-size:11px;padding:2px 0"><span style="color:${C.text_muted}">${l.level.replace(/_/g,' ')}</span><span>$${l.value.toFixed(2)} <span style="color:${above?C.danger:C.success}">${above?'\u2191':'\u2193'}${Math.abs(parseFloat(diff))}%</span></span></div>`;
      }).join('');
      cards += `<div style="background:${C.card_bg};padding:10px;border-radius:6px;border:1px solid ${C.border};min-width:140px"><div style="font-weight:600;margin-bottom:6px;font-size:11px;color:${C.text}">${name}</div>${rows}</div>`;
    }
    levelsHtml = `<div style="margin-top:16px"><h4 style="margin:0 0 10px;color:${C.text_muted};font-size:11px;text-transform:uppercase">Key Price Levels ($${lastClose.toFixed(2)})</h4><div style="display:flex;gap:10px;flex-wrap:wrap">${cards}</div></div>`;
  }

  // Divergences
  let divHtml;
  if (confluence.diverging_pairs?.length > 0) {
    divHtml = confluence.diverging_pairs.slice(0, 5).map(p => `<div style="padding:4px 0;font-size:12px"><span style="color:${C.text}">${p.ind1} - ${p.ind2}</span> <span style="color:${C.text_muted}">${p.reason}</span></div>`).join('');
  } else {
    divHtml = `<div style="color:${C.text_muted};font-size:12px">No divergences detected - indicators are aligned</div>`;
  }

  const content = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:16px">
      <div>
        <div class="alignment-meter"><div class="alignment-meter-marker" style="left:${alignPct}%"></div></div>
        <div style="text-align:center;font-size:24px;font-weight:bold;margin:8px 0;color:${scoreColor}">
          ${confluence.alignment_score > 0 ? '+' : ''}${Math.round(confluence.alignment_score)}
        </div>
        <div style="display:flex;justify-content:center;gap:16px">
          <div style="text-align:center"><div style="font-size:16px;color:${C.success}">\u25b2 ${confluence.bullish_count}</div><div style="font-size:10px;color:${C.text_muted}">Bullish</div></div>
          <div style="text-align:center"><div style="font-size:16px;color:${C.text_muted}">\u25cf ${confluence.neutral_count}</div><div style="font-size:10px;color:${C.text_muted}">Neutral</div></div>
          <div style="text-align:center"><div style="font-size:16px;color:${C.danger}">\u25bc ${confluence.bearish_count}</div><div style="font-size:10px;color:${C.text_muted}">Bearish</div></div>
        </div>
      </div>
      <div>${activeHtml}</div>
    </div>
    ${levelsHtml}
    <div style="margin-top:16px"><h4 style="margin:0 0 12px;color:${C.text_muted};font-size:11px;text-transform:uppercase">Diverging Indicators</h4>${divHtml}</div>`;

  return makeSection('sec-confluence', 'Confluence Analysis', content);
}

// ─── Section 2: DualMACD ─────────────────────────────────────────────────────

function renderDualMACD(data) {
  const rows = data.dual_macd_history || [];
  if (rows.length === 0) return makeSection('sec-dualmacd', 'DualMACD', `<div style="color:${C.text_muted};padding:16px">No DualMACD history</div>`);

  const trendColors = { BULLISH:'#10b981', BEARISH:'#ef4444', DETERIORATING:'#f59e0b', IMPROVING:'#06b6d4' };
  const trendIcons = { BULLISH:'\u25b2', BEARISH:'\u25bc', DETERIORATING:'\u25b6', IMPROVING:'\u25c0' };
  const rowBg = { DIP_BUY:'rgba(16,185,129,0.10)', RALLY_SELL:'rgba(239,68,68,0.10)' };

  const cur = rows[0];
  const curTrend = cur.trend_state || 'BEARISH';
  const curTactical = cur.tactical_signal || 'NONE';
  const curConf = Math.round((cur.confidence || 0) * 100);
  const curTrendColor = trendColors[curTrend] || C.text_muted;
  const curConfColor = curConf >= 50 ? '#10b981' : curConf >= 25 ? '#f59e0b' : C.text_muted;

  let tacticalBadge;
  if (curTactical === 'DIP_BUY') tacticalBadge = `<span style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid #10b981;padding:4px 12px;border-radius:6px;font-weight:700;font-size:14px">DIP BUY</span>`;
  else if (curTactical === 'RALLY_SELL') tacticalBadge = `<span style="background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid #ef4444;padding:4px 12px;border-radius:6px;font-weight:700;font-size:14px">RALLY SELL</span>`;
  else tacticalBadge = `<span style="color:${C.text_muted};border:1px solid ${C.border};padding:4px 12px;border-radius:6px;font-size:14px">No Signal</span>`;

  const cards = `
    <div class="metric-cards" style="grid-template-columns:repeat(4,1fr)">
      <div class="metric-card" style="text-align:center"><div class="metric-label">Trend State</div><div class="metric-value" style="color:${curTrendColor}">${trendIcons[curTrend]||''} ${curTrend}</div><div class="metric-sub">H_slow ${cur.slow_histogram>=0?'+':''}${(cur.slow_histogram||0).toFixed(2)}</div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Tactical Signal</div><div style="margin-top:4px">${tacticalBadge}</div><div class="metric-sub" style="margin-top:8px">H_fast ${cur.fast_histogram>=0?'+':''}${(cur.fast_histogram||0).toFixed(2)}</div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Momentum Balance</div><div class="metric-value">${(cur.momentum_balance||'BALANCED').replace('_',' ')}</div><div class="metric-sub">norm ${Math.abs(cur.slow_histogram||0).toFixed(2)}</div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Confidence</div><div class="metric-value" style="color:${curConfColor}">${curConf}%</div><div class="confidence-bar" style="width:80%;margin:6px auto 0"><div class="confidence-bar-fill" style="width:${curConf}%;background:${curConfColor}"></div></div></div>
    </div>`;

  let tbody = '';
  for (const row of rows) {
    const tactical = row.tactical_signal || 'NONE';
    const trend = row.trend_state || 'BEARISH';
    const bg = rowBg[tactical] || 'transparent';
    const tc = trendColors[trend] || C.text_muted;
    const conf = Math.round((row.confidence || 0) * 100);
    const cc = conf >= 50 ? '#10b981' : conf >= 25 ? '#f59e0b' : C.text_muted;
    const tacCell = tactical === 'DIP_BUY' ? `<span style="color:#10b981;font-weight:600">DIP_BUY</span>` :
                    tactical === 'RALLY_SELL' ? `<span style="color:#ef4444;font-weight:600">RALLY_SELL</span>` :
                    `<span style="color:${C.text_muted}">\u2014</span>`;
    tbody += `<tr class="${tactical==='DIP_BUY'?'row-buy':tactical==='RALLY_SELL'?'row-sell':''}" style="background:${bg}"><td>${row.date}</td><td style="text-align:right;font-family:monospace">${(row.slow_histogram>=0?'+':'')+(row.slow_histogram||0).toFixed(3)}</td><td style="text-align:right;font-family:monospace">${(row.fast_histogram>=0?'+':'')+(row.fast_histogram||0).toFixed(3)}</td><td style="text-align:right;font-family:monospace">${(row.slow_hist_delta>=0?'+':'')+(row.slow_hist_delta||0).toFixed(3)}</td><td style="text-align:right;font-family:monospace">${(row.fast_hist_delta>=0?'+':'')+(row.fast_hist_delta||0).toFixed(3)}</td><td style="color:${tc};font-weight:500">${trend}</td><td>${tacCell}</td><td style="color:${C.text_muted}">${row.momentum_balance||'\u2014'}</td><td><div style="display:flex;align-items:center;gap:4px"><div class="confidence-bar"><div class="confidence-bar-fill" style="width:${conf}%;background:${cc}"></div></div><span style="font-size:10px">${conf}</span></div></td></tr>`;
  }

  const table = `<div style="margin-top:12px;font-size:11px;color:${C.text_muted}">As of ${cur.date}</div>
    <div style="overflow-x:auto;margin-top:12px"><table class="signal-table"><thead><tr><th>Date</th><th>H_slow</th><th>H_fast</th><th>\u0394H_s</th><th>\u0394H_f</th><th>Trend</th><th>Tactical</th><th>Mom.Bal</th><th>Conf</th></tr></thead><tbody>${tbody}</tbody></table></div>`;

  return makeSection('sec-dualmacd', `DualMACD (${rows.length} bars)`, cards + table);
}

// ─── Section 3: TrendPulse ───────────────────────────────────────────────────

function renderTrendPulse(data) {
  const rows = data.trend_pulse_history || [];
  if (rows.length === 0) return makeSection('sec-trendpulse', 'TrendPulse', `<div style="color:${C.text_muted};padding:16px">No TrendPulse history</div>`);

  const trendColors = { BULLISH:'#10b981', BEARISH:'#ef4444', NEUTRAL: C.text_muted };
  const dmColors = { BULLISH:'#10b981', IMPROVING:'#06b6d4', DETERIORATING:'#f59e0b', BEARISH:'#ef4444' };

  const cur = rows[0];
  const curSwing = cur.swing_signal || 'NONE';
  const curTrend = cur.trend_filter || 'NEUTRAL';
  const curEntry = cur.entry_signal || false;
  const curConf = Math.round((cur.confidence_4f || 0) * 100);
  const curScore = Math.round(cur.score || 0);
  const curDmState = cur.dm_state || 'BEARISH';
  const curAdx = Math.round(cur.adx || 0);
  const curTop = cur.top_warning || 'NONE';
  const curExit = cur.exit_signal || 'none';
  const curAtrStop = cur.atr_stop_level || 0;
  const curCooldown = cur.cooldown_left || 0;

  const confColor = curConf >= 50 ? '#10b981' : curConf >= 25 ? '#f59e0b' : C.text_muted;
  const scoreColor = curScore >= 80 ? '#10b981' : curScore >= 50 ? '#f59e0b' : C.text_muted;

  let swingBadge;
  if (curEntry) swingBadge = `<span style="background:rgba(16,185,129,0.2);color:#10b981;border:1px solid #10b981;padding:4px 12px;border-radius:6px;font-weight:700">\u2713 ENTRY</span>`;
  else if (curSwing === 'BUY') swingBadge = `<span style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid #10b981;padding:4px 12px;border-radius:6px;font-weight:700">BUY</span>`;
  else if (curSwing === 'SELL') swingBadge = `<span style="background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid #ef4444;padding:4px 12px;border-radius:6px;font-weight:700">SELL</span>`;
  else swingBadge = `<span style="color:${C.text_muted};border:1px solid ${C.border};padding:4px 12px;border-radius:6px">No Signal</span>`;

  let topBadge = curTop === 'TOP_DETECTED' ? `<span style="color:#a855f7;font-weight:700">TOP_DETECTED</span>` :
                 curTop === 'TOP_ZONE' ? `<span style="color:#f59e0b;font-weight:700">TOP_ZONE</span>` :
                 `<span style="color:${C.text_muted}">\u2014</span>`;

  const cards = `
    <div class="metric-cards" style="grid-template-columns:repeat(4,1fr)">
      <div class="metric-card" style="text-align:center"><div class="metric-label">Signal</div><div style="margin-top:4px">${swingBadge}</div><div class="metric-sub" style="margin-top:6px">EMA: ${(cur.ema_alignment||'MIXED').replace('ALIGNED_','')}</div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Trend / DM</div><div class="metric-value" style="color:${trendColors[curTrend]||C.text_muted}">${curTrend==='BULLISH'?'\u25b2':curTrend==='BEARISH'?'\u25bc':'\u25c6'} ${curTrend}</div><div class="metric-sub" style="color:${dmColors[curDmState]||C.text_muted}">MACD: ${curDmState} \u00b7 ADX ${curAdx}${cur.adx_ok?'':' \u26a0'}</div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Risk</div><div style="margin-top:4px">${topBadge}</div><div class="metric-sub" style="margin-top:6px">ATR $${curAtrStop>0?curAtrStop.toFixed(2):'\u2014'} \u00b7 Exit: ${curExit!=='none'?`<span style="color:${C.danger}">${curExit}</span>`:'\u2014'} \u00b7 CD: <span style="color:${curCooldown===0?'#10b981':'#ef4444'};font-weight:600">${curCooldown}</span></div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Score / Confidence</div><div class="metric-value" style="color:${scoreColor}">${curScore}</div><div class="confidence-bar" style="width:80%;margin:6px auto 0"><div class="confidence-bar-fill" style="width:${curConf}%;background:${confColor}"></div></div><div class="metric-sub">4F Conf ${curConf}%</div></div>
    </div>`;

  const rowBgColors = { BUY:'rgba(16,185,129,0.10)', SELL:'rgba(239,68,68,0.10)', ENTRY:'rgba(16,185,129,0.15)', TOP_DETECTED:'rgba(168,85,247,0.10)' };
  let tbody = '';
  for (const row of rows) {
    const swing = row.swing_signal || 'NONE';
    const entry = row.entry_signal || false;
    const trend = row.trend_filter || 'NEUTRAL';
    const dm = row.dm_state || 'BEARISH';
    const top = row.top_warning || 'NONE';
    const conf4f = Math.round((row.confidence_4f || 0) * 100);
    const cc = conf4f >= 50 ? '#10b981' : conf4f >= 25 ? '#f59e0b' : C.text_muted;
    let bg = 'transparent';
    if (entry) bg = rowBgColors.ENTRY; else if (swing !== 'NONE') bg = rowBgColors[swing] || bg; else if (top === 'TOP_DETECTED') bg = rowBgColors.TOP_DETECTED;
    const swingCell = swing === 'BUY' ? `<span style="color:#10b981;font-weight:600">BUY</span>` : swing === 'SELL' ? `<span style="color:#ef4444;font-weight:600">SELL</span>` : `<span style="color:${C.text_muted}">\u2014</span>`;
    const entryCell = entry ? `<span style="color:#10b981;font-weight:700">\u2713</span>` : `<span style="color:${C.text_muted}">\u2014</span>`;
    tbody += `<tr style="background:${bg}"><td>${row.date}</td><td>${swingCell}</td><td style="text-align:center">${entryCell}</td><td style="color:${dmColors[dm]||C.text_muted}">${dm}</td><td style="text-align:right;font-family:monospace">${Math.round(row.adx||0)}</td><td style="text-align:right;font-family:monospace">${(row.trend_strength||0).toFixed(2)}</td><td style="color:${top!=='NONE'?(top==='TOP_DETECTED'?'#a855f7':'#f59e0b'):C.text_muted}">${top!=='NONE'?top:'\u2014'}</td><td><div style="display:flex;align-items:center;gap:4px"><div class="confidence-bar"><div class="confidence-bar-fill" style="width:${conf4f}%;background:${cc}"></div></div><span style="font-size:10px">${conf4f}</span></div></td><td style="text-align:right;font-family:monospace">${(row.atr_stop_level||0)>0?'$'+(row.atr_stop_level).toFixed(2):'\u2014'}</td><td style="text-align:center;color:${(row.cooldown_left||0)===0?'#10b981':'#ef4444'};font-weight:600">${row.cooldown_left||0}</td><td style="color:${(row.exit_signal||'none')!=='none'?C.danger:C.text_muted}">${(row.exit_signal||'none')!=='none'?row.exit_signal:'\u2014'}</td></tr>`;
  }

  const table = `<div style="margin-top:12px;font-size:11px;color:${C.text_muted}">As of ${cur.date}</div>
    <div style="overflow-x:auto;margin-top:12px"><table class="signal-table"><thead><tr><th>Date</th><th>Swing</th><th>Entry</th><th>MACD</th><th>ADX</th><th>Str</th><th>Top</th><th>Conf(4f)</th><th>ATR Stop</th><th>CD</th><th>Exit</th></tr></thead><tbody>${tbody}</tbody></table></div>`;

  return makeSection('sec-trendpulse', `TrendPulse (${rows.length} bars)`, cards + table);
}

// ─── Section 4: Regime Analysis ──────────────────────────────────────────────

async function renderRegimeAnalysis(symbol, tf) {
  const cacheKey = `${symbol}_${tf}`;
  if (regimeHtmlCache[cacheKey]) return makeSection('sec-regime', 'Regime Analysis', regimeHtmlCache[cacheKey]);

  const urls = [`data/regime/${symbol}_${tf}.html`, `data/regime/${symbol}.html`];
  for (const url of urls) {
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const html = await resp.text();
        regimeHtmlCache[cacheKey] = html;
        return makeSection('sec-regime', 'Regime Analysis', html);
      }
    } catch { /* continue */ }
  }

  // Fallback from summary
  const tickers = window.APEX?.summary?.tickers || [];
  const ticker = tickers.find(t => t.symbol === symbol);
  if (!ticker?.regime) return makeSection('sec-regime', 'Regime Analysis', `<div style="color:${C.text_muted};padding:16px">No regime data for ${symbol}</div>`);

  const regimeColors = { R0: C.success, R1: C.warning, R2: C.danger, R3: C.primary };
  const rc = regimeColors[ticker.regime] || C.text_muted;
  const components = ticker.component_states || {};
  const fallback = `
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
      <div style="background:${rc}20;color:${rc};border:2px solid ${rc};padding:12px 24px;border-radius:12px;font-size:24px;font-weight:700">${ticker.regime}</div>
      <div><div style="font-size:18px;font-weight:600">${ticker.regime_name||ticker.regime}</div><div style="color:${C.text_muted}">Confidence: ${ticker.confidence||0}%</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px">
      <div class="metric-card"><div class="metric-label">Trend</div><div class="metric-value" style="font-size:13px">${(components.trend_state||'N/A').toUpperCase()}</div></div>
      <div class="metric-card"><div class="metric-label">Volatility</div><div class="metric-value" style="font-size:13px">${(components.vol_state||'N/A').toUpperCase()}</div></div>
      <div class="metric-card"><div class="metric-label">Choppiness</div><div class="metric-value" style="font-size:13px">${(components.chop_state||'N/A').toUpperCase()}</div></div>
      <div class="metric-card"><div class="metric-label">Extension</div><div class="metric-value" style="font-size:13px">${(components.ext_state||'N/A').toUpperCase()}</div></div>
    </div>`;
  return makeSection('sec-regime', 'Regime Analysis', fallback);
}

// ─── Section 5: Signal History ───────────────────────────────────────────────

function renderSignalHistory(signals) {
  const sorted = [...(signals || [])].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  if (sorted.length === 0) return makeSection('sec-signals', 'Full Signal History (0 signals)', `<div style="color:${C.text_muted};padding:16px">No signals</div>`, false);

  let tbody = '';
  for (const sig of sorted.slice(0, 100)) {
    const dir = sig.direction || 'alert';
    const time = new Date(sig.timestamp).toLocaleString();
    tbody += `<tr><td>${time}</td><td>${sig.rule}</td><td><span class="signal-badge ${dir}">${dir}</span></td><td>${sig.indicator||''}</td><td>${sig.message||'\u2014'}</td></tr>`;
  }
  if (sorted.length > 100) {
    tbody += `<tr><td colspan="5" style="text-align:center;color:${C.text_muted}">... and ${sorted.length - 100} more signals</td></tr>`;
  }

  const table = `<table class="signal-table"><thead><tr><th>Time</th><th>Signal</th><th>Direction</th><th>Indicator</th><th>Message</th></tr></thead><tbody>${tbody}</tbody></table>`;
  return makeSection('sec-signals', `Full Signal History (${sorted.length} signals)`, table, false);
}

// ─── Section 6: RegimeFlex ───────────────────────────────────────────────────

function renderRegimeFlex(data) {
  const rows = data.regime_flex_history || [];
  if (rows.length === 0) return makeSection('sec-regimeflex', 'RegimeFlex Strategy', `<div style="color:${C.text_muted};padding:16px">No RegimeFlex history</div>`);

  const regimeColors = { R0:'#10b981', R1:'#f59e0b', R2:'#ef4444', R3:'#a855f7' };
  const regimeLabels = { R0:'Healthy Uptrend', R1:'Choppy/Extended', R2:'Risk-Off', R3:'Rebound Window' };

  const cur = rows[0];
  const curRegime = cur.regime || '--';
  const curExposure = cur.target_exposure || 0;
  const curSignal = cur.signal || 'HOLD';
  const rc = regimeColors[curRegime] || C.text_muted;
  const expColor = curExposure >= 80 ? '#10b981' : curExposure >= 40 ? '#f59e0b' : curExposure > 0 ? '#a855f7' : '#ef4444';

  let signalBadge;
  if (curSignal === 'BUY') signalBadge = `<span style="background:rgba(16,185,129,0.2);color:#10b981;border:1px solid #10b981;padding:4px 12px;border-radius:6px;font-weight:700">BUY</span>`;
  else if (curSignal === 'SELL') signalBadge = `<span style="background:rgba(239,68,68,0.2);color:#ef4444;border:1px solid #ef4444;padding:4px 12px;border-radius:6px;font-weight:700">SELL</span>`;
  else signalBadge = `<span style="color:${C.text_muted};border:1px solid ${C.border};padding:4px 12px;border-radius:6px">HOLD</span>`;

  const cards = `
    <div class="metric-cards" style="grid-template-columns:repeat(3,1fr)">
      <div class="metric-card" style="text-align:center"><div class="metric-label">Current Regime</div><div class="metric-value" style="color:${rc}">${curRegime}</div><div class="metric-sub">${regimeLabels[curRegime]||curRegime}</div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Target Exposure</div><div class="metric-value" style="color:${expColor}">${curExposure}%</div><div class="confidence-bar" style="width:80%;margin:6px auto 0"><div class="confidence-bar-fill" style="width:${Math.min(curExposure,100)}%;background:${expColor}"></div></div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Signal</div><div style="margin-top:4px">${signalBadge}</div></div>
    </div>`;

  let tbody = '';
  for (const row of rows) {
    const sig = row.signal || 'HOLD';
    const regime = row.regime || '--';
    const bg = sig === 'BUY' ? 'rgba(16,185,129,0.10)' : sig === 'SELL' ? 'rgba(239,68,68,0.10)' : 'transparent';
    const exposure = row.target_exposure || 0;
    const ec = exposure >= 80 ? '#10b981' : exposure >= 40 ? '#f59e0b' : exposure > 0 ? '#a855f7' : '#ef4444';
    tbody += `<tr style="background:${bg}"><td>${row.date}</td><td style="color:${regimeColors[regime]||C.text_muted};font-weight:600">${regime}</td><td style="text-align:right;font-family:monospace;color:${ec}">${exposure}%</td><td style="color:${sig==='BUY'?'#10b981':sig==='SELL'?'#ef4444':C.text_muted};font-weight:${sig!=='HOLD'?'600':'400'}">${sig}</td></tr>`;
  }

  const table = `<div style="overflow-x:auto;margin-top:12px"><table class="signal-table"><thead><tr><th>Date</th><th>Regime</th><th>Exposure</th><th>Signal</th></tr></thead><tbody>${tbody}</tbody></table></div>`;
  return makeSection('sec-regimeflex', `RegimeFlex Strategy (${rows.length} bars)`, cards + table);
}

// ─── Section 7: SectorPulse ──────────────────────────────────────────────────

function renderSectorPulse(data) {
  const rows = data.sector_pulse_history || [];
  if (rows.length === 0) return makeSection('sec-sectorpulse', 'SectorPulse Strategy', `<div style="color:${C.text_muted};padding:16px">No SectorPulse history</div>`);

  const regimeColors = { R0:'#10b981', R1:'#f59e0b', R2:'#ef4444', R3:'#a855f7' };
  const cur = rows[0];
  const mom = cur.momentum_score || 0;
  const momColor = mom > 5 ? '#10b981' : mom > 0 ? '#06b6d4' : mom > -5 ? '#f59e0b' : '#ef4444';
  const curRegime = cur.regime || '--';

  const cards = `
    <div class="metric-cards" style="grid-template-columns:repeat(2,1fr)">
      <div class="metric-card" style="text-align:center"><div class="metric-label">20-Day Momentum</div><div class="metric-value" style="color:${momColor}">${mom>=0?'+':''}${mom.toFixed(2)}%</div><div class="metric-sub">${mom>5?'Strong upside':mom>0?'Mild upside':mom>-5?'Mild downside':'Strong downside'}</div></div>
      <div class="metric-card" style="text-align:center"><div class="metric-label">Regime</div><div class="metric-value" style="color:${regimeColors[curRegime]||C.text_muted}">${curRegime}</div><div class="metric-sub">${curRegime==='R0'?'Full allocation OK':curRegime==='R1'?'Reduced allocation':curRegime==='R2'?'No new positions':curRegime==='R3'?'Small positions only':''}</div></div>
    </div>`;

  let tbody = '';
  for (const row of rows) {
    const m = row.momentum_score || 0;
    const mc = m > 5 ? '#10b981' : m > 0 ? '#06b6d4' : m > -5 ? '#f59e0b' : '#ef4444';
    const bg = m > 5 ? 'rgba(16,185,129,0.05)' : m < -5 ? 'rgba(239,68,68,0.05)' : 'transparent';
    const regime = row.regime || '--';
    tbody += `<tr style="background:${bg}"><td>${row.date}</td><td style="text-align:right;font-family:monospace;color:${mc}">${m>=0?'+':''}${m.toFixed(2)}%</td><td style="color:${regimeColors[regime]||C.text_muted};font-weight:600">${regime}</td></tr>`;
  }

  const table = `<div style="overflow-x:auto;margin-top:12px"><table class="signal-table"><thead><tr><th>Date</th><th>Momentum</th><th>Regime</th></tr></thead><tbody>${tbody}</tbody></table></div>`;
  return makeSection('sec-sectorpulse', `SectorPulse Strategy (${rows.length} bars)`, cards + table);
}

// ─── Section 8: Indicators ───────────────────────────────────────────────────

function renderIndicators(indicators) {
  if (!indicators?.categories?.length) return makeSection('sec-indicators', 'Indicators', `<div style="color:${C.text_muted};padding:16px">No indicators configured</div>`, false);

  let html = '';
  for (const cat of indicators.categories) {
    html += `<div style="margin-bottom:16px"><div style="font-size:12px;font-weight:600;color:${C.text_muted};text-transform:uppercase;margin-bottom:8px">${cat.name}</div><div class="metric-cards">`;
    for (const ind of cat.indicators) {
      const params = Object.entries(ind.params || {}).map(([k, v]) => `${k}=${v}`).join(', ');
      let rulesHtml = '';
      if (ind.rules?.length > 0) {
        rulesHtml = '<div style="margin-top:8px">' + ind.rules.map(r => {
          const dc = r.direction === 'buy' ? C.success : r.direction === 'sell' ? C.danger : C.text_muted;
          return `<div style="font-size:11px;padding:2px 0"><span style="color:${dc};font-weight:600">${r.id}</span> <span style="color:${C.text_muted}">${r.description||''}</span></div>`;
        }).join('') + '</div>';
      }
      html += `<div class="metric-card"><div class="metric-label">${ind.name}</div><div style="font-size:12px;color:${C.text_muted};margin-top:4px">${ind.description}${params ? '. Params: ' + params : ''}</div>${rulesHtml}</div>`;
    }
    html += '</div></div>';
  }

  return makeSection('sec-indicators', 'Indicators', html, false);
}

// ─── Page Lifecycle ──────────────────────────────────────────────────────────

async function loadAndRender(symbol, tf) {
  if (symbol === currentSymbol && tf === currentTf) return;
  currentSymbol = symbol;
  currentTf = tf;

  const select = document.getElementById('signals-symbol-select') as HTMLSelectElement;
  if (select) select.value = symbol;
  updateActiveTimeframe(tf);

  const data = await loadSymbolData(symbol, tf);
  if (!data) {
    document.getElementById('signals-chart')!.innerHTML = `<div class="empty-state">No data for ${symbol}. Run pipeline with this symbol.</div>`;
    document.getElementById('signals-sections')!.innerHTML = '';
    return;
  }

  // Render Plotly chart
  renderChart(data);

  // Build all 8 sections
  const sections = document.getElementById('signals-sections')!;
  const regimeHtml = await renderRegimeAnalysis(symbol, tf);

  sections.innerHTML = [
    renderConfluence(data, window.APEX?.summary),
    renderDualMACD(data),
    renderTrendPulse(data),
    regimeHtml,
    renderSignalHistory(data.signals),
    renderRegimeFlex(data),
    renderSectorPulse(data),
    renderIndicators(window.APEX?.indicators),
  ].join('');

  wireCollapsibles();
}

export async function init(apex, params) {
  populateSymbolSelect(apex);
  populateTimeframeTabs(apex);
  const symbol = params?.symbol || 'SPY';
  const tf = params?.tf || '1d';
  await loadAndRender(symbol, tf);
}

export async function update(apex, params) {
  const symbol = params?.symbol || currentSymbol || 'SPY';
  const tf = params?.tf || currentTf || '1d';
  // Force re-render on symbol/tf change
  if (symbol !== currentSymbol || tf !== currentTf) {
    currentSymbol = null;
    currentTf = null;
  }
  await loadAndRender(symbol, tf);
}
