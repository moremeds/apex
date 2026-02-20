/**
 * Chart Helpers — LC v5 candlestick, overlays, volume, signal markers,
 * RSI, DualMACD, and time-scale sync.
 * ES module — uses import map for lightweight-charts.
 */

import {
  createChart,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  AreaSeries,
  createSeriesMarkers,
} from 'lightweight-charts';

const DARK_THEME = {
  layout: {
    background: { color: '#0c0f14' },
    textColor: '#f0f4f8',
    fontSize: 12,
  },
  grid: {
    vertLines: { color: 'rgba(45,55,72,0.3)' },
    horzLines: { color: 'rgba(45,55,72,0.3)' },
  },
  crosshair: { mode: 0 },
  timeScale: { borderColor: '#2d3748', timeVisible: true },
  rightPriceScale: { borderColor: '#2d3748' },
};

/** Convert ISO timestamp to Unix seconds for LC v5 */
function toUnix(ts: string): number {
  return Math.floor(new Date(ts).getTime() / 1000);
}

/**
 * Create a candlestick chart in the given container.
 */
export function createCandlestickChart(container: HTMLElement, chartData: any) {
  container.innerHTML = '';

  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
  });

  const timestamps = chartData.timestamps || [];
  const data = timestamps.map((ts: string, i: number) => ({
    time: toUnix(ts),
    open: chartData.open[i],
    high: chartData.high[i],
    low: chartData.low[i],
    close: chartData.close[i],
  }));

  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: '#22c55e',
    downColor: '#ef4444',
    borderUpColor: '#22c55e',
    borderDownColor: '#ef4444',
    wickUpColor: '#22c55e',
    wickDownColor: '#ef4444',
  });
  candleSeries.setData(data);
  chart.timeScale().fitContent();

  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);

  return { chart, candleSeries, timestamps };
}

/** Overlay color palette */
const OVERLAY_COLORS: Record<string, string> = {
  ema_ema_fast: '#f59e0b',
  ema_ema_slow: '#3b82f6',
  sma_sma_fast: '#f59e0b',
  sma_sma_slow: '#3b82f6',
  bollinger_bb_upper: '#3b82f6',
  bollinger_bb_middle: '#6366f1',
  bollinger_bb_lower: '#3b82f6',
  supertrend_supertrend: '#f59e0b',
  vwap_vwap: '#ec4899',
  ichimoku_tenkan: '#f59e0b',
  ichimoku_kijun: '#3b82f6',
};

/** Add a line overlay series to the chart. */
export function addOverlay(chart: any, name: string, values: any[], timestamps: string[], color?: string) {
  const seriesColor = color || OVERLAY_COLORS[name] || '#a0aec0';
  const data: any[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (values[i] != null) {
      data.push({ time: toUnix(timestamps[i]), value: values[i] });
    }
  }

  const series = chart.addSeries(LineSeries, {
    color: seriesColor,
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
  });
  series.setData(data);
  return series;
}

/** Add volume histogram below the candlestick chart. */
export function addVolumeHistogram(chart: any, chartData: any) {
  const timestamps = chartData.timestamps || [];
  const data = timestamps.map((ts: string, i: number) => ({
    time: toUnix(ts),
    value: chartData.volume[i] || 0,
    color: (chartData.close[i] >= chartData.open[i])
      ? 'rgba(34,197,94,0.3)'
      : 'rgba(244,63,94,0.3)',
  }));

  const series = chart.addSeries(HistogramSeries, {
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
    priceLineVisible: false,
    lastValueVisible: false,
  });

  chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.8, bottom: 0 },
    drawTicks: false,
    borderVisible: false,
  });

  series.setData(data);
  return series;
}

/** Add signal markers to the candlestick series. */
export function addSignalMarkers(candleSeries: any, signals: any[], timestamps: string[]) {
  if (!signals || !signals.length) return;

  const markers: any[] = [];
  for (const sig of signals) {
    const ts = sig.timestamp;
    if (!ts) continue;
    const unixTs = toUnix(ts);
    const isBuy = sig.direction === 'buy';

    markers.push({
      time: unixTs,
      position: isBuy ? 'belowBar' : 'aboveBar',
      color: isBuy ? '#10b981' : '#f43f5e',
      shape: 'circle',
      text: sig.rule ? sig.rule.replace(/_/g, ' ') : (isBuy ? 'BUY' : 'SELL'),
      size: 1,
    });
  }

  markers.sort((a, b) => a.time - b.time);
  return createSeriesMarkers(candleSeries, markers);
}

/** Create a simple line chart (for regime timeline, etc.) */
export function createLineChart(container: HTMLElement, data: any[], options: any = {}) {
  container.innerHTML = '';
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
    ...options,
  });

  const series = chart.addSeries(LineSeries, {
    color: options.color || '#e07a3b',
    lineWidth: options.lineWidth || 2,
    priceLineVisible: false,
  });
  series.setData(data);
  chart.timeScale().fitContent();

  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);

  return { chart, series };
}

/** Create an RSI chart with 30/70 reference lines. */
export function createRSIChart(container: HTMLElement, rsiData: (number | null)[], timestamps: string[]) {
  container.innerHTML = '';
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
  });

  const data: any[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (rsiData[i] != null) {
      data.push({ time: toUnix(timestamps[i]), value: rsiData[i] });
    }
  }

  const series = chart.addSeries(LineSeries, {
    color: '#8b5cf6',
    lineWidth: 1.5,
    priceLineVisible: false,
    lastValueVisible: true,
  });
  series.setData(data);

  series.createPriceLine({ price: 70, color: 'rgba(239,68,68,0.5)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
  series.createPriceLine({ price: 30, color: 'rgba(34,197,94,0.5)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false });

  chart.timeScale().fitContent();

  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);

  return { chart, series };
}

/** Create a DualMACD chart with slow + fast histograms overlaid. */
export function createDualMACDChart(
  container: HTMLElement,
  slowData: (number | null)[],
  fastData: (number | null)[],
  timestamps: string[],
) {
  container.innerHTML = '';
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
  });

  // Slow histogram (55/89)
  const slowSeries = chart.addSeries(HistogramSeries, {
    priceLineVisible: false,
    lastValueVisible: false,
    priceScaleId: 'macd',
  });
  const slowPoints: any[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (slowData[i] != null) {
      slowPoints.push({
        time: toUnix(timestamps[i]),
        value: slowData[i],
        color: (slowData[i] as number) >= 0 ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)',
      });
    }
  }
  slowSeries.setData(slowPoints);

  // Fast histogram (13/21)
  const fastSeries = chart.addSeries(HistogramSeries, {
    priceLineVisible: false,
    lastValueVisible: false,
    priceScaleId: 'macd',
  });
  const fastPoints: any[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (fastData[i] != null) {
      fastPoints.push({
        time: toUnix(timestamps[i]),
        value: fastData[i],
        color: (fastData[i] as number) >= 0 ? 'rgba(59,130,246,0.8)' : 'rgba(224,122,59,0.8)',
      });
    }
  }
  fastSeries.setData(fastPoints);

  chart.timeScale().fitContent();

  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);

  return { chart, slowSeries, fastSeries };
}

/** Sync time scales across multiple LC charts. */
export function syncTimeScales(charts: any[]) {
  let isSyncing = false;
  for (const chart of charts) {
    chart.timeScale().subscribeVisibleLogicalRangeChange((range: any) => {
      if (isSyncing || !range) return;
      isSyncing = true;
      for (const other of charts) {
        if (other !== chart) {
          other.timeScale().setVisibleLogicalRange(range);
        }
      }
      isSyncing = false;
    });
  }
}

/** Create multi-line chart for equity curves (backtest). */
export function createMultiLineChart(
  container: HTMLElement,
  seriesDataList: { name: string; data: any[]; color: string }[],
  options: any = {},
) {
  container.innerHTML = '';
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
    ...options,
  });

  for (const s of seriesDataList) {
    const series = chart.addSeries(LineSeries, {
      color: s.color,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      title: s.name,
    });
    series.setData(s.data);
  }

  chart.timeScale().fitContent();

  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);

  return { chart };
}

/** Create an area chart (for drawdown curves). */
export function createAreaChart(
  container: HTMLElement,
  seriesDataList: { name: string; data: any[]; color: string }[],
) {
  container.innerHTML = '';
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
  });

  for (const s of seriesDataList) {
    const series = chart.addSeries(AreaSeries, {
      lineColor: s.color,
      topColor: 'transparent',
      bottomColor: s.color.replace(')', ',0.3)').replace('rgb', 'rgba'),
      lineWidth: 1.5,
      priceLineVisible: false,
      lastValueVisible: false,
      title: s.name,
    });
    series.setData(s.data);
  }

  chart.timeScale().fitContent();

  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);

  return { chart };
}
