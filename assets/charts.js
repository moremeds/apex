import {
  createChart,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  AreaSeries,
  createSeriesMarkers
} from "lightweight-charts";
const DARK_THEME = {
  layout: {
    background: { color: "#0c0f14" },
    textColor: "#f0f4f8",
    fontSize: 12
  },
  grid: {
    vertLines: { color: "rgba(45,55,72,0.3)" },
    horzLines: { color: "rgba(45,55,72,0.3)" }
  },
  crosshair: { mode: 0 },
  timeScale: { borderColor: "#2d3748", timeVisible: true },
  rightPriceScale: { borderColor: "#2d3748" }
};
function toUnix(ts) {
  return Math.floor(new Date(ts).getTime() / 1e3);
}
function createCandlestickChart(container, chartData) {
  container.innerHTML = "";
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true
  });
  const timestamps = chartData.timestamps || [];
  const data = timestamps.map((ts, i) => ({
    time: toUnix(ts),
    open: chartData.open[i],
    high: chartData.high[i],
    low: chartData.low[i],
    close: chartData.close[i]
  }));
  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: "#22c55e",
    downColor: "#ef4444",
    borderUpColor: "#22c55e",
    borderDownColor: "#ef4444",
    wickUpColor: "#22c55e",
    wickDownColor: "#ef4444"
  });
  candleSeries.setData(data);
  chart.timeScale().fitContent();
  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);
  return { chart, candleSeries, timestamps };
}
const OVERLAY_COLORS = {
  ema_ema_fast: "#f59e0b",
  ema_ema_slow: "#3b82f6",
  sma_sma_fast: "#f59e0b",
  sma_sma_slow: "#3b82f6",
  bollinger_bb_upper: "#3b82f6",
  bollinger_bb_middle: "#6366f1",
  bollinger_bb_lower: "#3b82f6",
  supertrend_supertrend: "#f59e0b",
  vwap_vwap: "#ec4899",
  ichimoku_tenkan: "#f59e0b",
  ichimoku_kijun: "#3b82f6"
};
function addOverlay(chart, name, values, timestamps, color) {
  const seriesColor = color || OVERLAY_COLORS[name] || "#a0aec0";
  const data = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (values[i] != null) {
      data.push({ time: toUnix(timestamps[i]), value: values[i] });
    }
  }
  const series = chart.addSeries(LineSeries, {
    color: seriesColor,
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false
  });
  series.setData(data);
  return series;
}
function addVolumeHistogram(chart, chartData) {
  const timestamps = chartData.timestamps || [];
  const data = timestamps.map((ts, i) => ({
    time: toUnix(ts),
    value: chartData.volume[i] || 0,
    color: chartData.close[i] >= chartData.open[i] ? "rgba(34,197,94,0.3)" : "rgba(244,63,94,0.3)"
  }));
  const series = chart.addSeries(HistogramSeries, {
    priceFormat: { type: "volume" },
    priceScaleId: "volume",
    priceLineVisible: false,
    lastValueVisible: false
  });
  chart.priceScale("volume").applyOptions({
    scaleMargins: { top: 0.8, bottom: 0 },
    drawTicks: false,
    borderVisible: false
  });
  series.setData(data);
  return series;
}
function addSignalMarkers(candleSeries, signals, timestamps) {
  if (!signals || !signals.length) return;
  const markers = [];
  for (const sig of signals) {
    const ts = sig.timestamp;
    if (!ts) continue;
    const unixTs = toUnix(ts);
    const isBuy = sig.direction === "buy";
    markers.push({
      time: unixTs,
      position: isBuy ? "belowBar" : "aboveBar",
      color: isBuy ? "#10b981" : "#f43f5e",
      shape: "circle",
      text: sig.rule ? sig.rule.replace(/_/g, " ") : isBuy ? "BUY" : "SELL",
      size: 1
    });
  }
  markers.sort((a, b) => a.time - b.time);
  return createSeriesMarkers(candleSeries, markers);
}
function createLineChart(container, data, options = {}) {
  container.innerHTML = "";
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
    ...options
  });
  const series = chart.addSeries(LineSeries, {
    color: options.color || "#e07a3b",
    lineWidth: options.lineWidth || 2,
    priceLineVisible: false
  });
  series.setData(data);
  chart.timeScale().fitContent();
  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);
  return { chart, series };
}
function createRSIChart(container, rsiData, timestamps) {
  container.innerHTML = "";
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true
  });
  const data = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (rsiData[i] != null) {
      data.push({ time: toUnix(timestamps[i]), value: rsiData[i] });
    }
  }
  const series = chart.addSeries(LineSeries, {
    color: "#8b5cf6",
    lineWidth: 1.5,
    priceLineVisible: false,
    lastValueVisible: true
  });
  series.setData(data);
  series.createPriceLine({ price: 70, color: "rgba(239,68,68,0.5)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
  series.createPriceLine({ price: 30, color: "rgba(34,197,94,0.5)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
  chart.timeScale().fitContent();
  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);
  return { chart, series };
}
function createDualMACDChart(container, slowData, fastData, timestamps) {
  container.innerHTML = "";
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true
  });
  const slowSeries = chart.addSeries(HistogramSeries, {
    priceLineVisible: false,
    lastValueVisible: false,
    priceScaleId: "macd"
  });
  const slowPoints = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (slowData[i] != null) {
      slowPoints.push({
        time: toUnix(timestamps[i]),
        value: slowData[i],
        color: slowData[i] >= 0 ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)"
      });
    }
  }
  slowSeries.setData(slowPoints);
  const fastSeries = chart.addSeries(HistogramSeries, {
    priceLineVisible: false,
    lastValueVisible: false,
    priceScaleId: "macd"
  });
  const fastPoints = [];
  for (let i = 0; i < timestamps.length; i++) {
    if (fastData[i] != null) {
      fastPoints.push({
        time: toUnix(timestamps[i]),
        value: fastData[i],
        color: fastData[i] >= 0 ? "rgba(59,130,246,0.8)" : "rgba(224,122,59,0.8)"
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
function syncTimeScales(charts) {
  let isSyncing = false;
  for (const chart of charts) {
    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
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
function createMultiLineChart(container, seriesDataList, options = {}) {
  container.innerHTML = "";
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true,
    ...options
  });
  for (const s of seriesDataList) {
    const series = chart.addSeries(LineSeries, {
      color: s.color,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      title: s.name
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
function createAreaChart(container, seriesDataList) {
  container.innerHTML = "";
  const chart = createChart(container, {
    ...DARK_THEME,
    width: container.clientWidth,
    height: container.clientHeight,
    autoSize: true
  });
  for (const s of seriesDataList) {
    const series = chart.addSeries(AreaSeries, {
      lineColor: s.color,
      topColor: "transparent",
      bottomColor: s.color.replace(")", ",0.3)").replace("rgb", "rgba"),
      lineWidth: 1.5,
      priceLineVisible: false,
      lastValueVisible: false,
      title: s.name
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
export {
  addOverlay,
  addSignalMarkers,
  addVolumeHistogram,
  createAreaChart,
  createCandlestickChart,
  createDualMACDChart,
  createLineChart,
  createMultiLineChart,
  createRSIChart,
  syncTimeScales
};
