import { useState, useMemo, useEffect, useRef } from "react"
import { useSearchParams } from "react-router-dom"
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  type IChartApi,
  type Time,
} from "lightweight-charts"
import { useSymbols, useHistory, useSignalData, useIndicators, useSummary } from "@/lib/api"
import { useMarketStore } from "@/stores/market"
import { useWebSocket } from "@/hooks/useWebSocket"
import { CandlestickChart } from "@/components/CandlestickChart"
import { Collapsible } from "@/components/Collapsible"
import type { OHLCV } from "@/lib/ws"

// ── Types ────────────────────────────────────────────────────────────────────

interface ChartData {
  timestamps: string[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  volume: number[]
  overlays?: Record<string, (number | null)[]>
  rsi?: Record<string, (number | null)[]>
  dual_macd?: Record<string, (number | null)[]>
  macd?: Record<string, (number | null)[]>
  price_levels?: Record<string, (number | null)[]>
}

interface R2Signal {
  timestamp: string
  rule: string
  direction: string
  indicator?: string
  message?: string
}

interface SignalDataResponse {
  symbol: string
  timeframe: string
  bar_count: number
  chart_data: ChartData
  signals?: R2Signal[]
  dual_macd_history?: DualMACDRow[]
  trend_pulse_history?: TrendPulseRow[]
  regime_flex_history?: RegimeFlexRow[]
  sector_pulse_history?: SectorPulseRow[]
}

interface DualMACDRow {
  date: string
  slow_histogram: number
  fast_histogram: number
  slow_hist_delta: number
  fast_hist_delta: number
  trend_state: string
  tactical_signal: string
  momentum_balance: string
  confidence: number
}

interface TrendPulseRow {
  date: string
  swing_signal: string
  entry_signal: boolean
  trend_filter: string
  dm_state: string
  adx: number
  trend_strength: number
  top_warning: string
  confidence_4f: number
  atr_stop_level: number
  cooldown_left: number
  exit_signal: string
  ema_alignment: string
  score: number
}

interface RegimeFlexRow {
  date: string
  regime: string
  target_exposure: number
  signal: string
}

interface SectorPulseRow {
  date: string
  momentum_score: number
  regime: string
}

interface ConfluenceData {
  alignment_score: number
  bullish_count: number
  bearish_count: number
  neutral_count: number
  diverging_pairs?: { ind1: string; ind2: string; reason: string }[]
}

// ── Constants ────────────────────────────────────────────────────────────────

const TIMEFRAMES = ["30m", "1h", "4h", "1d"] as const
const EMPTY_OHLCV: OHLCV[] = []
const EMPTY_STATES: Record<string, Record<string, unknown>[]> = {}

const DARK_CHART = {
  layout: { background: { color: "#0c0f14" }, textColor: "#94a3b8", fontSize: 11 },
  grid: { vertLines: { color: "rgba(45,55,72,0.3)" }, horzLines: { color: "rgba(45,55,72,0.3)" } },
  crosshair: { mode: 0 as const },
  timeScale: { borderColor: "#2d3748", timeVisible: true },
  rightPriceScale: { borderColor: "#2d3748" },
}

const OVERLAY_COLORS: Record<string, string> = {
  ema_ema_fast: "#f59e0b",
  ema_ema_slow: "#3b82f6",
  bollinger_bb_upper: "#3b82f6",
  bollinger_bb_middle: "#6366f1",
  bollinger_bb_lower: "#3b82f6",
  supertrend_supertrend: "#f59e0b",
  vwap_vwap: "#ec4899",
}

const REGIME_COLORS: Record<string, string> = {
  R0: "#22c55e", R1: "#f59e0b", R2: "#ef4444", R3: "#3b82f6",
}

const TREND_COLORS: Record<string, string> = {
  BULLISH: "#10b981", BEARISH: "#ef4444", DETERIORATING: "#f59e0b", IMPROVING: "#06b6d4", NEUTRAL: "#94a3b8",
}

// ── Utility ──────────────────────────────────────────────────────────────────

function toUnix(ts: string): number {
  return Math.floor(new Date(ts).getTime() / 1000)
}

function hasValues(arr?: (number | null)[]): boolean {
  return !!arr && arr.some((v) => v != null)
}

function fmtSigned(val: number | undefined, decimals: number): string {
  if (val == null) return "—"
  return `${val >= 0 ? "+" : ""}${val.toFixed(decimals)}`
}

/** Merge historical + live arrays, dedup by date field, most recent first. */
function mergeByDate<T extends { date: string }>(historical: T[], live: T[]): T[] {
  if (!live.length) return historical
  const seen = new Set(historical.map((r) => r.date))
  const merged = [...historical]
  for (const row of live) {
    if (!seen.has(row.date)) {
      merged.unshift(row)  // live rows are newer → prepend
      seen.add(row.date)
    }
  }
  return merged
}

// ── Main Component ───────────────────────────────────────────────────────────

export function Signals() {
  const [searchParams, setSearchParams] = useSearchParams()
  const { data: symbolsData } = useSymbols()
  const { data: summaryData } = useSummary()
  const { data: indicatorsData } = useIndicators()
  const { send } = useWebSocket()

  const [symbol, setSymbol] = useState(searchParams.get("symbol") ?? "")
  const [tf, setTf] = useState(searchParams.get("tf") ?? "1d")

  const symbolList = useMemo(() => {
    // Prefer live symbols (from Longbridge quote adapter)
    const liveSyms = Object.keys(symbolsData?.symbols ?? {})
    if (liveSyms.length > 0) return liveSyms.sort()
    // Fallback: summary tickers (from R2/static — always available)
    const summarySyms = ((summaryData as any)?.tickers ?? [])
      .map((t: any) => t.symbol)
      .filter(Boolean) as string[]
    return summarySyms.sort()
  }, [symbolsData, summaryData])

  // Auto-select first symbol
  useEffect(() => {
    if (!symbol && symbolList.length > 0) setSymbol(symbolList[0])
  }, [symbol, symbolList])

  // Update URL params
  useEffect(() => {
    if (symbol) setSearchParams({ symbol, tf }, { replace: true })
  }, [symbol, tf, setSearchParams])

  // WS subscribe — additive only, don't unsubscribe (global socket manages all symbols)
  useEffect(() => {
    if (!symbol) return
    send({ cmd: "subscribe", symbols: [symbol], types: ["quote", "bar", "indicator", "signal"] })
  }, [symbol, send])

  // R2 signal data (primary source)
  const { data: signalData, isLoading: sdLoading } = useSignalData(symbol, tf)
  const sd = signalData as SignalDataResponse | undefined
  const hasR2 = !!sd?.chart_data

  // Live history fallback
  const { data: historyData, isLoading: histLoading } = useHistory(symbol, tf)

  // Live WS data
  const quote = useMarketStore((s) => s.quotes[symbol])
  const wsSignals = useMarketStore((s) => s.signals)
  const liveBars = useMarketStore((s) => s.bars[symbol]?.[tf]) ?? EMPTY_OHLCV

  // Merge R2 + WS signals
  const allSignals = useMemo((): R2Signal[] => {
    const r2 = (sd?.signals ?? []) as R2Signal[]
    const ws: R2Signal[] = wsSignals
      .filter((s) => s.symbol === symbol)
      .map((s) => ({
        timestamp: s.timestamp,
        rule: s.rule,
        direction: s.direction === "bullish" ? "buy" : s.direction === "bearish" ? "sell" : "alert",
        indicator: "",
        message: "",
      }))
    return [...ws, ...r2]
  }, [sd?.signals, wsSignals, symbol])

  // Confluence from summary
  const confluence = useMemo(() => {
    const summary = summaryData as { confluence?: Record<string, ConfluenceData> } | undefined
    return summary?.confluence?.[`${symbol}_${tf}`]
  }, [summaryData, symbol, tf])

  // Live strategy states from WS (accumulated per bar close)
  const liveStates = useMarketStore((s) => s.strategyStates[symbol]?.[tf] ?? EMPTY_STATES)

  // Merge historical (REST) + live (WS) strategy states, dedup by date
  const mergedDualMacd = useMemo((): DualMACDRow[] => {
    const historical = sd?.dual_macd_history ?? []
    const live = (liveStates.dual_macd ?? []) as DualMACDRow[]
    return mergeByDate(historical, live)
  }, [sd?.dual_macd_history, liveStates.dual_macd])

  const mergedTrendPulse = useMemo((): TrendPulseRow[] => {
    const historical = sd?.trend_pulse_history ?? []
    const live = (liveStates.trend_pulse ?? []) as TrendPulseRow[]
    return mergeByDate(historical, live)
  }, [sd?.trend_pulse_history, liveStates.trend_pulse])

  const mergedRegimeFlex = useMemo((): RegimeFlexRow[] => {
    const historical = sd?.regime_flex_history ?? []
    const live = (liveStates.regime_detector ?? []) as RegimeFlexRow[]
    return mergeByDate(historical, live)
  }, [sd?.regime_flex_history, liveStates.regime_detector])

  const isLoading = histLoading  // signal-data computed on-demand, don't block

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-3">
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="rounded-md border border-input bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
        >
          {symbolList.length === 0 && <option value="">No symbols</option>}
          {symbolList.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>

        <div className="flex gap-1 rounded-lg bg-secondary p-1">
          {TIMEFRAMES.map((t) => (
            <button
              key={t}
              onClick={() => setTf(t)}
              className={`rounded-md px-3 py-1 text-sm transition-colors ${
                tf === t ? "bg-background text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {quote && (
          <div className="ml-auto flex items-baseline gap-2 text-sm">
            <span className="text-muted-foreground">Last: </span>
            <span className="font-medium">${quote.last.toFixed(2)}</span>
            {quote.prev_close != null && quote.prev_close > 0 && (() => {
              const chg = ((quote.last - quote.prev_close!) / quote.prev_close!) * 100
              return (
                <span className="font-medium" style={{ color: chg >= 0 ? "#10b981" : "#ef4444" }}>
                  {chg >= 0 ? "+" : ""}{chg.toFixed(2)}%
                </span>
              )
            })()}
            {quote.bid != null && quote.ask != null && (
              <span className="text-muted-foreground">
                {quote.bid.toFixed(2)} / {quote.ask.toFixed(2)}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Chart title */}
      {sd && (
        <p className="text-xs text-muted-foreground">
          {sd.symbol} - {sd.timeframe} ({sd.bar_count} bars)
        </p>
      )}

      {/* Multi-pane chart or fallback */}
      {isLoading ? (
        <div className="flex h-96 items-center justify-center rounded-lg border border-border text-sm text-muted-foreground">
          Loading chart data...
        </div>
      ) : hasR2 ? (
        <MultiPaneChart chartData={sd!.chart_data} signals={allSignals} />
      ) : (historyData?.bars?.length ?? 0) > 0 ? (
        <CandlestickChart bars={historyData!.bars} liveBars={liveBars} />
      ) : symbol ? (
        <div className="flex h-96 items-center justify-center rounded-lg border border-border text-sm text-muted-foreground">
          No data for {symbol} ({tf})
        </div>
      ) : null}

      {/* 8 Collapsible Sections */}
      <div className="space-y-3">
        <Collapsible title="Confluence Analysis">
          <ConfluenceSection confluence={confluence} signals={allSignals} chartData={sd?.chart_data} tf={tf} />
        </Collapsible>

        <Collapsible title={`DualMACD${mergedDualMacd.length ? ` (${mergedDualMacd.length} bars)` : ""}`}>
          <DualMACDSection rows={mergedDualMacd} />
        </Collapsible>

        <Collapsible title={`TrendPulse${mergedTrendPulse.length ? ` (${mergedTrendPulse.length} bars)` : ""}`}>
          <TrendPulseSection rows={mergedTrendPulse} />
        </Collapsible>

        <Collapsible title="Regime Analysis">
          <RegimeSection symbol={symbol} summaryData={summaryData as Record<string, unknown> | undefined} />
        </Collapsible>

        <Collapsible title={`Full Signal History (${allSignals.length} signals)`} defaultOpen={false}>
          <SignalHistorySection signals={allSignals} />
        </Collapsible>

        <Collapsible title={`RegimeFlex Strategy${mergedRegimeFlex.length ? ` (${mergedRegimeFlex.length} bars)` : ""}`}>
          <RegimeFlexSection rows={mergedRegimeFlex} />
        </Collapsible>

        <Collapsible title={`SectorPulse Strategy${sd?.sector_pulse_history?.length ? ` (${sd.sector_pulse_history.length} bars)` : ""}`}>
          <SectorPulseSection rows={sd?.sector_pulse_history ?? []} />
        </Collapsible>

        <Collapsible title="Indicators" defaultOpen={false}>
          <IndicatorsSection data={indicatorsData as Record<string, unknown> | undefined} />
        </Collapsible>
      </div>
    </div>
  )
}

// ── Multi-Pane Chart (Candlestick + RSI + DualMACD, synced) ──────────────────

function MultiPaneChart({ chartData, signals }: { chartData: ChartData; signals: R2Signal[] }) {
  const mainRef = useRef<HTMLDivElement>(null)
  const rsiRef = useRef<HTMLDivElement>(null)
  const macdRef = useRef<HTMLDivElement>(null)
  const chartsRef = useRef<IChartApi[]>([])

  useEffect(() => {
    if (!mainRef.current || !chartData?.timestamps?.length) return
    const ts = chartData.timestamps

    // Cleanup
    for (const c of chartsRef.current) try { c.remove() } catch { /* ok */ }
    chartsRef.current = []

    // === Main Candlestick Chart ===
    const mainChart = createChart(mainRef.current, { ...DARK_CHART, autoSize: true })
    chartsRef.current.push(mainChart)

    const candles = mainChart.addSeries(CandlestickSeries, {
      upColor: "#22c55e", downColor: "#ef4444",
      borderUpColor: "#22c55e", borderDownColor: "#ef4444",
      wickUpColor: "#22c55e", wickDownColor: "#ef4444",
    })
    candles.setData(ts.map((t, i) => ({
      time: toUnix(t) as Time,
      open: chartData.open[i], high: chartData.high[i],
      low: chartData.low[i], close: chartData.close[i],
    })))

    // Overlays (EMA, BB, SuperTrend)
    const overlayNames = [
      "ema_ema_fast", "ema_ema_slow",
      "bollinger_bb_upper", "bollinger_bb_lower",
      "supertrend_supertrend",
    ]
    for (const name of overlayNames) {
      const vals = chartData.overlays?.[name]
      if (!hasValues(vals)) continue
      const series = mainChart.addSeries(LineSeries, {
        color: OVERLAY_COLORS[name] ?? "#a0aec0",
        lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
      })
      const data: { time: Time; value: number }[] = []
      for (let i = 0; i < ts.length; i++) {
        if (vals![i] != null) data.push({ time: toUnix(ts[i]) as Time, value: vals![i]! })
      }
      series.setData(data)
    }

    // Volume histogram (bottom 20%)
    if (chartData.volume?.length > 0) {
      const vol = mainChart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" }, priceScaleId: "volume",
        priceLineVisible: false, lastValueVisible: false,
      })
      mainChart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 }, drawTicks: false, borderVisible: false,
      })
      vol.setData(ts.map((t, i) => ({
        time: toUnix(t) as Time,
        value: chartData.volume[i] ?? 0,
        color: chartData.close[i] >= chartData.open[i] ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
      })))
    }

    // === RSI Chart ===
    const rsiValues = chartData.rsi?.["rsi_rsi"]
    if (rsiRef.current && hasValues(rsiValues)) {
      rsiRef.current.innerHTML = ""
      const rsiChart = createChart(rsiRef.current, { ...DARK_CHART, autoSize: true })
      chartsRef.current.push(rsiChart)
      const rsiSeries = rsiChart.addSeries(LineSeries, {
        color: "#8b5cf6", lineWidth: 1.5, priceLineVisible: false,
      })
      const rsiData: { time: Time; value: number }[] = []
      for (let i = 0; i < ts.length; i++) {
        if (rsiValues![i] != null) rsiData.push({ time: toUnix(ts[i]) as Time, value: rsiValues![i]! })
      }
      rsiSeries.setData(rsiData)
      rsiSeries.createPriceLine({ price: 70, color: "rgba(239,68,68,0.5)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false })
      rsiSeries.createPriceLine({ price: 30, color: "rgba(34,197,94,0.5)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false })
      rsiChart.timeScale().fitContent()
    } else if (rsiRef.current) {
      rsiRef.current.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-size:12px">No RSI data</div>'
    }

    // === DualMACD Chart ===
    const slowHist = chartData.dual_macd?.["dual_macd_slow_histogram"]
    const fastHist = chartData.dual_macd?.["dual_macd_fast_histogram"]
    const hasDual = hasValues(slowHist) || hasValues(fastHist)

    if (macdRef.current && hasDual) {
      macdRef.current.innerHTML = ""
      const macdChart = createChart(macdRef.current, { ...DARK_CHART, autoSize: true })
      chartsRef.current.push(macdChart)

      // Slow histogram (green/red at 40% opacity)
      if (hasValues(slowHist)) {
        const s = macdChart.addSeries(HistogramSeries, { priceLineVisible: false, lastValueVisible: false, priceScaleId: "macd" })
        const d: { time: Time; value: number; color: string }[] = []
        for (let i = 0; i < ts.length; i++) {
          if (slowHist![i] != null)
            d.push({ time: toUnix(ts[i]) as Time, value: slowHist![i]!, color: slowHist![i]! >= 0 ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)" })
        }
        s.setData(d)
      }

      // Fast histogram (blue/orange at 80% opacity)
      if (hasValues(fastHist)) {
        const s = macdChart.addSeries(HistogramSeries, { priceLineVisible: false, lastValueVisible: false, priceScaleId: "macd" })
        const d: { time: Time; value: number; color: string }[] = []
        for (let i = 0; i < ts.length; i++) {
          if (fastHist![i] != null)
            d.push({ time: toUnix(ts[i]) as Time, value: fastHist![i]!, color: fastHist![i]! >= 0 ? "rgba(59,130,246,0.8)" : "rgba(224,122,59,0.8)" })
        }
        s.setData(d)
      }
      macdChart.timeScale().fitContent()
    } else if (macdRef.current) {
      // Fallback: single MACD
      const macdHist = chartData.macd?.["macd_histogram"]
      if (hasValues(macdHist)) {
        macdRef.current.innerHTML = ""
        const macdChart = createChart(macdRef.current, { ...DARK_CHART, autoSize: true })
        chartsRef.current.push(macdChart)
        const s = macdChart.addSeries(HistogramSeries, { priceLineVisible: false, lastValueVisible: false })
        const d: { time: Time; value: number; color: string }[] = []
        for (let i = 0; i < ts.length; i++) {
          if (macdHist![i] != null)
            d.push({ time: toUnix(ts[i]) as Time, value: macdHist![i]!, color: macdHist![i]! >= 0 ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)" })
        }
        s.setData(d)
        macdChart.timeScale().fitContent()
      } else if (macdRef.current) {
        macdRef.current.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-size:12px">No MACD data</div>'
      }
    }

    // Sync time scales across all panes
    if (chartsRef.current.length > 1) {
      let syncing = false
      for (const chart of chartsRef.current) {
        chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
          if (syncing || !range) return
          syncing = true
          for (const other of chartsRef.current) {
            if (other !== chart) other.timeScale().setVisibleLogicalRange(range)
          }
          syncing = false
        })
      }
    }

    mainChart.timeScale().fitContent()

    return () => {
      for (const c of chartsRef.current) try { c.remove() } catch { /* ok */ }
      chartsRef.current = []
    }
  }, [chartData, signals])

  return (
    <div>
      <div ref={mainRef} className="rounded-t-lg border border-border" style={{ height: 400 }} />
      <div ref={rsiRef} className="border-x border-border" style={{ height: 120 }} />
      <div ref={macdRef} className="rounded-b-lg border-x border-b border-border" style={{ height: 120 }} />
    </div>
  )
}

// ── Section 1: Confluence Analysis ───────────────────────────────────────────

function ConfluenceSection({
  confluence, signals, chartData, tf,
}: {
  confluence?: ConfluenceData
  signals: R2Signal[]
  chartData?: ChartData
  tf: string
}) {
  if (!confluence) return <p className="text-xs text-muted-foreground">No confluence data available</p>

  const alignPct = (confluence.alignment_score + 100) / 2
  const scoreColor = confluence.alignment_score > 20 ? "#10b981" : confluence.alignment_score < -20 ? "#ef4444" : "#94a3b8"

  // Active signals: latest per indicator
  const activeSignals = useMemo(() => {
    const byInd: Record<string, R2Signal> = {}
    const sorted = [...signals].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    for (const s of sorted) byInd[s.indicator ?? "unknown"] = s
    return Object.values(byInd).sort((a, b) => (a.indicator ?? "").localeCompare(b.indicator ?? ""))
  }, [signals])

  // Price levels grouped by type
  const priceLevels = chartData?.price_levels ?? {}
  const lastClose = chartData?.close?.length ? chartData.close[chartData.close.length - 1] : null

  const levelGroups = useMemo(() => {
    if (!Object.keys(priceLevels).length || !lastClose) return []
    const groups: Record<string, { level: string; value: number }[]> = {}
    for (const [col, vals] of Object.entries(priceLevels)) {
      if (!vals?.length) continue
      const lastVal = vals[vals.length - 1]
      if (lastVal == null) continue
      const parts = col.split("_")
      const ind = parts[0].toUpperCase()
      const level = parts.slice(1).join("_").toUpperCase()
      if (!groups[ind]) groups[ind] = []
      groups[ind].push({ level, value: lastVal })
    }
    return Object.entries(groups).map(([ind, levels]) => ({
      name: ind === "FIB" ? "Fibonacci" : ind === "SR" ? "Support/Res" : ind === "PIVOT" ? "Pivot Points" : ind,
      levels: levels.sort((a, b) => b.value - a.value).slice(0, 3),
    }))
  }, [priceLevels, lastClose])

  return (
    <div className="space-y-4">
      {/* Alignment meter + Active signals */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="relative h-3 rounded-full bg-gradient-to-r from-red-500 via-gray-600 to-emerald-500">
            <div
              className="absolute top-0 h-3 w-1 -translate-x-1/2 rounded bg-white"
              style={{ left: `${alignPct}%` }}
            />
          </div>
          <div className="mt-2 text-center text-2xl font-bold" style={{ color: scoreColor }}>
            {confluence.alignment_score > 0 ? "+" : ""}
            {Math.round(confluence.alignment_score)}
          </div>
          <div className="flex justify-center gap-4 text-xs">
            <span className="text-emerald-400">▲ {confluence.bullish_count} Bullish</span>
            <span className="text-muted-foreground">● {confluence.neutral_count} Neutral</span>
            <span className="text-red-400">▼ {confluence.bearish_count} Bearish</span>
          </div>
        </div>

        {activeSignals.length > 0 && (
          <div className="rounded-lg border border-border bg-card/50 p-3">
            <h4 className="mb-2 text-xs font-semibold">Active Signals - {tf.toUpperCase()}</h4>
            <div className="mb-2 flex gap-3 text-xs">
              <span className="text-emerald-400">▲ {activeSignals.filter((s) => s.direction === "buy").length} Bullish</span>
              <span className="text-red-400">▼ {activeSignals.filter((s) => s.direction === "sell").length} Bearish</span>
            </div>
            <div className="space-y-0.5">
              {activeSignals.slice(0, 8).map((s, i) => (
                <div key={i} className="flex justify-between text-[11px]">
                  <span className="text-muted-foreground">{s.indicator}</span>
                  <span className={s.direction === "buy" ? "text-emerald-400" : s.direction === "sell" ? "text-red-400" : "text-muted-foreground"}>
                    {s.rule}
                  </span>
                </div>
              ))}
              {activeSignals.length > 8 && (
                <div className="text-center text-[10px] text-muted-foreground">
                  +{activeSignals.length - 8} more
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Price Levels */}
      {levelGroups.length > 0 && lastClose && (
        <div>
          <h4 className="mb-2 text-[11px] font-semibold uppercase text-muted-foreground">
            Key Price Levels (${lastClose.toFixed(2)})
          </h4>
          <div className="flex flex-wrap gap-2">
            {levelGroups.map((g) => (
              <div key={g.name} className="rounded-md border border-border bg-card/50 p-2" style={{ minWidth: 140 }}>
                <div className="mb-1 text-[11px] font-semibold">{g.name}</div>
                {g.levels.map((l, i) => {
                  const diff = ((l.value - lastClose) / lastClose * 100).toFixed(2)
                  const above = l.value > lastClose
                  return (
                    <div key={i} className="flex justify-between text-[11px]">
                      <span className="text-muted-foreground">{l.level.replace(/_/g, " ")}</span>
                      <span>
                        ${l.value.toFixed(2)}{" "}
                        <span className={above ? "text-red-400" : "text-emerald-400"}>
                          {above ? "↑" : "↓"}{Math.abs(parseFloat(diff))}%
                        </span>
                      </span>
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Diverging Indicators */}
      <div>
        <h4 className="mb-1 text-[11px] font-semibold uppercase text-muted-foreground">Diverging Indicators</h4>
        {confluence.diverging_pairs?.length ? (
          confluence.diverging_pairs.slice(0, 5).map((p, i) => (
            <div key={i} className="text-xs">
              <span className="font-medium">{p.ind1} - {p.ind2}</span>{" "}
              <span className="text-muted-foreground">{p.reason}</span>
            </div>
          ))
        ) : (
          <p className="text-xs text-muted-foreground">No divergences detected - indicators are aligned</p>
        )}
      </div>
    </div>
  )
}

// ── Section 2: DualMACD ──────────────────────────────────────────────────────

function DualMACDSection({ rows }: { rows: DualMACDRow[] }) {
  if (rows.length === 0) return <p className="text-xs text-muted-foreground">No DualMACD history</p>

  const cur = rows[0]
  const trendColor = TREND_COLORS[cur.trend_state] ?? "#94a3b8"
  const confPct = Math.round((cur.confidence ?? 0) * 100)
  const confColor = confPct >= 50 ? "#10b981" : confPct >= 25 ? "#f59e0b" : "#94a3b8"

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-3">
        <SMetricCard label="Trend State" value={cur.trend_state} valueColor={trendColor} sub={`H_slow ${fmtSigned(cur.slow_histogram, 2)}`} />
        <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
          <p className="text-[10px] text-muted-foreground">Tactical Signal</p>
          <div className="mt-1"><TacticalBadge signal={cur.tactical_signal} /></div>
          <p className="mt-1 text-[10px] text-muted-foreground">H_fast {fmtSigned(cur.fast_histogram, 2)}</p>
        </div>
        <SMetricCard label="Momentum Balance" value={(cur.momentum_balance ?? "BALANCED").replace("_", " ")} />
        <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
          <p className="text-[10px] text-muted-foreground">Confidence</p>
          <p className="text-lg font-semibold" style={{ color: confColor }}>{confPct}%</p>
          <ConfBar pct={confPct} color={confColor} />
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground">As of {cur.date}</p>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="px-2 py-1.5">Date</th>
              <th className="px-2 py-1.5 text-right">H_slow</th>
              <th className="px-2 py-1.5 text-right">H_fast</th>
              <th className="px-2 py-1.5 text-right">ΔH_s</th>
              <th className="px-2 py-1.5 text-right">ΔH_f</th>
              <th className="px-2 py-1.5">Trend</th>
              <th className="px-2 py-1.5">Tactical</th>
              <th className="px-2 py-1.5">Mom.Bal</th>
              <th className="px-2 py-1.5">Conf</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const tc = TREND_COLORS[row.trend_state] ?? "#94a3b8"
              const conf = Math.round((row.confidence ?? 0) * 100)
              const cc = conf >= 50 ? "#10b981" : conf >= 25 ? "#f59e0b" : "#94a3b8"
              const bg = row.tactical_signal === "DIP_BUY" ? "rgba(16,185,129,0.10)"
                : row.tactical_signal === "RALLY_SELL" ? "rgba(239,68,68,0.10)" : undefined
              return (
                <tr key={i} className="border-b border-border/30" style={bg ? { background: bg } : undefined}>
                  <td className="px-2 py-1">{row.date}</td>
                  <td className="px-2 py-1 text-right font-mono">{fmtSigned(row.slow_histogram, 3)}</td>
                  <td className="px-2 py-1 text-right font-mono">{fmtSigned(row.fast_histogram, 3)}</td>
                  <td className="px-2 py-1 text-right font-mono">{fmtSigned(row.slow_hist_delta, 3)}</td>
                  <td className="px-2 py-1 text-right font-mono">{fmtSigned(row.fast_hist_delta, 3)}</td>
                  <td className="px-2 py-1 font-medium" style={{ color: tc }}>{row.trend_state}</td>
                  <td className="px-2 py-1"><TacticalCell signal={row.tactical_signal} /></td>
                  <td className="px-2 py-1 text-muted-foreground">{row.momentum_balance ?? "—"}</td>
                  <td className="px-2 py-1"><ConfBar pct={conf} color={cc} label /></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Section 3: TrendPulse ────────────────────────────────────────────────────

function TrendPulseSection({ rows }: { rows: TrendPulseRow[] }) {
  if (rows.length === 0) return <p className="text-xs text-muted-foreground">No TrendPulse history</p>

  const DM_COLORS: Record<string, string> = { BULLISH: "#10b981", IMPROVING: "#06b6d4", DETERIORATING: "#f59e0b", BEARISH: "#ef4444" }
  const cur = rows[0]
  const confPct = Math.round((cur.confidence_4f ?? 0) * 100)
  const confColor = confPct >= 50 ? "#10b981" : confPct >= 25 ? "#f59e0b" : "#94a3b8"
  const scoreColor = (cur.score ?? 0) >= 80 ? "#10b981" : (cur.score ?? 0) >= 50 ? "#f59e0b" : "#94a3b8"

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-3">
        {/* Signal */}
        <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
          <p className="text-[10px] text-muted-foreground">Signal</p>
          <div className="mt-1"><SwingBadge swing={cur.swing_signal} entry={cur.entry_signal} /></div>
          <p className="mt-1 text-[10px] text-muted-foreground">EMA: {(cur.ema_alignment ?? "MIXED").replace("ALIGNED_", "")}</p>
        </div>
        {/* Trend/DM */}
        <SMetricCard
          label="Trend / DM"
          value={cur.trend_filter ?? "NEUTRAL"}
          valueColor={TREND_COLORS[cur.trend_filter] ?? "#94a3b8"}
          sub={`MACD: ${cur.dm_state ?? "—"} · ADX ${Math.round(cur.adx ?? 0)}`}
        />
        {/* Risk */}
        <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
          <p className="text-[10px] text-muted-foreground">Risk</p>
          <div className="mt-1">
            {cur.top_warning === "TOP_DETECTED" ? <span className="text-sm font-bold text-purple-400">TOP_DETECTED</span>
              : cur.top_warning === "TOP_ZONE" ? <span className="text-sm font-bold text-amber-400">TOP_ZONE</span>
              : <span className="text-muted-foreground">—</span>}
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground">
            ATR ${(cur.atr_stop_level ?? 0) > 0 ? cur.atr_stop_level.toFixed(2) : "—"} · CD:{" "}
            <span className={`font-semibold ${(cur.cooldown_left ?? 0) === 0 ? "text-emerald-400" : "text-red-400"}`}>
              {cur.cooldown_left ?? 0}
            </span>
          </p>
        </div>
        {/* Score/Confidence */}
        <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
          <p className="text-[10px] text-muted-foreground">Score / Confidence</p>
          <p className="text-lg font-semibold" style={{ color: scoreColor }}>{Math.round(cur.score ?? 0)}</p>
          <ConfBar pct={confPct} color={confColor} />
          <p className="text-[10px] text-muted-foreground">4F Conf {confPct}%</p>
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground">As of {cur.date}</p>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="px-2 py-1.5">Date</th>
              <th className="px-2 py-1.5">Swing</th>
              <th className="px-2 py-1.5 text-center">Entry</th>
              <th className="px-2 py-1.5">MACD</th>
              <th className="px-2 py-1.5 text-right">ADX</th>
              <th className="px-2 py-1.5 text-right">Str</th>
              <th className="px-2 py-1.5">Top</th>
              <th className="px-2 py-1.5">Conf(4f)</th>
              <th className="px-2 py-1.5 text-right">ATR Stop</th>
              <th className="px-2 py-1.5 text-center">CD</th>
              <th className="px-2 py-1.5">Exit</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const dmc = DM_COLORS[row.dm_state] ?? "#94a3b8"
              const conf = Math.round((row.confidence_4f ?? 0) * 100)
              const cc = conf >= 50 ? "#10b981" : conf >= 25 ? "#f59e0b" : "#94a3b8"
              const bg = row.entry_signal ? "rgba(16,185,129,0.15)"
                : row.swing_signal === "BUY" ? "rgba(16,185,129,0.10)"
                : row.swing_signal === "SELL" ? "rgba(239,68,68,0.10)"
                : row.top_warning === "TOP_DETECTED" ? "rgba(168,85,247,0.10)" : undefined
              return (
                <tr key={i} className="border-b border-border/30" style={bg ? { background: bg } : undefined}>
                  <td className="px-2 py-1">{row.date}</td>
                  <td className="px-2 py-1">
                    {row.swing_signal === "BUY" ? <span className="font-semibold text-emerald-400">BUY</span>
                      : row.swing_signal === "SELL" ? <span className="font-semibold text-red-400">SELL</span>
                      : <span className="text-muted-foreground">—</span>}
                  </td>
                  <td className="px-2 py-1 text-center">
                    {row.entry_signal ? <span className="font-bold text-emerald-400">✓</span> : <span className="text-muted-foreground">—</span>}
                  </td>
                  <td className="px-2 py-1" style={{ color: dmc }}>{row.dm_state}</td>
                  <td className="px-2 py-1 text-right font-mono">{Math.round(row.adx ?? 0)}</td>
                  <td className="px-2 py-1 text-right font-mono">{(row.trend_strength ?? 0).toFixed(2)}</td>
                  <td className="px-2 py-1" style={{
                    color: row.top_warning && row.top_warning !== "NONE"
                      ? (row.top_warning === "TOP_DETECTED" ? "#a855f7" : "#f59e0b") : "#94a3b8"
                  }}>
                    {row.top_warning && row.top_warning !== "NONE" ? row.top_warning : "—"}
                  </td>
                  <td className="px-2 py-1"><ConfBar pct={conf} color={cc} label /></td>
                  <td className="px-2 py-1 text-right font-mono">
                    {(row.atr_stop_level ?? 0) > 0 ? `$${row.atr_stop_level.toFixed(2)}` : "—"}
                  </td>
                  <td className="px-2 py-1 text-center">
                    <span className={`font-semibold ${(row.cooldown_left ?? 0) === 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {row.cooldown_left ?? 0}
                    </span>
                  </td>
                  <td className="px-2 py-1" style={{ color: row.exit_signal && row.exit_signal !== "none" ? "#ef4444" : "#94a3b8" }}>
                    {row.exit_signal && row.exit_signal !== "none" ? row.exit_signal : "—"}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Section 4: Regime Analysis ───────────────────────────────────────────────

function RegimeSection({ symbol, summaryData }: { symbol: string; summaryData?: Record<string, unknown> }) {
  const tickers = (summaryData as { tickers?: { symbol: string; regime?: string; regime_name?: string; confidence?: number; component_states?: Record<string, string> }[] })?.tickers
  const ticker = tickers?.find((t) => t.symbol === symbol)

  if (!ticker?.regime)
    return <p className="text-xs text-muted-foreground">No regime data for {symbol}</p>

  const regime = ticker.regime
  const rc = REGIME_COLORS[regime] ?? "#94a3b8"
  const states = ticker.component_states ?? {}

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4">
        <div className="rounded-xl border-2 px-6 py-3" style={{ borderColor: rc, background: `${rc}20` }}>
          <span className="text-2xl font-bold" style={{ color: rc }}>{regime}</span>
        </div>
        <div>
          <div className="text-lg font-semibold">{ticker.regime_name ?? regime}</div>
          <div className="text-xs text-muted-foreground">Confidence: {ticker.confidence ?? "—"}%</div>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-3">
        {[
          { key: "trend_state", label: "Trend" },
          { key: "vol_state", label: "Volatility" },
          { key: "chop_state", label: "Choppiness" },
          { key: "ext_state", label: "Extension" },
        ].filter((c) => states[c.key]).map((c) => (
          <div key={c.key} className="rounded-lg border border-border bg-card/50 p-3 text-center">
            <p className="text-[10px] text-muted-foreground">{c.label}</p>
            <p className="mt-1 text-sm font-semibold">{(states[c.key] ?? "—").toUpperCase()}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Section 5: Full Signal History ───────────────────────────────────────────

function SignalHistorySection({ signals }: { signals: R2Signal[] }) {
  const sorted = useMemo(
    () => [...signals].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()).slice(0, 100),
    [signals],
  )

  if (sorted.length === 0)
    return <p className="text-xs text-muted-foreground">No signals</p>

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border text-left text-muted-foreground">
            <th className="px-2 py-1.5">Time</th>
            <th className="px-2 py-1.5">Signal</th>
            <th className="px-2 py-1.5">Direction</th>
            <th className="px-2 py-1.5">Indicator</th>
            <th className="px-2 py-1.5">Message</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((s, i) => (
            <tr key={i} className="border-b border-border/30">
              <td className="px-2 py-1 text-muted-foreground">{new Date(s.timestamp).toLocaleString()}</td>
              <td className="px-2 py-1">{s.rule}</td>
              <td className="px-2 py-1"><DirectionBadge dir={s.direction} /></td>
              <td className="px-2 py-1">{s.indicator ?? ""}</td>
              <td className="px-2 py-1 text-muted-foreground">{s.message ?? "—"}</td>
            </tr>
          ))}
          {signals.length > 100 && (
            <tr>
              <td colSpan={5} className="px-2 py-1 text-center text-muted-foreground">
                ... and {signals.length - 100} more signals
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

// ── Section 6: RegimeFlex ────────────────────────────────────────────────────

function RegimeFlexSection({ rows }: { rows: RegimeFlexRow[] }) {
  if (rows.length === 0) return <p className="text-xs text-muted-foreground">No RegimeFlex history</p>

  const REGIME_LABELS: Record<string, string> = { R0: "Healthy Uptrend", R1: "Choppy/Extended", R2: "Risk-Off", R3: "Rebound Window" }
  const cur = rows[0]
  const rc = REGIME_COLORS[cur.regime] ?? "#94a3b8"
  const expColor = (cur.target_exposure ?? 0) >= 80 ? "#10b981" : (cur.target_exposure ?? 0) >= 40 ? "#f59e0b" : (cur.target_exposure ?? 0) > 0 ? "#a855f7" : "#ef4444"

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3">
        <SMetricCard label="Current Regime" value={cur.regime ?? "—"} valueColor={rc} sub={REGIME_LABELS[cur.regime] ?? cur.regime} />
        <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
          <p className="text-[10px] text-muted-foreground">Target Exposure</p>
          <p className="text-lg font-semibold" style={{ color: expColor }}>{cur.target_exposure ?? 0}%</p>
          <ConfBar pct={Math.min(cur.target_exposure ?? 0, 100)} color={expColor} />
        </div>
        <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
          <p className="text-[10px] text-muted-foreground">Signal</p>
          <div className="mt-1">
            {cur.signal === "BUY" ? <span className="rounded-md border border-emerald-500 bg-emerald-900/20 px-3 py-1 text-sm font-bold text-emerald-400">BUY</span>
              : cur.signal === "SELL" ? <span className="rounded-md border border-red-500 bg-red-900/20 px-3 py-1 text-sm font-bold text-red-400">SELL</span>
              : <span className="rounded-md border border-border px-3 py-1 text-sm text-muted-foreground">HOLD</span>}
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="px-2 py-1.5">Date</th>
              <th className="px-2 py-1.5">Regime</th>
              <th className="px-2 py-1.5 text-right">Exposure</th>
              <th className="px-2 py-1.5">Signal</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const bg = row.signal === "BUY" ? "rgba(16,185,129,0.10)" : row.signal === "SELL" ? "rgba(239,68,68,0.10)" : undefined
              const ec = (row.target_exposure ?? 0) >= 80 ? "#10b981" : (row.target_exposure ?? 0) >= 40 ? "#f59e0b" : (row.target_exposure ?? 0) > 0 ? "#a855f7" : "#ef4444"
              return (
                <tr key={i} className="border-b border-border/30" style={bg ? { background: bg } : undefined}>
                  <td className="px-2 py-1">{row.date}</td>
                  <td className="px-2 py-1 font-semibold" style={{ color: REGIME_COLORS[row.regime] ?? "#94a3b8" }}>{row.regime}</td>
                  <td className="px-2 py-1 text-right font-mono" style={{ color: ec }}>{row.target_exposure ?? 0}%</td>
                  <td className="px-2 py-1">
                    {row.signal === "BUY" ? <span className="font-semibold text-emerald-400">BUY</span>
                      : row.signal === "SELL" ? <span className="font-semibold text-red-400">SELL</span>
                      : <span className="text-muted-foreground">HOLD</span>}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Section 7: SectorPulse ───────────────────────────────────────────────────

function SectorPulseSection({ rows }: { rows: SectorPulseRow[] }) {
  if (rows.length === 0) return <p className="text-xs text-muted-foreground">No SectorPulse history</p>

  const cur = rows[0]
  const mom = cur.momentum_score ?? 0
  const momColor = mom > 5 ? "#10b981" : mom > 0 ? "#06b6d4" : mom > -5 ? "#f59e0b" : "#ef4444"
  const rc = REGIME_COLORS[cur.regime] ?? "#94a3b8"

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <SMetricCard
          label="20-Day Momentum"
          value={`${mom >= 0 ? "+" : ""}${mom.toFixed(2)}%`}
          valueColor={momColor}
          sub={mom > 5 ? "Strong upside" : mom > 0 ? "Mild upside" : mom > -5 ? "Mild downside" : "Strong downside"}
        />
        <SMetricCard
          label="Regime"
          value={cur.regime ?? "—"}
          valueColor={rc}
          sub={cur.regime === "R0" ? "Full allocation OK" : cur.regime === "R1" ? "Reduced allocation" : cur.regime === "R2" ? "No new positions" : cur.regime === "R3" ? "Small positions only" : ""}
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-left text-muted-foreground">
              <th className="px-2 py-1.5">Date</th>
              <th className="px-2 py-1.5 text-right">Momentum</th>
              <th className="px-2 py-1.5">Regime</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const m = row.momentum_score ?? 0
              const mc = m > 5 ? "#10b981" : m > 0 ? "#06b6d4" : m > -5 ? "#f59e0b" : "#ef4444"
              const bg = m > 5 ? "rgba(16,185,129,0.05)" : m < -5 ? "rgba(239,68,68,0.05)" : undefined
              return (
                <tr key={i} className="border-b border-border/30" style={bg ? { background: bg } : undefined}>
                  <td className="px-2 py-1">{row.date}</td>
                  <td className="px-2 py-1 text-right font-mono" style={{ color: mc }}>{m >= 0 ? "+" : ""}{m.toFixed(2)}%</td>
                  <td className="px-2 py-1 font-semibold" style={{ color: REGIME_COLORS[row.regime] ?? "#94a3b8" }}>{row.regime}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Section 8: Indicators ────────────────────────────────────────────────────

interface IndicatorDef {
  name: string
  description: string
  params: Record<string, unknown>
  rules?: { id: string; direction: string; description: string }[]
}

interface IndicatorCategory {
  name: string
  indicators: IndicatorDef[]
}

function IndicatorsSection({ data }: { data?: Record<string, unknown> }) {
  const categories = (data as { categories?: IndicatorCategory[] })?.categories
  if (!categories?.length) return <p className="text-xs text-muted-foreground">No indicators configured</p>

  return (
    <div className="space-y-4">
      {categories.map((cat) => (
        <div key={cat.name}>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">{cat.name}</h4>
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-3">
            {cat.indicators.map((ind) => {
              const params = Object.entries(ind.params ?? {}).map(([k, v]) => `${k}=${v}`).join(", ")
              return (
                <div key={ind.name} className="rounded-lg border border-border bg-card/50 p-3">
                  <div className="text-xs font-semibold">{ind.name}</div>
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    {ind.description}{params ? `. Params: ${params}` : ""}
                  </div>
                  {ind.rules && ind.rules.length > 0 && (
                    <div className="mt-2 space-y-0.5">
                      {ind.rules.map((r) => (
                        <div key={r.id} className="text-[11px]">
                          <span className={`font-semibold ${r.direction === "buy" ? "text-emerald-400" : r.direction === "sell" ? "text-red-400" : "text-muted-foreground"}`}>
                            {r.id}
                          </span>{" "}
                          <span className="text-muted-foreground">{r.description ?? ""}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Shared Helper Components ─────────────────────────────────────────────────

function SMetricCard({ label, value, valueColor, sub }: {
  label: string; value: string | number; valueColor?: string; sub?: string
}) {
  return (
    <div className="rounded-lg border border-border bg-card/50 p-3 text-center">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold" style={valueColor ? { color: valueColor } : undefined}>{value}</p>
      {sub && <p className="mt-0.5 text-[10px] text-muted-foreground">{sub}</p>}
    </div>
  )
}

function ConfBar({ pct, color, label }: { pct: number; color: string; label?: boolean }) {
  return (
    <div className="flex items-center gap-1">
      <div className="h-1.5 flex-1 rounded-full bg-secondary">
        <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
      </div>
      {label && <span className="text-[10px]">{pct}</span>}
    </div>
  )
}

function DirectionBadge({ dir }: { dir: string }) {
  const cls =
    dir === "buy" ? "bg-emerald-900/50 text-emerald-400"
    : dir === "sell" ? "bg-red-900/50 text-red-400"
    : "bg-secondary text-muted-foreground"
  return <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${cls}`}>{dir}</span>
}

function TacticalBadge({ signal }: { signal: string }) {
  if (signal === "DIP_BUY")
    return <span className="rounded-md border border-emerald-500 bg-emerald-900/15 px-3 py-1 text-sm font-bold text-emerald-400">DIP BUY</span>
  if (signal === "RALLY_SELL")
    return <span className="rounded-md border border-red-500 bg-red-900/15 px-3 py-1 text-sm font-bold text-red-400">RALLY SELL</span>
  return <span className="rounded-md border border-border px-3 py-1 text-sm text-muted-foreground">No Signal</span>
}

function TacticalCell({ signal }: { signal: string }) {
  if (signal === "DIP_BUY") return <span className="font-semibold text-emerald-400">DIP_BUY</span>
  if (signal === "RALLY_SELL") return <span className="font-semibold text-red-400">RALLY_SELL</span>
  return <span className="text-muted-foreground">—</span>
}

function SwingBadge({ swing, entry }: { swing: string; entry?: boolean }) {
  if (entry)
    return <span className="rounded-md border border-emerald-500 bg-emerald-900/20 px-3 py-1 text-sm font-bold text-emerald-400">✓ ENTRY</span>
  if (swing === "BUY")
    return <span className="rounded-md border border-emerald-500 bg-emerald-900/15 px-3 py-1 text-sm font-bold text-emerald-400">BUY</span>
  if (swing === "SELL")
    return <span className="rounded-md border border-red-500 bg-red-900/15 px-3 py-1 text-sm font-bold text-red-400">SELL</span>
  return <span className="rounded-md border border-border px-3 py-1 text-sm text-muted-foreground">No Signal</span>
}
