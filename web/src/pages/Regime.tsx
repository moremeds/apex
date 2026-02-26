import { useState, useMemo, useEffect, useRef } from "react"
import { useNavigate } from "react-router-dom"
import { createChart, LineSeries, type IChartApi, type Time } from "lightweight-charts"
import { useSummary, useSignalData } from "@/lib/api"

const REGIME_NAMES: Record<string, string> = {
  R0: "Healthy Uptrend",
  R1: "Choppy / Extended",
  R2: "Risk-Off",
  R3: "Rebound Window",
}
const REGIME_COLORS: Record<string, string> = {
  R0: "#22c55e", R1: "#f59e0b", R2: "#ef4444", R3: "#3b82f6",
}

const COMPONENT_LABELS: Record<string, { label: string; valueKey: string }> = {
  trend_state: { label: "Trend", valueKey: "ma50_slope" },
  vol_state: { label: "Volatility", valueKey: "atr_pct_63" },
  chop_state: { label: "Choppiness", valueKey: "chop_pct_252" },
  ext_state: { label: "Extension", valueKey: "ext" },
  iv_state: { label: "IV", valueKey: "iv_pct_63" },
}

function toUnix(ts: string): number {
  return Math.floor(new Date(ts).getTime() / 1000)
}

interface Ticker {
  symbol: string
  regime?: string
  regime_name?: string
  confidence?: number
  composite_score_avg?: number
  component_states?: Record<string, string>
  component_values?: Record<string, number>
}

export function Regime() {
  const navigate = useNavigate()
  const { data: summaryRaw, isLoading } = useSummary()
  const [filter, setFilter] = useState("all")

  const tickers = useMemo(() => {
    const summary = summaryRaw as { tickers?: Ticker[] } | undefined
    return summary?.tickers ?? []
  }, [summaryRaw])

  const benchmark = useMemo(
    () => tickers.find((t) => t.symbol === "SPY") ?? tickers.find((t) => t.symbol === "QQQ") ?? tickers[0],
    [tickers],
  )

  const { data: benchData } = useSignalData(benchmark?.symbol ?? "SPY", "1d")

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: tickers.length, R0: 0, R1: 0, R2: 0, R3: 0 }
    for (const t of tickers) {
      const r = t.regime ?? "R0"
      if (r in c) c[r]++
    }
    return c
  }, [tickers])

  const filtered = useMemo(
    () => (filter === "all" ? tickers : tickers.filter((t) => t.regime === filter)),
    [tickers, filter],
  )

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading regime data...</p>
  if (tickers.length === 0) return <p className="text-sm text-muted-foreground">No regime data available</p>

  const regime = benchmark?.regime ?? "R0"
  const rc = REGIME_COLORS[regime] ?? "#94a3b8"
  const states = benchmark?.component_states ?? {}
  const values = benchmark?.component_values ?? {}

  return (
    <div className="space-y-6">
      {/* Regime Badge */}
      <div className="flex flex-col items-center gap-3">
        <div
          className="rounded-xl border-2 px-8 py-4 text-center"
          style={{ borderColor: rc, background: `${rc}20` }}
        >
          <div className="text-3xl font-bold" style={{ color: rc }}>
            {regime} — {REGIME_NAMES[regime] ?? regime}
          </div>
        </div>
        <div className="text-sm text-muted-foreground">
          {benchmark?.symbol} | Confidence:{" "}
          {benchmark?.confidence != null ? `${benchmark.confidence.toFixed(0)}%` : "—"} | Composite
          Score: {benchmark?.composite_score_avg?.toFixed(1) ?? "—"}
        </div>
      </div>

      {/* Component Cards */}
      <div className="grid grid-cols-5 gap-3">
        {Object.entries(COMPONENT_LABELS)
          .filter(([key]) => states[key] !== undefined)
          .map(([key, meta]) => {
            const state = states[key] ?? "—"
            const val = values[meta.valueKey]
            const sc = getStateColor(state)
            return (
              <div key={key} className="rounded-lg border border-border bg-card p-3 text-center">
                <p className="text-xs text-muted-foreground">{meta.label}</p>
                <p className="mt-1 text-sm font-semibold" style={{ color: sc }}>
                  {state}
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Value: {val != null ? Number(val).toFixed(1) : "—"}
                </p>
              </div>
            )
          })}
      </div>

      {/* Exposure Timeline */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Exposure Timeline
        </h3>
        <ExposureTimeline data={benchData as Record<string, unknown> | undefined} />
      </div>

      {/* Filter Buttons */}
      <div className="flex items-center gap-2">
        {Object.entries(counts).map(([f, count]) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              filter === f
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-muted-foreground hover:text-foreground"
            }`}
          >
            {f === "all" ? "All" : f} ({count})
          </button>
        ))}
      </div>

      {/* Symbol Grid */}
      <div
        className="grid gap-2"
        style={{ gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))" }}
      >
        {filtered.map((t) => {
          const r = t.regime ?? "R0"
          const color = REGIME_COLORS[r] ?? "#94a3b8"
          return (
            <button
              key={t.symbol}
              onClick={() => navigate(`/signals?symbol=${t.symbol}&tf=1d`)}
              className="rounded-md border p-2 text-center text-xs transition-colors hover:brightness-125"
              style={{ borderColor: color, background: `${color}20` }}
              title={`${t.symbol} — ${REGIME_NAMES[r] ?? r}`}
            >
              <div className="font-medium" style={{ color }}>
                {t.symbol}
              </div>
              <div className="text-[10px] text-muted-foreground">{r}</div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Sub-components ──

function getStateColor(state: string): string {
  const s = state.toLowerCase()
  if (s.includes("trend_up") || s.includes("trending") || s.includes("vol_normal") || s.includes("neutral"))
    return "#22c55e"
  if (s.includes("trend_down") || s.includes("choppy") || s.includes("vol_high") || s.includes("overbought"))
    return "#ef4444"
  return "#94a3b8"
}

function ExposureTimeline({ data }: { data: Record<string, unknown> | undefined }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  const history = useMemo(() => {
    if (!data) return []
    const h = (data as { regime_flex_history?: { date: string; target_exposure: number }[] })
      .regime_flex_history
    return h ?? []
  }, [data])

  useEffect(() => {
    if (!containerRef.current || history.length === 0) return

    if (chartRef.current) {
      try { chartRef.current.remove() } catch { /* ok */ }
    }

    const chart = createChart(containerRef.current, {
      layout: { background: { color: "transparent" }, textColor: "#94a3b8", fontSize: 11 },
      grid: { vertLines: { color: "rgba(45,55,72,0.3)" }, horzLines: { color: "rgba(45,55,72,0.3)" } },
      crosshair: { mode: 0 },
      timeScale: { borderColor: "#2d3748" },
      rightPriceScale: { borderColor: "#2d3748" },
      autoSize: true,
    })
    chartRef.current = chart

    const reversed = [...history].reverse()
    const lineData = reversed.map((row) => ({
      time: toUnix(row.date) as Time,
      value: row.target_exposure,
    }))

    const series = chart.addSeries(LineSeries, {
      color: "#e07a3b",
      lineWidth: 2,
      priceLineVisible: false,
    })
    series.setData(lineData)

    series.createPriceLine({ price: 80, color: "rgba(34,197,94,0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false })
    series.createPriceLine({ price: 40, color: "rgba(245,158,11,0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false })
    series.createPriceLine({ price: 20, color: "rgba(239,68,68,0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false })

    chart.timeScale().fitContent()

    return () => {
      try { chart.remove() } catch { /* ok */ }
      chartRef.current = null
    }
  }, [history])

  if (history.length === 0)
    return <p className="py-4 text-center text-xs text-muted-foreground">No exposure timeline data</p>

  return <div ref={containerRef} style={{ height: 200 }} />
}
