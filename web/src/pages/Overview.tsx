import { useMemo, useState, useCallback } from "react"
import { useNavigate } from "react-router-dom"
import type Plotly from "plotly.js"
import Plot from "react-plotly.js"
import { useSymbols, useSummary, useScoreHistory, useUniverse } from "@/lib/api"
import { useMarketStore } from "@/stores/market"

// ── ETF Configuration (mirrors static dashboard model.py) ────────────────────

const ETF_CONFIG = {
  market_indices: { name: "Market Indices", symbols: ["SPY", "QQQ", "IWM", "DIA"], style: "large" as const },
  commodities: { name: "Commodities & Safe Haven", symbols: ["GLD", "SLV"], style: "compact" as const },
  fixed_income: { name: "Fixed Income", symbols: ["TLT"], style: "compact" as const },
  volatility: { name: "Volatility", symbols: ["UVXY"], style: "compact" as const },
  sectors: { name: "Sector ETFs", symbols: ["XLK", "XLF", "XLV", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC", "XLY", "SMH"], style: "mini" as const },
} as const

const ETF_NAMES: Record<string, string> = {
  SPY: "S&P 500", QQQ: "NASDAQ 100", IWM: "Russell 2000", DIA: "Dow Jones",
  GLD: "Gold", SLV: "Silver", TLT: "Long Treasury", UVXY: "VIX Short",
  XLK: "Technology", XLC: "Communication", XLY: "Cons. Disc.", XLF: "Financials",
  XLV: "Healthcare", XLP: "Cons. Staples", XLE: "Energy", XLI: "Industrials",
  XLB: "Materials", XLRE: "Real Estate", XLU: "Utilities", SMH: "Semiconductors",
}

const ALL_DASHBOARD_ETFS = new Set(
  Object.values(ETF_CONFIG).flatMap((c) => c.symbols),
)

const REGIME_COLORS: Record<string, string> = {
  R0: "#22c55e", R1: "#f59e0b", R2: "#ef4444", R3: "#3b82f6",
}
const REGIME_NAMES: Record<string, string> = {
  R0: "Healthy Uptrend", R1: "Choppy / Extended", R2: "Risk-Off", R3: "Rebound Window",
}

const CATEGORY_ORDER = ["market_indices", "commodities", "fixed_income", "volatility", "sectors"] as const

// ── Color helpers (mirrors static dashboard) ─────────────────────────────────

function getScoreGradientColor(score: number): string {
  score = Math.max(0, Math.min(100, score))
  const hue = score <= 50 ? (score / 50) * 45 : 45 + ((score - 50) / 50) * 75
  const s = 0.7, l = 0.5
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1))
  const m = l - c / 2
  let r1: number, g1: number, b1: number
  if (hue < 60) { r1 = c; g1 = x; b1 = 0 }
  else if (hue < 120) { r1 = x; g1 = c; b1 = 0 }
  else { r1 = 0; g1 = c; b1 = x }
  const r = Math.round((r1 + m) * 255)
  const g = Math.round((g1 + m) * 255)
  const b = Math.round((b1 + m) * 255)
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
}

function getDailyChangeColor(pct: number | null): string {
  if (pct == null) return "#9ca3af"
  const clamped = Math.max(-5, Math.min(5, pct))
  if (clamped >= 0) {
    const t = clamped / 5
    const r = Math.round(200 - t * (200 - 34))
    const g = Math.round(220 - t * (220 - 197))
    const b = Math.round(180 - t * (180 - 94))
    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
  } else {
    const t = Math.abs(clamped) / 5
    const r = Math.round(200 + t * (239 - 200))
    const g = Math.round(200 - t * (200 - 68))
    const b = Math.round(200 - t * (200 - 68))
    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
  }
}

function getAlignmentColor(score: number | null): string {
  if (score == null) return "#9ca3af"
  const clamped = Math.max(-100, Math.min(100, score))
  if (clamped >= 0) {
    const g = Math.round(180 + (clamped / 100) * 40)
    return `#22${g.toString(16).padStart(2, "0")}5e`
  } else {
    const r = Math.round(200 + (Math.abs(clamped) / 100) * 44)
    return `#${r.toString(16).padStart(2, "0")}4444`
  }
}

// ── Types ────────────────────────────────────────────────────────────────────

interface TickerData {
  symbol: string
  close?: number
  last_close?: number
  daily_change_pct?: number
  trending_score?: number
  composite_score_avg?: number
  regime?: string
  alignment_score?: number
  market_cap?: number
  volume?: number
  sector?: string
}

interface ScoreSnapshot {
  date: string
  scores: Record<string, number>
}

type ColorBy = "trending" | "daily_change" | "alignment" | "regime"
type SizeBy = "market_cap" | "volume" | "equal"

// ── Main Component ───────────────────────────────────────────────────────────

export function Overview() {
  const navigate = useNavigate()
  const { data: symbolsData } = useSymbols()
  const { data: summaryRaw } = useSummary()
  const { data: scoreHistoryRaw } = useScoreHistory()
  const { data: universeRaw } = useUniverse()
  const quotes = useMarketStore((s) => s.quotes)
  const signals = useMarketStore((s) => s.signals)

  const [colorBy, setColorBy] = useState<ColorBy>("trending")
  const [sizeBy, setSizeBy] = useState<SizeBy>("market_cap")

  // Parse summary data
  const summary = summaryRaw as {
    generated_at?: string
    tickers?: TickerData[]
    signal_count_24h?: number
    regime_counts?: Record<string, number>
  } | undefined

  // Parse universe for sector + market cap enrichment
  const universeMap = useMemo(() => {
    const map: Record<string, { sector?: string; marketCap?: number }> = {}
    const udata = universeRaw as {
      tickers?: { symbol: string; sector?: string; marketCap?: number }[]
    } | undefined
    for (const t of udata?.tickers ?? []) {
      map[t.symbol] = { sector: t.sector, marketCap: t.marketCap }
    }
    return map
  }, [universeRaw])

  // Build tickerMap from summary, enriched with universe sector + market cap
  const tickerMap = useMemo(() => {
    const map: Record<string, TickerData> = {}
    for (const t of summary?.tickers ?? []) {
      if (!t.symbol) continue
      const uni = universeMap[t.symbol]
      map[t.symbol] = {
        ...t,
        sector: t.sector ?? uni?.sector,
        market_cap: t.market_cap ?? uni?.marketCap,
      }
    }
    return map
  }, [summary, universeMap])

  // Parse score history for sparklines
  const sparklines = useMemo(() => {
    const result: Record<string, number[]> = {}
    const snapshots = (scoreHistoryRaw as { snapshots?: ScoreSnapshot[] })?.snapshots
    if (!snapshots || snapshots.length < 2) return result
    const allSyms = new Set<string>()
    for (const snap of snapshots) {
      for (const sym of Object.keys(snap.scores ?? {})) allSyms.add(sym)
    }
    for (const sym of allSyms) {
      const pts = snapshots.map((s) => s.scores?.[sym]).filter((v): v is number => v != null)
      if (pts.length >= 2) result[sym] = pts
    }
    return result
  }, [scoreHistoryRaw])

  // Regime counts
  const regimeCounts = summary?.regime_counts ?? { R0: 0, R1: 0, R2: 0, R3: 0 }

  // Get price for a symbol (live quote → summary close → last_close → symbolsData)
  const getPrice = useCallback(
    (sym: string) =>
      quotes[sym]?.last ?? tickerMap[sym]?.close ?? tickerMap[sym]?.last_close ?? (symbolsData?.symbols?.[sym] as { last?: number })?.last,
    [quotes, tickerMap, symbolsData],
  )

  const getChange = useCallback(
    (sym: string) => {
      // If we have a live quote, compute change from previous close (last_close)
      const livePrice = quotes[sym]?.last
      const prevClose = tickerMap[sym]?.last_close ?? tickerMap[sym]?.close
      if (livePrice && prevClose && prevClose > 0) {
        return Math.round(((livePrice - prevClose) / prevClose) * 100 * 100) / 100
      }
      return tickerMap[sym]?.daily_change_pct ?? null
    },
    [quotes, tickerMap],
  )

  const getRegime = useCallback(
    (sym: string) => tickerMap[sym]?.regime ?? null,
    [tickerMap],
  )

  // Symbols not in ETF dashboard (for treemap)
  const allSymbolKeys = Object.keys(symbolsData?.symbols ?? {})
  const treemapSymbols = useMemo(() => {
    // Include tickers from summary that aren't ETFs
    const syms = new Set<string>()
    for (const t of summary?.tickers ?? []) {
      if (t.symbol && !ALL_DASHBOARD_ETFS.has(t.symbol)) syms.add(t.symbol)
    }
    // Also include from live symbols
    for (const s of allSymbolKeys) {
      if (!ALL_DASHBOARD_ETFS.has(s)) syms.add(s)
    }
    return Array.from(syms).sort()
  }, [summary, allSymbolKeys])

  // Missing caps count
  const missingCaps = useMemo(
    () => treemapSymbols.filter((s) => !tickerMap[s]?.market_cap).length,
    [treemapSymbols, tickerMap],
  )

  // Treemap data
  const treemapData = useMemo(() => {
    if (treemapSymbols.length === 0) return null

    const labels: string[] = []
    const parents: string[] = []
    const values: number[] = []
    const colors: string[] = []
    const texts: string[] = []

    // Group by sector
    const sectors = new Map<string, string[]>()
    for (const sym of treemapSymbols) {
      const sector = tickerMap[sym]?.sector ?? "Other"
      if (!sectors.has(sector)) sectors.set(sector, [])
      sectors.get(sector)!.push(sym)
    }

    // Build stock entries first (to compute sector totals for branchvalues="total")
    const sectorTotals = new Map<string, number>()
    const stockEntries: { sym: string; sector: string; size: number; color: string; text: string }[] = []

    for (const [sector, syms] of sectors) {
      let sectorTotal = 0
      for (const sym of syms) {
        const td = tickerMap[sym]
        const price = getPrice(sym) ?? 1
        const vol = td?.volume ?? 1_000_000
        const mcap = td?.market_cap ?? price * vol

        let size: number
        if (sizeBy === "market_cap") size = Math.max(mcap, 1)
        else if (sizeBy === "volume") size = Math.max(vol, 1)
        else size = 1

        let color: string
        if (colorBy === "trending") color = getScoreGradientColor(td?.trending_score ?? 50)
        else if (colorBy === "daily_change") color = getDailyChangeColor(td?.daily_change_pct ?? null)
        else if (colorBy === "regime") color = getScoreGradientColor(td?.composite_score_avg ?? 50)
        else color = getAlignmentColor(td?.alignment_score ?? null)

        const text = `${sym}<br>$${price.toFixed(2)}<br>${td?.daily_change_pct != null ? `${td.daily_change_pct >= 0 ? "+" : ""}${td.daily_change_pct.toFixed(2)}%` : ""}`
        stockEntries.push({ sym, sector, size, color, text })
        sectorTotal += size
      }
      sectorTotals.set(sector, sectorTotal)
    }

    // Root node (value = sum of all sectors)
    const rootTotal = Array.from(sectorTotals.values()).reduce((a, b) => a + b, 0)
    labels.push("Universe")
    parents.push("")
    values.push(rootTotal)
    colors.push("")
    texts.push("")

    // Sector nodes (value = sum of their stocks)
    for (const [sector] of sectors) {
      labels.push(sector)
      parents.push("Universe")
      values.push(sectorTotals.get(sector)!)
      colors.push("")
      texts.push("")
    }

    // Stock leaf nodes
    for (const entry of stockEntries) {
      labels.push(entry.sym)
      parents.push(entry.sector)
      values.push(entry.size)
      colors.push(entry.color)
      texts.push(entry.text)
    }

    return { labels, parents, values, colors, texts }
  }, [treemapSymbols, tickerMap, getPrice, colorBy, sizeBy])

  const sectorNames = useMemo(() => {
    const names = new Set<string>()
    for (const sym of treemapSymbols) {
      names.add(tickerMap[sym]?.sector ?? "Other")
    }
    return names
  }, [treemapSymbols, tickerMap])

  const handleTreemapClick = useCallback(
    (event: Plotly.PlotMouseEvent) => {
      const point = event.points?.[0]
      const label = (point as unknown as { label?: string })?.label
      if (label && label !== "Universe" && !sectorNames.has(label)) {
        navigate(`/signals?symbol=${label}&tf=1d`)
      }
    },
    [navigate, sectorNames],
  )

  const totalSymbols = Math.max(allSymbolKeys.length, summary?.tickers?.length ?? 0)

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold uppercase tracking-wider text-muted-foreground">
          Market Overview
        </h2>
        {summary?.generated_at && (
          <span className="text-xs text-muted-foreground">
            Updated: {new Date(summary.generated_at).toLocaleString()}
          </span>
        )}
      </div>

      {/* ETF Dashboard Cards */}
      {CATEGORY_ORDER.map((catKey) => {
        const cat = ETF_CONFIG[catKey]
        return (
          <section key={catKey}>
            <SectionTitle>{cat.name}</SectionTitle>
            <div
              className={
                cat.style === "large"
                  ? "grid grid-cols-4 gap-3"
                  : cat.style === "mini"
                    ? "grid grid-cols-6 gap-2"
                    : "grid grid-cols-4 gap-3"
              }
            >
              {cat.symbols.map((sym) => (
                <EtfCard
                  key={sym}
                  symbol={sym}
                  name={ETF_NAMES[sym] ?? sym}
                  price={getPrice(sym)}
                  change={getChange(sym)}
                  regime={getRegime(sym)}
                  sparkline={sparklines[sym]}
                  style={cat.style}
                  onClick={() => navigate(`/signals?symbol=${sym}&tf=1d`)}
                />
              ))}
            </div>
          </section>
        )
      })}

      {/* Stats bar */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 rounded-lg border border-border bg-card px-4 py-2 text-xs">
        <Stat label="Symbols" value={totalSymbols} />
        <Stat label="Signals (live)" value={signals.length} />
        <Stat label="Signals (24h)" value={summary?.signal_count_24h ?? "—"} />
        {missingCaps > 0 && <Stat label="Missing Caps" value={missingCaps} />}
        <div className="ml-auto flex gap-4">
          {(["R0", "R1", "R2", "R3"] as const).map((r) => (
            <span key={r} style={{ color: REGIME_COLORS[r] }}>
              {r}: <strong>{regimeCounts[r] ?? 0}</strong>
            </span>
          ))}
        </div>
      </div>

      {/* Controls bar */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          Color By
          <select
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value as ColorBy)}
            className="rounded border border-input bg-background px-2 py-1 text-xs text-foreground"
          >
            <option value="trending">Trending Score</option>
            <option value="daily_change">Daily Change</option>
            <option value="alignment">Alignment Score</option>
            <option value="regime">Regime</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          Size By
          <select
            value={sizeBy}
            onChange={(e) => setSizeBy(e.target.value as SizeBy)}
            className="rounded border border-input bg-background px-2 py-1 text-xs text-foreground"
          >
            <option value="market_cap">Market Cap</option>
            <option value="volume">Volume</option>
            <option value="equal">Equal Weight</option>
          </select>
        </label>
      </div>

      {/* Treemap */}
      {treemapData ? (
        <section>
          <SectionTitle>Stock Universe</SectionTitle>
          <div className="rounded-lg border border-border bg-card p-1">
            <Plot
              data={[
                {
                  type: "treemap",
                  labels: treemapData.labels,
                  parents: treemapData.parents,
                  values: treemapData.values,
                  text: treemapData.texts,
                  hoverinfo: "text",
                  textinfo: "label",
                  marker: {
                    colors: treemapData.colors,
                    line: { color: "#1a2332", width: 0.5 },
                  },
                  branchvalues: "total" as const,
                  textfont: { color: "#c0cad8", size: 12 },
                  pathbar: { visible: false },
                } as unknown as Plotly.Data,
              ]}
              layout={{
                margin: { t: 5, l: 5, r: 5, b: 5 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#8899aa", size: 11 },
                height: 450,
              } as Partial<Plotly.Layout>}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
              onClick={handleTreemapClick}
            />
          </div>
        </section>
      ) : totalSymbols === 0 ? (
        <div className="flex h-48 items-center justify-center rounded-lg border border-border text-sm text-muted-foreground">
          Waiting for symbol data...
        </div>
      ) : null}
    </div>
  )
}

// ── Sub-components ───────────────────────────────────────────────────────────

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
      {children}
    </h3>
  )
}

function EtfCard({
  symbol,
  name,
  price,
  change,
  regime,
  sparkline,
  style,
  onClick,
}: {
  symbol: string
  name: string
  price: number | undefined
  change: number | null
  regime: string | null
  sparkline?: number[]
  style: "large" | "compact" | "mini"
  onClick: () => void
}) {
  if (style === "mini") {
    return (
      <button
        onClick={onClick}
        className="flex flex-col items-center rounded-lg border border-border bg-card p-2 text-center transition-colors hover:border-primary/30"
      >
        <div className="flex items-center gap-1">
          <span className="text-xs font-bold">{symbol}</span>
          {regime && <RegimePill regime={regime} />}
        </div>
        {price != null && (
          <span className="mt-0.5 text-[10px] text-muted-foreground">
            ${price.toFixed(2)}
          </span>
        )}
        {change != null && (
          <span
            className="text-[9px] font-medium"
            style={{ color: change >= 0 ? "#10b981" : "#ef4444" }}
          >
            {change >= 0 ? "+" : ""}{change.toFixed(2)}%
          </span>
        )}
        <span className="mt-0.5 text-[9px] text-muted-foreground">{name}</span>
      </button>
    )
  }

  return (
    <button
      onClick={onClick}
      className="rounded-lg border border-border bg-card p-3 text-left transition-colors hover:border-primary/30"
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-bold">{symbol}</span>
        <span className="text-[10px] text-muted-foreground">{name}</span>
        {regime && <RegimePill regime={regime} />}
        {sparkline && <Sparkline points={sparkline} />}
      </div>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span className="text-lg font-semibold">
          {price != null ? `$${price.toFixed(2)}` : "—"}
        </span>
        {change != null && (
          <span
            className="text-sm font-medium"
            style={{ color: change >= 0 ? "#10b981" : "#ef4444" }}
          >
            {change >= 0 ? "+" : ""}{change.toFixed(2)}%
          </span>
        )}
      </div>
    </button>
  )
}

function RegimePill({ regime }: { regime: string }) {
  const color = REGIME_COLORS[regime] ?? "#6b7280"
  return (
    <span
      className="rounded px-1 py-0.5 text-[9px] font-bold"
      style={{ backgroundColor: `${color}22`, color }}
      title={REGIME_NAMES[regime] ?? regime}
    >
      {regime}
    </span>
  )
}

function Sparkline({ points, width = 60, height = 20 }: { points: number[]; width?: number; height?: number }) {
  if (points.length < 2) return null
  const delta = points[points.length - 1] - points[0]
  const color = delta > 3 ? "#10b981" : delta < -3 ? "#ef4444" : "#94a3b8"
  const min = Math.max(0, Math.min(...points) - 5)
  const max = Math.min(100, Math.max(...points) + 5)
  const range = max - min || 1
  const d = points
    .map(
      (v, i) =>
        `${i === 0 ? "M" : "L"}${((i / (points.length - 1)) * width).toFixed(1)},${(height - ((v - min) / range) * height).toFixed(1)}`,
    )
    .join(" ")
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="ml-1 inline-block align-middle">
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <span>
      <span className="text-muted-foreground">{label}: </span>
      <strong>{value}</strong>
    </span>
  )
}
