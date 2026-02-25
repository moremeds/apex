import { useMemo } from "react"
import Plot from "react-plotly.js"
import { useMonitor, useSymbols } from "@/lib/api"
import { useMarketStore } from "@/stores/market"

/** Sector mapping — symbols not in this map fall under "Other". */
const SECTOR_MAP: Record<string, string> = {
  AAPL: "Technology", MSFT: "Technology", NVDA: "Technology", GOOG: "Technology",
  GOOGL: "Technology", META: "Technology", AVGO: "Technology", ADBE: "Technology",
  CRM: "Technology", AMD: "Technology", INTC: "Technology", QCOM: "Technology",
  AMAT: "Technology", MU: "Technology", LRCX: "Technology", KLAC: "Technology",
  MRVL: "Technology", SNPS: "Technology", CDNS: "Technology", PANW: "Technology",
  AMZN: "Consumer", TSLA: "Consumer", HD: "Consumer", NKE: "Consumer",
  SBUX: "Consumer", MCD: "Consumer", TGT: "Consumer", COST: "Consumer",
  JPM: "Financials", GS: "Financials", MS: "Financials", BAC: "Financials",
  V: "Financials", MA: "Financials", AXP: "Financials", BLK: "Financials",
  JNJ: "Healthcare", UNH: "Healthcare", PFE: "Healthcare", ABBV: "Healthcare",
  LLY: "Healthcare", TMO: "Healthcare", ABT: "Healthcare", MRK: "Healthcare",
  XOM: "Energy", CVX: "Energy", COP: "Energy", SLB: "Energy", OXY: "Energy",
  BA: "Industrials", CAT: "Industrials", GE: "Industrials", UNP: "Industrials",
  SPY: "ETFs", QQQ: "ETFs", IWM: "ETFs", DIA: "ETFs",
  XLK: "ETFs", XLF: "ETFs", XLE: "ETFs", XLV: "ETFs",
}

export function Overview() {
  const { data: symbolsData } = useSymbols()
  const { data: monitorData } = useMonitor()
  const quotes = useMarketStore((s) => s.quotes)
  const providers = useMarketStore((s) => s.providers)
  const signals = useMarketStore((s) => s.signals)

  const treemapData = useMemo(() => {
    const allSymbols = symbolsData?.symbols ?? {}
    const labels: string[] = ["APEX"]
    const parents: string[] = [""]
    const values: number[] = [0]
    const colors: number[] = [0]
    const hoverTexts: string[] = [""]

    const sectors = new Map<string, string[]>()

    for (const sym of Object.keys(allSymbols)) {
      const sector = SECTOR_MAP[sym] ?? "Other"
      if (!sectors.has(sector)) sectors.set(sector, [])
      sectors.get(sector)!.push(sym)
    }

    for (const [sector, syms] of sectors) {
      labels.push(sector)
      parents.push("APEX")
      values.push(0)
      colors.push(0)
      hoverTexts.push("")

      for (const sym of syms) {
        const info = allSymbols[sym]
        const liveQuote = quotes[sym]
        const price = liveQuote?.last ?? info?.last ?? 1
        const vol = liveQuote?.volume ?? info?.volume ?? 1_000_000
        const size = Math.max(price * vol, 1)

        labels.push(sym)
        parents.push(sector)
        values.push(size)
        colors.push(0) // placeholder — will use daily change when available
        hoverTexts.push(
          `${sym}<br>$${price.toFixed(2)}<br>Vol: ${(vol / 1e6).toFixed(1)}M`,
        )
      }
    }

    return { labels, parents, values, colors, hoverTexts }
  }, [symbolsData, quotes])

  const providerCount = providers.length > 0
    ? providers.filter((p) => p.connected).length
    : monitorData?.providers.filter((p) => p.connected).length ?? 0

  const totalSymbols = Object.keys(symbolsData?.symbols ?? {}).length

  return (
    <div className="space-y-4">
      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="Symbols" value={totalSymbols} />
        <StatCard label="Providers" value={`${providerCount} online`} />
        <StatCard label="Signals" value={signals.length} />
        <StatCard
          label="Uptime"
          value={monitorData ? formatUptime(monitorData.uptime_sec) : "--"}
        />
      </div>

      {/* Treemap */}
      {treemapData.labels.length > 1 ? (
        <div className="rounded-lg border border-border bg-card p-2">
          <Plot
            data={[
              {
                type: "treemap",
                labels: treemapData.labels,
                parents: treemapData.parents,
                values: treemapData.values,
                text: treemapData.hoverTexts,
                hoverinfo: "text",
                textinfo: "label",
                marker: {
                  colorscale: [
                    [0, "#ef4444"],
                    [0.5, "#f59e0b"],
                    [1, "#22c55e"],
                  ],
                },
                branchvalues: "total" as const,
              },
            ]}
            layout={{
              margin: { t: 10, l: 10, r: 10, b: 10 },
              paper_bgcolor: "transparent",
              font: { color: "#e5e5e5", size: 11 },
              height: 500,
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </div>
      ) : (
        <div className="flex h-64 items-center justify-center rounded-lg border border-border text-sm text-muted-foreground">
          Waiting for symbol data...
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  )
}

function formatUptime(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.round(sec / 60)}m`
  return `${(sec / 3600).toFixed(1)}h`
}
