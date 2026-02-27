import { useState, useMemo } from "react"
import { useAdvisor } from "@/lib/api"
import { useMarketStore } from "@/stores/market"
import type { AdvisorMarketContext, PremiumAdvice, EquityAdvice } from "@/lib/ws"

// ── Helpers ─────────────────────────────────────────

const actionColor: Record<string, string> = {
  STRONG_BUY: "bg-emerald-600 text-white",
  BUY: "bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/30",
  HOLD: "bg-amber-500/20 text-amber-400 ring-1 ring-amber-500/30",
  SELL: "bg-red-500/20 text-red-400 ring-1 ring-red-500/30",
  STRONG_SELL: "bg-red-600 text-white",
  BLOCKED: "bg-zinc-700 text-zinc-400 ring-1 ring-zinc-600",
}

const regimeColor: Record<string, string> = {
  R0: "bg-emerald-500/20 text-emerald-400",
  R1: "bg-amber-500/20 text-amber-400",
  R2: "bg-red-500/20 text-red-400",
  R3: "bg-blue-500/20 text-blue-400",
}

function Badge({ label, className }: { label: string; className?: string }) {
  return (
    <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ${className ?? ""}`}>
      {label}
    </span>
  )
}

function ConfidenceBar({ value }: { value: number }) {
  const clamped = Math.max(0, Math.min(100, value))
  const color = clamped >= 60 ? "bg-emerald-500" : clamped >= 30 ? "bg-amber-500" : "bg-red-500"
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 rounded-full bg-zinc-700">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${clamped}%` }} />
      </div>
      <span className="text-xs text-muted-foreground">{Math.round(clamped)}%</span>
    </div>
  )
}

// ── Market Context Bar ──────────────────────────────

function MarketContextBar({ ctx }: { ctx: AdvisorMarketContext | null }) {
  if (!ctx) return null
  return (
    <div className="mb-6 grid grid-cols-2 gap-4 rounded-lg border border-border bg-card p-4 sm:grid-cols-4 lg:grid-cols-7">
      <div>
        <div className="text-xs text-muted-foreground">Regime</div>
        <Badge label={`${ctx.regime} — ${ctx.regime_name}`} className={regimeColor[ctx.regime] ?? "bg-zinc-700 text-zinc-300"} />
      </div>
      <div>
        <div className="text-xs text-muted-foreground">Confidence</div>
        <span className="text-sm font-medium">{ctx.regime_confidence}%</span>
      </div>
      <div>
        <div className="text-xs text-muted-foreground">VIX</div>
        <span className="text-sm font-medium">{ctx.vix?.toFixed(1) ?? "—"}</span>
      </div>
      <div>
        <div className="text-xs text-muted-foreground">VIX %ile</div>
        <span className="text-sm font-medium">{ctx.vix_percentile ?? "—"}%</span>
      </div>
      <div>
        <div className="text-xs text-muted-foreground">VRP Z-Score</div>
        <span className={`text-sm font-medium ${(ctx.vrp_zscore ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
          {ctx.vrp_zscore?.toFixed(2) ?? "—"}
        </span>
      </div>
      <div>
        <div className="text-xs text-muted-foreground">Term Structure</div>
        <span className="text-sm font-medium capitalize">{ctx.term_structure_state ?? "—"}</span>
      </div>
      <div>
        <div className="text-xs text-muted-foreground">TS Ratio</div>
        <span className="text-sm font-medium">{ctx.term_structure_ratio?.toFixed(3) ?? "—"}</span>
      </div>
    </div>
  )
}

// ── Premium Strategy Cards ──────────────────────────

function PremiumCard({ advice }: { advice: PremiumAdvice }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold">{advice.symbol}</span>
          <Badge label={advice.action} className={actionColor[advice.action] ?? actionColor.HOLD} />
        </div>
        <ConfidenceBar value={advice.confidence} />
      </div>

      {advice.strategy && (
        <div className="mt-2 text-sm text-muted-foreground">
          Strategy: <span className="text-foreground">{advice.display_name ?? advice.strategy}</span>
        </div>
      )}

      <div className="mt-2 flex gap-4 text-xs text-muted-foreground">
        <span>VRP Z: {advice.vrp_zscore?.toFixed(2)}</span>
        <span>IV %ile: {advice.iv_percentile}%</span>
        <span>Regime: {advice.regime}</span>
      </div>

      {advice.earnings_warning && (
        <div className="mt-2 text-xs text-amber-400">Warning: {advice.earnings_warning}</div>
      )}

      {advice.legs.length > 0 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-xs text-primary hover:underline"
        >
          {expanded ? "Hide" : "Show"} legs ({advice.legs.length})
        </button>
      )}

      {expanded && advice.legs.length > 0 && (
        <div className="mt-2 space-y-1">
          {advice.legs.map((leg, i) => (
            <div key={i} className="rounded bg-zinc-800/50 px-2 py-1 text-xs">
              <span className={leg.side === "sell" ? "text-red-400" : "text-emerald-400"}>
                {leg.side.toUpperCase()}
              </span>{" "}
              {leg.option_type.toUpperCase()} | Delta: {leg.target_delta} | DTE: {leg.target_dte} |
              Strike: ${leg.estimated_strike?.toFixed(2)}
            </div>
          ))}
        </div>
      )}

      {advice.reasoning.length > 0 && (
        <div className="mt-2 space-y-0.5">
          {advice.reasoning.map((r, i) => (
            <div key={i} className="text-xs text-muted-foreground">
              &bull; {r}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Equity Table ────────────────────────────────────

function EquityTable({
  rows,
  sectorFilter,
  actionFilter,
}: {
  rows: EquityAdvice[]
  sectorFilter: string
  actionFilter: string
}) {
  const filtered = useMemo(() => {
    let result = rows
    if (sectorFilter) result = result.filter((r) => r.sector === sectorFilter)
    if (actionFilter) result = result.filter((r) => r.action === actionFilter)
    return result.sort((a, b) => b.confidence - a.confidence)
  }, [rows, sectorFilter, actionFilter])

  if (filtered.length === 0) {
    return <div className="py-8 text-center text-sm text-muted-foreground">No equity recommendations available</div>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-muted-foreground">
            <th className="pb-2 pr-4">Symbol</th>
            <th className="pb-2 pr-4">Sector</th>
            <th className="pb-2 pr-4">Action</th>
            <th className="pb-2 pr-4">Confidence</th>
            <th className="pb-2 pr-4">Regime</th>
            <th className="pb-2 pr-4">Signals</th>
            <th className="pb-2">Top Signal</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((row) => (
            <tr key={row.symbol} className="border-b border-border/50 hover:bg-zinc-800/30">
              <td className="py-2 pr-4 font-medium">{row.symbol}</td>
              <td className="py-2 pr-4 text-muted-foreground">{row.sector || "—"}</td>
              <td className="py-2 pr-4">
                <Badge label={row.action} className={actionColor[row.action] ?? actionColor.HOLD} />
              </td>
              <td className="py-2 pr-4">
                <ConfidenceBar value={row.confidence} />
              </td>
              <td className="py-2 pr-4">
                <Badge label={row.regime} className={regimeColor[row.regime] ?? "bg-zinc-700 text-zinc-300"} />
              </td>
              <td className="py-2 pr-4 text-xs text-muted-foreground">
                {row.signal_summary
                  ? `${row.signal_summary.bullish ?? 0}↑ ${row.signal_summary.bearish ?? 0}↓ ${row.signal_summary.neutral ?? 0}—`
                  : "—"}
              </td>
              <td className="py-2 text-xs text-muted-foreground">{row.top_signals?.[0]?.rule ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Main Page ───────────────────────────────────────

export function Advisor() {
  const { data, isLoading, error } = useAdvisor()

  // Live WS data (may be newer than REST)
  const wsCtx = useMarketStore((s) => s.advisorContext)
  const wsPremium = useMarketStore((s) => s.advisorPremium)
  const wsEquity = useMarketStore((s) => s.advisorEquity)

  // Prefer WS data when available
  const ctx = wsCtx ?? data?.market_context ?? null
  const premium = wsPremium.length > 0 ? wsPremium : data?.premium ?? []
  const equity = wsEquity.length > 0 ? wsEquity : data?.equity ?? []

  const [sectorFilter, setSectorFilter] = useState("")
  const [actionFilter, setActionFilter] = useState("")

  // Derive unique sectors and actions for filters
  const sectors = useMemo(() => [...new Set(equity.map((e) => e.sector).filter(Boolean))].sort(), [equity])
  const actions = useMemo(() => [...new Set(equity.map((e) => e.action).filter(Boolean))].sort(), [equity])

  if (isLoading) {
    return <div className="py-12 text-center text-muted-foreground">Loading advisor data...</div>
  }

  if (error) {
    return (
      <div className="py-12 text-center text-muted-foreground">
        Advisor service not available — data loads after market open
      </div>
    )
  }

  return (
    <div>
      <h1 className="mb-4 text-lg font-bold">Trading Advisor</h1>

      {/* Market Context */}
      <MarketContextBar ctx={ctx} />

      {/* Premium Strategies */}
      {premium.length > 0 && (
        <section className="mb-6">
          <h2 className="mb-3 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            Premium Strategies
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {premium.map((p) => (
              <PremiumCard key={p.symbol} advice={p} />
            ))}
          </div>
        </section>
      )}

      {/* Equity Recommendations */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            Equity Recommendations ({equity.length})
          </h2>
          <div className="flex gap-2">
            <select
              value={sectorFilter}
              onChange={(e) => setSectorFilter(e.target.value)}
              className="rounded border border-border bg-card px-2 py-1 text-xs text-foreground"
            >
              <option value="">All Sectors</option>
              {sectors.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <select
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
              className="rounded border border-border bg-card px-2 py-1 text-xs text-foreground"
            >
              <option value="">All Actions</option>
              {actions.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </div>
        </div>
        <EquityTable rows={equity} sectorFilter={sectorFilter} actionFilter={actionFilter} />
      </section>
    </div>
  )
}
