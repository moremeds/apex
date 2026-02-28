import { Fragment, useState, useMemo } from "react"
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

const directionIcon: Record<string, string> = {
  bullish: "text-emerald-400",
  bearish: "text-red-400",
  neutral: "text-zinc-400",
}

type SortKey = "confidence" | "symbol" | "sector" | "action" | "regime"

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

function formatTimestamp(ts: string | undefined | null): string {
  if (!ts) return "—"
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

// ── Market Context Bar ──────────────────────────────

function MarketContextBar({ ctx }: { ctx: AdvisorMarketContext | null }) {
  if (!ctx) return null
  return (
    <div className="mb-6 rounded-lg border border-border bg-card p-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <div>
          <div className="text-xs text-muted-foreground">Regime</div>
          <div className="mt-1 flex items-center gap-2">
            <Badge label={ctx.regime} className={regimeColor[ctx.regime] ?? "bg-zinc-700 text-zinc-300"} />
            <span className="text-sm">{ctx.regime_name}</span>
          </div>
          <div className="mt-0.5 text-xs text-muted-foreground">Confidence: {ctx.regime_confidence}%</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">VIX</div>
          <div className="mt-1 text-lg font-semibold">{ctx.vix?.toFixed(1) ?? "—"}</div>
          <div className="text-xs text-muted-foreground">{Math.round(ctx.vix_percentile ?? 0)}th percentile</div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">VRP</div>
          <div className={`mt-1 text-lg font-semibold ${(ctx.vrp_zscore ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
            z={ctx.vrp_zscore?.toFixed(2) ?? "—"}
          </div>
          <div className="text-xs text-muted-foreground">
            {(ctx.vrp_zscore ?? 0) > 0.5 ? "Premium rich" : (ctx.vrp_zscore ?? 0) < -0.5 ? "Premium cheap" : "Fair value"}
          </div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Term Structure</div>
          <div className="mt-1 text-lg font-semibold">{ctx.term_structure_ratio?.toFixed(3) ?? "—"}</div>
          <div className="text-xs text-muted-foreground capitalize">
            {ctx.term_structure_state ?? "—"}
            {ctx.term_structure_state === "contango" && " ✓"}
            {ctx.term_structure_state === "inverted" && " ⚠"}
          </div>
        </div>
        <div>
          <div className="text-xs text-muted-foreground">Last Updated</div>
          <div className="mt-1 text-sm font-medium">{formatTimestamp(ctx.timestamp)}</div>
        </div>
      </div>
    </div>
  )
}

// ── Premium Strategy Cards ──────────────────────────

function PremiumCard({ advice }: { advice: PremiumAdvice }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div
      className="cursor-pointer rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/30"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-base font-bold">{advice.symbol}</span>
          <Badge label={advice.action} className={actionColor[advice.action] ?? actionColor.HOLD} />
        </div>
        <ConfidenceBar value={advice.confidence} />
      </div>

      {advice.strategy && (
        <div className="mt-2 text-sm">
          <span className="text-muted-foreground">Strategy: </span>
          <span className="font-medium">{advice.display_name ?? advice.strategy}</span>
        </div>
      )}

      {/* Quick metrics row */}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>VRP Z: <span className={(advice.vrp_zscore ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}>{advice.vrp_zscore?.toFixed(2)}</span></span>
        <span>IV %ile: {Math.round(advice.iv_percentile)}%</span>
        <span>TS: {advice.term_structure_ratio?.toFixed(3)}</span>
        <span>Regime: <Badge label={advice.regime} className={`${regimeColor[advice.regime] ?? ""} !py-0 !text-[10px]`} /></span>
      </div>

      {advice.earnings_warning && (
        <div className="mt-2 rounded bg-amber-500/10 px-2 py-1 text-xs text-amber-400">⚠ {advice.earnings_warning}</div>
      )}

      {/* Expanded: legs + reasoning */}
      {expanded && (
        <div className="mt-3 border-t border-border/50 pt-3">
          {advice.legs.length > 0 && (
            <div className="mb-3">
              <div className="mb-1 text-xs font-semibold text-muted-foreground uppercase">Legs</div>
              <div className="space-y-1">
                {advice.legs.map((leg, i) => (
                  <div key={i} className="flex items-center gap-3 rounded bg-zinc-800/50 px-3 py-1.5 text-xs">
                    <span className={`font-semibold ${leg.side === "sell" ? "text-red-400" : "text-emerald-400"}`}>
                      {leg.side.toUpperCase()}
                    </span>
                    <span className="font-medium">{leg.option_type.toUpperCase()}</span>
                    <span className="text-muted-foreground">Δ{leg.target_delta}</span>
                    <span className="text-muted-foreground">{leg.target_dte}d</span>
                    <span className="font-medium">~${leg.estimated_strike?.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {advice.reasoning.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-semibold text-muted-foreground uppercase">Reasoning</div>
              <ul className="space-y-0.5">
                {advice.reasoning.map((r, i) => (
                  <li key={i} className="text-xs text-muted-foreground">&bull; {r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Equity Detail Panel (expanded row) ──────────────

function EquityDetail({ row }: { row: EquityAdvice }) {
  const kl = row.key_levels ?? {}
  const tp = row.trend_pulse
  const hasKeyLevels = Object.keys(kl).length > 0
  const hasTopSignals = (row.top_signals ?? []).length > 0
  const hasReasoning = (row.reasoning ?? []).length > 0

  return (
    <div className="grid gap-4 px-4 pb-4 pt-2 sm:grid-cols-2 lg:grid-cols-4">
      {/* Top signals */}
      {hasTopSignals && (
        <div>
          <div className="mb-1.5 text-xs font-semibold text-muted-foreground uppercase">Top Signals</div>
          <div className="space-y-1">
            {row.top_signals.map((sig, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className={`${directionIcon[sig.direction] ?? "text-zinc-400"}`}>
                  {sig.direction === "bullish" ? "▲" : sig.direction === "bearish" ? "▼" : "─"}
                </span>
                <span className="font-medium">{sig.rule}</span>
                <span className="text-muted-foreground">({(sig.strength * 100).toFixed(0)}%)</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key levels */}
      {hasKeyLevels && (
        <div>
          <div className="mb-1.5 text-xs font-semibold text-muted-foreground uppercase">Key Levels</div>
          <div className="space-y-1 text-xs">
            {kl.resistance != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Resistance</span>
                <span className="font-medium text-red-400">${Number(kl.resistance).toFixed(2)}</span>
              </div>
            )}
            {kl.support != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Support</span>
                <span className="font-medium text-emerald-400">${Number(kl.support).toFixed(2)}</span>
              </div>
            )}
            {kl.atr_stop != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">ATR Stop</span>
                <span className="font-medium text-amber-400">${Number(kl.atr_stop).toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TrendPulse state */}
      {tp && Object.keys(tp).length > 0 && (
        <div>
          <div className="mb-1.5 text-xs font-semibold text-muted-foreground uppercase">TrendPulse</div>
          <div className="space-y-1 text-xs">
            {tp.zig_state != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Zig State</span>
                <span className="font-medium">{String(tp.zig_state)}</span>
              </div>
            )}
            {tp.macd_state != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">MACD State</span>
                <span className="font-medium">{String(tp.macd_state)}</span>
              </div>
            )}
            {tp.atr_trail != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">ATR Trail</span>
                <span className="font-medium">${Number(tp.atr_trail).toFixed(2)}</span>
              </div>
            )}
            {tp.confidence_score != null && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-medium">{Number(tp.confidence_score).toFixed(0)}%</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Reasoning */}
      {hasReasoning && (
        <div className={tp && Object.keys(tp).length > 0 ? "" : "lg:col-span-2"}>
          <div className="mb-1.5 text-xs font-semibold text-muted-foreground uppercase">Reasoning</div>
          <ul className="space-y-0.5">
            {row.reasoning.map((r, i) => (
              <li key={i} className="text-xs text-muted-foreground">&bull; {r}</li>
            ))}
          </ul>
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
  sortKey,
  sortDesc,
  onToggleSort,
  searchQuery,
}: {
  rows: EquityAdvice[]
  sectorFilter: string
  actionFilter: string
  sortKey: SortKey
  sortDesc: boolean
  onToggleSort: (key: SortKey) => void
  searchQuery: string
}) {
  const [expandedSymbol, setExpandedSymbol] = useState<string | null>(null)

  const filtered = useMemo(() => {
    let result = rows
    if (sectorFilter) result = result.filter((r) => r.sector === sectorFilter)
    if (actionFilter) result = result.filter((r) => r.action === actionFilter)
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      result = result.filter(
        (r) => r.symbol.toLowerCase().includes(q) || (r.sector ?? "").toLowerCase().includes(q),
      )
    }
    return [...result].sort((a, b) => {
      let cmp = 0
      switch (sortKey) {
        case "confidence":
          cmp = a.confidence - b.confidence
          break
        case "symbol":
          cmp = a.symbol.localeCompare(b.symbol)
          break
        case "sector":
          cmp = (a.sector ?? "").localeCompare(b.sector ?? "")
          break
        case "action":
          cmp = a.action.localeCompare(b.action)
          break
        case "regime":
          cmp = a.regime.localeCompare(b.regime)
          break
      }
      return sortDesc ? -cmp : cmp
    })
  }, [rows, sectorFilter, actionFilter, searchQuery, sortKey, sortDesc])

  if (filtered.length === 0) {
    return <div className="py-8 text-center text-sm text-muted-foreground">No equity recommendations available</div>
  }

  const columns: { key: SortKey; label: string; className?: string }[] = [
    { key: "symbol", label: "Symbol" },
    { key: "sector", label: "Sector" },
    { key: "action", label: "Action" },
    { key: "confidence", label: "Confidence" },
    { key: "regime", label: "Regime" },
  ]

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-card">
          <tr className="border-b border-border text-left text-xs text-muted-foreground">
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => onToggleSort(col.key)}
                className="cursor-pointer select-none px-3 py-2 hover:text-foreground"
              >
                <div className="flex items-center gap-1">
                  {col.label}
                  {sortKey === col.key && (sortDesc ? " ↓" : " ↑")}
                </div>
              </th>
            ))}
            <th className="px-3 py-2">Signals</th>
            <th className="px-3 py-2">Top Signal</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((row) => {
            const isExpanded = expandedSymbol === row.symbol
            return (
              <Fragment key={row.symbol}>
                <tr
                  onClick={() => setExpandedSymbol(isExpanded ? null : row.symbol)}
                  className={`cursor-pointer border-b transition-colors ${
                    isExpanded ? "border-primary/30 bg-zinc-800/40" : "border-border/50 hover:bg-zinc-800/30"
                  }`}
                >
                  <td className="px-3 py-2 font-medium">
                    <div className="flex items-center gap-1.5">
                      <span className={`text-[10px] transition-transform ${isExpanded ? "" : "-rotate-90"}`}>▼</span>
                      {row.symbol}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{row.sector || "—"}</td>
                  <td className="px-3 py-2">
                    <Badge label={row.action} className={actionColor[row.action] ?? actionColor.HOLD} />
                  </td>
                  <td className="px-3 py-2">
                    <ConfidenceBar value={row.confidence} />
                  </td>
                  <td className="px-3 py-2">
                    <Badge label={row.regime} className={regimeColor[row.regime] ?? "bg-zinc-700 text-zinc-300"} />
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    <span className="text-emerald-400">{row.signal_summary?.bullish ?? 0}↑</span>{" "}
                    <span className="text-red-400">{row.signal_summary?.bearish ?? 0}↓</span>{" "}
                    <span>{row.signal_summary?.neutral ?? 0}─</span>
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {row.top_signals?.[0] ? (
                      <span className={directionIcon[row.top_signals[0].direction] ?? ""}>
                        {row.top_signals[0].rule}
                      </span>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>
                {isExpanded && (
                  <tr className="border-b border-primary/20 bg-zinc-800/30">
                    <td colSpan={7}>
                      <EquityDetail row={row} />
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
      <div className="px-3 py-1.5 text-xs text-muted-foreground">{filtered.length} rows</div>
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
  const [searchQuery, setSearchQuery] = useState("")
  const [sortKey, setSortKey] = useState<SortKey>("confidence")
  const [sortDesc, setSortDesc] = useState(true)

  // Derive unique sectors and actions for filters
  const sectors = useMemo(() => [...new Set(equity.map((e) => e.sector).filter(Boolean))].sort(), [equity])
  const actions = useMemo(() => [...new Set(equity.map((e) => e.action).filter(Boolean))].sort(), [equity])

  // Counts by action for summary bar
  const actionCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const e of equity) {
      counts[e.action] = (counts[e.action] ?? 0) + 1
    }
    return counts
  }, [equity])

  const handleToggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDesc(!sortDesc)
    } else {
      setSortKey(key)
      setSortDesc(key === "confidence") // default desc for confidence, asc for others
    }
  }

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

      {/* Action summary bar */}
      {equity.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-3">
          {["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"].map((act) =>
            actionCounts[act] ? (
              <button
                key={act}
                onClick={() => setActionFilter(actionFilter === act ? "" : act)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  actionFilter === act ? "ring-2 ring-primary" : ""
                } ${actionColor[act] ?? ""}`}
              >
                {act.replace("_", " ")} ({actionCounts[act]})
              </button>
            ) : null,
          )}
          <span className="flex items-center text-xs text-muted-foreground">
            {equity.length} total
          </span>
        </div>
      )}

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

      {premium.length === 0 && ctx?.regime === "R2" && (
        <section className="mb-6">
          <h2 className="mb-3 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            Premium Strategies
          </h2>
          <div className="rounded-lg border border-border bg-card p-6 text-center text-sm text-muted-foreground">
            Premium selling blocked — regime is <Badge label="R2" className={regimeColor.R2} /> Risk-Off
          </div>
        </section>
      )}

      {/* Equity Recommendations */}
      <section>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            Equity Recommendations ({equity.length})
          </h2>
          <div className="flex flex-wrap gap-2">
            <input
              type="text"
              placeholder="Search symbol..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-40 rounded border border-border bg-card px-2 py-1 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
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
            <select
              value={`${sortKey}:${sortDesc ? "desc" : "asc"}`}
              onChange={(e) => {
                const [k, d] = e.target.value.split(":")
                setSortKey(k as SortKey)
                setSortDesc(d === "desc")
              }}
              className="rounded border border-border bg-card px-2 py-1 text-xs text-foreground"
            >
              <option value="confidence:desc">Sort: Conf ↓</option>
              <option value="confidence:asc">Sort: Conf ↑</option>
              <option value="symbol:asc">Sort: Symbol A-Z</option>
              <option value="symbol:desc">Sort: Symbol Z-A</option>
              <option value="sector:asc">Sort: Sector A-Z</option>
              <option value="regime:asc">Sort: Regime</option>
            </select>
          </div>
        </div>
        <EquityTable
          rows={equity}
          sectorFilter={sectorFilter}
          actionFilter={actionFilter}
          sortKey={sortKey}
          sortDesc={sortDesc}
          onToggleSort={handleToggleSort}
          searchQuery={searchQuery}
        />
      </section>
    </div>
  )
}
