import { useMemo, useState } from "react"
import { useMarketStore } from "@/stores/market"
import { usePortfolioSnapshot } from "@/lib/api"
import type { PositionData } from "@/lib/ws"

type SortKey = "symbol" | "asset_type" | "quantity" | "avg_price" | "mark_price" | "unrealized_pnl" | "daily_pnl" | "delta" | "source" | "days_to_expiry"
type SortDir = "asc" | "desc"

function formatPnl(v: number | null): string {
  if (v == null) return "-"
  const sign = v >= 0 ? "+" : ""
  return `${sign}${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function formatNum(v: number | null, digits = 2): string {
  if (v == null) return "-"
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function pnlColor(v: number | null): string {
  if (v == null || v === 0) return "text-muted-foreground"
  return v > 0 ? "text-emerald-400" : "text-red-400"
}

// ── Broker Status Bar ───────────────────────────────────

function BrokerStatusBar() {
  const brokerStatus = useMarketStore((s) => s.brokerStatus)
  if (brokerStatus.length === 0) return null

  return (
    <div className="flex items-center gap-4 mb-6">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Brokers</span>
      {brokerStatus.map((b) => (
        <div key={b.name} className="flex items-center gap-2">
          <span className={`inline-block h-2 w-2 rounded-full ${b.connected ? "bg-emerald-500" : "bg-red-500"}`} />
          <span className="text-sm">{b.name.toUpperCase()}</span>
        </div>
      ))}
    </div>
  )
}

// ── Account Summary Cards ───────────────────────────────

function AccountSummary() {
  const account = useMarketStore((s) => s.account)
  const pnl = useMarketStore((s) => s.portfolioPnl)

  const cards = [
    { label: "Net Liquidation", value: formatNum(pnl.net_liquidation, 0), sub: null },
    { label: "Buying Power", value: formatNum(account?.buying_power ?? null, 0), sub: null },
    { label: "Unrealized P&L", value: formatPnl(pnl.unrealized), sub: null, color: pnlColor(pnl.unrealized) },
    { label: "Daily P&L", value: formatPnl(pnl.daily), sub: null, color: pnlColor(pnl.daily) },
    { label: "Margin Used", value: formatNum(account?.margin_used ?? null, 0), sub: account ? `${(account.margin_utilization * 100).toFixed(1)}% util` : null },
  ]

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
      {cards.map((c) => (
        <div key={c.label} className="rounded-lg border border-border bg-card p-4">
          <div className="text-xs text-muted-foreground mb-1">{c.label}</div>
          <div className={`text-lg font-mono font-semibold ${c.color ?? "text-foreground"}`}>
            {c.value}
          </div>
          {c.sub && <div className="text-xs text-muted-foreground mt-1">{c.sub}</div>}
        </div>
      ))}
    </div>
  )
}

// ── Greeks Summary ──────────────────────────────────────

function GreeksSummary() {
  const greeks = useMarketStore((s) => s.portfolioGreeks)

  const items = [
    { label: "Delta", value: greeks.delta },
    { label: "Gamma", value: greeks.gamma },
    { label: "Vega", value: greeks.vega },
    { label: "Theta", value: greeks.theta },
  ]

  return (
    <div className="flex items-center gap-6 mb-6 p-3 rounded-lg border border-border bg-card">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Portfolio Greeks</span>
      {items.map((g) => (
        <div key={g.label} className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">{g.label}</span>
          <span className={`text-sm font-mono font-medium ${pnlColor(g.value)}`}>
            {formatNum(g.value, 2)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Positions Table ─────────────────────────────────────

function PositionsTable() {
  const positions = useMarketStore((s) => s.positions)
  const [sortKey, setSortKey] = useState<SortKey>("unrealized_pnl")
  const [sortDir, setSortDir] = useState<SortDir>("desc")

  const sorted = useMemo(() => {
    const copy = [...positions]
    copy.sort((a, b) => {
      const av = a[sortKey] ?? 0
      const bv = b[sortKey] ?? 0
      if (typeof av === "string" && typeof bv === "string") {
        return sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av)
      }
      const diff = (av as number) - (bv as number)
      return sortDir === "asc" ? diff : -diff
    })
    return copy
  }, [positions, sortKey, sortDir])

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"))
    } else {
      setSortKey(key)
      setSortDir("desc")
    }
  }

  function SortHeader({ k, label, className }: { k: SortKey; label: string; className?: string }) {
    const arrow = sortKey === k ? (sortDir === "asc" ? " ↑" : " ↓") : ""
    return (
      <th
        className={`px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase cursor-pointer hover:text-foreground select-none ${className ?? ""}`}
        onClick={() => toggleSort(k)}
      >
        {label}{arrow}
      </th>
    )
  }

  if (positions.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No positions. Connect a broker to see live portfolio data.
      </div>
    )
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50">
          <tr>
            <SortHeader k="symbol" label="Symbol" />
            <SortHeader k="asset_type" label="Type" />
            <SortHeader k="quantity" label="Qty" className="text-right" />
            <SortHeader k="avg_price" label="Avg Price" className="text-right" />
            <SortHeader k="mark_price" label="Mark" className="text-right" />
            <SortHeader k="unrealized_pnl" label="Unrealized P&L" className="text-right" />
            <SortHeader k="daily_pnl" label="Daily P&L" className="text-right" />
            <SortHeader k="delta" label="Delta" className="text-right" />
            <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase">Gamma</th>
            <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase">Vega</th>
            <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase">Theta</th>
            <SortHeader k="days_to_expiry" label="DTE" className="text-right" />
            <SortHeader k="source" label="Broker" />
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {sorted.map((p) => (
            <PositionRow key={`${p.symbol}-${p.source}-${p.expiry ?? ""}-${p.strike ?? ""}`} position={p} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PositionRow({ position: p }: { position: PositionData }) {
  const optionLabel = p.asset_type === "OPTION"
    ? `${p.underlying} ${p.expiry ?? ""} ${p.strike ?? ""}${p.right ?? ""}`
    : p.symbol

  return (
    <tr className="hover:bg-muted/30 transition-colors">
      <td className="px-3 py-2 font-mono font-medium">
        {optionLabel}
        {p.is_stale && <span className="ml-1 text-amber-500 text-xs" title="Stale data">!</span>}
      </td>
      <td className="px-3 py-2 text-muted-foreground">{p.asset_type}</td>
      <td className="px-3 py-2 text-right font-mono">{p.quantity}</td>
      <td className="px-3 py-2 text-right font-mono">{formatNum(p.avg_price)}</td>
      <td className="px-3 py-2 text-right font-mono">{formatNum(p.mark_price)}</td>
      <td className={`px-3 py-2 text-right font-mono font-medium ${pnlColor(p.unrealized_pnl)}`}>
        {formatPnl(p.unrealized_pnl)}
      </td>
      <td className={`px-3 py-2 text-right font-mono font-medium ${pnlColor(p.daily_pnl)}`}>
        {formatPnl(p.daily_pnl)}
      </td>
      <td className="px-3 py-2 text-right font-mono">{formatNum(p.delta)}</td>
      <td className="px-3 py-2 text-right font-mono">{formatNum(p.gamma, 4)}</td>
      <td className="px-3 py-2 text-right font-mono">{formatNum(p.vega)}</td>
      <td className={`px-3 py-2 text-right font-mono ${pnlColor(p.theta)}`}>{formatNum(p.theta)}</td>
      <td className="px-3 py-2 text-right font-mono text-muted-foreground">
        {p.days_to_expiry != null ? `${p.days_to_expiry}d` : "-"}
      </td>
      <td className="px-3 py-2 text-muted-foreground">{p.source}</td>
    </tr>
  )
}

// ── Main Page ───────────────────────────────────────────

export function Portfolio() {
  const portfolioEnabled = useMarketStore((s) => s.portfolioEnabled)
  const positionCount = useMarketStore((s) => s.positions.length)

  // Fetch initial snapshot via REST (WS will keep it updated after)
  usePortfolioSnapshot()

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold">Portfolio</h1>
        <span className="text-sm text-muted-foreground">
          {portfolioEnabled
            ? `${positionCount} position${positionCount !== 1 ? "s" : ""}`
            : "Portfolio not connected"}
        </span>
      </div>

      {!portfolioEnabled && (
        <div className="rounded-lg border border-border bg-card p-8 text-center text-muted-foreground">
          <p className="mb-2">No broker connections detected.</p>
          <p className="text-xs">
            Configure IB Gateway, Futu, or LongBridge in <code className="bg-muted px-1 rounded">config/base.yaml</code> and restart the server.
          </p>
        </div>
      )}

      {portfolioEnabled && (
        <>
          <BrokerStatusBar />
          <AccountSummary />
          <GreeksSummary />
          <PositionsTable />
        </>
      )}
    </div>
  )
}
