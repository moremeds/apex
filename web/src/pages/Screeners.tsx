import { useState, useMemo } from "react"
import { createColumnHelper } from "@tanstack/react-table"
import { useScreeners } from "@/lib/api"
import { DataTable } from "@/components/DataTable"

type Tab = "momentum" | "pead"

interface MomentumRow {
  rank: number
  symbol: string
  momentum_12_1: number
  fip: number
  composite_rank: number
  quality_label: string
  regime: string
  last_close: number
  market_cap: number
  liquidity_tier: string
}

interface PeadRow {
  symbol: string
  report_date: string
  sue_score: number
  mq_sue: number
  earnings_day_return: number
  earnings_day_volume_ratio: number
  revenue_surprise_pct: number
  quality_label: string
  gap_held: boolean
  entry_price: number
  stop_loss_pct: number
  profit_target_pct: number
  max_hold_days: number
}

const momHelper = createColumnHelper<MomentumRow>()
const momColumns = [
  momHelper.accessor("rank", { header: "#", size: 40 }),
  momHelper.accessor("symbol", {
    header: "Symbol",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  momHelper.accessor("momentum_12_1", {
    header: "Mom 12-1",
    cell: (info) => pctCell(info.getValue()),
  }),
  momHelper.accessor("fip", {
    header: "FIP",
    cell: (info) => info.getValue()?.toFixed(2) ?? "--",
  }),
  momHelper.accessor("composite_rank", {
    header: "Rank",
    cell: (info) => pctCell(info.getValue()),
  }),
  momHelper.accessor("quality_label", {
    header: "Quality",
    cell: (info) => <QualityBadge label={info.getValue()} />,
  }),
  momHelper.accessor("regime", {
    header: "Regime",
    cell: (info) => <RegimeBadge regime={info.getValue()} />,
  }),
  momHelper.accessor("last_close", {
    header: "Price",
    cell: (info) => `$${info.getValue()?.toFixed(2) ?? "--"}`,
  }),
  momHelper.accessor("market_cap", {
    header: "Mkt Cap",
    cell: (info) => formatBigNumber(info.getValue()),
  }),
  momHelper.accessor("liquidity_tier", { header: "Liquidity" }),
]

const peadHelper = createColumnHelper<PeadRow>()
const peadColumns = [
  peadHelper.accessor("symbol", {
    header: "Symbol",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  peadHelper.accessor("report_date", { header: "Report Date" }),
  peadHelper.accessor("sue_score", {
    header: "SUE",
    cell: (info) => info.getValue()?.toFixed(2) ?? "--",
  }),
  peadHelper.accessor("mq_sue", {
    header: "MQ SUE",
    cell: (info) => info.getValue()?.toFixed(2) ?? "--",
  }),
  peadHelper.accessor("earnings_day_return", {
    header: "ED Return",
    cell: (info) => pctCell(info.getValue()),
  }),
  peadHelper.accessor("earnings_day_volume_ratio", {
    header: "Vol Ratio",
    cell: (info) => `${info.getValue()?.toFixed(1) ?? "--"}x`,
  }),
  peadHelper.accessor("quality_label", {
    header: "Quality",
    cell: (info) => <QualityBadge label={info.getValue()} />,
  }),
  peadHelper.accessor("gap_held", {
    header: "Gap Held",
    cell: (info) => (info.getValue() ? "Yes" : "No"),
  }),
  peadHelper.accessor("entry_price", {
    header: "Entry",
    cell: (info) => `$${info.getValue()?.toFixed(2) ?? "--"}`,
  }),
  peadHelper.accessor("stop_loss_pct", {
    header: "Stop %",
    cell: (info) => `${info.getValue()?.toFixed(1) ?? "--"}%`,
  }),
  peadHelper.accessor("profit_target_pct", {
    header: "Target %",
    cell: (info) => `${info.getValue()?.toFixed(1) ?? "--"}%`,
  }),
]

export function Screeners() {
  const [tab, setTab] = useState<Tab>("momentum")
  const { data, isLoading, error } = useScreeners()

  const momData = useMemo(
    () => ((data as Record<string, unknown>)?.momentum as { candidates?: MomentumRow[] })?.candidates ?? [],
    [data],
  )
  const peadData = useMemo(
    () => ((data as Record<string, unknown>)?.pead as { candidates?: PeadRow[] })?.candidates ?? [],
    [data],
  )

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading screener data...</p>
  }
  if (error) {
    return (
      <p className="text-sm text-red-400">
        Failed to load screeners: {(error as Error).message}
      </p>
    )
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Screeners</h2>

      {/* Tabs */}
      <div className="flex gap-1 rounded-lg bg-secondary p-1">
        {(["momentum", "pead"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-md px-4 py-1.5 text-sm transition-colors ${
              tab === t
                ? "bg-background text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t === "momentum" ? "Momentum" : "PEAD"}
          </button>
        ))}
      </div>

      {tab === "momentum" ? (
        <DataTable
          data={momData}
          columns={momColumns}
          searchPlaceholder="Search momentum..."
        />
      ) : (
        <DataTable
          data={peadData}
          columns={peadColumns}
          searchPlaceholder="Search PEAD..."
        />
      )}
    </div>
  )
}

// ── Helpers ──────────────────────────────────────

function pctCell(val: number | null | undefined) {
  if (val == null) return "--"
  const pct = val * 100
  const color = pct > 0 ? "text-emerald-400" : pct < 0 ? "text-red-400" : ""
  return <span className={color}>{pct.toFixed(1)}%</span>
}

function QualityBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    STRONG: "bg-emerald-900/50 text-emerald-400",
    MODERATE: "bg-amber-900/50 text-amber-400",
    MARGINAL: "bg-red-900/50 text-red-400",
  }
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
        colors[label?.toUpperCase()] ?? "bg-secondary text-muted-foreground"
      }`}
    >
      {label}
    </span>
  )
}

function RegimeBadge({ regime }: { regime: string }) {
  const colors: Record<string, string> = {
    R0: "bg-emerald-900/50 text-emerald-400",
    R1: "bg-amber-900/50 text-amber-400",
    R2: "bg-red-900/50 text-red-400",
    R3: "bg-blue-900/50 text-blue-400",
  }
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
        colors[regime] ?? "bg-secondary text-muted-foreground"
      }`}
    >
      {regime}
    </span>
  )
}

function formatBigNumber(val: number | null | undefined): string {
  if (val == null) return "--"
  if (val >= 1e12) return `$${(val / 1e12).toFixed(1)}T`
  if (val >= 1e9) return `$${(val / 1e9).toFixed(1)}B`
  if (val >= 1e6) return `$${(val / 1e6).toFixed(0)}M`
  return `$${val.toLocaleString()}`
}
