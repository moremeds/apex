import { useState, useMemo } from "react"
import { useNavigate } from "react-router-dom"
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
  addv: number
  position_size_factor: number
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

// ── CSV Export ────────────────────────────────────────

function exportCSV(data: Record<string, unknown>[], filename: string) {
  if (data.length === 0) return
  const headers = Object.keys(data[0])
  const rows = data.map((row) =>
    headers.map((h) => String(row[h] ?? "")).join(","),
  )
  const csv = [headers.join(","), ...rows].join("\n")
  const blob = new Blob([csv], { type: "text/csv" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// ── Column Definitions (hoisted, use navigate via closure) ──

function useMomColumns() {
  const navigate = useNavigate()
  return useMemo(() => {
    const h = createColumnHelper<MomentumRow>()
    return [
      h.accessor("rank", { header: "#", size: 40 }),
      h.accessor("symbol", {
        header: "Symbol",
        cell: (info) => (
          <button
            onClick={() => navigate(`/signals?symbol=${info.getValue()}`)}
            className="font-medium text-primary hover:underline"
          >
            {info.getValue()}
          </button>
        ),
      }),
      h.accessor("momentum_12_1", {
        header: "Mom 12-1",
        cell: (info) => pctCell(info.getValue()),
      }),
      h.accessor("fip", {
        header: "FIP",
        cell: (info) => info.getValue()?.toFixed(2) ?? "--",
      }),
      h.accessor("composite_rank", {
        header: "Rank",
        cell: (info) => pctCell(info.getValue()),
      }),
      h.accessor("quality_label", {
        header: "Quality",
        cell: (info) => <QualityBadge label={info.getValue()} />,
      }),
      h.accessor("regime", {
        header: "Regime",
        cell: (info) => <RegimeBadge regime={info.getValue()} />,
      }),
      h.accessor("last_close", {
        header: "Price",
        cell: (info) => `$${info.getValue()?.toFixed(2) ?? "--"}`,
      }),
      h.accessor("market_cap", {
        header: "Mkt Cap",
        cell: (info) => formatBigNumber(info.getValue()),
      }),
      h.accessor("addv", {
        header: "ADDV",
        cell: (info) => formatBigNumber(info.getValue()),
      }),
      h.accessor("position_size_factor", {
        header: "Pos Size",
        cell: (info) => info.getValue()?.toFixed(2) ?? "--",
      }),
      h.accessor("liquidity_tier", { header: "Liquidity" }),
    ]
  }, [navigate])
}

function usePeadColumns() {
  const navigate = useNavigate()
  return useMemo(() => {
    const h = createColumnHelper<PeadRow>()
    return [
      h.accessor("symbol", {
        header: "Symbol",
        cell: (info) => (
          <button
            onClick={() => navigate(`/signals?symbol=${info.getValue()}`)}
            className="font-medium text-primary hover:underline"
          >
            {info.getValue()}
          </button>
        ),
      }),
      h.accessor("report_date", { header: "Report Date" }),
      h.accessor("sue_score", {
        header: "SUE",
        cell: (info) => info.getValue()?.toFixed(2) ?? "--",
      }),
      h.accessor("mq_sue", {
        header: "MQ SUE",
        cell: (info) => info.getValue()?.toFixed(2) ?? "--",
      }),
      h.accessor("earnings_day_return", {
        header: "ED Return",
        cell: (info) => pctCell(info.getValue()),
      }),
      h.accessor("earnings_day_volume_ratio", {
        header: "Vol Ratio",
        cell: (info) => `${info.getValue()?.toFixed(1) ?? "--"}x`,
      }),
      h.accessor("revenue_surprise_pct", {
        header: "Rev Surprise",
        cell: (info) => pctCell(info.getValue()),
      }),
      h.accessor("quality_label", {
        header: "Quality",
        cell: (info) => <QualityBadge label={info.getValue()} />,
      }),
      h.accessor("gap_held", {
        header: "Gap Held",
        cell: (info) => (info.getValue() ? "Yes" : "No"),
      }),
      h.accessor("entry_price", {
        header: "Entry",
        cell: (info) => `$${info.getValue()?.toFixed(2) ?? "--"}`,
      }),
      h.accessor("stop_loss_pct", {
        header: "Stop %",
        cell: (info) => `${info.getValue()?.toFixed(1) ?? "--"}%`,
      }),
      h.accessor("profit_target_pct", {
        header: "Target %",
        cell: (info) => `${info.getValue()?.toFixed(1) ?? "--"}%`,
      }),
    ]
  }, [navigate])
}

// ── Summary Card Components ──────────────────────────

function SummaryCard({
  label,
  value,
}: {
  label: string
  value: string | number
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  )
}

interface MomentumMeta {
  universe_size?: number
  top_n?: number
}

function MomentumSummary({
  data,
  meta,
}: {
  data: MomentumRow[]
  meta: MomentumMeta
}) {
  const universeSize = meta.universe_size ?? 0
  const passed = data.length
  const topN = meta.top_n ?? data.length
  const avgMom =
    data.length > 0
      ? data.reduce((sum, r) => sum + (r.momentum_12_1 ?? 0), 0) / data.length
      : 0
  const avgFip =
    data.length > 0
      ? data.reduce((sum, r) => sum + (r.fip ?? 0), 0) / data.length
      : 0

  return (
    <div className="grid grid-cols-5 gap-3">
      <SummaryCard label="Universe" value={universeSize.toLocaleString()} />
      <SummaryCard label="Passed Filters" value={passed.toLocaleString()} />
      <SummaryCard label="Top-N" value={topN.toLocaleString()} />
      <SummaryCard
        label="Avg Momentum"
        value={`${(avgMom * 100).toFixed(1)}%`}
      />
      <SummaryCard label="Avg FIP" value={avgFip.toFixed(2)} />
    </div>
  )
}

function PeadSummary({ data }: { data: PeadRow[] }) {
  const total = data.length
  const strong = data.filter(
    (r) => r.quality_label?.toUpperCase() === "STRONG",
  ).length
  const moderate = data.filter(
    (r) => r.quality_label?.toUpperCase() === "MODERATE",
  ).length
  const marginal = data.filter(
    (r) => r.quality_label?.toUpperCase() === "MARGINAL",
  ).length

  // Find dominant regime from quality labels
  const regimeCounts: Record<string, number> = {}
  for (const r of data) {
    const label = r.quality_label?.toUpperCase() ?? "UNKNOWN"
    regimeCounts[label] = (regimeCounts[label] ?? 0) + 1
  }
  const dominantRegime =
    Object.entries(regimeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "--"

  return (
    <div className="grid grid-cols-3 gap-3">
      <SummaryCard label="Candidates" value={total.toLocaleString()} />
      <SummaryCard
        label="Quality Mix"
        value={`${strong}S / ${moderate}M / ${marginal}G`}
      />
      <SummaryCard label="Dominant Quality" value={dominantRegime} />
    </div>
  )
}

// ── Main Component ───────────────────────────────────

export function Screeners() {
  const [tab, setTab] = useState<Tab>("momentum")
  const { data, isLoading, error } = useScreeners()
  const momColumns = useMomColumns()
  const peadColumns = usePeadColumns()

  const momRaw = useMemo(
    () =>
      (data as Record<string, unknown>)?.momentum as
        | { candidates?: MomentumRow[]; universe_size?: number; top_n?: number }
        | undefined,
    [data],
  )
  const momData = useMemo(() => momRaw?.candidates ?? [], [momRaw])
  const momMeta: MomentumMeta = useMemo(
    () => ({
      universe_size: momRaw?.universe_size,
      top_n: momRaw?.top_n,
    }),
    [momRaw],
  )

  const peadData = useMemo(
    () =>
      (
        (data as Record<string, unknown>)?.pead as
          | { candidates?: PeadRow[] }
          | undefined
      )?.candidates ?? [],
    [data],
  )

  if (isLoading) {
    return (
      <p className="text-sm text-muted-foreground">Loading screener data...</p>
    )
  }
  if (error) {
    return (
      <p className="text-sm text-red-400">
        Failed to load screeners: {(error as Error).message}
      </p>
    )
  }

  const handleExport = () => {
    if (tab === "momentum") {
      exportCSV(
        momData as unknown as Record<string, unknown>[],
        "momentum_screener.csv",
      )
    } else {
      exportCSV(
        peadData as unknown as Record<string, unknown>[],
        "pead_screener.csv",
      )
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Screeners</h2>

      {/* Tabs + Export */}
      <div className="flex items-center justify-between">
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
        <button
          onClick={handleExport}
          className="rounded-md border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          Export CSV
        </button>
      </div>

      {tab === "momentum" ? (
        <div className="space-y-4">
          <MomentumSummary data={momData} meta={momMeta} />
          <DataTable
            data={momData}
            columns={momColumns}
            searchPlaceholder="Search momentum..."
          />
        </div>
      ) : (
        <div className="space-y-4">
          <PeadSummary data={peadData} />
          <DataTable
            data={peadData}
            columns={peadColumns}
            searchPlaceholder="Search PEAD..."
          />
          {/* PEAD Methodology */}
          <details className="rounded-lg border border-border bg-card">
            <summary className="cursor-pointer px-4 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              PEAD Methodology
            </summary>
            <div className="space-y-2 px-4 pb-4 text-xs text-muted-foreground">
              <p>
                Post-Earnings Announcement Drift (PEAD) exploits the tendency of
                stock prices to drift in the direction of earnings surprises for
                60-90 days after announcements.
              </p>
              <p>
                <strong>Entry Criteria:</strong> SUE score &ge; 2.0, earnings day
                return &ge; 1%, volume ratio &ge; 1.5x, gap held
              </p>
              <p>
                <strong>Position Sizing:</strong> Based on quality label (STRONG:
                full, MODERATE: 75%, MARGINAL: 50%)
              </p>
              <p>
                <strong>Exit:</strong> Stop loss at -5%, profit target at +15%,
                max hold 60 days
              </p>
            </div>
          </details>
        </div>
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
  if (val >= 1e6) return `$${(val / 1e6).toFixed(1)}M`
  return `$${val.toLocaleString()}`
}
