import { useState, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import { useMonitor, useDataQuality, useUniverse } from "@/lib/api"
import { useMarketStore } from "@/stores/market"
import type { ProviderStatus } from "@/lib/ws"

const EMPTY_PROVIDERS: ProviderStatus[] = []

type SortDir = "asc" | "desc"
interface SortState {
  key: string
  dir: SortDir
}

const STATUS_COLORS: Record<string, string> = {
  PASS: "bg-emerald-900/50 text-emerald-400",
  WARN: "bg-amber-900/50 text-amber-400",
  CAUTION: "bg-orange-900/50 text-orange-400",
  FAIL: "bg-red-900/50 text-red-400",
  EMPTY: "bg-secondary text-muted-foreground",
}

const STATUS_LABEL_COLORS: Record<string, string> = {
  PASS: "text-emerald-400",
  WARN: "text-amber-400",
  CAUTION: "text-orange-400",
  FAIL: "text-red-400",
  EMPTY: "text-muted-foreground",
}

interface DqRow {
  symbol: string
  timeframe?: string
  tier?: string
  sector?: string
  bars?: number
  coverage_pct?: number
  gaps?: number
  last_bar?: string
  status?: string
  [key: string]: unknown
}

const PER_PAGE_OPTIONS = [25, 50, 100, 200]

export function Monitor() {
  const navigate = useNavigate()
  const { data: monitor } = useMonitor()
  const { data: dq, isLoading: dqLoading } = useDataQuality()
  const { data: universeRaw } = useUniverse()

  const wsProviders = useMarketStore((s) => s.providers)
  const wsStatus = useMarketStore((s) => s.wsStatus)

  const providers = wsProviders.length > 0 ? wsProviders : (monitor?.providers ?? EMPTY_PROVIDERS)

  // Parse universe for enrichment (universe.json has tickers[] array)
  const universeMap = useMemo(() => {
    const map: Record<string, { tier?: string; sector?: string }> = {}
    const udata = universeRaw as {
      tickers?: { symbol: string; tier?: string; sector?: string }[]
      symbols?: Record<string, { tier?: string; sector?: string }>
    } | undefined
    if (udata?.tickers && Array.isArray(udata.tickers)) {
      for (const t of udata.tickers) {
        map[t.symbol] = { tier: t.tier, sector: t.sector }
      }
    } else if (udata?.symbols) {
      for (const [sym, info] of Object.entries(udata.symbols)) {
        map[sym] = info
      }
    }
    return map
  }, [universeRaw])

  // Parse data quality metadata
  const dqMeta = useMemo(() => {
    if (!dq) return { generatedAt: null as string | null, totalEntries: 0 }
    const raw = dq as Record<string, unknown>
    return {
      generatedAt: (raw.generated_at as string) ?? null,
      totalEntries: (raw.total_entries as number) ?? 0,
    }
  }, [dq])

  // Parse data quality rows
  const allRows = useMemo((): DqRow[] => {
    if (!dq) return []
    const raw = dq as Record<string, unknown>
    const entries = (raw.entries ?? raw.summary) as DqRow[] | undefined
    if (!Array.isArray(entries)) return []
    return entries.map((row) => ({
      ...row,
      tier: row.tier ?? universeMap[row.symbol]?.tier ?? "—",
      sector: row.sector ?? universeMap[row.symbol]?.sector ?? "—",
    }))
  }, [dq, universeMap])

  // Filters
  const [search, setSearch] = useState("")
  const [filterTf, setFilterTf] = useState("All")
  const [filterTier, setFilterTier] = useState("All")
  const [filterStatus, setFilterStatus] = useState("All")
  const [filterSector, setFilterSector] = useState("All")
  const [sort, setSort] = useState<SortState>({ key: "symbol", dir: "asc" })
  const [page, setPage] = useState(0)
  const [perPage, setPerPage] = useState(50)

  // Unique values for dropdowns
  const uniqueTfs = useMemo(() => [...new Set(allRows.map((r) => r.timeframe).filter(Boolean))].sort(), [allRows])
  const uniqueTiers = useMemo(() => [...new Set(allRows.map((r) => r.tier).filter(Boolean))].sort(), [allRows])
  const uniqueSectors = useMemo(() => [...new Set(allRows.map((r) => r.sector).filter((s) => s && s !== "—"))].sort(), [allRows])
  const uniqueStatuses = useMemo(() => [...new Set(allRows.map((r) => r.status).filter(Boolean))].sort(), [allRows])

  // Filter + sort
  const filteredRows = useMemo(() => {
    let rows = allRows
    if (search) {
      const q = search.toLowerCase()
      rows = rows.filter((r) => r.symbol.toLowerCase().includes(q))
    }
    if (filterTf !== "All") rows = rows.filter((r) => r.timeframe === filterTf)
    if (filterTier !== "All") rows = rows.filter((r) => r.tier === filterTier)
    if (filterStatus !== "All") rows = rows.filter((r) => r.status === filterStatus)
    if (filterSector !== "All") rows = rows.filter((r) => r.sector === filterSector)

    // Sort
    rows = [...rows].sort((a, b) => {
      const av = a[sort.key] as string | number | undefined
      const bv = b[sort.key] as string | number | undefined
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      if (typeof av === "number" && typeof bv === "number") return sort.dir === "asc" ? av - bv : bv - av
      return sort.dir === "asc" ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
    })

    return rows
  }, [allRows, search, filterTf, filterTier, filterStatus, filterSector, sort])

  // Status counts
  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const r of allRows) {
      const s = r.status ?? "EMPTY"
      counts[s] = (counts[s] ?? 0) + 1
    }
    return counts
  }, [allRows])

  // Pagination
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / perPage))
  const pagedRows = filteredRows.slice(page * perPage, (page + 1) * perPage)

  const toggleSort = (key: string) => {
    setSort((prev) => ({
      key,
      dir: prev.key === key && prev.dir === "asc" ? "desc" : "asc",
    }))
  }

  const sortIndicator = (key: string) => {
    if (sort.key !== key) return ""
    return sort.dir === "asc" ? " ▲" : " ▼"
  }

  return (
    <div className="space-y-6">
      {/* ── System Health ── */}
      <section>
        <h2 className="mb-3 text-base font-semibold uppercase tracking-wider text-muted-foreground">
          System Health
        </h2>

        <div className="grid grid-cols-4 gap-3">
          <MetricCard label="Uptime" value={monitor ? formatUptime(monitor.uptime_sec) : "--"} />
          <MetricCard label="WS Clients" value={monitor?.ws_clients ?? 0} />
          <MetricCard
            label="Pipeline"
            value={monitor?.pipeline.running ? "Running" : "Stopped"}
            color={monitor?.pipeline.running ? "text-emerald-400" : "text-red-400"}
          />
          <MetricCard
            label="WebSocket"
            value={wsStatus}
            color={wsStatus === "connected" ? "text-emerald-400" : wsStatus === "connecting" ? "text-amber-400" : "text-red-400"}
          />
        </div>

        {/* Provider cards */}
        {providers.length > 0 && (
          <div className="mt-3 grid grid-cols-2 gap-3">
            {providers.map((p) => (
              <div key={p.name} className="rounded-lg border border-border bg-card px-4 py-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium capitalize">{p.name}</span>
                  <span className={`inline-block h-2 w-2 rounded-full ${p.connected ? "bg-emerald-500" : "bg-red-500"}`} />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{p.symbols} symbols subscribed</p>
                {p.subscribed_symbols?.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {p.subscribed_symbols.slice(0, 20).map((sym) => (
                      <span key={sym} className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground">
                        {sym}
                      </span>
                    ))}
                    {p.subscribed_symbols.length > 20 && (
                      <span className="text-[10px] text-muted-foreground">+{p.subscribed_symbols.length - 20} more</span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Timeframes */}
        {monitor?.pipeline.timeframes && monitor.pipeline.timeframes.length > 0 && (
          <div className="mt-3">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Timeframes: </span>
            {monitor.pipeline.timeframes.map((tf) => (
              <span key={tf} className="ml-1 rounded bg-secondary px-2 py-0.5 text-xs text-secondary-foreground">
                {tf}
              </span>
            ))}
          </div>
        )}
      </section>

      {/* ── Data Quality ── */}
      <section>
        <h2 className="mb-3 text-base font-semibold uppercase tracking-wider text-muted-foreground">
          Data Quality
        </h2>

        {allRows.length === 0 ? (
          <p className="text-sm text-muted-foreground">{dqLoading ? "Loading data quality report..." : "No data quality report available"}</p>
        ) : (
          <>
            {/* Generated timestamp */}
            {dqMeta.generatedAt && (
              <p className="mb-2 text-[11px] text-muted-foreground">
                Report generated: {new Date(dqMeta.generatedAt).toLocaleString()}
              </p>
            )}

            {/* Summary bar */}
            <div className="mb-3 flex flex-wrap items-center gap-4 rounded-lg border border-border bg-card px-4 py-2 text-xs">
              {["PASS", "WARN", "CAUTION", "FAIL", "EMPTY"].map((s) => (
                <span key={s}>
                  <span className={STATUS_LABEL_COLORS[s] ?? "text-muted-foreground"}>{s}: </span>
                  <strong>{statusCounts[s] ?? 0}</strong>
                </span>
              ))}
              <span className="ml-auto text-muted-foreground">Total: {allRows.length}</span>
            </div>

            {/* Filter controls */}
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <input
                type="text"
                placeholder="Search symbol..."
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(0) }}
                className="rounded border border-input bg-background px-2 py-1 text-xs text-foreground placeholder:text-muted-foreground"
              />
              <FilterSelect label="TF" value={filterTf} options={uniqueTfs} onChange={(v) => { setFilterTf(v); setPage(0) }} />
              <FilterSelect label="Tier" value={filterTier} options={uniqueTiers} onChange={(v) => { setFilterTier(v); setPage(0) }} />
              <FilterSelect label="Status" value={filterStatus} options={uniqueStatuses} onChange={(v) => { setFilterStatus(v); setPage(0) }} />
              <FilterSelect label="Sector" value={filterSector} options={uniqueSectors} onChange={(v) => { setFilterSector(v); setPage(0) }} />
              <span className="ml-auto text-xs text-muted-foreground">
                {filteredRows.length} results
              </span>
            </div>

            {/* Table */}
            <div className="overflow-auto rounded-lg border border-border">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-card">
                  <tr className="border-b border-border text-left text-muted-foreground">
                    {[
                      { key: "symbol", label: "Symbol" },
                      { key: "tier", label: "Tier" },
                      { key: "sector", label: "Sector" },
                      { key: "timeframe", label: "TF" },
                      { key: "bars", label: "Bars" },
                      { key: "coverage_pct", label: "Coverage %" },
                      { key: "gaps", label: "Gaps" },
                      { key: "last_bar", label: "Last Bar" },
                      { key: "status", label: "Status" },
                    ].map((col) => (
                      <th
                        key={col.key}
                        className="cursor-pointer px-3 py-2 hover:text-foreground"
                        onClick={() => toggleSort(col.key)}
                      >
                        {col.label}{sortIndicator(col.key)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pagedRows.map((row, i) => (
                    <tr key={`${row.symbol}-${row.timeframe}-${i}`} className="border-b border-border/50 hover:bg-accent/30">
                      <td className="px-3 py-1.5">
                        <button
                          onClick={() => navigate(`/signals?symbol=${row.symbol}`)}
                          className="font-medium text-primary hover:underline"
                        >
                          {row.symbol}
                        </button>
                      </td>
                      <td className="px-3 py-1.5">{row.tier}</td>
                      <td className="px-3 py-1.5">{row.sector}</td>
                      <td className="px-3 py-1.5">{row.timeframe ?? "—"}</td>
                      <td className="px-3 py-1.5 text-right">{row.bars ?? "—"}</td>
                      <td className="px-3 py-1.5 text-right">{row.coverage_pct != null ? `${row.coverage_pct.toFixed(1)}%` : "—"}</td>
                      <td className="px-3 py-1.5 text-right">{row.gaps ?? "—"}</td>
                      <td className="px-3 py-1.5">{row.last_bar ?? "—"}</td>
                      <td className="px-3 py-1.5">
                        <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${STATUS_COLORS[row.status ?? ""] ?? "bg-secondary text-muted-foreground"}`}>
                          {row.status ?? "—"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <span>Per page:</span>
                {PER_PAGE_OPTIONS.map((n) => (
                  <button
                    key={n}
                    onClick={() => { setPerPage(n); setPage(0) }}
                    className={`rounded px-2 py-0.5 ${perPage === n ? "bg-secondary text-foreground" : "hover:bg-secondary/50"}`}
                  >
                    {n}
                  </button>
                ))}
              </div>
              <span>
                Showing {page * perPage + 1}–{Math.min((page + 1) * perPage, filteredRows.length)} of {filteredRows.length}
              </span>
              <div className="flex gap-1">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="rounded border border-border px-2 py-0.5 disabled:opacity-30"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="rounded border border-border px-2 py-0.5 disabled:opacity-30"
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  )
}

// ── Sub-components ───────────────────────────────────────────────────────────

function MetricCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`text-lg font-semibold ${color ?? ""}`}>{value}</p>
    </div>
  )
}

function FilterSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: (string | undefined)[]
  onChange: (v: string) => void
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-input bg-background px-2 py-1 text-xs text-foreground"
      aria-label={label}
    >
      <option value="All">{label}: All</option>
      {options.filter(Boolean).map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  )
}

function formatUptime(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.round(sec / 60)}m`
  const h = Math.floor(sec / 3600)
  const m = Math.round((sec % 3600) / 60)
  return `${h}h ${m}m`
}
