import { useState, useMemo } from "react"
import Plot from "react-plotly.js"
import { useBacktest } from "@/lib/api"

type Tab =
  | "overview"
  | "metrics"
  | "per-stock"
  | "sector"
  | "regime"
  | "monthly"

interface StrategyData {
  name: string
  tier?: string
  sharpe: number
  sortino: number
  calmar: number
  total_return: number
  max_drawdown: number
  win_rate: number
  profit_factor: number
  trade_count: number
  avg_trade_pnl: number
  equity_curve?: [number, number][]
  drawdown_curve?: [number, number][]
  monthly_returns?: Record<string, number>
  per_symbol_metrics?: Record<string, { sharpe: number; total_return: number; max_drawdown: number; win_rate: number; trade_count: number }>
  per_regime_sharpe?: Record<string, number>
  per_regime_return?: Record<string, number>
  per_regime_trades?: Record<string, number>
}

interface BacktestBundle {
  title?: string
  period?: string
  strategies?: Record<string, StrategyData>
}

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "metrics", label: "Metrics" },
  { id: "per-stock", label: "Per-Stock" },
  { id: "sector", label: "Regime Perf" },
  { id: "monthly", label: "Monthly" },
]

const STRATEGY_COLORS: Record<string, string> = {
  trend_pulse: "#3b82f6",
  regime_flex: "#22c55e",
  sector_pulse: "#f59e0b",
  rsi_mean_reversion: "#a855f7",
  pead: "#ef4444",
  buy_and_hold: "#6b7280",
}

export function Backtest() {
  const [tab, setTab] = useState<Tab>("overview")
  const { data: raw, isLoading, error } = useBacktest()

  const bundle = raw as BacktestBundle | undefined
  const strategies = useMemo(() => {
    if (!bundle?.strategies) return []
    return Object.values(bundle.strategies)
  }, [bundle])

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading backtest data...</p>
  }
  if (error) {
    return (
      <p className="text-sm text-red-400">
        Failed to load backtest: {(error as Error).message}
      </p>
    )
  }
  if (strategies.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No backtest data available. R2 strategies.json may not be configured.
      </p>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">{bundle?.title ?? "Backtest"}</h2>
        {bundle?.period && (
          <p className="text-xs text-muted-foreground">{bundle.period}</p>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-lg bg-secondary p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
              tab === t.id
                ? "bg-background text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && <OverviewTab strategies={strategies} />}
      {tab === "metrics" && <MetricsTab strategies={strategies} />}
      {tab === "per-stock" && <PerStockTab strategies={strategies} />}
      {tab === "sector" && <RegimeTab strategies={strategies} />}
      {tab === "monthly" && <MonthlyTab strategies={strategies} />}
    </div>
  )
}

// ── Overview: scorecards + equity curves ──

function OverviewTab({ strategies }: { strategies: StrategyData[] }) {
  return (
    <div className="space-y-4">
      {/* Scorecards */}
      <div className="grid grid-cols-3 gap-3">
        {strategies.map((s) => (
          <div key={s.name} className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">{s.name.replace(/_/g, " ")}</h4>
              {s.tier && (
                <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  {s.tier}
                </span>
              )}
            </div>
            <div className="mt-3 grid grid-cols-2 gap-y-1.5 text-xs">
              <Metric label="Sharpe" value={s.sharpe?.toFixed(2)} />
              <Metric label="Return" value={pctStr(s.total_return)} colored />
              <Metric label="Max DD" value={pctStr(s.max_drawdown)} colored />
              <Metric label="Win Rate" value={pctStr(s.win_rate)} />
              <Metric label="Trades" value={s.trade_count?.toString()} />
              <Metric label="P/F" value={s.profit_factor?.toFixed(2)} />
            </div>
          </div>
        ))}
      </div>

      {/* Equity curves */}
      {strategies.some((s) => s.equity_curve) && (
        <div className="rounded-lg border border-border bg-card p-2">
          <Plot
            data={strategies
              .filter((s) => s.equity_curve)
              .map((s) => ({
                x: s.equity_curve!.map(([t]) => new Date(t * 1000)),
                y: s.equity_curve!.map(([, v]) => v),
                type: "scatter" as const,
                mode: "lines" as const,
                name: s.name,
                line: { color: STRATEGY_COLORS[s.name] ?? "#888", width: 1.5 },
              }))}
            layout={{
              height: 350,
              margin: { t: 20, l: 60, r: 20, b: 40 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#a3a3a3", size: 10 },
              xaxis: { gridcolor: "rgba(255,255,255,0.05)" },
              yaxis: { gridcolor: "rgba(255,255,255,0.05)", title: "Equity ($)" },
              legend: { orientation: "h", y: -0.15 },
              showlegend: true,
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </div>
      )}
    </div>
  )
}

// ── Metrics table ──

function MetricsTab({ strategies }: { strategies: StrategyData[] }) {
  const metrics = [
    { key: "sharpe", label: "Sharpe", fmt: (v: number) => v?.toFixed(2) },
    { key: "sortino", label: "Sortino", fmt: (v: number) => v?.toFixed(2) },
    { key: "calmar", label: "Calmar", fmt: (v: number) => v?.toFixed(2) },
    { key: "total_return", label: "Total Return", fmt: (v: number) => pctStr(v) },
    { key: "max_drawdown", label: "Max Drawdown", fmt: (v: number) => pctStr(v) },
    { key: "win_rate", label: "Win Rate", fmt: (v: number) => pctStr(v) },
    { key: "profit_factor", label: "Profit Factor", fmt: (v: number) => v?.toFixed(2) },
    { key: "trade_count", label: "Trades", fmt: (v: number) => v?.toString() },
    { key: "avg_trade_pnl", label: "Avg Trade P&L", fmt: (v: number) => pctStr(v) },
  ]

  return (
    <div className="overflow-auto rounded-lg border border-border">
      <table className="w-full text-xs">
        <thead className="bg-card">
          <tr className="border-b border-border">
            <th className="px-3 py-2 text-left text-muted-foreground">Metric</th>
            {strategies.map((s) => (
              <th key={s.name} className="px-3 py-2 text-right text-muted-foreground">
                {s.name.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={m.key} className="border-b border-border/50">
              <td className="px-3 py-1.5 text-muted-foreground">{m.label}</td>
              {strategies.map((s) => (
                <td key={s.name} className="px-3 py-1.5 text-right">
                  {m.fmt((s as Record<string, number>)[m.key]) ?? "--"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Per-Stock breakdown ──

function PerStockTab({ strategies }: { strategies: StrategyData[] }) {
  const withPerStock = strategies.filter((s) => s.per_symbol_metrics)
  if (withPerStock.length === 0) {
    return <p className="text-sm text-muted-foreground">No per-stock data available</p>
  }

  const strat = withPerStock[0]
  const entries = Object.entries(strat.per_symbol_metrics ?? {})
    .sort(([, a], [, b]) => b.sharpe - a.sharpe)

  return (
    <div className="overflow-auto rounded-lg border border-border">
      <table className="w-full text-xs">
        <thead className="bg-card">
          <tr className="border-b border-border text-left text-muted-foreground">
            <th className="px-3 py-2">Symbol</th>
            <th className="px-3 py-2 text-right">Sharpe</th>
            <th className="px-3 py-2 text-right">Return</th>
            <th className="px-3 py-2 text-right">Max DD</th>
            <th className="px-3 py-2 text-right">Win Rate</th>
            <th className="px-3 py-2 text-right">Trades</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([sym, m]) => (
            <tr key={sym} className="border-b border-border/50">
              <td className="px-3 py-1.5 font-medium">{sym}</td>
              <td className="px-3 py-1.5 text-right">{m.sharpe?.toFixed(2)}</td>
              <td className="px-3 py-1.5 text-right">{pctStr(m.total_return)}</td>
              <td className="px-3 py-1.5 text-right">{pctStr(m.max_drawdown)}</td>
              <td className="px-3 py-1.5 text-right">{pctStr(m.win_rate)}</td>
              <td className="px-3 py-1.5 text-right">{m.trade_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Regime performance ──

function RegimeTab({ strategies }: { strategies: StrategyData[] }) {
  const regimes = ["R0", "R1", "R2", "R3"]
  const withRegime = strategies.filter((s) => s.per_regime_sharpe)

  if (withRegime.length === 0) {
    return <p className="text-sm text-muted-foreground">No regime data available</p>
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-2">
        <Plot
          data={withRegime.map((s) => ({
            x: regimes,
            y: regimes.map((r) => s.per_regime_sharpe?.[r] ?? 0),
            type: "bar" as const,
            name: s.name,
            marker: { color: STRATEGY_COLORS[s.name] ?? "#888" },
          }))}
          layout={{
            height: 300,
            margin: { t: 20, l: 50, r: 20, b: 40 },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#a3a3a3", size: 10 },
            barmode: "group",
            xaxis: { title: "Regime" },
            yaxis: { title: "Sharpe Ratio", gridcolor: "rgba(255,255,255,0.05)" },
            legend: { orientation: "h", y: -0.2 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </div>

      {/* Regime trades table */}
      <div className="overflow-auto rounded-lg border border-border">
        <table className="w-full text-xs">
          <thead className="bg-card">
            <tr className="border-b border-border">
              <th className="px-3 py-2 text-left text-muted-foreground">Strategy</th>
              {regimes.map((r) => (
                <th key={r} className="px-3 py-2 text-right text-muted-foreground">
                  {r} Trades
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {withRegime.map((s) => (
              <tr key={s.name} className="border-b border-border/50">
                <td className="px-3 py-1.5">{s.name.replace(/_/g, " ")}</td>
                {regimes.map((r) => (
                  <td key={r} className="px-3 py-1.5 text-right">
                    {s.per_regime_trades?.[r] ?? 0}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Monthly returns heatmap ──

function MonthlyTab({ strategies }: { strategies: StrategyData[] }) {
  const withMonthly = strategies.filter((s) => s.monthly_returns)

  if (withMonthly.length === 0) {
    return <p className="text-sm text-muted-foreground">No monthly return data available</p>
  }

  const strat = withMonthly[0]
  const months = Object.keys(strat.monthly_returns ?? {}).sort()
  const values = months.map((m) => strat.monthly_returns![m] * 100)

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-card p-2">
        <Plot
          data={[
            {
              x: months,
              y: [strat.name],
              z: [values],
              type: "heatmap" as const,
              colorscale: [
                [0, "#ef4444"],
                [0.5, "#1c1c1c"],
                [1, "#22c55e"],
              ],
              zmin: -10,
              zmax: 10,
              hovertemplate: "%{x}: %{z:.1f}%<extra></extra>",
            },
          ]}
          layout={{
            height: 120,
            margin: { t: 20, l: 120, r: 20, b: 40 },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#a3a3a3", size: 10 },
            xaxis: { dtick: 1 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </div>

      {/* Multi-strategy monthly comparison table */}
      {withMonthly.length > 1 && (
        <div className="overflow-auto rounded-lg border border-border">
          <table className="w-full text-xs">
            <thead className="bg-card">
              <tr className="border-b border-border">
                <th className="px-3 py-2 text-left text-muted-foreground">Month</th>
                {withMonthly.map((s) => (
                  <th key={s.name} className="px-3 py-2 text-right text-muted-foreground">
                    {s.name.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {months.map((m) => (
                <tr key={m} className="border-b border-border/50">
                  <td className="px-3 py-1.5 text-muted-foreground">{m}</td>
                  {withMonthly.map((s) => {
                    const v = (s.monthly_returns?.[m] ?? 0) * 100
                    return (
                      <td
                        key={s.name}
                        className={`px-3 py-1.5 text-right ${
                          v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : ""
                        }`}
                      >
                        {v.toFixed(1)}%
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── Helpers ──

function Metric({
  label,
  value,
  colored,
}: {
  label: string
  value: string | undefined
  colored?: boolean
}) {
  let color = ""
  if (colored && value) {
    const num = parseFloat(value)
    if (num > 0) color = "text-emerald-400"
    else if (num < 0) color = "text-red-400"
  }
  return (
    <div>
      <span className="text-muted-foreground">{label}: </span>
      <span className={color}>{value ?? "--"}</span>
    </div>
  )
}

function pctStr(val: number | undefined | null): string {
  if (val == null) return "--"
  return `${(val * 100).toFixed(1)}%`
}
