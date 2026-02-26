import { useState, useMemo } from "react"
import Plot from "react-plotly.js"
import { useBacktest } from "@/lib/api"

type Tab = "overview" | "metrics" | "per-stock" | "sector" | "regime" | "regime-analysis" | "symbol-map" | "trades"

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
  per_sector_sharpe?: Record<string, number>
  per_sector_return?: Record<string, number>
  per_regime_sharpe?: Record<string, number>
  per_regime_return?: Record<string, number>
  per_regime_trades?: Record<string, number>
  per_regime_win_rate?: Record<string, number>
  rolling_sharpe?: [number, number][]
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
  { id: "sector", label: "Sector" },
  { id: "regime", label: "Regime Perf" },
  { id: "regime-analysis", label: "Regime Analysis" },
  { id: "symbol-map", label: "Symbol Map" },
  { id: "trades", label: "Trades" },
]

const STRATEGY_COLORS: Record<string, string> = {
  trend_pulse: "#3b82f6",
  regime_flex: "#22c55e",
  sector_pulse: "#f59e0b",
  rsi_mean_reversion: "#a855f7",
  pead: "#ef4444",
  buy_and_hold: "#6b7280",
}

const DARK_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#8899aa", size: 10 },
}

const DIVERGING_COLORSCALE: [number, string][] = [
  [0, "#ef4444"],
  [0.5, "#1c1c1c"],
  [1, "#22c55e"],
]

export function Backtest() {
  const [tab, setTab] = useState<Tab>("overview")
  const { data: raw, isLoading, error } = useBacktest()

  const bundle = raw as BacktestBundle | undefined
  const strategies = useMemo(() => {
    if (!bundle?.strategies) return []
    return Object.values(bundle.strategies)
  }, [bundle])

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading backtest data...</p>
  if (error) return <p className="text-sm text-red-400">Failed to load backtest: {(error as Error).message}</p>
  if (strategies.length === 0) {
    return <p className="text-sm text-muted-foreground">No backtest data available. R2 strategies.json may not be configured.</p>
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">{bundle?.title ?? "Backtest"}</h2>
        {bundle?.period && <p className="text-xs text-muted-foreground">{bundle.period}</p>}
      </div>

      <div className="flex flex-wrap gap-1 rounded-lg bg-secondary p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
              tab === t.id ? "bg-background text-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && <OverviewTab strategies={strategies} />}
      {tab === "metrics" && <MetricsTab strategies={strategies} />}
      {tab === "per-stock" && <PerStockTab strategies={strategies} />}
      {tab === "sector" && <SectorTab strategies={strategies} />}
      {tab === "regime" && <RegimeTab strategies={strategies} />}
      {tab === "regime-analysis" && <RegimeAnalysisTab strategies={strategies} />}
      {tab === "symbol-map" && <SymbolMapTab strategies={strategies} />}
      {tab === "trades" && <TradesTab strategies={strategies} />}
    </div>
  )
}

// ── Overview: scorecards + equity + drawdown curves ──

function OverviewTab({ strategies }: { strategies: StrategyData[] }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        {strategies.map((s) => (
          <div key={s.name} className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">{s.name.replace(/_/g, " ")}</h4>
              {s.tier && <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-muted-foreground">{s.tier}</span>}
            </div>
            <div className="mt-3 grid grid-cols-2 gap-y-1.5 text-xs">
              <MetricItem label="Sharpe" value={s.sharpe?.toFixed(2)} />
              <MetricItem label="Return" value={pctStr(s.total_return)} colored />
              <MetricItem label="Max DD" value={pctStr(s.max_drawdown)} colored />
              <MetricItem label="Win Rate" value={pctStr(s.win_rate)} />
              <MetricItem label="Trades" value={s.trade_count?.toString()} />
              <MetricItem label="P/F" value={s.profit_factor?.toFixed(2)} />
            </div>
          </div>
        ))}
      </div>

      {/* Equity curves */}
      {strategies.some((s) => s.equity_curve) && (
        <ChartCard title="Equity Curves">
          <Plot
            data={strategies.filter((s) => s.equity_curve).map((s) => ({
              x: s.equity_curve!.map(([t]) => new Date(t * 1000)),
              y: s.equity_curve!.map(([, v]) => v),
              type: "scatter" as const,
              mode: "lines" as const,
              name: s.name,
              line: { color: STRATEGY_COLORS[s.name] ?? "#888", width: 1.5 },
            }))}
            layout={{ ...DARK_LAYOUT, height: 350, margin: { t: 20, l: 60, r: 20, b: 40 }, xaxis: { gridcolor: "rgba(255,255,255,0.05)" }, yaxis: { gridcolor: "rgba(255,255,255,0.05)", title: { text: "Equity ($)" } }, legend: { orientation: "h", y: -0.15 }, showlegend: true }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </ChartCard>
      )}

      {/* Drawdown curves */}
      {strategies.some((s) => s.drawdown_curve) && (
        <ChartCard title="Drawdown">
          <Plot
            data={strategies.filter((s) => s.drawdown_curve).map((s) => ({
              x: s.drawdown_curve!.map(([t]) => new Date(t * 1000)),
              y: s.drawdown_curve!.map(([, v]) => v * 100),
              type: "scatter" as const,
              mode: "lines" as const,
              name: s.name,
              fill: "tozeroy" as const,
              line: { color: STRATEGY_COLORS[s.name] ?? "#888", width: 1 },
              fillcolor: `${STRATEGY_COLORS[s.name] ?? "#888"}22`,
            }))}
            layout={{ ...DARK_LAYOUT, height: 250, margin: { t: 10, l: 60, r: 20, b: 40 }, xaxis: { gridcolor: "rgba(255,255,255,0.05)" }, yaxis: { gridcolor: "rgba(255,255,255,0.05)", title: { text: "Drawdown (%)" }, rangemode: "tozero" }, legend: { orientation: "h", y: -0.2 }, showlegend: true }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </ChartCard>
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
              <th key={s.name} className="px-3 py-2 text-right text-muted-foreground">{s.name.replace(/_/g, " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={m.key} className="border-b border-border/50">
              <td className="px-3 py-1.5 text-muted-foreground">{m.label}</td>
              {strategies.map((s) => (
                <td key={s.name} className="px-3 py-1.5 text-right">{m.fmt((s as unknown as Record<string, number>)[m.key]) ?? "--"}</td>
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
  if (withPerStock.length === 0) return <EmptyState text="No per-stock data available" />

  const strat = withPerStock[0]
  const entries = Object.entries(strat.per_symbol_metrics ?? {}).sort(([, a], [, b]) => b.sharpe - a.sharpe)

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

// ── NEW: Sector heatmap (sector × strategy for Sharpe + Return) ──

function SectorTab({ strategies }: { strategies: StrategyData[] }) {
  const withSector = strategies.filter((s) => s.per_sector_sharpe)
  if (withSector.length === 0) return <EmptyState text="No sector data available" />

  const allSectors = [...new Set(withSector.flatMap((s) => Object.keys(s.per_sector_sharpe ?? {})))].sort()
  const stratNames = withSector.map((s) => s.name)

  // Sharpe heatmap: rows = sectors, cols = strategies
  const sharpeZ = allSectors.map((sector) => stratNames.map((sn) => withSector.find((s) => s.name === sn)?.per_sector_sharpe?.[sector] ?? 0))
  const returnZ = allSectors.map((sector) => stratNames.map((sn) => (withSector.find((s) => s.name === sn)?.per_sector_return?.[sector] ?? 0) * 100))

  return (
    <div className="space-y-4">
      <ChartCard title="Sector Sharpe Ratio">
        <Plot
          data={[{
            x: stratNames.map((n) => n.replace(/_/g, " ")),
            y: allSectors,
            z: sharpeZ,
            type: "heatmap" as const,
            colorscale: DIVERGING_COLORSCALE,
            zmin: -1, zmax: 2,
            hovertemplate: "%{y} / %{x}: Sharpe %{z:.2f}<extra></extra>",
          }]}
          layout={{ ...DARK_LAYOUT, height: Math.max(250, allSectors.length * 25 + 80), margin: { t: 10, l: 100, r: 30, b: 60 } }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </ChartCard>

      <ChartCard title="Sector Return (%)">
        <Plot
          data={[{
            x: stratNames.map((n) => n.replace(/_/g, " ")),
            y: allSectors,
            z: returnZ,
            type: "heatmap" as const,
            colorscale: DIVERGING_COLORSCALE,
            zmin: -20, zmax: 20,
            hovertemplate: "%{y} / %{x}: Return %{z:.1f}%<extra></extra>",
          }]}
          layout={{ ...DARK_LAYOUT, height: Math.max(250, allSectors.length * 25 + 80), margin: { t: 10, l: 100, r: 30, b: 60 } }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </ChartCard>
    </div>
  )
}

// ── Regime performance (existing, enhanced) ──

function RegimeTab({ strategies }: { strategies: StrategyData[] }) {
  const regimes = ["R0", "R1", "R2", "R3"]
  const withRegime = strategies.filter((s) => s.per_regime_sharpe)
  if (withRegime.length === 0) return <EmptyState text="No regime data available" />

  return (
    <div className="space-y-4">
      <ChartCard title="Sharpe by Regime">
        <Plot
          data={withRegime.map((s) => ({
            x: regimes,
            y: regimes.map((r) => s.per_regime_sharpe?.[r] ?? 0),
            type: "bar" as const,
            name: s.name,
            marker: { color: STRATEGY_COLORS[s.name] ?? "#888" },
          }))}
          layout={{ ...DARK_LAYOUT, height: 300, margin: { t: 20, l: 50, r: 20, b: 40 }, barmode: "group", xaxis: { title: { text: "Regime" } }, yaxis: { title: { text: "Sharpe" }, gridcolor: "rgba(255,255,255,0.05)" }, legend: { orientation: "h", y: -0.2 } }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </ChartCard>

      <div className="overflow-auto rounded-lg border border-border">
        <table className="w-full text-xs">
          <thead className="bg-card">
            <tr className="border-b border-border">
              <th className="px-3 py-2 text-left text-muted-foreground">Strategy</th>
              {regimes.map((r) => (
                <th key={r} className="px-3 py-2 text-right text-muted-foreground">{r} Trades</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {withRegime.map((s) => (
              <tr key={s.name} className="border-b border-border/50">
                <td className="px-3 py-1.5">{s.name.replace(/_/g, " ")}</td>
                {regimes.map((r) => (
                  <td key={r} className="px-3 py-1.5 text-right">{s.per_regime_trades?.[r] ?? 0}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── NEW: Regime Analysis (trade count + win rate grouped bars) ──

function RegimeAnalysisTab({ strategies }: { strategies: StrategyData[] }) {
  const regimes = ["R0", "R1", "R2", "R3"]
  const withRegime = strategies.filter((s) => s.per_regime_trades || s.per_regime_win_rate)
  if (withRegime.length === 0) return <EmptyState text="No regime analysis data available" />

  return (
    <div className="space-y-4">
      <ChartCard title="Trade Count by Regime">
        <Plot
          data={withRegime.map((s) => ({
            x: regimes,
            y: regimes.map((r) => s.per_regime_trades?.[r] ?? 0),
            type: "bar" as const,
            name: s.name,
            marker: { color: STRATEGY_COLORS[s.name] ?? "#888" },
          }))}
          layout={{ ...DARK_LAYOUT, height: 300, margin: { t: 20, l: 50, r: 20, b: 40 }, barmode: "group", xaxis: { title: { text: "Regime" } }, yaxis: { title: { text: "Trade Count" }, gridcolor: "rgba(255,255,255,0.05)" }, legend: { orientation: "h", y: -0.2 } }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </ChartCard>

      <ChartCard title="Win Rate by Regime">
        <Plot
          data={[
            // 50% reference line
            { x: regimes, y: regimes.map(() => 50), type: "scatter" as const, mode: "lines" as const, name: "50% ref", line: { color: "#666", dash: "dash", width: 1 }, showlegend: false },
            ...withRegime.map((s) => ({
              x: regimes,
              y: regimes.map((r) => ((s.per_regime_win_rate?.[r]) ?? (s.win_rate ?? 0)) * 100),
              type: "bar" as const,
              name: s.name,
              marker: { color: STRATEGY_COLORS[s.name] ?? "#888" },
            })),
          ]}
          layout={{ ...DARK_LAYOUT, height: 300, margin: { t: 20, l: 50, r: 20, b: 40 }, barmode: "group", xaxis: { title: { text: "Regime" } }, yaxis: { title: { text: "Win Rate (%)" }, gridcolor: "rgba(255,255,255,0.05)", range: [0, 100] }, legend: { orientation: "h", y: -0.2 } }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </ChartCard>
    </div>
  )
}

// ── NEW: Symbol Map heatmap (symbol × strategy, Sharpe colorscale) ──

function SymbolMapTab({ strategies }: { strategies: StrategyData[] }) {
  const withPerStock = strategies.filter((s) => s.per_symbol_metrics)
  if (withPerStock.length === 0) return <EmptyState text="No per-stock data available" />

  const allSymbols = [...new Set(withPerStock.flatMap((s) => Object.keys(s.per_symbol_metrics ?? {})))].sort()
  const stratNames = withPerStock.map((s) => s.name)
  const z = allSymbols.map((sym) => stratNames.map((sn) => withPerStock.find((s) => s.name === sn)?.per_symbol_metrics?.[sym]?.sharpe ?? 0))

  return (
    <ChartCard title="Symbol × Strategy Sharpe">
      <Plot
        data={[{
          x: stratNames.map((n) => n.replace(/_/g, " ")),
          y: allSymbols,
          z,
          type: "heatmap" as const,
          colorscale: DIVERGING_COLORSCALE,
          zmin: -1, zmax: 3,
          hovertemplate: "%{y} / %{x}: Sharpe %{z:.2f}<extra></extra>",
        }]}
        layout={{ ...DARK_LAYOUT, height: Math.max(400, allSymbols.length * 18 + 80), margin: { t: 10, l: 70, r: 30, b: 60 } }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%" }}
      />
    </ChartCard>
  )
}

// ── NEW: Trades tab (rolling Sharpe + monthly heatmap) ──

function TradesTab({ strategies }: { strategies: StrategyData[] }) {
  const withMonthly = strategies.filter((s) => s.monthly_returns)
  const withRolling = strategies.filter((s) => s.rolling_sharpe)

  return (
    <div className="space-y-4">
      {/* Rolling Sharpe */}
      {withRolling.length > 0 && (
        <ChartCard title="Rolling 60-Day Sharpe">
          <Plot
            data={withRolling.map((s) => ({
              x: s.rolling_sharpe!.map(([t]) => new Date(t * 1000)),
              y: s.rolling_sharpe!.map(([, v]) => v),
              type: "scatter" as const,
              mode: "lines" as const,
              name: s.name,
              line: { color: STRATEGY_COLORS[s.name] ?? "#888", width: 1.5 },
            }))}
            layout={{ ...DARK_LAYOUT, height: 300, margin: { t: 20, l: 50, r: 20, b: 40 }, xaxis: { gridcolor: "rgba(255,255,255,0.05)" }, yaxis: { title: { text: "Sharpe" }, gridcolor: "rgba(255,255,255,0.05)" }, legend: { orientation: "h", y: -0.2 }, showlegend: true }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </ChartCard>
      )}

      {/* Monthly returns heatmap */}
      {withMonthly.length > 0 && (
        <ChartCard title="Monthly Returns">
          <Plot
            data={withMonthly.map((s) => {
              const months = Object.keys(s.monthly_returns ?? {}).sort()
              return {
                x: months,
                y: [s.name.replace(/_/g, " ")],
                z: [months.map((m) => (s.monthly_returns![m] ?? 0) * 100)],
                type: "heatmap" as const,
                colorscale: DIVERGING_COLORSCALE,
                zmin: -10, zmax: 10,
                hovertemplate: "%{x}: %{z:.1f}%<extra>%{y}</extra>",
                showscale: s === withMonthly[0],
              }
            })}
            layout={{ ...DARK_LAYOUT, height: Math.max(120, withMonthly.length * 40 + 80), margin: { t: 20, l: 120, r: 40, b: 40 }, xaxis: { dtick: 1 } }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </ChartCard>
      )}

      {/* Monthly comparison table */}
      {withMonthly.length > 1 && (() => {
        const months = Object.keys(withMonthly[0].monthly_returns ?? {}).sort()
        return (
          <div className="overflow-auto rounded-lg border border-border">
            <table className="w-full text-xs">
              <thead className="bg-card">
                <tr className="border-b border-border">
                  <th className="px-3 py-2 text-left text-muted-foreground">Month</th>
                  {withMonthly.map((s) => (
                    <th key={s.name} className="px-3 py-2 text-right text-muted-foreground">{s.name.replace(/_/g, " ")}</th>
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
                        <td key={s.name} className={`px-3 py-1.5 text-right ${v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : ""}`}>
                          {v.toFixed(1)}%
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      })()}

      {withRolling.length === 0 && withMonthly.length === 0 && (
        <EmptyState text="No trades data available" />
      )}
    </div>
  )
}

// ── Shared components ──

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <h4 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{title}</h4>
      <div className="rounded-lg border border-border bg-card p-2">{children}</div>
    </div>
  )
}

function EmptyState({ text }: { text: string }) {
  return <p className="text-sm text-muted-foreground">{text}</p>
}

function MetricItem({ label, value, colored }: { label: string; value: string | undefined; colored?: boolean }) {
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
