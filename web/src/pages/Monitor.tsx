import { useMonitor, useDataQuality } from "@/lib/api"
import { useMarketStore } from "@/stores/market"

export function Monitor() {
  const { data: monitor, isLoading } = useMonitor()
  const { data: dq } = useDataQuality()
  const wsProviders = useMarketStore((s) => s.providers)
  const signals = useMarketStore((s) => s.signals)
  const wsStatus = useMarketStore((s) => s.wsStatus)

  const providers =
    wsProviders.length > 0 ? wsProviders : monitor?.providers ?? []

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">System Monitor</h2>

      {/* System stats */}
      <div className="grid grid-cols-4 gap-3">
        <MetricCard
          label="Uptime"
          value={monitor ? formatUptime(monitor.uptime_sec) : "--"}
        />
        <MetricCard
          label="WS Clients"
          value={monitor?.ws_clients ?? 0}
        />
        <MetricCard
          label="Pipeline"
          value={monitor?.pipeline.running ? "Running" : "Stopped"}
          color={monitor?.pipeline.running ? "text-emerald-400" : "text-red-400"}
        />
        <MetricCard
          label="WebSocket"
          value={wsStatus}
          color={
            wsStatus === "connected"
              ? "text-emerald-400"
              : wsStatus === "connecting"
                ? "text-amber-400"
                : "text-red-400"
          }
        />
      </div>

      {/* Provider cards */}
      <section>
        <h3 className="mb-2 text-sm font-medium text-muted-foreground">
          Data Providers
        </h3>
        {providers.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {isLoading ? "Loading..." : "No providers configured"}
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {providers.map((p) => (
              <div
                key={p.name}
                className="rounded-lg border border-border bg-card px-4 py-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium capitalize">
                    {p.name}
                  </span>
                  <span
                    className={`inline-block h-2 w-2 rounded-full ${
                      p.connected ? "bg-emerald-500" : "bg-red-500"
                    }`}
                  />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {p.symbols} symbols subscribed
                </p>
                {p.subscribed_symbols && p.subscribed_symbols.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {p.subscribed_symbols.slice(0, 10).map((sym) => (
                      <span
                        key={sym}
                        className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground"
                      >
                        {sym}
                      </span>
                    ))}
                    {p.subscribed_symbols.length > 10 && (
                      <span className="text-[10px] text-muted-foreground">
                        +{p.subscribed_symbols.length - 10} more
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Timeframes */}
      {monitor?.pipeline.timeframes && monitor.pipeline.timeframes.length > 0 && (
        <section>
          <h3 className="mb-2 text-sm font-medium text-muted-foreground">
            Active Timeframes
          </h3>
          <div className="flex gap-2">
            {monitor.pipeline.timeframes.map((tf) => (
              <span
                key={tf}
                className="rounded bg-secondary px-2 py-1 text-xs text-secondary-foreground"
              >
                {tf}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Recent signals */}
      <section>
        <h3 className="mb-2 text-sm font-medium text-muted-foreground">
          Recent Signals ({signals.length})
        </h3>
        {signals.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No signals received yet
          </p>
        ) : (
          <div className="max-h-64 overflow-auto rounded-lg border border-border">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Symbol</th>
                  <th className="px-3 py-2">Rule</th>
                  <th className="px-3 py-2">Direction</th>
                  <th className="px-3 py-2">TF</th>
                </tr>
              </thead>
              <tbody>
                {signals.slice(0, 50).map((s, i) => (
                  <tr key={i} className="border-b border-border/50">
                    <td className="px-3 py-1.5 text-muted-foreground">
                      {new Date(s.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-3 py-1.5 font-medium">{s.symbol}</td>
                    <td className="px-3 py-1.5">{s.rule}</td>
                    <td className="px-3 py-1.5">
                      <span
                        className={
                          s.direction === "bullish"
                            ? "text-emerald-400"
                            : s.direction === "bearish"
                              ? "text-red-400"
                              : "text-muted-foreground"
                        }
                      >
                        {s.direction}
                      </span>
                    </td>
                    <td className="px-3 py-1.5">{s.timeframe}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Data quality */}
      {dq && (
        <section>
          <h3 className="mb-2 text-sm font-medium text-muted-foreground">
            Data Quality
          </h3>
          <pre className="max-h-48 overflow-auto rounded-lg border border-border bg-card p-3 text-xs text-muted-foreground">
            {JSON.stringify(dq, null, 2)}
          </pre>
        </section>
      )}
    </div>
  )
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string
  value: string | number
  color?: string
}) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`text-lg font-semibold ${color ?? ""}`}>{value}</p>
    </div>
  )
}

function formatUptime(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.round(sec / 60)}m`
  const h = Math.floor(sec / 3600)
  const m = Math.round((sec % 3600) / 60)
  return `${h}h ${m}m`
}
