import { useState, useMemo, useEffect } from "react"
import { useSymbols, useHistory } from "@/lib/api"
import { useMarketStore } from "@/stores/market"
import { useWebSocket } from "@/hooks/useWebSocket"
import { CandlestickChart } from "@/components/CandlestickChart"

const TIMEFRAMES = ["1d", "4h", "1h"] as const

export function Signals() {
  const { data: symbolsData } = useSymbols()
  const [symbol, setSymbol] = useState("")
  const [tf, setTf] = useState<string>("1d")
  const { send } = useWebSocket()

  const symbolList = useMemo(
    () => Object.keys(symbolsData?.symbols ?? {}).sort(),
    [symbolsData],
  )

  // Auto-select first symbol
  useEffect(() => {
    if (!symbol && symbolList.length > 0) {
      setSymbol(symbolList[0])
    }
  }, [symbol, symbolList])

  // Subscribe/unsubscribe on symbol change
  useEffect(() => {
    if (!symbol) return
    send({ cmd: "subscribe", symbols: [symbol], types: ["quote", "bar", "signal"] })
    return () => {
      send({ cmd: "unsubscribe", symbols: [symbol] })
    }
  }, [symbol, send])

  const { data: historyData, isLoading } = useHistory(symbol, tf)
  const bars = historyData?.bars ?? []

  const liveBars = useMarketStore((s) => s.bars[symbol]?.[tf] ?? [])
  const quote = useMarketStore((s) => s.quotes[symbol])
  const signals = useMarketStore((s) =>
    s.signals.filter((sig) => sig.symbol === symbol).slice(0, 20),
  )

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-3">
        {/* Symbol selector */}
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="rounded-md border border-input bg-background px-3 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
        >
          {symbolList.length === 0 && (
            <option value="">No symbols</option>
          )}
          {symbolList.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        {/* Timeframe tabs */}
        <div className="flex gap-1 rounded-lg bg-secondary p-1">
          {TIMEFRAMES.map((t) => (
            <button
              key={t}
              onClick={() => setTf(t)}
              className={`rounded-md px-3 py-1 text-sm transition-colors ${
                tf === t
                  ? "bg-background text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Live quote */}
        {quote && (
          <div className="ml-auto text-sm">
            <span className="text-muted-foreground">Last: </span>
            <span className="font-medium">${quote.last.toFixed(2)}</span>
            <span className="ml-2 text-muted-foreground">
              {quote.bid.toFixed(2)} / {quote.ask.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Chart */}
      {isLoading ? (
        <div className="flex h-96 items-center justify-center rounded-lg border border-border text-sm text-muted-foreground">
          Loading chart data...
        </div>
      ) : bars.length > 0 ? (
        <CandlestickChart bars={bars} liveBars={liveBars} />
      ) : symbol ? (
        <div className="flex h-96 items-center justify-center rounded-lg border border-border text-sm text-muted-foreground">
          No history available for {symbol} ({tf})
        </div>
      ) : null}

      {/* Signals table */}
      {signals.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-medium text-muted-foreground">
            Recent Signals
          </h3>
          <div className="max-h-48 overflow-auto rounded-lg border border-border">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="px-3 py-2">Time</th>
                  <th className="px-3 py-2">Rule</th>
                  <th className="px-3 py-2">Direction</th>
                  <th className="px-3 py-2">Strength</th>
                  <th className="px-3 py-2">TF</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((s, i) => (
                  <tr key={i} className="border-b border-border/50">
                    <td className="px-3 py-1.5 text-muted-foreground">
                      {new Date(s.timestamp).toLocaleTimeString()}
                    </td>
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
                    <td className="px-3 py-1.5">{(s.strength * 100).toFixed(0)}%</td>
                    <td className="px-3 py-1.5">{s.timeframe}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
