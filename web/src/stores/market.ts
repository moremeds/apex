import { create } from "zustand"
import type { QuoteData, OHLCV, SignalData, ProviderStatus } from "@/lib/ws"

const MAX_SIGNALS = 200

interface MarketState {
  quotes: Record<string, QuoteData>
  bars: Record<string, Record<string, OHLCV[]>>
  signals: SignalData[]
  providers: ProviderStatus[]
  wsStatus: "connecting" | "connected" | "disconnected"

  updateQuote: (symbol: string, quote: QuoteData) => void
  appendBar: (symbol: string, tf: string, bar: OHLCV) => void
  addSignal: (signal: SignalData) => void
  setProviders: (providers: ProviderStatus[]) => void
  setWsStatus: (status: MarketState["wsStatus"]) => void
}

export const useMarketStore = create<MarketState>((set) => ({
  quotes: {},
  bars: {},
  signals: [],
  providers: [],
  wsStatus: "disconnected",

  updateQuote: (symbol, quote) =>
    set((state) => ({
      quotes: { ...state.quotes, [symbol]: quote },
    })),

  appendBar: (symbol, tf, bar) =>
    set((state) => {
      const symbolBars = state.bars[symbol] ?? {}
      const tfBars = symbolBars[tf] ?? []
      return {
        bars: {
          ...state.bars,
          [symbol]: {
            ...symbolBars,
            [tf]: [...tfBars, bar],
          },
        },
      }
    }),

  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, MAX_SIGNALS),
    })),

  setProviders: (providers) => set({ providers }),

  setWsStatus: (wsStatus) => set({ wsStatus }),
}))
