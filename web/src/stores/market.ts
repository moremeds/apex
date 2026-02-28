import { create } from "zustand"
import type { QuoteData, OHLCV, SignalData, ProviderStatus, AdvisorMarketContext, PremiumAdvice, EquityAdvice } from "@/lib/ws"

const MAX_SIGNALS = 200
const MAX_BARS_PER_TF = 500

/** Indicator values keyed by symbol → timeframe → indicator_name → value */
type IndicatorMap = Record<string, Record<string, Record<string, number>>>

interface MarketState {
  quotes: Record<string, QuoteData>
  bars: Record<string, Record<string, OHLCV[]>>
  indicators: IndicatorMap
  signals: SignalData[]
  providers: ProviderStatus[]
  wsStatus: "connecting" | "connected" | "disconnected"
  advisorContext: AdvisorMarketContext | null
  advisorPremium: PremiumAdvice[]
  advisorEquity: EquityAdvice[]

  updateQuote: (symbol: string, quote: QuoteData) => void
  appendBar: (symbol: string, tf: string, bar: OHLCV) => void
  updateIndicator: (symbol: string, tf: string, name: string, value: number) => void
  addSignal: (signal: SignalData) => void
  setProviders: (providers: ProviderStatus[]) => void
  setWsStatus: (status: MarketState["wsStatus"]) => void
  updateAdvisor: (ctx: AdvisorMarketContext, premium: PremiumAdvice[], equity: EquityAdvice[]) => void
}

export const useMarketStore = create<MarketState>((set) => ({
  quotes: {},
  bars: {},
  indicators: {},
  signals: [],
  providers: [],
  wsStatus: "disconnected",
  advisorContext: null,
  advisorPremium: [],
  advisorEquity: [],

  updateQuote: (symbol, quote) =>
    set((state) => ({
      quotes: { ...state.quotes, [symbol]: quote },
    })),

  appendBar: (symbol, tf, bar) =>
    set((state) => {
      const symbolBars = state.bars[symbol] ?? {}
      const tfBars = symbolBars[tf] ?? []
      const updated = [...tfBars, bar]
      return {
        bars: {
          ...state.bars,
          [symbol]: {
            ...symbolBars,
            [tf]: updated.length > MAX_BARS_PER_TF ? updated.slice(-MAX_BARS_PER_TF) : updated,
          },
        },
      }
    }),

  updateIndicator: (symbol, tf, name, value) =>
    set((state) => {
      const symInd = state.indicators[symbol] ?? {}
      const tfInd = symInd[tf] ?? {}
      return {
        indicators: {
          ...state.indicators,
          [symbol]: {
            ...symInd,
            [tf]: { ...tfInd, [name]: value },
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

  updateAdvisor: (ctx, premium, equity) => set({ advisorContext: ctx, advisorPremium: premium, advisorEquity: equity }),
}))
