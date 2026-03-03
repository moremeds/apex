import { create } from "zustand"
import type {
  QuoteData, OHLCV, SignalData, ProviderStatus,
  AdvisorMarketContext, PremiumAdvice, EquityAdvice,
  PositionData, AccountData, PortfolioGreeks, PortfolioPnl, BrokerStatusData,
} from "@/lib/ws"

const MAX_SIGNALS = 200
const MAX_BARS_PER_TF = 500
const MAX_STRATEGY_STATES = 500

/** Indicator values keyed by symbol → timeframe → indicator_name → value */
type IndicatorMap = Record<string, Record<string, Record<string, number>>>

/** Strategy states keyed by symbol → timeframe → indicator → state[] */
type StrategyStateMap = Record<string, Record<string, Record<string, Record<string, unknown>[]>>>

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
  strategyStates: StrategyStateMap

  // Portfolio state
  positions: PositionData[]
  account: AccountData | null
  brokerStatus: BrokerStatusData[]
  portfolioGreeks: PortfolioGreeks
  portfolioPnl: PortfolioPnl
  portfolioEnabled: boolean

  updateQuote: (symbol: string, quote: QuoteData) => void
  appendBar: (symbol: string, tf: string, bar: OHLCV) => void
  updateIndicator: (symbol: string, tf: string, name: string, value: number) => void
  addSignal: (signal: SignalData) => void
  setProviders: (providers: ProviderStatus[]) => void
  setWsStatus: (status: MarketState["wsStatus"]) => void
  updateAdvisor: (ctx: AdvisorMarketContext, premium: PremiumAdvice[], equity: EquityAdvice[]) => void
  appendStrategyState: (symbol: string, tf: string, indicator: string, state: Record<string, unknown>) => void

  // Portfolio actions
  updatePortfolioSnapshot: (
    positions: PositionData[],
    account: AccountData | null,
    greeks: PortfolioGreeks,
    pnl: PortfolioPnl,
    brokerStatus: BrokerStatusData[],
  ) => void
  applyPositionDelta: (delta: {
    symbol: string
    new_mark_price: number
    pnl_change: number
    daily_pnl_change: number
    delta_change: number
    gamma_change: number
    vega_change: number
    theta_change: number
  }) => void
  updateAccount: (account: Partial<AccountData>) => void
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
  strategyStates: {},

  // Portfolio initial state
  positions: [],
  account: null,
  brokerStatus: [],
  portfolioGreeks: { delta: 0, gamma: 0, vega: 0, theta: 0 },
  portfolioPnl: { unrealized: 0, daily: 0, net_liquidation: 0 },
  portfolioEnabled: false,

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

  appendStrategyState: (symbol, tf, indicator, state) =>
    set((s) => {
      const symStates = s.strategyStates[symbol] ?? {}
      const tfStates = symStates[tf] ?? {}
      const indStates = tfStates[indicator] ?? []
      const updated = [...indStates, state]
      return {
        strategyStates: {
          ...s.strategyStates,
          [symbol]: {
            ...symStates,
            [tf]: {
              ...tfStates,
              [indicator]: updated.length > MAX_STRATEGY_STATES
                ? updated.slice(-MAX_STRATEGY_STATES)
                : updated,
            },
          },
        },
      }
    }),

  // Portfolio actions
  updatePortfolioSnapshot: (positions, account, greeks, pnl, brokerStatus) =>
    set({
      positions,
      account,
      portfolioGreeks: greeks,
      portfolioPnl: pnl,
      brokerStatus,
      portfolioEnabled: true,
    }),

  applyPositionDelta: (delta) =>
    set((state) => {
      // Apply incremental update to matching position
      const positions = state.positions.map((p) => {
        if (p.symbol !== delta.symbol) return p
        return {
          ...p,
          mark_price: delta.new_mark_price,
          unrealized_pnl: (p.unrealized_pnl ?? 0) + delta.pnl_change,
          daily_pnl: (p.daily_pnl ?? 0) + delta.daily_pnl_change,
          delta: (p.delta ?? 0) + delta.delta_change,
          gamma: (p.gamma ?? 0) + delta.gamma_change,
          vega: (p.vega ?? 0) + delta.vega_change,
          theta: (p.theta ?? 0) + delta.theta_change,
        }
      })

      // Update aggregated Greeks
      const portfolioGreeks = {
        delta: state.portfolioGreeks.delta + delta.delta_change,
        gamma: state.portfolioGreeks.gamma + delta.gamma_change,
        vega: state.portfolioGreeks.vega + delta.vega_change,
        theta: state.portfolioGreeks.theta + delta.theta_change,
      }

      // Update aggregated P&L
      const portfolioPnl = {
        ...state.portfolioPnl,
        unrealized: state.portfolioPnl.unrealized + delta.pnl_change,
        daily: state.portfolioPnl.daily + delta.daily_pnl_change,
      }

      return { positions, portfolioGreeks, portfolioPnl }
    }),

  updateAccount: (partial) =>
    set((state) => ({
      account: state.account ? { ...state.account, ...partial } : null,
    })),
}))
