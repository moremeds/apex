/** WebSocket protocol types matching the FastAPI backend. */

export interface WsCommand {
  cmd: "subscribe" | "unsubscribe"
  symbols: string[]
  types?: ("quote" | "bar" | "indicator" | "signal")[]
}

/** Server → Client message envelope. */
export type WsMessage =
  | { type: "quote"; symbol: string; data: QuoteData }
  | { type: "bar"; symbol: string; timeframe: string; data: OHLCV }
  | { type: "indicator"; symbol: string; timeframe: string; name: string; value: number }
  | { type: "signal"; data: SignalData }
  | { type: "status"; providers: ProviderStatus[] }
  | { type: "advisor"; market_context: AdvisorMarketContext; premium: PremiumAdvice[]; equity: EquityAdvice[] }
  | { type: "strategy_state"; symbol: string; timeframe: string; indicator: string; state: Record<string, unknown> }

export interface QuoteData {
  last: number
  bid: number
  ask: number
  volume: number
  ts: string
  prev_close?: number
}

export interface OHLCV {
  t: string
  o: number
  h: number
  l: number
  c: number
  v: number
}

export interface SignalData {
  symbol: string
  rule: string
  direction: "bullish" | "bearish" | "neutral"
  strength: number
  timeframe: string
  timestamp: string
}

export interface ProviderStatus {
  name: string
  connected: boolean
  symbols: number
  subscribed_symbols?: string[]
}

export interface AdvisorMarketContext {
  regime: string
  regime_name: string
  regime_confidence: number
  vix: number
  vix_percentile: number
  vrp_zscore: number
  term_structure_ratio: number
  term_structure_state: string
  timestamp: string
}

export interface PremiumAdvice {
  symbol: string
  action: string
  strategy: string | null
  display_name: string | null
  confidence: number
  legs: LegSpec[]
  vrp_zscore: number
  iv_percentile: number
  term_structure_ratio: number
  regime: string
  earnings_warning: string | null
  reasoning: string[]
}

export interface LegSpec {
  side: string
  option_type: string
  target_delta: number
  target_dte: number
  estimated_strike: number
}

export interface EquityAdvice {
  symbol: string
  sector: string
  action: string
  confidence: number
  regime: string
  signal_summary: Record<string, number>
  top_signals: { rule: string; direction: string; strength: number }[]
  trend_pulse: Record<string, unknown> | null
  key_levels: Record<string, number>
  reasoning: string[]
}
