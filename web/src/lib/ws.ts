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
  | { type: "portfolio"; positions: PositionData[]; account: AccountData | null; greeks: PortfolioGreeks; pnl: PortfolioPnl; broker_status: BrokerStatusData[]; position_count: number; timestamp: string | null }
  | { type: "position_delta"; symbol: string; underlying: string; new_mark_price: number; pnl_change: number; daily_pnl_change: number; delta_change: number; gamma_change: number; vega_change: number; theta_change: number }
  | { type: "account_update"; account_id: string; net_liquidation: number; total_cash: number; buying_power: number; margin_used: number; margin_available: number; unrealized_pnl: number; realized_pnl: number; daily_pnl: number }

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

// ── Portfolio types ──────────────────────────────────────

export interface PositionData {
  symbol: string
  underlying: string
  asset_type: string
  quantity: number
  avg_price: number
  multiplier: number
  mark_price: number | null
  market_value: number | null
  unrealized_pnl: number | null
  daily_pnl: number | null
  delta: number | null
  gamma: number | null
  vega: number | null
  theta: number | null
  iv: number | null
  delta_dollars: number | null
  notional: number | null
  expiry: string | null
  strike: number | null
  right: string | null
  days_to_expiry: number | null
  source: string
  account_id: string | null
  has_market_data: boolean
  has_greeks: boolean
  is_stale: boolean
}

export interface AccountData {
  net_liquidation: number
  total_cash: number
  buying_power: number
  margin_used: number
  margin_available: number
  maintenance_margin: number
  init_margin_req: number
  excess_liquidity: number
  unrealized_pnl: number
  realized_pnl: number
  margin_utilization: number
  account_id: string | null
  timestamp: string | null
}

export interface PortfolioGreeks {
  delta: number
  gamma: number
  vega: number
  theta: number
}

export interface PortfolioPnl {
  unrealized: number
  daily: number
  net_liquidation: number
}

export interface BrokerStatusData {
  name: string
  connected: boolean
  position_count: number
  last_error: string | null
}
