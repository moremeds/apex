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

export interface QuoteData {
  last: number
  bid: number
  ask: number
  volume: number
  ts: string
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
