import { useQuery } from "@tanstack/react-query"
import type { AdvisorMarketContext, PremiumAdvice, EquityAdvice } from "./ws"

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

// ── Types ──────────────────────────────────────────────

export interface AdvisorResponse {
  market_context: AdvisorMarketContext
  premium: PremiumAdvice[]
  equity: EquityAdvice[]
  timestamp: string
}

export interface SymbolsResponse {
  symbols: Record<string, SymbolInfo>
  count: number
}

export interface SymbolInfo {
  symbol: string
  last: number
  bid: number
  ask: number
  volume: number
  timestamp: string | null
  source: string
}

export interface HistoryResponse {
  symbol: string
  timeframe: string
  bars: HistoryBar[]
  count: number
}

export interface HistoryBar {
  t: string
  o: number
  h: number
  l: number
  c: number
  v: number
}

export interface MonitorResponse {
  status: string
  uptime_sec: number
  providers: ProviderInfo[]
  ws_clients: number
  pipeline: { running: boolean; timeframes: string[] }
}

export interface ProviderInfo {
  name: string
  connected: boolean
  symbols: number
  subscribed_symbols: string[]
}

// ── API functions ──────────────────────────────────────

export const api = {
  health: () => fetchJson<{ status: string; uptime: number; ws_clients: number }>("/api/health"),
  symbols: () => fetchJson<SymbolsResponse>("/api/symbols"),
  history: (symbol: string, tf = "1d", bars = 500) =>
    fetchJson<HistoryResponse>(`/api/history/${symbol}?tf=${tf}&bars=${bars}`),
  screeners: () => fetchJson<Record<string, unknown>>("/api/screeners"),
  backtest: () => fetchJson<Record<string, unknown>>("/api/backtest"),
  monitor: () => fetchJson<MonitorResponse>("/api/monitor"),
  dataQuality: () => fetchJson<Record<string, unknown>>("/api/monitor/data-quality"),
  signalData: (symbol: string, tf = "1d") =>
    fetchJson<Record<string, unknown>>(`/api/signal-data/${symbol}?tf=${tf}`),
  summary: () => fetchJson<Record<string, unknown>>("/api/summary"),
  scoreHistory: () => fetchJson<Record<string, unknown>>("/api/score-history"),
  indicators: () => fetchJson<Record<string, unknown>>("/api/indicators"),
  universe: () => fetchJson<Record<string, unknown>>("/api/universe"),
  advisor: () => fetchJson<AdvisorResponse>("/api/advisor"),
  advisorSymbol: (symbol: string) => fetchJson<Record<string, unknown>>(`/api/advisor/${symbol}`),
}

// ── Query hooks ────────────────────────────────────────

export function useSymbols() {
  return useQuery({ queryKey: ["symbols"], queryFn: api.symbols, refetchInterval: 10_000 })
}

export function useHistory(symbol: string, tf = "1d", bars = 500) {
  return useQuery({
    queryKey: ["history", symbol, tf, bars],
    queryFn: () => api.history(symbol, tf, bars),
    enabled: !!symbol,
  })
}

export function useScreeners() {
  return useQuery({ queryKey: ["screeners"], queryFn: api.screeners, staleTime: 5 * 60_000 })
}

export function useBacktest() {
  return useQuery({ queryKey: ["backtest"], queryFn: api.backtest, staleTime: 5 * 60_000 })
}

export function useMonitor() {
  return useQuery({ queryKey: ["monitor"], queryFn: api.monitor, refetchInterval: 5_000 })
}

export function useDataQuality() {
  return useQuery({
    queryKey: ["data-quality"],
    queryFn: api.dataQuality,
    staleTime: 5 * 60_000,
  })
}

export function useSignalData(symbol: string, tf = "1d") {
  return useQuery({
    queryKey: ["signal-data", symbol, tf],
    queryFn: () => api.signalData(symbol, tf),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
  })
}

export function useSummary() {
  return useQuery({ queryKey: ["summary"], queryFn: api.summary, staleTime: 5 * 60_000 })
}

export function useScoreHistory() {
  return useQuery({ queryKey: ["score-history"], queryFn: api.scoreHistory, staleTime: 5 * 60_000 })
}

export function useIndicators() {
  return useQuery({ queryKey: ["indicators"], queryFn: api.indicators, staleTime: 5 * 60_000 })
}

export function useUniverse() {
  return useQuery({ queryKey: ["universe"], queryFn: api.universe, staleTime: 5 * 60_000 })
}

export function useAdvisor() {
  return useQuery({ queryKey: ["advisor"], queryFn: api.advisor, staleTime: 30_000 })
}
