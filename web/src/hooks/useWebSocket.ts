import { useEffect, useCallback } from "react"
import { useMarketStore } from "@/stores/market"
import type { WsCommand, WsMessage } from "@/lib/ws"

const WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30_000

// Module-level singleton — only one WS connection regardless of how many
// components call useWebSocket()
let ws: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | undefined
let attempt = 0
let refCount = 0

// Queue commands sent before the socket reaches OPEN state
let pendingCommands: WsCommand[] = []

/** Fetch symbols from multiple sources, falling back gracefully. */
async function fetchSubscriptionSymbols(): Promise<string[]> {
  // Try /api/symbols first (live Longbridge quotes)
  try {
    const resp = await fetch("/api/symbols")
    const data = (await resp.json()) as { symbols?: Record<string, unknown> }
    const syms = Object.keys(data.symbols ?? {})
    if (syms.length > 0) return syms
  } catch {
    // fall through
  }

  // Fallback: /api/summary tickers (always available from R2/static)
  try {
    const resp = await fetch("/api/summary")
    const data = (await resp.json()) as { tickers?: { symbol: string }[] }
    const syms = (data.tickers ?? []).map((t) => t.symbol).filter(Boolean)
    if (syms.length > 0) return syms
  } catch {
    // fall through
  }

  return []
}

function flushPendingCommands() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return
  for (const cmd of pendingCommands) {
    ws.send(JSON.stringify(cmd))
  }
  pendingCommands = []
}

function doConnect() {
  if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return

  const store = useMarketStore.getState()
  store.setWsStatus("connecting")

  const socket = new WebSocket(WS_URL)
  ws = socket

  socket.onopen = () => {
    useMarketStore.getState().setWsStatus("connected")
    attempt = 0

    // Flush any commands queued before OPEN
    flushPendingCommands()

    // Auto-subscribe to all symbols for live price updates
    fetchSubscriptionSymbols().then((syms) => {
      if (syms.length > 0 && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ cmd: "subscribe", symbols: syms }))
      }
    })
  }

  socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data) as WsMessage
      const s = useMarketStore.getState()
      switch (msg.type) {
        case "quote":
          s.updateQuote(msg.symbol, msg.data)
          break
        case "bar":
          s.appendBar(msg.symbol, msg.timeframe, msg.data)
          break
        case "indicator":
          s.updateIndicator(msg.symbol, msg.timeframe, msg.name, msg.value)
          break
        case "signal":
          s.addSignal(msg.data)
          break
        case "status":
          s.setProviders(msg.providers)
          break
        case "advisor":
          s.updateAdvisor(msg.market_context, msg.premium ?? [], msg.equity ?? [])
          break
        case "strategy_state":
          s.appendStrategyState(msg.symbol, msg.timeframe, msg.indicator, msg.state)
          break
      }
    } catch {
      // ignore malformed messages
    }
  }

  socket.onclose = () => {
    useMarketStore.getState().setWsStatus("disconnected")
    ws = null
    if (refCount > 0) {
      const delay = Math.min(RECONNECT_BASE_MS * 2 ** attempt, RECONNECT_MAX_MS)
      attempt++
      reconnectTimer = setTimeout(doConnect, delay)
    }
  }

  socket.onerror = () => {
    socket.close()
  }
}

function doDisconnect() {
  clearTimeout(reconnectTimer)
  reconnectTimer = undefined
  pendingCommands = []
  ws?.close()
  ws = null
}

export function useWebSocket() {
  useEffect(() => {
    refCount++
    if (refCount === 1) {
      doConnect()
    }
    return () => {
      refCount--
      if (refCount === 0) {
        doDisconnect()
      }
    }
  }, [])

  const send = useCallback((cmd: WsCommand) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd))
    } else {
      // Queue for delivery when socket opens
      pendingCommands.push(cmd)
    }
  }, [])

  return { send, status: useMarketStore((s) => s.wsStatus) }
}
