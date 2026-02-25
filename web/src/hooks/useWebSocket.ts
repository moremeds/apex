import { useEffect, useRef, useCallback } from "react"
import { useMarketStore } from "@/stores/market"
import type { WsCommand, WsMessage } from "@/lib/ws"

const WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30_000

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>()
  const attemptRef = useRef(0)

  const setWsStatus = useMarketStore((s) => s.setWsStatus)
  const updateQuote = useMarketStore((s) => s.updateQuote)
  const appendBar = useMarketStore((s) => s.appendBar)
  const addSignal = useMarketStore((s) => s.addSignal)
  const setProviders = useMarketStore((s) => s.setProviders)

  const dispatch = useCallback(
    (msg: WsMessage) => {
      switch (msg.type) {
        case "quote":
          updateQuote(msg.symbol, msg.data)
          break
        case "bar":
          appendBar(msg.symbol, msg.timeframe, msg.data)
          break
        case "signal":
          addSignal(msg.data)
          break
        case "status":
          setProviders(msg.providers)
          break
      }
    },
    [updateQuote, appendBar, addSignal, setProviders],
  )

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setWsStatus("connecting")
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setWsStatus("connected")
      attemptRef.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage
        dispatch(msg)
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      setWsStatus("disconnected")
      wsRef.current = null
      // Exponential backoff
      const delay = Math.min(
        RECONNECT_BASE_MS * 2 ** attemptRef.current,
        RECONNECT_MAX_MS,
      )
      attemptRef.current++
      reconnectTimer.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [dispatch, setWsStatus])

  const send = useCallback(
    (cmd: WsCommand) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(cmd))
      }
    },
    [],
  )

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { send, status: useMarketStore((s) => s.wsStatus) }
}
