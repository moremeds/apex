import { useEffect, useRef } from "react"
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type HistogramData,
  type Time,
  ColorType,
  CrosshairMode,
} from "lightweight-charts"
import type { HistoryBar } from "@/lib/api"
import type { OHLCV } from "@/lib/ws"

interface Props {
  bars: HistoryBar[]
  liveBars?: OHLCV[]
  height?: number
}

export function CandlestickChart({ bars, liveBars, height = 400 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null)
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null)

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0d1520" },
        textColor: "#8899aa",
      },
      grid: {
        vertLines: { color: "rgba(100,140,180,0.06)" },
        horzLines: { color: "rgba(100,140,180,0.06)" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: { borderColor: "rgba(100,140,180,0.15)" },
      rightPriceScale: { borderColor: "rgba(100,140,180,0.15)" },
      height,
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    })

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    })

    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    })

    chartRef.current = chart
    candleRef.current = candleSeries
    volumeRef.current = volumeSeries

    const ro = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect
      chart.applyOptions({ width })
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      candleRef.current = null
      volumeRef.current = null
    }
  }, [height])

  // Update data
  useEffect(() => {
    if (!candleRef.current || !volumeRef.current) return

    const allBars = [
      ...bars.map((b) => ({
        time: toChartTime(b.t),
        open: b.o,
        high: b.h,
        low: b.l,
        close: b.c,
        volume: b.v,
      })),
      ...(liveBars ?? []).map((b) => ({
        time: toChartTime(b.t),
        open: b.o,
        high: b.h,
        low: b.l,
        close: b.c,
        volume: b.v,
      })),
    ]

    const candleData: CandlestickData[] = allBars.map((b) => ({
      time: b.time as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))

    const volumeData: HistogramData[] = allBars.map((b) => ({
      time: b.time as Time,
      value: b.volume,
      color: b.close >= b.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
    }))

    candleRef.current.setData(candleData)
    volumeRef.current.setData(volumeData)
    chartRef.current?.timeScale().fitContent()
  }, [bars, liveBars])

  return (
    <div
      ref={containerRef}
      className="rounded-lg border border-border"
    />
  )
}

function toChartTime(ts: string): string {
  // Lightweight charts expects "YYYY-MM-DD" for daily or UTC timestamp
  if (!ts) return ""
  const d = new Date(ts)
  return d.toISOString().slice(0, 10)
}
