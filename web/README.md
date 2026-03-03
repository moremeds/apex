# APEX Live Dashboard — Frontend

React + TypeScript + Vite frontend for the APEX live trading dashboard.

## Architecture

- **React 19** with TypeScript strict mode
- **Vite** for dev server and bundling
- **TanStack Query** for API data fetching with caching
- **Zustand** for WebSocket-driven market state
- **Tailwind CSS** for styling
- **Plotly.js** for treemaps and charts
- **lightweight-charts** for candlestick/indicator charts

## Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `Overview.tsx` | ETF dashboard cards, regime counts, stock universe treemap |
| `/signals` | `Signals.tsx` | Per-symbol candlestick charts with technical indicators |
| `/advisor` | `Advisor.tsx` | Trading advisor with equity signals, VRP, and premium strategies |
| `/monitor` | `Monitor.tsx` | Live WebSocket feed monitor |

## Development

```bash
# Install dependencies
npm install

# Dev server (proxies API to localhost:8000)
npm run dev

# Type check
npx tsc --noEmit

# Build for production
npm run build
```

## API Integration

The frontend connects to the FastAPI backend:
- REST endpoints at `/api/*` (summary, symbols, screeners, jobs)
- WebSocket at `/ws` for live quotes, bars, indicators, and signals
- API hooks in `src/lib/api.ts`, WebSocket hook in `src/hooks/useWebSocket.ts`
- Market state store in `src/stores/market.ts` (Zustand)

## Running with Backend

```bash
# From project root — starts both backend and frontend
make live
```
