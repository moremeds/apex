import { BrowserRouter, Routes, Route } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Layout } from "@/components/Layout"
import { Overview } from "@/pages/Overview"
import { Signals } from "@/pages/Signals"
import { Screeners } from "@/pages/Screeners"
import { Backtest } from "@/pages/Backtest"
import { Monitor } from "@/pages/Monitor"
import { Advisor } from "@/pages/Advisor"
import { Portfolio } from "@/pages/Portfolio"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="signals" element={<Signals />} />
            <Route path="screeners" element={<Screeners />} />
            <Route path="backtest" element={<Backtest />} />
            <Route path="monitor" element={<Monitor />} />
            <Route path="advisor" element={<Advisor />} />
            <Route path="portfolio" element={<Portfolio />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
