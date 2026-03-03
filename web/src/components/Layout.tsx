import { NavLink, Outlet } from "react-router-dom"
import { useMarketStore } from "@/stores/market"
import { useWebSocket } from "@/hooks/useWebSocket"

const navItems = [
  { to: "/", label: "Overview", end: true },
  { to: "/portfolio", label: "Portfolio" },
  { to: "/signals", label: "Signals" },
  { to: "/screeners", label: "Screeners" },
  { to: "/monitor", label: "Monitor" },
  { to: "/backtest", label: "Backtest" },
  { to: "/advisor", label: "Advisor" },
]

function WsIndicator() {
  const status = useMarketStore((s) => s.wsStatus)
  const color =
    status === "connected"
      ? "bg-emerald-500"
      : status === "connecting"
        ? "bg-amber-500 animate-pulse"
        : "bg-red-500"
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className={`inline-block h-2 w-2 rounded-full ${color}`} />
      {status}
    </div>
  )
}

export function Layout() {
  useWebSocket()

  return (
    <div className="flex min-h-screen flex-col">
      {/* Top nav bar */}
      <header className="fixed inset-x-0 top-0 z-50 border-b border-border bg-card">
        <div className="flex h-12 items-center justify-between px-6">
          {/* Left: brand + nav links */}
          <div className="flex items-center gap-8">
            <span className="text-sm font-bold tracking-wider text-primary">
              APEX
            </span>

            <nav className="flex items-center gap-1">
              {navItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `relative px-3 py-3 text-sm transition-colors ${
                      isActive
                        ? "text-foreground after:absolute after:inset-x-1 after:bottom-0 after:h-0.5 after:rounded-full after:bg-primary"
                        : "text-muted-foreground hover:text-foreground"
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </div>

          {/* Right: WS status */}
          <WsIndicator />
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 pt-12">
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
