import { NavLink, Outlet } from "react-router-dom"
import { useMarketStore } from "@/stores/market"

const navItems = [
  { to: "/", label: "Overview", end: true },
  { to: "/signals", label: "Signals" },
  { to: "/screeners", label: "Screeners" },
  { to: "/backtest", label: "Backtest" },
  { to: "/monitor", label: "Monitor" },
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
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-48 flex-col border-r border-border bg-card">
        <div className="border-b border-border px-4 py-3">
          <h1 className="text-sm font-bold tracking-wider text-primary">
            APEX
          </h1>
          <p className="text-[10px] text-muted-foreground">Live Dashboard</p>
        </div>

        <nav className="flex-1 space-y-0.5 px-2 py-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `block rounded-md px-3 py-1.5 text-sm transition-colors ${
                  isActive
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-border px-4 py-3">
          <WsIndicator />
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">
        <Outlet />
      </main>
    </div>
  )
}
