import { useState, type ReactNode } from "react"

interface Props {
  title: string
  defaultOpen?: boolean
  children: ReactNode
}

export function Collapsible({ title, defaultOpen = true, children }: Props) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:bg-accent/20"
      >
        <span className={`text-[10px] transition-transform ${open ? "" : "-rotate-90"}`}>▼</span>
        <span>{title}</span>
      </button>
      {open && <div className="border-t border-border px-4 pb-4 pt-3">{children}</div>}
    </div>
  )
}
