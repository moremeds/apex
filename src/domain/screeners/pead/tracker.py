"""PEAD tracker domain models.

Tracks PEAD candidates over time and resolves outcomes using OHLC first-touch logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class TrackedCandidate:
    """A PEAD candidate being tracked to resolution."""

    symbol: str
    entry_date: date
    entry_price: float
    profit_target_pct: float
    stop_loss_pct: float
    max_hold_days: int
    quality_score: float
    quality_label: str
    sue_score: float
    multi_quarter_sue: float | None
    regime: str
    # Outcome (filled on resolution)
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str | None = None  # "profit_target" | "stop_loss" | "timeout"
    pnl_pct: float | None = None
    status: str = "open"  # "open" | "won" | "lost" | "timeout"

    @property
    def target_price(self) -> float:
        """Absolute price for profit target."""
        return self.entry_price * (1 + self.profit_target_pct)

    @property
    def stop_price(self) -> float:
        """Absolute price for stop loss."""
        return self.entry_price * (1 + self.stop_loss_pct)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat(),
            "entry_price": self.entry_price,
            "profit_target_pct": self.profit_target_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "max_hold_days": self.max_hold_days,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "sue_score": self.sue_score,
            "multi_quarter_sue": self.multi_quarter_sue,
            "regime": self.regime,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pct": self.pnl_pct,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrackedCandidate:
        """Deserialize from dict."""
        return cls(
            symbol=d["symbol"],
            entry_date=date.fromisoformat(d["entry_date"]),
            entry_price=d["entry_price"],
            profit_target_pct=d["profit_target_pct"],
            stop_loss_pct=d["stop_loss_pct"],
            max_hold_days=d["max_hold_days"],
            quality_score=d["quality_score"],
            quality_label=d["quality_label"],
            sue_score=d["sue_score"],
            multi_quarter_sue=d.get("multi_quarter_sue"),
            regime=d["regime"],
            exit_date=date.fromisoformat(d["exit_date"]) if d.get("exit_date") else None,
            exit_price=d.get("exit_price"),
            exit_reason=d.get("exit_reason"),
            pnl_pct=d.get("pnl_pct"),
            status=d.get("status", "open"),
        )


@dataclass
class TrackerStats:
    """Aggregate performance statistics."""

    total: int = 0
    open: int = 0
    won: int = 0
    lost: int = 0
    timeout: int = 0
    win_rate: float | None = None
    avg_pnl_pct: float | None = None
    avg_hold_days: float | None = None
    by_quality: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "open": self.open,
            "won": self.won,
            "lost": self.lost,
            "timeout": self.timeout,
            "win_rate": round(self.win_rate, 4) if self.win_rate is not None else None,
            "avg_pnl_pct": round(self.avg_pnl_pct, 4) if self.avg_pnl_pct is not None else None,
            "avg_hold_days": (
                round(self.avg_hold_days, 1) if self.avg_hold_days is not None else None
            ),
            "by_quality": self.by_quality,
        }
