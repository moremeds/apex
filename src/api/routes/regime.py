"""Regime state endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/regime", tags=["regime"])


@router.get("/{symbol}")
async def regime_for_symbol(symbol: str, request: Request) -> dict:
    """Get latest regime state for a symbol from PG score_history table."""
    pg_pool = getattr(request.app.state, "pg_pool", None)
    if pg_pool is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT symbol, ts, score, trend_state, regime "
            "FROM score_history WHERE symbol = $1 ORDER BY ts DESC LIMIT 1",
            symbol.upper(),
        )

    if row is None:
        raise HTTPException(status_code=404, detail=f"No regime data for '{symbol}'")

    return {
        "symbol": row["symbol"],
        "regime": row["regime"] or "R1",
        "score": row["score"] or 0,
        "trend_state": row["trend_state"] or "unknown",
        "ts": str(row["ts"]),
    }
