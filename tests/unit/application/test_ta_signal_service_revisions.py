from __future__ import annotations

import pytest

from src.application.services.ta_signal_service import TASignalService


class _FakeIndicatorEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def replace_symbol_histories(self, symbol: str, histories: dict) -> dict[str, int]:
        self.calls.append((symbol, histories))
        return {timeframe: len(rows) for timeframe, rows in histories.items()}


@pytest.mark.asyncio
async def test_service_delegates_symbol_history_replacement() -> None:
    service = TASignalService(event_bus=object())
    engine = _FakeIndicatorEngine()
    service._indicator_engine = engine
    histories = {"1d": [{"timestamp": "2026-01-02", "close": 100.0}]}

    result = await service.replace_symbol_histories("NVDA", histories)

    assert result == {"1d": 1}
    assert engine.calls == [("NVDA", histories)]


@pytest.mark.asyncio
async def test_service_replacement_requires_initialized_engine() -> None:
    service = TASignalService(event_bus=object())

    with pytest.raises(RuntimeError, match="IndicatorEngine not initialized"):
        await service.replace_symbol_histories("NVDA", {"1d": []})
