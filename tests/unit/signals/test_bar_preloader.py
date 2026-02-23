"""Tests for BarPreloader concurrency and dependency ordering."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.orchestrator.signal_pipeline.bar_preloader import BarPreloader

# =============================================================================
# Helpers
# =============================================================================


def _make_bar(
    bar_start: datetime | None = None,
    open_: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    volume: int = 1_000_000,
) -> MagicMock:
    """Create a mock bar object with OHLCV attributes."""
    bar = MagicMock()
    bar.bar_start = bar_start or datetime(2024, 6, 1, 9, 30, tzinfo=timezone.utc)
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    return bar


def _make_historical_data_manager(
    bars_per_call: int = 5,
    max_history_days: int = 365,
) -> MagicMock:
    """Create a mock HistoricalDataManager.

    ensure_data is async, get_max_history_days is sync.
    """
    mgr = MagicMock()
    mgr.get_max_history_days = MagicMock(return_value=max_history_days)
    bars = [_make_bar() for _ in range(bars_per_call)]
    mgr.ensure_data = AsyncMock(return_value=bars)
    return mgr


def _make_indicator_engine(inject_count: int = 5) -> MagicMock:
    """Create a mock IndicatorEngine.

    inject_historical_bars is sync (returns int).
    compute_on_history is async (returns int).
    """
    engine = MagicMock()
    engine.inject_historical_bars = MagicMock(return_value=inject_count)
    engine.compute_on_history = AsyncMock(return_value=inject_count)
    return engine


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hdm() -> MagicMock:
    """Historical data manager mock."""
    return _make_historical_data_manager()


@pytest.fixture
def engine() -> MagicMock:
    """Indicator engine mock."""
    return _make_indicator_engine()


# =============================================================================
# Tests: empty / guard-clause paths
# =============================================================================


@pytest.mark.asyncio
async def test_preload_empty_symbols_returns_empty(hdm: MagicMock, engine: MagicMock) -> None:
    """preload_startup([]) should return {} without touching hdm or engine."""
    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d", "1h"],
    )

    result = await preloader.preload_startup([])

    assert result == {}
    hdm.ensure_data.assert_not_called()
    engine.inject_historical_bars.assert_not_called()


@pytest.mark.asyncio
async def test_preload_no_indicator_engine_returns_empty(hdm: MagicMock) -> None:
    """When indicator_engine is None/falsy, preload_startup returns {} immediately."""
    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=None,  # type: ignore[arg-type]
        timeframes=["1d"],
    )

    result = await preloader.preload_startup(["AAPL"])

    assert result == {}
    hdm.ensure_data.assert_not_called()


@pytest.mark.asyncio
async def test_preload_no_historical_manager_returns_empty(engine: MagicMock) -> None:
    """When historical_data_manager is None/falsy, preload_startup returns {} immediately."""
    preloader = BarPreloader(
        historical_data_manager=None,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    result = await preloader.preload_startup(["AAPL"])

    assert result == {}
    engine.inject_historical_bars.assert_not_called()


# =============================================================================
# Tests: sequential path (concurrency=1, default)
# =============================================================================


@pytest.mark.asyncio
async def test_preload_sequential_default(hdm: MagicMock, engine: MagicMock) -> None:
    """With default concurrency=1, _preload_sequential is called for each (symbol, tf)."""
    symbols = ["AAPL", "TSLA"]
    timeframes = ["1d", "1h"]

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=timeframes,
    )

    result = await preloader.preload_startup(symbols)

    # Each symbol should have bars injected for each timeframe
    assert "AAPL" in result
    assert "TSLA" in result

    # ensure_data called once per (symbol, timeframe) = 2 symbols x 2 timeframes = 4
    assert hdm.ensure_data.call_count == 4

    # inject_historical_bars and compute_on_history called same number of times
    assert engine.inject_historical_bars.call_count == 4
    assert engine.compute_on_history.call_count == 4

    # Each symbol gets 5 bars x 2 timeframes = 10
    assert result["AAPL"] == 10
    assert result["TSLA"] == 10


@pytest.mark.asyncio
async def test_preload_sequential_single_timeframe(hdm: MagicMock, engine: MagicMock) -> None:
    """Sequential preload with one symbol and one timeframe."""
    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    result = await preloader.preload_startup(["SPY"])

    assert result == {"SPY": 5}
    hdm.ensure_data.call_count == 1
    engine.inject_historical_bars.assert_called_once()
    engine.compute_on_history.assert_called_once()


@pytest.mark.asyncio
async def test_preload_sequential_no_bars_returned(engine: MagicMock) -> None:
    """When ensure_data returns empty list, symbol not added to results."""
    hdm = _make_historical_data_manager()
    hdm.ensure_data = AsyncMock(return_value=[])

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    result = await preloader.preload_startup(["AAPL"])

    # No bars means no injection
    assert result == {}
    engine.inject_historical_bars.assert_not_called()


# =============================================================================
# Tests: concurrent path (concurrency > 1)
# =============================================================================


@pytest.mark.asyncio
async def test_preload_concurrent_phases_ordering() -> None:
    """Verify 4h processing happens AFTER all 1h/1d tasks complete.

    We track the order of ensure_data calls and assert that every 1h and 1d call
    appears before any 4h call in the sequence.
    """
    call_order: list[tuple[str, str]] = []

    async def _tracking_ensure_data(
        symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[MagicMock]:
        # Small sleep to let concurrent tasks interleave
        await asyncio.sleep(0.001)
        call_order.append((symbol, timeframe))
        return [_make_bar()]

    hdm = _make_historical_data_manager()
    hdm.ensure_data = AsyncMock(side_effect=_tracking_ensure_data)

    engine = _make_indicator_engine(inject_count=1)

    symbols = ["AAPL", "TSLA"]
    timeframes = ["1d", "1h", "4h"]

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=timeframes,
        preload_config={"preload_concurrency": 4},
    )

    result = await preloader.preload_startup(symbols)

    # All symbols should have bars
    assert "AAPL" in result
    assert "TSLA" in result

    # Separate calls into phases
    non_4h_calls = [(s, tf) for s, tf in call_order if tf != "4h"]
    four_h_calls = [(s, tf) for s, tf in call_order if tf == "4h"]

    # Phase 1 should have all 1d and 1h calls
    assert len(non_4h_calls) == 4  # 2 symbols x 2 non-4h timeframes
    assert len(four_h_calls) == 2  # 2 symbols x 1 (4h)

    # Every non-4h call must appear before every 4h call
    if four_h_calls:
        first_4h_index = call_order.index(four_h_calls[0])
        last_non_4h_index = max(call_order.index(c) for c in non_4h_calls)
        assert last_non_4h_index < first_4h_index, (
            f"Phase ordering violated: last non-4h at index {last_non_4h_index}, "
            f"first 4h at index {first_4h_index}. Order: {call_order}"
        )


@pytest.mark.asyncio
async def test_preload_concurrent_no_4h_skips_phase2() -> None:
    """When timeframes have no 4h, only Phase 1 runs."""
    call_order: list[tuple[str, str]] = []

    async def _tracking_ensure_data(
        symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[MagicMock]:
        call_order.append((symbol, timeframe))
        return [_make_bar()]

    hdm = _make_historical_data_manager()
    hdm.ensure_data = AsyncMock(side_effect=_tracking_ensure_data)
    engine = _make_indicator_engine(inject_count=1)

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d", "1h"],
        preload_config={"preload_concurrency": 2},
    )

    result = await preloader.preload_startup(["AAPL"])

    assert "AAPL" in result
    # Only 1d and 1h calls, no 4h
    timeframes_called = {tf for _, tf in call_order}
    assert timeframes_called == {"1d", "1h"}
    assert len(call_order) == 2


@pytest.mark.asyncio
async def test_preload_concurrent_only_4h() -> None:
    """When timeframes is only ['4h'], Phase 1 is empty, Phase 2 runs."""
    hdm = _make_historical_data_manager()
    engine = _make_indicator_engine(inject_count=3)

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["4h"],
        preload_config={"preload_concurrency": 2},
    )

    result = await preloader.preload_startup(["AAPL", "TSLA"])

    assert result["AAPL"] == 3
    assert result["TSLA"] == 3
    # ensure_data called twice (one per symbol, one timeframe)
    assert hdm.ensure_data.call_count == 2


# =============================================================================
# Tests: error isolation
# =============================================================================


@pytest.mark.asyncio
async def test_preload_error_isolation_sequential() -> None:
    """One symbol failing in sequential mode doesn't crash others."""
    call_count = 0

    async def _ensure_data_with_error(
        symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[MagicMock]:
        nonlocal call_count
        call_count += 1
        if symbol == "BAD":
            raise RuntimeError("Network error for BAD")
        return [_make_bar()]

    hdm = _make_historical_data_manager()
    hdm.ensure_data = AsyncMock(side_effect=_ensure_data_with_error)
    engine = _make_indicator_engine(inject_count=1)

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    result = await preloader.preload_startup(["AAPL", "BAD", "TSLA"])

    # AAPL and TSLA should succeed; BAD should be isolated
    assert "AAPL" in result
    assert "TSLA" in result
    assert "BAD" not in result

    # All three symbols were attempted
    assert call_count == 3


@pytest.mark.asyncio
async def test_preload_error_isolation_concurrent() -> None:
    """One symbol failing in concurrent mode doesn't crash others."""
    call_count = 0

    async def _ensure_data_with_error(
        symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[MagicMock]:
        nonlocal call_count
        call_count += 1
        if symbol == "BAD":
            raise RuntimeError("Network error for BAD")
        return [_make_bar()]

    hdm = _make_historical_data_manager()
    hdm.ensure_data = AsyncMock(side_effect=_ensure_data_with_error)
    engine = _make_indicator_engine(inject_count=1)

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d", "1h"],
        preload_config={"preload_concurrency": 3},
    )

    result = await preloader.preload_startup(["AAPL", "BAD", "TSLA"])

    # AAPL and TSLA should succeed across both timeframes
    assert result["AAPL"] == 2  # 1 bar x 2 timeframes
    assert result["TSLA"] == 2
    assert "BAD" not in result

    # All (symbol, tf) pairs attempted: 3 symbols x 2 timeframes = 6
    assert call_count == 6


# =============================================================================
# Tests: bar dict conversion
# =============================================================================


@pytest.mark.asyncio
async def test_preload_bar_dict_conversion(hdm: MagicMock, engine: MagicMock) -> None:
    """Verify bars are converted to dicts with correct keys before injection."""
    ts = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
    bar = _make_bar(bar_start=ts, open_=150.0, high=155.0, low=148.0, close=153.0, volume=500_000)
    hdm.ensure_data = AsyncMock(return_value=[bar])

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    await preloader.preload_startup(["AAPL"])

    # Verify the dict passed to inject_historical_bars
    call_args = engine.inject_historical_bars.call_args
    symbol_arg, tf_arg, bar_dicts_arg = call_args[0]

    assert symbol_arg == "AAPL"
    assert tf_arg == "1d"
    assert len(bar_dicts_arg) == 1

    d = bar_dicts_arg[0]
    assert d["timestamp"] == ts
    assert d["open"] == 150.0
    assert d["high"] == 155.0
    assert d["low"] == 148.0
    assert d["close"] == 153.0
    assert d["volume"] == 500_000


@pytest.mark.asyncio
async def test_preload_bar_volume_none_defaults_to_zero(hdm: MagicMock, engine: MagicMock) -> None:
    """When bar.volume is None, it should be converted to 0 in the dict."""
    bar = _make_bar(volume=None)  # type: ignore[arg-type]
    hdm.ensure_data = AsyncMock(return_value=[bar])

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    await preloader.preload_startup(["AAPL"])

    bar_dicts = engine.inject_historical_bars.call_args[0][2]
    assert bar_dicts[0]["volume"] == 0


# =============================================================================
# Tests: config and metadata
# =============================================================================


@pytest.mark.asyncio
async def test_preload_sets_last_cache_refresh(hdm: MagicMock, engine: MagicMock) -> None:
    """After successful preload, last_cache_refresh should be set."""
    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    assert preloader.last_cache_refresh is None

    await preloader.preload_startup(["AAPL"])

    assert preloader.last_cache_refresh is not None


@pytest.mark.asyncio
async def test_preload_uses_max_history_days_per_timeframe() -> None:
    """Verify get_max_history_days is called per timeframe for lookback calculation."""
    hdm = _make_historical_data_manager()

    def _max_days(tf: str) -> int:
        return {"1d": 730, "1h": 90, "4h": 180}.get(tf, 365)

    hdm.get_max_history_days = MagicMock(side_effect=_max_days)
    engine = _make_indicator_engine()

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d", "1h"],
    )

    await preloader.preload_startup(["AAPL"])

    # get_max_history_days called once per timeframe
    assert hdm.get_max_history_days.call_count == 2
    hdm.get_max_history_days.assert_any_call("1d")
    hdm.get_max_history_days.assert_any_call("1h")


@pytest.mark.asyncio
async def test_preload_concurrency_routing() -> None:
    """Verify concurrency=1 routes to sequential, concurrency>1 routes to concurrent."""
    hdm = _make_historical_data_manager()
    engine = _make_indicator_engine()

    # Test concurrency=1 (default) -> sequential
    preloader_seq = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
    )

    with patch.object(BarPreloader, "_preload_sequential", new_callable=AsyncMock) as mock_seq:
        mock_seq.return_value = {"AAPL": 5}
        result = await preloader_seq.preload_startup(["AAPL"])
        mock_seq.assert_called_once()
        assert result == {"AAPL": 5}

    # Test concurrency>1 -> concurrent
    preloader_conc = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d"],
        preload_config={"preload_concurrency": 4},
    )

    with patch.object(BarPreloader, "_preload_concurrent", new_callable=AsyncMock) as mock_conc:
        mock_conc.return_value = {"AAPL": 5}
        result = await preloader_conc.preload_startup(["AAPL"])
        mock_conc.assert_called_once()
        assert result == {"AAPL": 5}


# =============================================================================
# Tests: accumulation across timeframes
# =============================================================================


@pytest.mark.asyncio
async def test_preload_accumulates_bars_across_timeframes(
    hdm: MagicMock,
) -> None:
    """Bar counts for the same symbol accumulate across timeframes."""
    engine = MagicMock()
    # Return different counts per call to verify accumulation
    engine.inject_historical_bars = MagicMock(side_effect=[10, 20])
    engine.compute_on_history = AsyncMock(return_value=0)

    preloader = BarPreloader(
        historical_data_manager=hdm,
        indicator_engine=engine,
        timeframes=["1d", "1h"],
    )

    result = await preloader.preload_startup(["AAPL"])

    # 10 (from 1d) + 20 (from 1h) = 30
    assert result["AAPL"] == 30
