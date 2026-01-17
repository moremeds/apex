"""
Backtest engine interface (Protocol).

Defines the contract that all backtest engines must implement:
- VectorBTEngine: Fast vectorized backtesting for screening
- ApexEngine: Event-driven backtesting with full feature support

The two-stage pipeline uses VectorBT for fast screening (10,000+ variants)
and Apex for final validation (top 100 candidates).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from ...core import RunMetrics, RunResult, RunSpec, RunStatus


class EngineType(str, Enum):
    """Available backtest engine types."""

    VECTORBT = "vectorbt"
    APEX = "apex"


@dataclass
class EngineConfig:
    """Configuration for a backtest engine."""

    engine_type: EngineType = EngineType.VECTORBT

    # Data settings - default to IB (no Yahoo Finance dependency)
    data_source: str = "ib"  # ib, yahoo, local
    data_cache_path: Optional[str] = None

    # IB-specific settings (only used when data_source="ib")
    ib_host: str = "127.0.0.1"
    ib_port: int = 4001  # Gateway Paper (4002=Gateway Live, 7497=TWS Paper, 7496=TWS Live)
    ib_client_id: Optional[int] = None  # None = pick from pool (4-10)
    ib_rate_limit: bool = True  # Apply IB pacing rules

    # Execution profile
    slippage_bps: float = 0.0  # 0 for fast_track, realistic value for validation
    commission_per_share: float = 0.0
    fill_model: str = "close"  # close, vwap, open

    # Performance settings
    enable_caching: bool = True
    parallel_symbols: bool = True


@runtime_checkable
class BacktestEngine(Protocol):
    """
    Protocol defining the backtest engine interface.

    All backtest engines (VectorBT, Apex, etc.) must implement this interface
    to be usable in the systematic runner and parity harness.
    """

    def run(
        self,
        spec: RunSpec,
        data: Optional[pd.DataFrame] = None,
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> RunResult:
        """
        Execute a single backtest run.

        Args:
            spec: Run specification (symbol, window, params)
            data: Optional pre-loaded primary timeframe OHLCV data. If None, engine loads it.
            secondary_data: Optional secondary timeframe data for MTF strategies.
                Keys are timeframe strings (e.g., "1h", "4h").
                Values are DataFrames with OHLCV columns.

        Returns:
            RunResult with metrics and optional equity curve
        """
        ...

    def run_batch(
        self,
        specs: List[RunSpec],
        data: Optional[Dict[str, pd.DataFrame]] = None,
        secondary_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> List[RunResult]:
        """
        Execute multiple runs (potentially in parallel/vectorized).

        VectorBT can vectorize across parameters for massive speedup.
        Apex runs sequentially but with full feature support.

        Args:
            specs: List of run specifications
            data: Optional dict of symbol -> OHLCV DataFrame (primary timeframe)
            secondary_data: Optional dict of symbol -> {timeframe: DataFrame}
                for multi-timeframe strategies

        Returns:
            List of RunResults in same order as specs
        """
        ...

    @property
    def engine_type(self) -> EngineType:
        """Return the engine type identifier."""
        ...

    @property
    def supports_vectorization(self) -> bool:
        """Whether this engine supports vectorized parameter sweeps."""
        ...


class BaseEngine(ABC):
    """
    Abstract base class for backtest engines.

    Provides common functionality like data loading and metric calculation.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._data_cache: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def run(
        self,
        spec: RunSpec,
        data: Optional[pd.DataFrame] = None,
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> RunResult:
        """Execute a single backtest run with optional MTF data."""
        pass

    def run_batch(
        self,
        specs: List[RunSpec],
        data: Optional[Dict[str, pd.DataFrame]] = None,
        secondary_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> List[RunResult]:
        """Default batch implementation - override for vectorization."""
        results = []
        for spec in specs:
            symbol_data = data.get(spec.symbol) if data else None
            symbol_secondary = secondary_data.get(spec.symbol) if secondary_data else None
            results.append(self.run(spec, symbol_data, symbol_secondary))
        return results

    @property
    @abstractmethod
    def engine_type(self) -> EngineType:
        """Return the engine type identifier."""
        pass

    @property
    def supports_vectorization(self) -> bool:
        """Override in engines that support vectorization."""
        return False

    def load_data(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a symbol.

        Uses caching if enabled to avoid repeated data loads.
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"

        if self.config.enable_caching and cache_key in self._data_cache:
            return self._data_cache[cache_key]

        df = self._fetch_data(symbol, start_date, end_date)

        if df is not None and self.config.enable_caching:
            self._data_cache[cache_key] = df

        return df

    def _fetch_data(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from configured source.

        Override for custom data sources.
        """
        if self.config.data_source == "ib":
            return self._fetch_ib(symbol, start_date, end_date)
        elif self.config.data_source == "yahoo":
            return self._fetch_yahoo(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {self.config.data_source}")

    def _fetch_ib(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Fetch data from IB using existing adapter infrastructure."""
        try:
            from datetime import datetime

            from ...data.providers import IbBacktestDataProvider

            provider = IbBacktestDataProvider(
                host=self.config.ib_host,
                port=self.config.ib_port,
                client_id=self.config.ib_client_id,
                rate_limit=self.config.ib_rate_limit,
            )

            # Convert date to datetime for IB adapter
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())

            # Use sync wrapper - handles connect/fetch/disconnect
            df = provider.fetch_bars_single_sync(
                symbol=symbol,
                start=start_dt,
                end=end_dt,
                timeframe="1d",
            )

            if df.empty:
                return None

            return df

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"IB fetch failed for {symbol}: {e}")
            return None

    def _fetch_yahoo(self, symbol: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance via yfinance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=True,
            )

            if df.empty:
                return None

            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            return df

        except Exception:
            return None

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
