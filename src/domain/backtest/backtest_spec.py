"""
BacktestSpec: Complete backtest configuration.

A single config object that captures the complete backtest definition
for reproducibility. This spec becomes the shared interface between:
- Humans (design docs + reproducible research)
- CI (integration parity tests)
- Production backtests (Backtrader)
- Live trading (same strategy with real execution)

Usage:
    # Load from YAML file
    spec = BacktestSpec.from_yaml("config/backtest/my_strategy.yaml")

    # Create programmatically
    spec = BacktestSpec(
        strategy=StrategySpecConfig(name="ma_cross", params={"short_window": 10}),
        universe={"symbols": ["AAPL", "MSFT"]},
        data=DataSpecConfig(source="csv", start_date=date(2024, 1, 1)),
        execution=ExecutionSpecConfig(initial_capital=100000),
    )

    # Run backtest
    runner = BacktestRunner(spec)
    result = await runner.run()
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StrategySpecConfig:
    """Strategy configuration from spec."""

    name: str  # Strategy name in registry
    id: str = ""  # Unique ID for this run
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.name


@dataclass
class DataSpecConfig:
    """Data source configuration from spec."""

    source: str = "csv"  # historical, csv, parquet
    bar_size: str = "1d"  # 1m, 5m, 15m, 1h, 1d
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # Multi-timeframe config (for MTF strategies)
    secondary_timeframes: List[str] = field(default_factory=list)

    # Historical store config (for source="historical")
    coverage_mode: Optional[str] = None  # off, check, download (default: download)
    historical_dir: Optional[str] = None  # Base dir for Parquet store
    source_priority: Optional[List[str]] = None  # e.g., ["ib", "yahoo"]

    # Source-specific config
    csv_dir: Optional[str] = None
    parquet_dir: Optional[str] = None
    ib_config: Optional[Dict[str, Any]] = None
    streaming: bool = True  # OPT-009: Use streaming data feeds


@dataclass
class ExecutionSpecConfig:
    """Execution configuration from spec."""

    initial_capital: float = 100000.0
    currency: str = "USD"
    allowed_order_types: List[str] = field(default_factory=lambda: ["MARKET", "LIMIT"])

    # Reality model pack (e.g., "ib", "futu_us", "simple", "zero_cost", "conservative")
    reality_pack: Optional[str] = None

    # Latency model (legacy - superseded by reality_pack)
    latency_model: Optional[Dict[str, Any]] = None


@dataclass
class BacktestSpec:
    """
    Complete backtest specification.

    Captures everything needed to reproduce a backtest:
    - Strategy class and parameters
    - Data source and date range
    - Reality model (fees, slippage, fills)
    - Risk controls
    - Reporting configuration
    """

    # Required
    strategy: StrategySpecConfig

    # Universe
    universe: Dict[str, Any] = field(default_factory=dict)

    # Data source
    data: DataSpecConfig = field(default_factory=DataSpecConfig)

    # Execution
    execution: ExecutionSpecConfig = field(default_factory=ExecutionSpecConfig)

    # Reality model (fees, slippage)
    reality_model: Dict[str, Any] = field(default_factory=dict)

    # Risk controls
    risk: Dict[str, Any] = field(default_factory=dict)

    # Scheduler config
    scheduler: Dict[str, Any] = field(default_factory=dict)

    # Reporting config
    reporting: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed fields
    spec_file: Optional[str] = None
    loaded_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_yaml(cls, path: str) -> "BacktestSpec":
        """
        Load spec from YAML file.

        Args:
            path: Path to YAML spec file.

        Returns:
            Loaded BacktestSpec.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Parse strategy
        strat_raw = raw.get("strategy", {})
        strategy = StrategySpecConfig(
            name=strat_raw.get("name", ""),
            id=strat_raw.get("id", strat_raw.get("name", "")),
            params=strat_raw.get("params", {}),
        )

        # Parse data
        data_raw = raw.get("data", {})
        source_priority = data_raw.get("source_priority")
        if isinstance(source_priority, str):
            # Support comma-separated string format
            source_priority = [s.strip() for s in source_priority.split(",") if s.strip()]
        data = DataSpecConfig(
            source=data_raw.get("source", "csv"),
            bar_size=data_raw.get("bar_size", "1d"),
            start_date=cls._parse_date(data_raw.get("start_date")),
            end_date=cls._parse_date(data_raw.get("end_date")),
            coverage_mode=data_raw.get("coverage_mode"),
            historical_dir=data_raw.get("historical_dir"),
            source_priority=source_priority,
            csv_dir=data_raw.get("csv_dir"),
            parquet_dir=data_raw.get("parquet_dir"),
            ib_config=data_raw.get("ib_config"),
            streaming=data_raw.get("streaming", True),
        )

        # Parse execution
        exec_raw = raw.get("execution", {})
        execution = ExecutionSpecConfig(
            initial_capital=exec_raw.get("initial_capital", 100000),
            currency=exec_raw.get("currency", "USD"),
            allowed_order_types=exec_raw.get("allowed_order_types", ["MARKET", "LIMIT"]),
            reality_pack=exec_raw.get("reality_pack"),
            latency_model=exec_raw.get("latency_model"),
        )

        return cls(
            strategy=strategy,
            universe=raw.get("universe", {}),
            data=data,
            execution=execution,
            reality_model=raw.get("reality_model", {}),
            risk=raw.get("risk", {}),
            scheduler=raw.get("scheduler", {}),
            reporting=raw.get("reporting", {}),
            metadata=raw.get("metadata", {}),
            spec_file=path,
        )

    @staticmethod
    def _parse_date(val) -> Optional[date]:
        """Parse date from various formats."""
        if val is None:
            return None
        if isinstance(val, date):
            return val
        if isinstance(val, datetime):
            return val.date()
        return datetime.strptime(str(val), "%Y-%m-%d").date()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": {
                "name": self.strategy.name,
                "id": self.strategy.id,
                "params": self.strategy.params,
            },
            "universe": self.universe,
            "data": {
                "source": self.data.source,
                "bar_size": self.data.bar_size,
                "start_date": str(self.data.start_date) if self.data.start_date else None,
                "end_date": str(self.data.end_date) if self.data.end_date else None,
                "coverage_mode": self.data.coverage_mode,
                "historical_dir": self.data.historical_dir,
                "source_priority": self.data.source_priority,
                "csv_dir": self.data.csv_dir,
                "parquet_dir": self.data.parquet_dir,
                "streaming": self.data.streaming,
            },
            "execution": {
                "initial_capital": self.execution.initial_capital,
                "currency": self.execution.currency,
                "allowed_order_types": self.execution.allowed_order_types,
                "reality_pack": self.execution.reality_pack,
            },
            "reality_model": self.reality_model,
            "risk": self.risk,
            "scheduler": self.scheduler,
            "reporting": self.reporting,
            "metadata": self.metadata,
            "spec_file": self.spec_file,
            "loaded_at": self.loaded_at.isoformat(),
        }

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def save_yaml(self, path: str) -> None:
        """Save spec to YAML file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())
        logger.info(f"Saved backtest spec to {path}")

    def get_symbols(self) -> List[str]:
        """Get list of symbols from universe config."""
        return self.universe.get("symbols", [])

    def get_hash(self) -> str:
        """
        Get hash of spec for reproducibility verification.

        Returns:
            SHA256 hash of spec configuration.
        """
        # Create deterministic representation
        spec_dict = self.to_dict()
        spec_dict.pop("loaded_at", None)  # Don't include timestamp
        spec_dict.pop("spec_file", None)  # Don't include file path

        spec_json = json.dumps(spec_dict, sort_keys=True)
        return hashlib.sha256(spec_json.encode()).hexdigest()[:16]

    def validate(self) -> List[str]:
        """
        Validate spec configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if not self.strategy.name:
            errors.append("Strategy name is required")

        if not self.get_symbols():
            errors.append("At least one symbol required in universe.symbols")

        if self.data.start_date and self.data.end_date:
            if self.data.start_date >= self.data.end_date:
                errors.append("start_date must be before end_date")

        if self.execution.initial_capital <= 0:
            errors.append("initial_capital must be positive")

        return errors

    def __repr__(self) -> str:
        return (
            f"BacktestSpec(strategy={self.strategy.name}, "
            f"symbols={self.get_symbols()}, "
            f"period={self.data.start_date} to {self.data.end_date})"
        )
