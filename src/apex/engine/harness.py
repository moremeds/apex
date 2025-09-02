"""Engine harness for unified backtesting interface."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Type

import structlog

from apex.core.types import MarketDataFrame
from apex.engine.base import BaseEngine, BacktestResult, EngineConfig, StrategyProtocol
from apex.engine.vectorbt import VectorbtEngine

logger = structlog.get_logger(__name__)


class EngineType(str, Enum):
    """Supported backtesting engines."""
    
    VECTORBT = "vectorbt"
    # Future engines can be added here
    # BACKTRADER = "backtrader"
    # ZIPLINE = "zipline"


class EngineHarness:
    """Unified interface for different backtesting engines."""

    _engines: Dict[EngineType, Type[BaseEngine]] = {
        EngineType.VECTORBT: VectorbtEngine,
    }

    def __init__(
        self, 
        config: EngineConfig, 
        engine_type: EngineType = EngineType.VECTORBT
    ) -> None:
        """Initialize the engine harness."""
        self.config = config
        self.engine_type = engine_type
        
        # Create the specified engine
        engine_class = self._engines.get(engine_type)
        if engine_class is None:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        
        self.engine = engine_class(config)
        
        logger.info(
            "Engine harness initialized",
            engine_type=engine_type.value,
            initial_cash=config.initial_cash,
            commission=config.commission
        )

    async def run_backtest(
        self, 
        strategy: StrategyProtocol, 
        data: MarketDataFrame,
        validate_data: bool = True
    ) -> BacktestResult:
        """Run backtest using the configured engine."""
        
        if validate_data:
            # Validate data quality
            if data.overall_quality_score < 0.7:
                logger.warning(
                    "Data quality below recommended threshold",
                    symbol=data.symbol,
                    quality_score=data.overall_quality_score
                )
                
                # Could implement different enforcement levels here
                if hasattr(self.config, 'enforcement_level'):
                    if getattr(self.config, 'enforcement_level') == 'block':
                        raise ValueError(
                            f"Data quality score {data.overall_quality_score:.3f} "
                            "below threshold 0.7"
                        )

        # Validate strategy compatibility
        if not self.engine.validate_strategy(strategy, data):
            raise ValueError(
                f"Strategy {strategy.__class__.__name__} is not compatible "
                f"with {self.engine_type.value} engine"
            )

        # Run the backtest
        result = await self.engine.run_backtest(strategy, data)
        
        logger.info(
            "Backtest completed via harness",
            engine=self.engine_type.value,
            strategy=strategy.__class__.__name__,
            symbol=data.symbol,
            total_return=f"{result.total_return:.2%}" if result.total_return else "N/A"
        )
        
        return result

    def get_available_engines(self) -> list[EngineType]:
        """Get list of available engine types."""
        return list(self._engines.keys())

    def switch_engine(self, engine_type: EngineType) -> None:
        """Switch to a different engine type."""
        if engine_type not in self._engines:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        
        self.engine_type = engine_type
        engine_class = self._engines[engine_type]
        self.engine = engine_class(self.config)
        
        logger.info("Switched engine", new_engine=engine_type.value)

    @classmethod
    def register_engine(
        cls, 
        engine_type: EngineType, 
        engine_class: Type[BaseEngine]
    ) -> None:
        """Register a new engine type."""
        cls._engines[engine_type] = engine_class
        logger.info("Registered new engine", engine_type=engine_type.value)


# Convenience function for quick backtesting
async def quick_backtest(
    strategy: StrategyProtocol,
    data: MarketDataFrame,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    engine_type: EngineType = EngineType.VECTORBT
) -> BacktestResult:
    """Run a quick backtest with default settings."""
    config = EngineConfig(
        initial_cash=initial_cash,
        commission=commission
    )
    
    harness = EngineHarness(config, engine_type)
    return await harness.run_backtest(strategy, data)