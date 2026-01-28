"""
Regime Service - Application Layer Orchestration.

Orchestrates the 3-level hierarchical regime detection:
1. Fetches regime for market benchmarks (QQQ, SPY)
2. Resolves market disagreement
3. Fetches sector regime
4. Fetches stock regime
5. Synthesizes action with context

Usage:
    service = RegimeService(indicator_engine)
    result = await service.get_hierarchical_regime("NVDA", account_type=AccountType.SHORT_PUT)
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.domain.services.regime import (
    AccountType,
    HierarchicalRegime,
    apply_weekly_veto,
    get_4h_alerts,
    get_hierarchy_level,
    get_sector_for_symbol,
    resolve_action,
    resolve_market_action,
)
from src.domain.signals.indicators.regime import (
    MarketRegime,
    RegimeDetectorIndicator,
    RegimeOutput,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class RegimeService:
    """
    Application service for hierarchical regime detection.

    Coordinates regime calculation across market, sector, and stock levels
    to produce actionable trading decisions.
    """

    def __init__(
        self,
        indicator_engine: Optional[Any] = None,
        market_data_store: Optional[Any] = None,
    ) -> None:
        """
        Initialize regime service.

        Args:
            indicator_engine: Optional IndicatorEngine for calculation
            market_data_store: Optional market data store for bar history
        """
        self._indicator_engine = indicator_engine
        self._market_data_store = market_data_store
        self._regime_detector = RegimeDetectorIndicator()

        # Cache for regime calculations (avoid recalculating same symbol)
        # Thread-safe: protected by _cache_lock
        self._regime_cache: Dict[str, RegimeOutput] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_lock = threading.Lock()

        # Weekly veto tracking per symbol
        # Thread-safe: protected by _veto_lock
        self._weekly_veto_states: Dict[str, Dict[str, Any]] = {}
        self._veto_lock = threading.Lock()

    def calculate_regime(
        self,
        symbol: str,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        is_market_level: bool = False,
        iv_data: Optional[pd.Series] = None,
        timeframe: str = "1d",
    ) -> RegimeOutput:
        """
        Calculate regime for a single symbol.

        Args:
            symbol: Symbol to analyze
            data: OHLC DataFrame with at least 252 bars
            params: Optional parameter overrides
            is_market_level: Whether this is a market benchmark
            iv_data: Optional VIX/VXN data for IV state (market level only)
            timeframe: Bar interval (e.g., "1d", "1h", "5m")

        Returns:
            RegimeOutput with classification and details
        """
        # Use minimum_bars threshold to allow newer tickers (6+ months)
        minimum_bars = self._regime_detector.minimum_bars
        if len(data) < minimum_bars:
            logger.warning(
                f"Insufficient data for {symbol}: {len(data)} bars < {minimum_bars} minimum required"
            )
            return RegimeOutput(
                symbol=symbol,
                final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
                regime_name="Choppy/Extended",
                confidence=0,
            )

        # Calculate regime with error handling
        params = params or {}
        try:
            result_df = self._regime_detector.calculate(data, params)
        except Exception as e:
            logger.error(f"Regime calculation failed for {symbol}: {e}", exc_info=True)
            return RegimeOutput(
                symbol=symbol,
                final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
                regime_name="Choppy/Extended",
                confidence=0,
            )

        # Get state for last bar
        current = result_df.iloc[-1]
        previous = result_df.iloc[-2] if len(result_df) >= 2 else None

        try:
            state = self._regime_detector.get_state(current, previous, params)
        except Exception as e:
            logger.error(f"Get state failed for {symbol}: {e}", exc_info=True)
            return RegimeOutput(
                symbol=symbol,
                final_regime=MarketRegime.R1_CHOPPY_EXTENDED,
                regime_name="Choppy/Extended",
                confidence=0,
            )

        # Handle IV state for market level
        if is_market_level and iv_data is not None:
            state = self._add_iv_state(state, iv_data, params)
            state["is_market_level"] = True

        # Use hysteresis for stable transitions
        output = self._regime_detector.update_with_hysteresis(
            symbol=symbol,
            state=state,
            timestamp=data.index[-1] if hasattr(data.index[-1], "isoformat") else datetime.now(),
            timeframe=timeframe,
        )

        # Cache the result (thread-safe)
        with self._cache_lock:
            self._regime_cache[symbol] = output
            self._cache_timestamp = datetime.now()

        return output

    def get_hierarchical_regime(
        self,
        symbol: str,
        symbol_data: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        sector_data: Optional[pd.DataFrame] = None,
        account_type: AccountType = AccountType.SHORT_PUT,
        iv_data: Optional[Dict[str, pd.Series]] = None,
        weekly_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> HierarchicalRegime:
        """
        Get complete hierarchical regime for a symbol.

        Args:
            symbol: Stock symbol to analyze
            symbol_data: OHLC data for the symbol
            market_data: Dict of market benchmark OHLC data {"QQQ": df, "SPY": df}
            sector_data: Optional OHLC data for sector ETF
            account_type: Account type for action resolution
            iv_data: Optional IV data {"VIX": series, "VXN": series}
            weekly_data: Optional weekly OHLC for veto logic

        Returns:
            HierarchicalRegime with synthesized action
        """
        timestamp = datetime.now()
        if len(symbol_data) > 0 and hasattr(symbol_data.index[-1], "isoformat"):
            timestamp = symbol_data.index[-1]

        # Determine hierarchy level
        level = get_hierarchy_level(symbol)

        # === Level 1: Market Regime ===
        qqq_regime = None
        spy_regime = None
        market_confidence = 50

        # Calculate QQQ regime
        if "QQQ" in market_data and len(market_data["QQQ"]) >= self._regime_detector.warmup_periods:
            qqq_iv = iv_data.get("VXN") if iv_data else None
            qqq_output = self.calculate_regime(
                "QQQ", market_data["QQQ"], is_market_level=True, iv_data=qqq_iv
            )
            qqq_regime = qqq_output.regime
            market_confidence = qqq_output.confidence

        # Calculate SPY regime
        if "SPY" in market_data and len(market_data["SPY"]) >= self._regime_detector.warmup_periods:
            spy_iv = iv_data.get("VIX") if iv_data else None
            spy_output = self.calculate_regime(
                "SPY", market_data["SPY"], is_market_level=True, iv_data=spy_iv
            )
            spy_regime = spy_output.regime
            if qqq_regime is None:
                market_confidence = spy_output.confidence

        # Resolve market disagreement
        if qqq_regime is not None and spy_regime is not None:
            _, market_regime = resolve_market_action(qqq_regime, spy_regime, account_type)
            market_symbol = "QQQ+SPY"
        elif qqq_regime is not None:
            market_regime = qqq_regime
            market_symbol = "QQQ"
        elif spy_regime is not None:
            market_regime = spy_regime
            market_symbol = "SPY"
        else:
            # No market data - default to R1
            market_regime = MarketRegime.R1_CHOPPY_EXTENDED
            market_symbol = "N/A"
            market_confidence = 0

        # === Level 2: Sector Regime (if applicable) ===
        sector_regime = None
        sector_confidence = None
        sector_symbol = None

        if level == "stock":
            sector_symbol = get_sector_for_symbol(symbol)
            if (
                sector_symbol
                and sector_data is not None
                and len(sector_data) >= self._regime_detector.warmup_periods
            ):
                sector_output = self.calculate_regime(sector_symbol, sector_data)
                sector_regime = sector_output.regime
                sector_confidence = sector_output.confidence
        elif level == "sector":
            # The symbol IS a sector ETF
            sector_symbol = symbol
            sector_output = self.calculate_regime(symbol, symbol_data)
            sector_regime = sector_output.regime
            sector_confidence = sector_output.confidence

        # === Level 3: Stock Regime (if applicable) ===
        stock_regime = None
        stock_confidence = None

        if level == "stock":
            stock_output = self.calculate_regime(symbol, symbol_data)
            stock_regime = stock_output.regime
            stock_confidence = stock_output.confidence
        elif level == "market":
            # For market benchmarks, stock regime = market regime
            stock_regime = market_regime
            stock_confidence = market_confidence

        # === Weekly Veto ===
        weekly_veto_active = False
        if weekly_data and market_symbol in weekly_data:
            weekly_df = weekly_data.get(market_symbol.split("+")[0], weekly_data.get("QQQ"))
            if weekly_df is not None and len(weekly_df) >= 52:
                weekly_veto_active, market_regime = self._apply_weekly_veto(
                    symbol, market_regime, weekly_df
                )

        # === Synthesize Action ===
        action, context = resolve_action(market_regime, sector_regime, stock_regime, account_type)

        # === Generate Alerts ===
        alerts = []
        if symbol in self._regime_cache:
            # Get alerts if we have history
            cached = self._regime_cache.get(symbol)
            if cached:
                alerts = get_4h_alerts(
                    cached.component_values.to_dict(),
                    None,  # Would need previous state
                )

        return HierarchicalRegime(
            symbol=symbol,
            timestamp=timestamp,
            market_regime=market_regime,
            market_confidence=market_confidence,
            market_symbol=market_symbol,
            sector_regime=sector_regime,
            sector_confidence=sector_confidence,
            sector_symbol=sector_symbol,
            stock_regime=stock_regime,
            stock_confidence=stock_confidence,
            action=action,
            action_context=context,
            account_type=account_type,
            weekly_veto_active=weekly_veto_active,
            alerts=alerts,
        )

    def get_batch_regimes(
        self,
        symbols: List[str],
        data_by_symbol: Dict[str, pd.DataFrame],
        market_data: Dict[str, pd.DataFrame],
        account_type: AccountType = AccountType.SHORT_PUT,
    ) -> Dict[str, HierarchicalRegime]:
        """
        Calculate hierarchical regimes for multiple symbols.

        Args:
            symbols: List of symbols to analyze
            data_by_symbol: Dict of symbol -> OHLC DataFrame
            market_data: Market benchmark data
            account_type: Account type for all calculations

        Returns:
            Dict of symbol -> HierarchicalRegime
        """
        results = {}

        for symbol in symbols:
            if symbol not in data_by_symbol:
                logger.warning(f"No data for {symbol}, skipping")
                continue

            # Get sector data if available
            sector_symbol = get_sector_for_symbol(symbol)
            sector_data = data_by_symbol.get(sector_symbol) if sector_symbol else None

            results[symbol] = self.get_hierarchical_regime(
                symbol=symbol,
                symbol_data=data_by_symbol[symbol],
                market_data=market_data,
                sector_data=sector_data,
                account_type=account_type,
            )

        return results

    def _add_iv_state(
        self,
        state: Dict[str, Any],
        iv_data: pd.Series,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add IV state to existing state dict."""
        from src.domain.signals.indicators.regime.components import calculate_iv_state

        iv_state, iv_details, is_available = calculate_iv_state(
            iv_data.values if hasattr(iv_data, "values") else iv_data, params
        )

        if is_available and len(iv_state) > 0:
            state["iv_state"] = iv_state[-1]
            if "components" in state:
                state["components"]["iv_value"] = (
                    float(iv_data.iloc[-1]) if len(iv_data) > 0 else None
                )
                state["components"]["iv_pct_63"] = (
                    float(iv_details["iv_pct_63"][-1]) if len(iv_details["iv_pct_63"]) > 0 else None
                )

        return state

    def _apply_weekly_veto(
        self,
        symbol: str,
        daily_regime: MarketRegime,
        weekly_data: pd.DataFrame,
    ) -> Tuple[bool, MarketRegime]:
        """Apply weekly veto logic. Thread-safe."""
        # Calculate weekly trend and vol state (outside lock)
        weekly_result = self._regime_detector.calculate(weekly_data, {})

        with self._veto_lock:
            # Get or initialize veto state
            if symbol not in self._weekly_veto_states:
                self._weekly_veto_states[symbol] = {
                    "veto_active": False,
                    "bars_since_veto": 0,
                }

            veto_state = self._weekly_veto_states[symbol]

            if len(weekly_result) == 0:
                return veto_state["veto_active"], daily_regime

            weekly_trend = weekly_result.iloc[-1].get("trend_state", "neutral")
            weekly_vol = weekly_result.iloc[-1].get("vol_state", "vol_normal")

            # Apply veto
            effective_regime, veto_active = apply_weekly_veto(
                daily_regime=daily_regime,
                weekly_trend_state=weekly_trend,
                weekly_vol_state=weekly_vol,
                veto_active=veto_state["veto_active"],
                bars_since_veto=veto_state["bars_since_veto"],
            )

            # Update veto state
            if veto_active and not veto_state["veto_active"]:
                veto_state["bars_since_veto"] = 0
            elif veto_active:
                veto_state["bars_since_veto"] += 1
            veto_state["veto_active"] = veto_active

            return veto_active, effective_regime

    def clear_cache(self) -> None:
        """Clear the regime cache. Thread-safe."""
        with self._cache_lock:
            self._regime_cache.clear()
            self._cache_timestamp = None

    def get_cached_regime(self, symbol: str) -> Optional[RegimeOutput]:
        """Get cached regime for a symbol. Thread-safe."""
        with self._cache_lock:
            return self._regime_cache.get(symbol)

    def reset_hysteresis(self, symbol: Optional[str] = None) -> None:
        """Reset hysteresis state for symbol(s). Thread-safe."""
        self._regime_detector.reset_state(symbol)
        with self._veto_lock:
            if symbol:
                self._weekly_veto_states.pop(symbol, None)
            else:
                self._weekly_veto_states.clear()
