"""
Turning Point Integration for Regime Detector.

Handles:
- Model loading and caching
- Auto-training on demand
- Feature extraction from regime state
- Prediction generation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


class TurningPointIntegrator:
    """
    Manages turning point model loading, training, and prediction.

    Caches models per-symbol to avoid repeated loading.
    Auto-trains models on demand when not found.
    """

    def __init__(self) -> None:
        """Initialize with empty caches."""
        self._models: Dict[str, Any] = {}
        self._load_attempted: Dict[str, bool] = {}

    def get_prediction(
        self,
        symbol: str,
        flat_state: Dict[str, Any],
        auto_train: bool = True,
    ) -> Optional[Any]:
        """
        Get turning point prediction for the current bar.

        Loads model on demand from models/turning_point/{symbol.lower()}_logistic.pkl.
        If model not found and auto_train=True, trains a new model automatically.

        Args:
            symbol: Symbol to predict for
            flat_state: Flattened state dict with component values
            auto_train: Whether to auto-train if model not found

        Returns:
            TurningPointOutput or None if model unavailable
        """
        # Lazy import to avoid circular dependency
        from .turning_point.features import TurningPointFeatures
        from .turning_point.model import TurningPointModel

        symbol_key = symbol.upper()

        # Try to load model if not attempted yet
        if symbol_key not in self._load_attempted:
            self._load_attempted[symbol_key] = True
            model = self._try_load_model(symbol, TurningPointModel)
            if model:
                self._models[symbol_key] = model
            elif auto_train:
                trained_model = self._train_model(symbol)
                if trained_model:
                    self._models[symbol_key] = trained_model

        # Get model if available
        model = self._models.get(symbol_key)
        if model is None:
            return None

        # Build features and predict
        try:
            features = self._extract_features(flat_state, TurningPointFeatures)
            return model.predict(features)
        except Exception as e:
            logger.warning(f"Turning point prediction failed for {symbol}: {e}")
            return None

    def _try_load_model(self, symbol: str, model_class: Any) -> Optional[Any]:
        """Try to load an existing model from disk."""
        # Check both new format (symbol/active.pkl) and legacy format (symbol_logistic.pkl)
        new_model_path = PROJECT_ROOT / "models/turning_point" / symbol.lower() / "active.pkl"
        legacy_model_path = PROJECT_ROOT / "models/turning_point" / f"{symbol.lower()}_logistic.pkl"
        model_path = new_model_path if new_model_path.exists() else legacy_model_path

        if model_path.exists():
            try:
                model = model_class.load(model_path)
                logger.info(f"Loaded turning point model for {symbol} from {model_path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load turning point model for {symbol}: {e}")
        else:
            logger.debug(f"No turning point model found at {model_path}")

        return None

    def _train_model(self, symbol: str, days: int = 750) -> Optional[Any]:
        """
        Train a turning point model for a symbol on-demand.

        Args:
            symbol: Symbol to train model for
            days: Days of historical data to use

        Returns:
            Trained TurningPointModel or None if training fails
        """
        from .turning_point import TurningPointLabeler, TurningPointModel
        from .turning_point.features import extract_features

        logger.info(f"Auto-training turning point model for {symbol}...")

        try:
            # Fetch historical data
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days + 50}d", interval="1d")
            df.columns = df.columns.str.lower()

            if len(df) < 300:
                logger.warning(f"Insufficient data for {symbol}: only {len(df)} bars")
                return None

            # Generate labels
            labeler = TurningPointLabeler(
                atr_period=14,
                zigzag_threshold=2.0,
                risk_horizon=10,
                risk_threshold=1.5,
            )
            y_top, y_bottom, _ = labeler.generate_combined_labels(df)

            # Extract features
            features_df = extract_features(df)

            # Align data
            valid_mask = ~features_df.isna().any(axis=1)
            valid_idx = features_df.index[valid_mask][:-10]  # Exclude last horizon bars

            X = features_df.loc[valid_idx].values
            y_top_arr = y_top.loc[valid_idx].values
            y_bottom_arr = y_bottom.loc[valid_idx].values

            if len(X) < 200:
                logger.warning(f"Insufficient training samples for {symbol}: {len(X)}")
                return None

            # Train model
            model = TurningPointModel(model_type="logistic", confidence_threshold=0.7)
            model.train(
                X=X,
                y_top=y_top_arr,
                y_bottom=y_bottom_arr,
                cv_splits=5,
                label_horizon=10,
                embargo=2,
            )

            # Save model
            model_dir = PROJECT_ROOT / "models/turning_point"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{symbol.lower()}_logistic.pkl"
            model.save(model_path)

            logger.info(f"Auto-trained and saved turning point model for {symbol} to {model_path}")
            return model

        except Exception as e:
            logger.warning(f"Auto-training failed for {symbol}: {e}")
            return None

    def _extract_features(self, flat_state: Dict[str, Any], features_class: Any) -> Any:
        """Extract TurningPointFeatures from flat state dict."""
        # Map vol_state to vol_regime integer
        vol_state_str = flat_state.get("vol_state", "vol_normal")
        if vol_state_str == "vol_high":
            vol_regime = 1
        elif vol_state_str == "vol_low":
            vol_regime = -1
        else:
            vol_regime = 0

        close = flat_state.get("close", 0.0)
        ma20 = flat_state.get("ma20", close)
        ma50 = flat_state.get("ma50", close)
        ma200 = flat_state.get("ma200", close)
        atr = flat_state.get("atr20", 1.0) or 1.0  # Avoid division by zero

        return features_class(
            # Trend features (normalized by ATR where applicable)
            price_vs_ma20=(close - ma20) / atr if atr > 0 else 0.0,
            price_vs_ma50=(close - ma50) / atr if atr > 0 else 0.0,
            price_vs_ma200=(close - ma200) / atr if atr > 0 else 0.0,
            ma20_slope=flat_state.get("ma50_slope", 0.0),  # Using ma50_slope as proxy
            ma50_slope=flat_state.get("ma50_slope", 0.0),
            ma20_vs_ma50=(ma20 - ma50) / atr if atr > 0 else 0.0,
            # Volatility features
            atr_pct_63=flat_state.get("atr_pct_63", 50.0),
            atr_pct_252=flat_state.get("atr_pct_252", 50.0),
            atr_expansion_rate=0.0,  # Would need historical ATR
            vol_regime=vol_regime,
            # Chop/Range features
            chop_pct_252=flat_state.get("chop_pct_252", 50.0),
            adx_value=25.0,  # Default - would need ADX indicator
            range_position=0.5,  # Default - would need range calculation
            # Extension features
            ext_atr_units=flat_state.get("ext", 0.0),
            ext_zscore=flat_state.get("ext", 0.0),  # Using ext as proxy
            rsi_14=50.0,  # Default - would need RSI indicator
            # Rate of change features (defaults)
            roc_5=0.0,
            roc_10=0.0,
            roc_20=0.0,
            # Delta features (defaults - would need previous bar)
            delta_atr_pct=0.0,
            delta_chop_pct=0.0,
            delta_ext=0.0,
        )

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset model cache for a symbol or all symbols."""
        if symbol is None:
            self._models.clear()
            self._load_attempted.clear()
        else:
            symbol_key = symbol.upper()
            self._models.pop(symbol_key, None)
            self._load_attempted.pop(symbol_key, None)
