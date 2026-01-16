"""Feature engineering for trading models."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger("trading_bot.features")


class FeatureEngineer:
    """
    Calculates technical indicator features for ML model.

    Features:
    - EMA crossovers (50, 200 periods)
    - RSI and RSI momentum (slope)
    - ATR ratio (volatility relative to price)
    - ADX-like directional movement
    - Candlestick patterns (body size, range)
    - Previous returns (momentum)

    Note: Removed placeholder features (hour, session) that added no value
    for daily data. Add them back only if using intraday data with proper values.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.debug("FeatureEngineer initialized")

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicator features.

        Args:
            df: DataFrame with OHLC data (columns: open, high, low, close)

        Returns:
            DataFrame with added feature columns

        Raises:
            ValueError: If required columns are missing
        """
        logger.info(f"Engineering features for {len(df)} rows")

        required_cols = ["open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()

        # Calculate features in order
        df = self._ema(df)
        df = self._rsi(df)
        df = self._atr(df)
        df = self._adx_like(df)
        df = self._candle(df)
        df = self._returns(df)

        feature_cols = self.default_feature_columns()
        logger.info(f"Created {len(feature_cols)} features: {feature_cols}")

        return df

    def _ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA-based features."""
        df["close_ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["close_ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        # EMA crossover signal (positive = bullish, negative = bearish)
        df["ema50_ema200"] = df["close_ema50"] - df["close_ema200"]

        return df

    def _rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate RSI and RSI slope.

        RSI: Measures momentum (0-100 scale)
        RSI slope: Rate of change of RSI (momentum of momentum)
        """
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        roll_up = up.rolling(period).mean()
        roll_down = down.rolling(period).mean()

        rs = roll_up / (roll_down + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI slope: momentum of momentum
        df["rsi_slope"] = df["rsi"].diff()

        return df

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range ratio.

        ATR ratio: ATR as percentage of price (normalized volatility measure)
        """
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        # True Range is max of the three
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df["atr"] = tr.rolling(period).mean()

        # Ratio to price for normalization (scale-independent)
        df["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)

        return df

    def _adx_like(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate simplified ADX-like directional movement indicator.

        Measures the strength of the trend regardless of direction.
        """
        # Directional movement
        df["plus_dm"] = df["high"].diff().clip(lower=0)
        df["minus_dm"] = (-df["low"].diff()).clip(lower=0)

        # Simplified ADX: smoothed absolute difference of directional movements
        df["adx"] = (df["plus_dm"] - df["minus_dm"]).abs().rolling(period).mean()

        return df

    def _candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candlestick pattern features.

        Body: Close - Open (positive = bullish, negative = bearish)
        Range: High - Low (volatility within the bar)
        """
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]

        return df

    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate previous period return.

        Captures recent momentum.
        """
        df["prev_return"] = df["close"].pct_change()

        return df

    @staticmethod
    def default_feature_columns() -> list[str]:
        """
        Return list of feature column names used by the model.

        Note: Removed 'hour' and 'session' as they were placeholder values (always 0)
        that added no predictive value. Re-add them only if using intraday data
        with actual hour/session values.
        """
        return [
            "close_ema50",
            "ema50_ema200",
            "rsi",
            "rsi_slope",
            "atr_ratio",
            "adx",
            "body",
            "range",
            "prev_return",
        ]
