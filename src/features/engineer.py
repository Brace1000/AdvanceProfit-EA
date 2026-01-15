"""Feature engineering for trading models."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger("trading_bot.features")


class FeatureEngineer:
    """
    Calculates technical indicator features.

    Transforms raw OHLC data into features:
    - EMA crossovers (50, 200)
    - RSI and RSI slope
    - ATR ratio
    - ADX-like measure
    - Candlestick patterns (body, range)
    - Previous returns
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.debug("FeatureEngineer initialized")

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Engineering features for {len(df)} rows")
        required_cols = ["open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df = self._ema(df)
        df = self._rsi(df)
        df = self._atr(df)
        df = self._adx_like(df)
        df = self._candle(df)
        df = self._returns(df)
        logger.info(f"Created {len(df.columns)} columns after feature engineering")
        return df

    def _ema(self, df: pd.DataFrame) -> pd.DataFrame:
        df["close_ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["close_ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["ema50_ema200"] = df["close_ema50"] - df["close_ema200"]
        return df

    def _rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_slope"] = df["rsi"].diff()
        return df

    def _atr(self, df: pd.DataFrame) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df["atr"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)
        return df

    def _adx_like(self, df: pd.DataFrame) -> pd.DataFrame:
        df["plus_dm"] = df["high"].diff().clip(lower=0)
        df["minus_dm"] = (-df["low"].diff()).clip(lower=0)
        df["adx"] = (df["plus_dm"] - df["minus_dm"]).abs().rolling(14).mean()
        return df

    def _candle(self, df: pd.DataFrame) -> pd.DataFrame:
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        # For daily data we won't have hour/session; keep placeholders for model compatibility
        df["hour"] = 0
        df["session"] = 0
        return df

    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prev_return"] = df["close"].pct_change()
        return df

    @staticmethod
    def default_feature_columns() -> list[str]:
        return [
            "close_ema50",
            "ema50_ema200",
            "rsi",
            "rsi_slope",
            "atr_ratio",
            "adx",
            "body",
            "range",
            "hour",
            "session",
            "prev_return",
        ]
