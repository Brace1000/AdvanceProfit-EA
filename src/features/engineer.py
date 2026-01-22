"""Feature engineering for trading models with multi-timeframe support."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger("trading_bot.features")


class FeatureEngineer:
    """
    Calculates technical indicator features for ML model.

    Supports multi-timeframe features (H1 + H4) to match what the EA sends.
    Features are calculated on both H1 data and resampled H4 data.

    20 Features total:
    - H1: close_ema50, ema50_ema200, rsi, rsi_slope, atr_ratio, adx, body, range
    - hour, session (time-based)
    - prev_return_h1
    - H4: close_ema50, ema50_ema200, rsi, rsi_slope, atr_ratio, adx, body, range
    - prev_return_h4
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.debug("FeatureEngineer initialized")

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicator features including multi-timeframe.

        Args:
            df: DataFrame with H1 OHLC data (columns: time/date, open, high, low, close)

        Returns:
            DataFrame with added feature columns (20 features total)
        """
        logger.info(f"Engineering features for {len(df)} rows")

        required_cols = ["open", "high", "low", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()

        # Ensure we have a datetime index for resampling
        # Check various possible datetime column names
        datetime_col = None
        for col in ["time", "date", "datetime", "timestamp", "price"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    # Check if conversion was successful (not all NaT)
                    if df[col].notna().sum() > 0:
                        # Drop rows where datetime parsing failed
                        df = df.dropna(subset=[col])
                        datetime_col = col
                        logger.debug(f"Using '{col}' as datetime column")
                        break
                except Exception as e:
                    logger.debug(f"Could not parse '{col}' as datetime: {e}")
                    continue

        if datetime_col:
            df = df.set_index(datetime_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"Available columns: {df.columns.tolist()}")
            logger.error(f"First row sample: {df.iloc[0].to_dict() if len(df) > 0 else 'empty'}")
            raise ValueError(
                "No datetime column found. Expected one of: time, date, datetime, timestamp, price"
            )

        # Calculate H1 features
        df = self._calculate_features(df, suffix="_h1")

        # Calculate H4 features by resampling
        df = self._add_h4_features(df)

        # Add time-based features
        df = self._add_time_features(df)

        # Reset index
        df = df.reset_index()

        feature_cols = self.default_feature_columns()
        logger.info(f"Created {len(feature_cols)} features")

        return df

    def _calculate_features(self, df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        """Calculate all technical features with optional suffix."""
        df = self._ema(df, suffix)
        df = self._rsi(df, suffix)
        df = self._atr(df, suffix)
        df = self._adx_like(df, suffix)
        df = self._candle(df, suffix)
        df = self._returns(df, suffix)
        return df

    def _add_h4_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add H4 timeframe features by resampling H1 data."""
        # Resample H1 to H4
        h4 = df[["open", "high", "low", "close"]].resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna()

        if len(h4) < 200:
            logger.warning(f"Only {len(h4)} H4 bars after resampling. Features may be incomplete.")

        # Calculate features on H4 data
        h4 = self._calculate_features(h4, suffix="_h4")

        # Forward-fill H4 features to H1 timeframe
        h4_features = [c for c in h4.columns if c.endswith("_h4")]

        for col in h4_features:
            # Reindex H4 to H1 index and forward-fill
            df[col] = h4[col].reindex(df.index, method="ffill")

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features with One-Hot Encoding for categorical variables."""
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour

            # Session mapping: Asian: 0-8 UTC, European: 8-16 UTC, American: 16-24 UTC
            session_raw = pd.cut(
                df.index.hour,
                bins=[-1, 8, 16, 24],
                labels=["asian", "european", "american"]
            )

            # One-hot encode sessions (no drop_first for clear feature importance)
            df["session_asian"] = (session_raw == "asian").astype(int)
            df["session_european"] = (session_raw == "european").astype(int)
            df["session_american"] = (session_raw == "american").astype(int)

            # Day of week (0=Monday, 4=Friday)
            day_raw = df.index.dayofweek

            # One-hot encode day of week (Monday-Friday only, trading days)
            df["dow_monday"] = (day_raw == 0).astype(int)
            df["dow_tuesday"] = (day_raw == 1).astype(int)
            df["dow_wednesday"] = (day_raw == 2).astype(int)
            df["dow_thursday"] = (day_raw == 3).astype(int)
            df["dow_friday"] = (day_raw == 4).astype(int)
        else:
            df["hour"] = 0
            # Default to all zeros if no datetime index
            for session in ["asian", "european", "american"]:
                df[f"session_{session}"] = 0
            for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
                df[f"dow_{day}"] = 0

        return df

    def _ema(self, df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        """Calculate EMA-based features."""
        df[f"close_ema50{suffix}"] = df["close"].ewm(span=50, adjust=False).mean()
        df[f"close_ema200{suffix}"] = df["close"].ewm(span=200, adjust=False).mean()
        df[f"ema50_ema200{suffix}"] = df[f"close_ema50{suffix}"] - df[f"close_ema200{suffix}"]
        return df

    def _rsi(self, df: pd.DataFrame, suffix: str = "", period: int = 14) -> pd.DataFrame:
        """Calculate RSI and RSI slope."""
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        roll_up = up.rolling(period).mean()
        roll_down = down.rolling(period).mean()

        rs = roll_up / (roll_down + 1e-10)
        df[f"rsi{suffix}"] = 100 - (100 / (1 + rs))
        df[f"rsi_slope{suffix}"] = df[f"rsi{suffix}"].diff()
        return df

    def _atr(self, df: pd.DataFrame, suffix: str = "", period: int = 14) -> pd.DataFrame:
        """Calculate ATR ratio."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        tr = high_low.combine(high_close, max).combine(low_close, max)
        df[f"atr{suffix}"] = tr.rolling(period).mean()
        df[f"atr_ratio{suffix}"] = df[f"atr{suffix}"] / (df["close"] + 1e-10)
        return df

    def _adx_like(self, df: pd.DataFrame, suffix: str = "", period: int = 14) -> pd.DataFrame:
        """Calculate simplified ADX-like indicator."""
        df[f"plus_dm{suffix}"] = df["high"].diff().clip(lower=0)
        df[f"minus_dm{suffix}"] = (-df["low"].diff()).clip(lower=0)
        df[f"adx{suffix}"] = (df[f"plus_dm{suffix}"] - df[f"minus_dm{suffix}"]).abs().rolling(period).mean()
        return df

    def _candle(self, df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        """Calculate candlestick features."""
        df[f"body{suffix}"] = df["close"] - df["open"]
        df[f"range{suffix}"] = df["high"] - df["low"]
        return df

    def _returns(self, df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        """Calculate previous period return."""
        df[f"prev_return{suffix}"] = df["close"].pct_change()
        return df

    @staticmethod
    def default_feature_columns() -> list[str]:
        """
        Return list of 27 feature column names with OHE for categorical variables.

        Note: 'session' and 'day_of_week' are now one-hot encoded.
        """
        return [
            # H1 features (8)
            "close_ema50_h1",
            "ema50_ema200_h1",
            "rsi_h1",
            "rsi_slope_h1",
            "atr_ratio_h1",
            "adx_h1",
            "body_h1",
            "range_h1",
            # Time features (9): hour + 3 sessions + 5 days
            "hour",
            "session_asian",
            "session_european",
            "session_american",
            "dow_monday",
            "dow_tuesday",
            "dow_wednesday",
            "dow_thursday",
            "dow_friday",
            # H1 return (1)
            "prev_return_h1",
            # H4 features (8)
            "close_ema50_h4",
            "ema50_ema200_h4",
            "rsi_h4",
            "rsi_slope_h4",
            "atr_ratio_h4",
            "adx_h4",
            "body_h4",
            "range_h4",
            # H4 return (1)
            "prev_return_h4",
        ]
