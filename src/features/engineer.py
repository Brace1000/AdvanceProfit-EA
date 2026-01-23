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

        # ANTI-LEAKAGE: Shift all non-calendar features by 1
        # This ensures feature[t] can only predict label[t+1], not label[t]
        calendar_features = ["hour", "session_asian", "session_european", "session_american",
                           "dow_monday", "dow_tuesday", "dow_wednesday", "dow_thursday", "dow_friday"]

        feature_cols = self.default_feature_columns()
        ts_features = [f for f in feature_cols if f not in calendar_features and f in df.columns]

        if ts_features:
            logger.debug(f"Applying shift(1) to {len(ts_features)} time-series features")
            df[ts_features] = df[ts_features].shift(1)

        # ADDITIONAL LAG for inherently lagging indicators
        # These features use smoothed/averaged values, so they're backward-looking
        # Extra shift helps model learn them as they appear in real-time (with delay)
        lagging_indicators = [
            "ema50_ema200_h1", "ema50_ema200_h4",  # EMA crossovers
            "rsi_slope_h1", "rsi_slope_h4",         # RSI slopes
            "z_close_h1", "z_close_h4"              # Z-scores (100-bar lookback)
        ]
        lagging_present = [f for f in lagging_indicators if f in df.columns]

        if lagging_present:
            logger.debug(f"Applying additional shift(1) to {len(lagging_present)} lagging indicators")
            df[lagging_present] = df[lagging_present].shift(1)

        # Reset index
        df = df.reset_index()

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

        # P1 regime detection features
        df = self._squeeze_ratio(df, suffix)
        df = self._choppiness(df, suffix)
        df = self._z_score_price(df, suffix)

        return df

    def _add_h4_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add H4 timeframe features by resampling H1 data.

        CRITICAL: Uses shift(1) to prevent look-ahead bias. At time T (H1),
        we only see H4 features from the *completed* H4 bar, not the bar
        currently in progress.
        """
        # Resample H1 to H4 (default label='left' means bar labeled 12:00 covers 12:00-16:00)
        h4 = df[["open", "high", "low", "close"]].resample("4h", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna()

        if len(h4) < 200:
            logger.warning(f"Only {len(h4)} H4 bars after resampling. Features may be incomplete.")

        # Calculate features on H4 data
        h4 = self._calculate_features(h4, suffix="_h4")

        # ANTI-LEAKAGE: Shift H4 features by 1 period
        # This ensures at 13:00 H1, we see the H4 bar from 08:00 (08:00-12:00, completed)
        # NOT the bar from 12:00 (12:00-16:00, still in progress)
        h4 = h4.shift(1)

        # Forward-fill H4 features to H1 timeframe
        h4_features = [c for c in h4.columns if c.endswith("_h4")]

        for col in h4_features:
            # Reindex H4 to H1 index and forward-fill
            df[col] = h4[col].reindex(df.index, method="ffill")

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features with One-Hot Encoding for categorical variables.

        Calendar features provide market structure context:
        - Sessions capture real volatility patterns (London > Asian)
        - Days capture week patterns (Friday closings, Monday gaps)
        Combined with P1 regime features for best results.
        """
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

    def _squeeze_ratio(self, df: pd.DataFrame, suffix: str = "", bb_period: int = 20, kc_period: int = 20) -> pd.DataFrame:
        """
        Calculate Squeeze Ratio (Bollinger Band width / Keltner Channel width).

        Low values (<1.0) indicate "squeeze" - market compression before breakout.
        High values (>1.0) indicate expansion.

        ANTI-LEAKAGE: Uses only backward-looking rolling windows.
        """
        # Bollinger Bands width (2 std devs)
        bb_mid = df["close"].rolling(window=bb_period, min_periods=bb_period).mean()
        bb_std = df["close"].rolling(window=bb_period, min_periods=bb_period).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = bb_upper - bb_lower

        # Keltner Channel width (ATR-based)
        if f"atr{suffix}" not in df.columns:
            logger.warning(f"ATR not found for squeeze ratio{suffix}, skipping")
            return df

        kc_atr = df[f"atr{suffix}"].rolling(window=kc_period, min_periods=kc_period).mean()
        kc_width = kc_atr * 2  # Standard KC multiplier

        # Squeeze ratio
        df[f"squeeze_ratio{suffix}"] = bb_width / (kc_width + 1e-10)

        return df

    def _choppiness(self, df: pd.DataFrame, suffix: str = "", period: int = 14) -> pd.DataFrame:
        """
        Calculate Choppiness Index.

        Values near 100 = choppy/ranging market
        Values near 0 = trending market

        ANTI-LEAKAGE: Uses only backward-looking rolling sums and max/min.
        """
        # True range (already calculated for ATR)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        # Sum of true range over period
        tr_sum = tr.rolling(window=period, min_periods=period).sum()

        # Price range (high - low) over period
        high_max = df["high"].rolling(window=period, min_periods=period).max()
        low_min = df["low"].rolling(window=period, min_periods=period).min()
        price_range = high_max - low_min

        # Choppiness index formula
        chop = 100 * np.log10(tr_sum / (price_range + 1e-10)) / np.log10(period)

        df[f"choppiness{suffix}"] = chop.clip(0, 100)  # Bound to [0, 100]

        return df

    def _z_score_price(self, df: pd.DataFrame, suffix: str = "", lookback: int = 100) -> pd.DataFrame:
        """
        Calculate causal Z-score for close price.

        Z-score = (price - rolling_mean) / rolling_std

        ANTI-LEAKAGE: Uses only backward-looking rolling statistics.
        Each bar only knows the mean/std of the PAST lookback bars.
        """
        rolling_mean = df["close"].rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = df["close"].rolling(window=lookback, min_periods=lookback).std()

        z_score = (df["close"] - rolling_mean) / (rolling_std + 1e-10)

        # Clip extreme outliers for robustness
        df[f"z_close{suffix}"] = z_score.clip(-5, 5)

        return df

    @staticmethod
    def default_feature_columns() -> list[str]:
        """
        Return list of feature column names with OHE and P1 regime features.

        Best tested configuration combines:
        - Calendar features (sessions, days) for market structure context
        - P1 regime features (squeeze, choppiness, z-score) for price action
        - Technical indicators (EMAs, RSI, ATR) for momentum/volatility

        Includes:
        - H1/H4 technical indicators
        - Time context (OHE for sessions and days)
        - P1 regime features (squeeze, choppiness, z-score)
        """
        return [
            # H1 features (11): technical + regime
            "close_ema50_h1",
            "ema50_ema200_h1",
            "rsi_h1",
            "rsi_slope_h1",
            "atr_ratio_h1",
            "adx_h1",
            "body_h1",
            "range_h1",
            "squeeze_ratio_h1",     # P1: Volatility compression
            "choppiness_h1",        # P1: Trend vs range
            "z_close_h1",           # P1: Price level normalization
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
            # H4 features (11): technical + regime
            "close_ema50_h4",
            "ema50_ema200_h4",
            "rsi_h4",
            "rsi_slope_h4",
            "atr_ratio_h4",
            "adx_h4",
            "body_h4",
            "range_h4",
            "squeeze_ratio_h4",     # P1: Volatility compression
            "choppiness_h4",        # P1: Trend vs range
            "z_close_h4",           # P1: Price level normalization
            # H4 return (1)
            "prev_return_h4",
        ]
