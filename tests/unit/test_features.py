"""
Unit tests for feature engineering functions.

These tests verify that individual feature calculation functions
work correctly with various inputs.
"""

import pytest
import pandas as pd
import numpy as np


class TestFeatureEngineering:
    """Test suite for feature engineering functions."""

    def test_ema_calculation(self, sample_ohlcv_data):
        """Test EMA calculation."""
        df = sample_ohlcv_data.copy()

        # Calculate EMA 50
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        assert 'ema50' in df.columns
        assert not df['ema50'].isna().all()
        assert len(df['ema50']) == len(df)

    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI calculation."""
        df = sample_ohlcv_data.copy()

        # Calculate RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        assert 'rsi' in df.columns
        # RSI should be between 0 and 100
        assert df['rsi'].dropna().min() >= 0
        assert df['rsi'].dropna().max() <= 100

    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR calculation."""
        df = sample_ohlcv_data.copy()

        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df['atr'] = tr.rolling(14).mean()

        assert 'atr' in df.columns
        # ATR should be positive
        assert (df['atr'].dropna() >= 0).all()

    def test_label_creation(self, sample_ohlcv_data):
        """Test label creation with thresholds."""
        df = sample_ohlcv_data.copy()

        future_return = df['close'].shift(-1) / df['close'] - 1
        buy_threshold = 0.002
        sell_threshold = -0.002

        df['label'] = 0
        df.loc[future_return > buy_threshold, 'label'] = 1
        df.loc[future_return < sell_threshold, 'label'] = -1

        assert 'label' in df.columns
        # Labels should be -1, 0, or 1
        unique_labels = df['label'].unique()
        assert all(label in [-1, 0, 1] for label in unique_labels)


class TestFeatureValidation:
    """Test feature validation and error handling."""

    def test_handle_missing_data(self, sample_ohlcv_data):
        """Test handling of missing data in features."""
        df = sample_ohlcv_data.copy()

        # Introduce some NaN values
        df.loc[10:15, 'close'] = np.nan

        # Calculate feature
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        # Should handle NaN appropriately
        assert df['ema50'].isna().sum() >= 5

    def test_feature_ranges(self, sample_features):
        """Test that features are within expected ranges."""
        df = sample_features

        # RSI should be 0-100
        assert (df['rsi'] >= 0).all() and (df['rsi'] <= 100).all()

        # Hour should be 0-23
        assert (df['hour'] >= 0).all() and (df['hour'] <= 23).all()

        # Session should be 0-2
        assert (df['session'] >= 0).all() and (df['session'] <= 2).all()
