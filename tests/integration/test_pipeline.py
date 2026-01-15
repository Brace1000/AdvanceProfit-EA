"""
Integration tests for the complete training pipeline.

These tests verify that data loading, feature engineering,
and model training work together correctly.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataPipeline:
    """Test the complete data processing pipeline."""

    def test_load_and_clean_data(self, sample_ohlcv_data, test_data_dir):
        """Test loading and cleaning raw data."""
        # Save sample data
        raw_file = test_data_dir / "test_raw.csv"
        sample_ohlcv_data.to_csv(raw_file, index=False)

        # Load data
        df = pd.read_csv(raw_file)
        df.columns = df.columns.str.strip().str.lower()

        # Clean data
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=price_cols, inplace=True)

        assert len(df) > 0
        assert all(col in df.columns for col in price_cols)
        assert df[price_cols].notna().all().all()


class TestFeaturePipeline:
    """Test the feature engineering pipeline."""

    def test_full_feature_engineering(self, sample_ohlcv_data):
        """Test creating all features from raw OHLCV data."""
        df = sample_ohlcv_data.copy()

        # EMA features
        df["close_ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["close_ema200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["ema50_ema200"] = df["close_ema50"] - df["close_ema200"]

        # RSI
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_slope"] = df["rsi"].diff()

        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df["atr"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)

        # All features should be created
        expected_features = [
            "close_ema50", "ema50_ema200", "rsi", "rsi_slope",
            "atr_ratio"
        ]

        for feature in expected_features:
            assert feature in df.columns


class TestModelPipeline:
    """Test the model training pipeline."""

    def test_train_predict_pipeline(self, sample_features, sample_labels):
        """Test complete train and predict pipeline."""
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        # Prepare data
        y = sample_labels.map({-1: 0, 0: 1, 1: 2})
        X = sample_features

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            objective="multi:softprob",
            num_class=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_val)
        probs = model.predict_proba(X_val)

        # Verify predictions
        assert len(predictions) == len(X_val)
        assert probs.shape == (len(X_val), 3)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_model_evaluation_metrics(self, sample_features, sample_labels):
        """Test calculating evaluation metrics."""
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report

        y = sample_labels.map({-1: 0, 0: 1, 1: 2})
        X = sample_features

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)

        # Accuracy should be between 0 and 1
        assert 0 <= accuracy <= 1

        # Should be able to generate classification report
        report = classification_report(y, predictions, output_dict=True, zero_division=0)
        assert 'accuracy' in report
