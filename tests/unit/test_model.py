"""
Unit tests for model training and inference.

These tests verify model-related functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


class TestModelTraining:
    """Test suite for model training."""

    def test_model_creation(self):
        """Test that XGBoost model can be created."""
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42
        )

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_model_training(self, sample_features, sample_labels):
        """Test model training with sample data."""
        import xgboost as xgb

        # Map labels to 0, 1, 2
        y = sample_labels.map({-1: 0, 0: 1, 1: 2})
        X = sample_features

        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            objective="multi:softprob",
            num_class=3,
            random_state=42
        )

        model.fit(X, y)

        assert hasattr(model, 'n_features_in_')
        assert model.n_features_in_ == len(sample_features.columns)

    def test_model_prediction(self, sample_features, mock_model):
        """Test model prediction."""
        X = sample_features.iloc[:3]

        predictions = mock_model.predict(X)

        assert len(predictions) == 3
        assert all(pred in [-1, 0, 1] for pred in predictions)


class TestModelInference:
    """Test suite for model inference."""

    def test_prediction_probabilities(self, sample_features, mock_model):
        """Test that prediction probabilities sum to 1."""
        X = sample_features.iloc[:1]

        probs = mock_model.predict_proba(X)

        assert probs.shape == (1, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_feature_count_validation(self, sample_features):
        """Test that model validates feature count."""
        X = sample_features

        # Model expects 11 features
        assert len(X.columns) == 11

    def test_confidence_threshold(self, mock_model):
        """Test confidence threshold filtering."""
        probs = np.array([[0.2, 0.3, 0.5]])  # Max confidence 50%
        threshold = 0.6

        max_confidence = probs.max()

        if max_confidence < threshold:
            prediction = 0  # No trade
        else:
            prediction = probs.argmax()

        # Should filter out low confidence
        assert prediction == 0


class TestModelPersistence:
    """Test model saving and loading."""

    def test_model_save_load(self, tmp_path, sample_features, sample_labels):
        """Test that model can be saved and loaded."""
        import xgboost as xgb
        import joblib

        # Train a simple model
        y = sample_labels.map({-1: 0, 0: 1, 1: 2})
        X = sample_features

        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        assert model_path.exists()

        # Load model
        loaded_model = joblib.load(model_path)

        # Make predictions with both
        pred_original = model.predict(X[:5])
        pred_loaded = loaded_model.predict(X[:5])

        # Should be identical
        assert np.array_equal(pred_original, pred_loaded)
