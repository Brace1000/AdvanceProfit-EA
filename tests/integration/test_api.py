"""
Integration tests for FastAPI prediction endpoint.

These tests verify that the API correctly integrates with the ML model.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def mock_loaded_model():
    """Mock a loaded XGBoost model."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])
    model.n_features_in_ = 11
    return model


@pytest.fixture
def api_client(mock_loaded_model):
    """Create a test client for the FastAPI app."""
    # Import the app
    import sys
    from pathlib import Path

    # Add parent directory to path to import main
    # In real tests, you'd organize imports better
    with patch('main.model', mock_loaded_model):
        from main import app
        client = TestClient(app)
        yield client


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint(self, api_client):
        """Test that health endpoint returns 200."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestPredictionEndpoint:
    """Test prediction endpoint."""

    def test_prediction_with_valid_features(self, api_client):
        """Test prediction with valid feature input."""
        payload = {
            "features": [
                1.08, 0.001, 50.0, 0.1, 0.0005,
                25.0, 0.0001, 0.0003, 14, 1, 0.0002
            ]
        }

        response = api_client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "sell" in data
        assert "range" in data
        assert "buy" in data
        assert "prediction" in data
        assert "confidence" in data

        # Check value ranges
        assert 0 <= data["sell"] <= 1
        assert 0 <= data["range"] <= 1
        assert 0 <= data["buy"] <= 1
        assert data["prediction"] in ["sell", "range", "buy"]
        assert 0 <= data["confidence"] <= 1

    def test_prediction_with_wrong_feature_count(self, api_client):
        """Test that API rejects wrong number of features."""
        payload = {
            "features": [1.08, 0.001, 50.0]  # Only 3 features instead of 11
        }

        response = api_client.post("/predict", json=payload)

        # Should return error
        assert response.status_code in [400, 422]

    def test_prediction_with_invalid_data_type(self, api_client):
        """Test that API rejects invalid data types."""
        payload = {
            "features": ["invalid", "data", "types"] * 4
        }

        response = api_client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error


class TestPredictionLogic:
    """Test prediction logic."""

    def test_probabilities_sum_to_one(self, api_client):
        """Test that returned probabilities sum to approximately 1."""
        payload = {
            "features": [1.08, 0.001, 50.0, 0.1, 0.0005, 25.0, 0.0001, 0.0003, 14, 1, 0.0002]
        }

        response = api_client.post("/predict", json=payload)
        data = response.json()

        prob_sum = data["sell"] + data["range"] + data["buy"]

        # Should sum to approximately 1 (allowing for floating point errors)
        assert abs(prob_sum - 1.0) < 0.01

    def test_confidence_is_max_probability(self, api_client):
        """Test that confidence equals the maximum probability."""
        payload = {
            "features": [1.08, 0.001, 50.0, 0.1, 0.0005, 25.0, 0.0001, 0.0003, 14, 1, 0.0002]
        }

        response = api_client.post("/predict", json=payload)
        data = response.json()

        max_prob = max(data["sell"], data["range"], data["buy"])

        assert abs(data["confidence"] - max_prob) < 0.01
