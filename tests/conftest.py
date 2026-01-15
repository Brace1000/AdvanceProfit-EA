"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides fixtures
that can be used across all test files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'time': dates,
        'open': np.random.uniform(1.08, 1.10, 100),
        'high': np.random.uniform(1.10, 1.12, 100),
        'low': np.random.uniform(1.06, 1.08, 100),
        'close': np.random.uniform(1.08, 1.10, 100),
    })
    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1) + 0.0001
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1) - 0.0001
    return data


@pytest.fixture
def sample_features():
    """Generate sample feature data for model testing."""
    return pd.DataFrame({
        'close_ema50': np.random.uniform(1.08, 1.10, 50),
        'ema50_ema200': np.random.uniform(-0.001, 0.001, 50),
        'rsi': np.random.uniform(30, 70, 50),
        'rsi_slope': np.random.uniform(-2, 2, 50),
        'atr_ratio': np.random.uniform(0.0001, 0.001, 50),
        'adx': np.random.uniform(15, 35, 50),
        'body': np.random.uniform(-0.001, 0.001, 50),
        'range': np.random.uniform(0.001, 0.003, 50),
        'hour': np.random.randint(0, 24, 50),
        'session': np.random.randint(0, 3, 50),
        'prev_return': np.random.uniform(-0.01, 0.01, 50),
    })


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    return pd.Series(np.random.choice([-1, 0, 1], 50))


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_model():
    """Create a mock XGBoost model for testing."""
    from unittest.mock import MagicMock
    model = MagicMock()
    model.predict.return_value = np.array([1, 0, -1])
    model.predict_proba.return_value = np.array([
        [0.1, 0.2, 0.7],
        [0.3, 0.5, 0.2],
        [0.8, 0.1, 0.1]
    ])
    return model
