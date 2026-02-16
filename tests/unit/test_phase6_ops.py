"""
Unit tests for Phase 6: Operational Safety, Observability, and Staleness Controls.

Tests cover:
- 6A: Enhanced /health endpoint (status, staleness, feature count)
- 6B: Structured trade logging (CSV format, features hash)
- 6C: Model staleness logic
- 6D: Magic number centralization in MQL5
"""

import json
import os
import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _make_client(model_obj=None, contract=None, model_path_exists=True, model_mtime=None):
    """Create a TestClient with controlled model/contract state."""
    import main as main_mod
    from main import app

    main_mod.feature_contract = contract
    if contract:
        main_mod.expected_n_features = contract["feature_count"]
        main_mod.feature_names = contract["features"]
    else:
        main_mod.expected_n_features = None
        main_mod.feature_names = None

    main_mod.model = model_obj
    return TestClient(app)


def _default_contract():
    """Return a minimal feature contract matching production."""
    return {
        "version": "2.30",
        "feature_count": 23,
        "features": [f"feat_{i}" for i in range(23)],
    }


# ──────────────────────────────────────────────────────────────────
# 6A — TestHealthEndpoint
# ──────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_all_fields(self):
        """Verify response contains all required keys."""
        contract = _default_contract()
        stub = MagicMock()
        stub.n_features_in_ = 23
        client = _make_client(model_obj=stub, contract=contract)

        with patch("os.path.getmtime", return_value=time.time()):
            resp = client.get("/health")

        data = resp.json()
        expected_keys = {"status", "model_loaded", "feature_count",
                         "model_age_hours", "model_path", "contract_version"}
        assert expected_keys.issubset(set(data.keys())), \
            f"Missing keys: {expected_keys - set(data.keys())}"

    def test_health_model_loaded_true(self):
        """With a stub model loaded, status should be 'healthy'."""
        contract = _default_contract()
        stub = MagicMock()
        stub.n_features_in_ = 23
        client = _make_client(model_obj=stub, contract=contract)

        with patch("os.path.getmtime", return_value=time.time()), \
             patch("pathlib.Path.exists", return_value=True):
            resp = client.get("/health")

        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_model_not_loaded(self):
        """Without a model, status should be 'model_not_loaded'."""
        contract = _default_contract()
        client = _make_client(model_obj=None, contract=contract)

        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "model_not_loaded"
        assert data["model_loaded"] is False

    def test_health_stale_model(self):
        """Model older than staleness threshold should return 'stale'."""
        contract = _default_contract()
        stub = MagicMock()
        stub.n_features_in_ = 23
        client = _make_client(model_obj=stub, contract=contract)

        # Mock model file as 200 hours old (threshold is 168)
        old_mtime = time.time() - (200 * 3600)
        with patch("os.path.getmtime", return_value=old_mtime), \
             patch("pathlib.Path.exists", return_value=True):
            resp = client.get("/health")

        data = resp.json()
        assert data["status"] == "stale"

    def test_health_model_age_positive(self):
        """model_age_hours should be >= 0."""
        contract = _default_contract()
        stub = MagicMock()
        stub.n_features_in_ = 23
        client = _make_client(model_obj=stub, contract=contract)

        with patch("os.path.getmtime", return_value=time.time() - 3600), \
             patch("pathlib.Path.exists", return_value=True):
            resp = client.get("/health")

        data = resp.json()
        assert data["model_age_hours"] >= 0


# ──────────────────────────────────────────────────────────────────
# 6B — TestTradeLogging
# ──────────────────────────────────────────────────────────────────

# Python replica of MQL5 ComputeFeaturesHash (FNV-1a 64-bit)
def compute_features_hash(features_json: str) -> str:
    """Python replica of the MQL5 ComputeFeaturesHash function."""
    FNV_OFFSET = 14695981039346656037
    FNV_PRIME = 1099511628211
    MASK_64 = (1 << 64) - 1

    h = FNV_OFFSET
    for ch in features_json:
        h = (h ^ ord(ch)) & MASK_64
        h = (h * FNV_PRIME) & MASK_64

    return format(h, '016x')


TRADE_LOG_COLUMNS = [
    "timestamp", "ticket", "direction", "entry_price", "sl", "tp",
    "lot_size", "ml_prediction", "ml_confidence", "tech_signal", "features_hash",
]


class TestTradeLogging:

    def test_log_row_has_all_columns(self):
        """CSV header must have exactly 11 columns matching spec."""
        assert len(TRADE_LOG_COLUMNS) == 11

    def test_log_entry_format(self):
        """A simulated log row has correct types/format."""
        row = {
            "timestamp": "2025.01.15 14:00:00",
            "ticket": "12345",
            "direction": "BUY",
            "entry_price": "1.08500",
            "sl": "1.08200",
            "tp": "1.09100",
            "lot_size": "0.10",
            "ml_prediction": "buy",
            "ml_confidence": "0.7200",
            "tech_signal": "1",
            "features_hash": "a1b2c3d4e5f67890",
        }
        # All columns present
        assert set(row.keys()) == set(TRADE_LOG_COLUMNS)
        # Types
        assert float(row["entry_price"]) > 0
        assert int(row["ticket"]) > 0
        assert row["direction"] in ("BUY", "SELL")
        assert float(row["ml_confidence"]) >= 0
        assert len(row["features_hash"]) == 16

    def test_features_hash_deterministic(self):
        """Same features string must produce the same hash."""
        features = "[1.0,2.0,3.0,4.0,5.0]"
        h1 = compute_features_hash(features)
        h2 = compute_features_hash(features)
        assert h1 == h2

    def test_features_hash_changes(self):
        """Different features strings must produce different hashes."""
        h1 = compute_features_hash("[1.0,2.0,3.0]")
        h2 = compute_features_hash("[1.0,2.0,4.0]")
        assert h1 != h2


# ──────────────────────────────────────────────────────────────────
# 6C — TestStalenessLogic
# ──────────────────────────────────────────────────────────────────

def staleness_status(model_age_hours: float, threshold: float = 168) -> tuple:
    """
    Python replica of the staleness logic used in both the /health
    endpoint and the EA's CheckAPIHealth.

    Returns (status, ml_enabled).
    """
    if model_age_hours > threshold * 2:
        return "stale", False   # CRITICAL — disable ML
    elif model_age_hours > threshold:
        return "stale", True    # WARNING — still enabled
    else:
        return "healthy", True


class TestStalenessLogic:

    def test_within_threshold_is_healthy(self):
        """Age < threshold -> healthy, ML enabled."""
        status, ml_on = staleness_status(100, threshold=168)
        assert status == "healthy"
        assert ml_on is True

    def test_above_threshold_is_stale(self):
        """Age > threshold -> stale (warning), ML still enabled."""
        status, ml_on = staleness_status(200, threshold=168)
        assert status == "stale"
        assert ml_on is True

    def test_above_2x_threshold_disables(self):
        """Age > 2x threshold -> stale, ML disabled."""
        status, ml_on = staleness_status(400, threshold=168)
        assert status == "stale"
        assert ml_on is False


# ──────────────────────────────────────────────────────────────────
# 6D — TestMagicNumber
# ──────────────────────────────────────────────────────────────────

class TestMagicNumber:

    def test_magic_number_centralized(self):
        """Literal 123456 should appear only in the #define EA_MAGIC line."""
        mq5_path = Path("AdvanceEA.mq5")
        assert mq5_path.exists(), "AdvanceEA.mq5 not found"

        content = mq5_path.read_text()
        matches = [
            (i + 1, line)
            for i, line in enumerate(content.splitlines())
            if "123456" in line
        ]

        # Exactly one match: the #define line
        assert len(matches) == 1, (
            f"Expected exactly 1 occurrence of '123456' (the #define), "
            f"found {len(matches)}:\n"
            + "\n".join(f"  L{n}: {l.strip()}" for n, l in matches)
        )
        line_num, line_text = matches[0]
        assert "#define" in line_text and "EA_MAGIC" in line_text, (
            f"The single occurrence at L{line_num} is not the #define: {line_text.strip()}"
        )
