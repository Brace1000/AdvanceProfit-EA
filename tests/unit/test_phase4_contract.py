"""
Unit tests for Phase 4: Feature contract, /contract endpoint, and structural validation.

Tests cover:
- 4A: feature_contract.json schema correctness
- 4B: /contract endpoint and /predict feature-count validation
- 4C: MQL5 structural JSON validation (Python replicas)
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────
# 4A — feature_contract.json schema validation
# ──────────────────────────────────────────────────────────────────

CONTRACT_PATH = Path("feature_contract.json")


@pytest.fixture(scope="module")
def contract():
    assert CONTRACT_PATH.exists(), "feature_contract.json not found"
    return json.loads(CONTRACT_PATH.read_text())


class TestContractSchema:

    def test_required_keys_present(self, contract):
        """Contract must have all required top-level keys."""
        required = {
            "version", "features", "feature_count",
            "all_features_before_pruning", "pruned_features",
            "label_mapping", "prediction_mapping", "confidence_threshold",
        }
        missing = required - set(contract.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_feature_count_matches_list(self, contract):
        """feature_count must equal len(features)."""
        assert contract["feature_count"] == len(contract["features"])

    def test_features_are_unique(self, contract):
        """No duplicate feature names."""
        feats = contract["features"]
        assert len(feats) == len(set(feats))

    def test_pruned_not_in_features(self, contract):
        """Pruned features must NOT appear in active features."""
        active = set(contract["features"])
        for pruned in contract["pruned_features"]:
            assert pruned not in active, f"Pruned feature '{pruned}' still in active list"

    def test_pruned_in_all_features(self, contract):
        """Pruned features must appear in all_features_before_pruning."""
        all_feats = set(contract["all_features_before_pruning"])
        for pruned in contract["pruned_features"]:
            assert pruned in all_feats, f"Pruned feature '{pruned}' not in all_features list"

    def test_active_features_subset_of_all(self, contract):
        """Active features must be a subset of all_features_before_pruning."""
        all_feats = set(contract["all_features_before_pruning"])
        active = set(contract["features"])
        diff = active - all_feats
        assert not diff, f"Active features not in all_features: {diff}"

    def test_all_minus_pruned_equals_active(self, contract):
        """all_features - pruned = active features (as sets)."""
        all_set = set(contract["all_features_before_pruning"])
        pruned_set = set(contract["pruned_features"])
        expected = all_set - pruned_set
        actual = set(contract["features"])
        assert expected == actual

    def test_label_mapping_has_three_classes(self, contract):
        """Label mapping must define sell/range/buy."""
        lm = contract["label_mapping"]
        assert set(lm.values()) == {"sell", "range", "buy"}

    def test_prediction_mapping_has_three_classes(self, contract):
        """Prediction mapping must define sell/range/buy."""
        pm = contract["prediction_mapping"]
        assert set(pm.values()) == {"sell", "range", "buy"}

    def test_confidence_threshold_is_050(self, contract):
        """Confidence threshold must be >= 0.50."""
        assert contract["confidence_threshold"] >= 0.50

    def test_version_is_string(self, contract):
        assert isinstance(contract["version"], str)


# ──────────────────────────────────────────────────────────────────
# 4A — Contract matches config.yaml
# ──────────────────────────────────────────────────────────────────

class TestContractConfigConsistency:

    def test_contract_features_match_config(self, contract):
        """Contract all_features must match config.yaml model.features."""
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        config_features = cfg["model"]["features"]
        assert contract["all_features_before_pruning"] == config_features

    def test_contract_threshold_matches_config(self, contract):
        """Contract confidence_threshold must match config.yaml."""
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert contract["confidence_threshold"] == cfg["trading"]["ml"]["confidence_threshold"]


# ──────────────────────────────────────────────────────────────────
# 4B — /contract endpoint + /predict feature count validation
# ──────────────────────────────────────────────────────────────────

class TestContractEndpoint:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app, _load_contract
        # Manually trigger startup state for testing
        import main as main_mod
        main_mod.feature_contract = _load_contract()
        if main_mod.feature_contract:
            main_mod.expected_n_features = main_mod.feature_contract["feature_count"]
            main_mod.feature_names = main_mod.feature_contract["features"]
        return TestClient(app)

    def test_contract_returns_200(self, client):
        resp = client.get("/contract")
        assert resp.status_code == 200

    def test_contract_has_required_keys(self, client):
        data = client.get("/contract").json()
        assert "version" in data
        assert "feature_count" in data
        assert "feature_names" in data

    def test_contract_feature_count_correct(self, client, contract):
        data = client.get("/contract").json()
        assert data["feature_count"] == contract["feature_count"]
        assert len(data["feature_names"]) == contract["feature_count"]

    def test_predict_wrong_feature_count_returns_400(self, client, contract):
        """Sending wrong number of features must return 400."""
        wrong_count = contract["feature_count"] - 1
        resp = client.post("/predict", json={"features": [0.0] * wrong_count})
        assert resp.status_code == 400

    def test_predict_extra_feature_returns_400(self, client, contract):
        """Sending one extra feature must return 400."""
        extra_count = contract["feature_count"] + 1
        resp = client.post("/predict", json={"features": [0.0] * extra_count})
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────
# 4C — Structural JSON validation (Python replicas of MQL5 logic)
# ──────────────────────────────────────────────────────────────────

def validate_api_response(response_json: str) -> tuple[bool, str]:
    """
    Python replica of the MQL5 structural validation logic.

    Returns (is_valid, error_message).
    """
    # Check required keys
    for key in ("sell", "range", "buy", "confidence", "prediction"):
        if f'"{key}"' not in response_json:
            return False, f"Missing required key: {key}"

    # Parse values
    try:
        data = json.loads(response_json)
    except json.JSONDecodeError:
        return False, "Invalid JSON"

    # Probabilities must sum to ~1.0
    prob_sum = data["sell"] + data["range"] + data["buy"]
    if abs(prob_sum - 1.0) > 0.05:
        return False, f"Probabilities sum to {prob_sum:.4f}, expected ~1.0"

    # Prediction must be a known class
    if data["prediction"] not in ("buy", "sell", "range"):
        return False, f"Unknown prediction: {data['prediction']}"

    # Confidence must be in [0, 1]
    if not (0.0 <= data["confidence"] <= 1.0):
        return False, f"Confidence {data['confidence']} out of [0,1]"

    return True, ""


class TestStructuralValidation:

    def test_valid_response_passes(self):
        resp = json.dumps({
            "sell": 0.15, "range": 0.25, "buy": 0.60,
            "prediction": "buy", "confidence": 0.60
        })
        ok, msg = validate_api_response(resp)
        assert ok, msg

    def test_missing_sell_key_fails(self):
        resp = json.dumps({
            "range": 0.25, "buy": 0.60,
            "prediction": "buy", "confidence": 0.60
        })
        ok, msg = validate_api_response(resp)
        assert not ok
        assert "sell" in msg

    def test_missing_prediction_key_fails(self):
        resp = json.dumps({
            "sell": 0.15, "range": 0.25, "buy": 0.60,
            "confidence": 0.60
        })
        ok, msg = validate_api_response(resp)
        assert not ok
        assert "prediction" in msg

    def test_probabilities_sum_too_low_fails(self):
        resp = json.dumps({
            "sell": 0.10, "range": 0.10, "buy": 0.10,
            "prediction": "buy", "confidence": 0.10
        })
        ok, msg = validate_api_response(resp)
        assert not ok
        assert "sum" in msg.lower()

    def test_probabilities_sum_too_high_fails(self):
        resp = json.dumps({
            "sell": 0.50, "range": 0.50, "buy": 0.50,
            "prediction": "buy", "confidence": 0.50
        })
        ok, msg = validate_api_response(resp)
        assert not ok
        assert "sum" in msg.lower()

    def test_unknown_prediction_fails(self):
        resp = json.dumps({
            "sell": 0.15, "range": 0.25, "buy": 0.60,
            "prediction": "hold", "confidence": 0.60
        })
        ok, msg = validate_api_response(resp)
        assert not ok
        assert "Unknown" in msg

    def test_confidence_above_1_fails(self):
        resp = json.dumps({
            "sell": 0.15, "range": 0.25, "buy": 0.60,
            "prediction": "buy", "confidence": 1.5
        })
        ok, msg = validate_api_response(resp)
        assert not ok
        assert "confidence" in msg.lower()

    def test_confidence_negative_fails(self):
        resp = json.dumps({
            "sell": 0.15, "range": 0.25, "buy": 0.60,
            "prediction": "buy", "confidence": -0.1
        })
        ok, msg = validate_api_response(resp)
        assert not ok
        assert "confidence" in msg.lower()

    def test_malformed_json_fails(self):
        ok, msg = validate_api_response("{broken json")
        assert not ok
        assert "Invalid JSON" in msg or "Missing" in msg

    def test_edge_probabilities_within_tolerance_passes(self):
        """Probabilities summing to 0.96 (within 0.05 tolerance) should pass."""
        resp = json.dumps({
            "sell": 0.14, "range": 0.24, "buy": 0.58,
            "prediction": "buy", "confidence": 0.58
        })
        ok, msg = validate_api_response(resp)
        assert ok, msg

    def test_all_three_prediction_values_accepted(self):
        """All valid prediction values should pass."""
        for pred in ("buy", "sell", "range"):
            resp = json.dumps({
                "sell": 0.33, "range": 0.34, "buy": 0.33,
                "prediction": pred, "confidence": 0.34
            })
            ok, msg = validate_api_response(resp)
            assert ok, f"Valid prediction '{pred}' rejected: {msg}"
