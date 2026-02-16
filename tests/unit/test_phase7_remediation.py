"""
Unit tests for adversarial audit remediation.

Tests cover:
- Domain validation: per-feature range checks on /predict
- Plausibility checks: DOW exactly-one, session exactly-one, binary constraints
- Confidence threshold: raised to 0.80
- MQL5 structural fixes: g_UseML, daily limit timing, partial close tracking,
  stale value ordering, API timeout reduction
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

CONTRACT_PATH = Path("feature_contract.json")


@pytest.fixture(scope="module")
def contract():
    assert CONTRACT_PATH.exists(), "feature_contract.json not found"
    return json.loads(CONTRACT_PATH.read_text())


def _make_client(contract_data):
    """Create a TestClient with a stub model and the given contract."""
    import main as main_mod
    from main import app

    main_mod.feature_contract = contract_data
    main_mod.expected_n_features = contract_data["feature_count"]
    main_mod.feature_names = contract_data["features"]

    stub = MagicMock()
    stub.n_features_in_ = contract_data["feature_count"]
    stub.predict_proba.return_value = np.array([[0.15, 0.25, 0.60]])
    main_mod.model = stub

    return TestClient(app)


def _valid_features(contract_data):
    """Return a plausible 23-feature vector that passes all validation."""
    n = contract_data["feature_count"]
    names = contract_data["features"]
    vec = [0.0] * n
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Set plausible values for bounded features
    vec[name_to_idx["rsi_h1"]] = 55.0
    vec[name_to_idx["rsi_h4"]] = 48.0
    vec[name_to_idx["rsi_slope_h1"]] = 2.5
    vec[name_to_idx["rsi_slope_h4"]] = -1.0
    vec[name_to_idx["hour"]] = 10.0
    vec[name_to_idx["atr_ratio_h1"]] = 0.005
    vec[name_to_idx["atr_ratio_h4"]] = 0.004
    vec[name_to_idx["range_h1"]] = 0.003
    vec[name_to_idx["range_h4"]] = 0.006
    vec[name_to_idx["dm_momentum_h4"]] = 0.001

    # Set close/ema features
    vec[name_to_idx["close_ema50_h1"]] = 1.0914
    vec[name_to_idx["ema50_ema200_h1"]] = 0.0003
    vec[name_to_idx["ema50_ema200_h4"]] = 0.001
    vec[name_to_idx["body_h1"]] = 0.0005
    vec[name_to_idx["body_h4"]] = -0.001

    # Exactly one session and one DOW
    vec[name_to_idx["session_european"]] = 1.0
    vec[name_to_idx["dow_wednesday"]] = 1.0

    return vec


# ──────────────────────────────────────────────────────────────────
# Contract schema: feature_domains + plausibility_rules present
# ──────────────────────────────────────────────────────────────────

class TestContractDomains:

    def test_contract_has_feature_domains(self, contract):
        """Contract must include feature_domains section."""
        assert "feature_domains" in contract
        assert isinstance(contract["feature_domains"], dict)
        assert len(contract["feature_domains"]) > 0

    def test_contract_has_plausibility_rules(self, contract):
        """Contract must include plausibility_rules section."""
        assert "plausibility_rules" in contract
        rules = contract["plausibility_rules"]
        assert "exactly_one_dow" in rules
        assert "exactly_one_session" in rules

    def test_all_binary_features_in_domains(self, contract):
        """All session and DOW features must have binary domain."""
        domains = contract["feature_domains"]
        binary_names = [
            "session_asian", "session_european", "session_american",
            "dow_monday", "dow_tuesday", "dow_wednesday",
            "dow_thursday", "dow_friday",
        ]
        for name in binary_names:
            assert name in domains, f"Missing domain for {name}"
            assert domains[name].get("binary") is True, f"{name} not marked binary"

    def test_rsi_domains_correct(self, contract):
        """RSI features must have [0, 100] domain."""
        domains = contract["feature_domains"]
        for name in ("rsi_h1", "rsi_h4"):
            assert domains[name]["min"] == 0
            assert domains[name]["max"] == 100

    def test_confidence_threshold_080(self, contract):
        """Confidence threshold must be 0.80 after remediation."""
        assert contract["confidence_threshold"] == 0.80


# ──────────────────────────────────────────────────────────────────
# Domain validation on /predict
# ──────────────────────────────────────────────────────────────────

class TestDomainValidation:

    @pytest.fixture(scope="class")
    def setup(self):
        c = json.loads(CONTRACT_PATH.read_text())
        client = _make_client(c)
        return client, c

    def test_valid_features_accepted(self, setup):
        """A valid feature vector should return 200."""
        client, c = setup
        features = _valid_features(c)
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200

    def test_rsi_above_100_rejected(self, setup):
        """RSI > 100 must return 400."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("rsi_h1")
        features[idx] = 150.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "rsi_h1" in resp.json()["detail"]["violations"][0]

    def test_rsi_negative_rejected(self, setup):
        """RSI < 0 must return 400."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("rsi_h4")
        features[idx] = -5.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "rsi_h4" in resp.json()["detail"]["violations"][0]

    def test_hour_negative_rejected(self, setup):
        """hour = -1 must return 400."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("hour")
        features[idx] = -1.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "hour" in resp.json()["detail"]["violations"][0]

    def test_hour_999_rejected(self, setup):
        """hour = 999 must return 400."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("hour")
        features[idx] = 999.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400

    def test_binary_feature_103_rejected(self, setup):
        """session_european = 1.03 (scaling drift) must return 400."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("session_european")
        features[idx] = 1.03
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "session_european" in resp.json()["detail"]["violations"][0]

    def test_negative_range_rejected(self, setup):
        """range_h1 < 0 must return 400."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("range_h1")
        features[idx] = -0.001
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "range_h1" in resp.json()["detail"]["violations"][0]

    def test_atr_ratio_above_max_rejected(self, setup):
        """atr_ratio_h1 > 0.1 must return 400."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("atr_ratio_h1")
        features[idx] = 0.5
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "atr_ratio_h1" in resp.json()["detail"]["violations"][0]

    def test_extreme_rsi_1e15_rejected(self, setup):
        """RSI = 1e15 must return 400 (adversarial attack #3)."""
        client, c = setup
        features = _valid_features(c)
        idx = c["features"].index("rsi_h1")
        features[idx] = 1e15
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────
# Plausibility checks on /predict
# ──────────────────────────────────────────────────────────────────

class TestPlausibilityChecks:

    @pytest.fixture(scope="class")
    def setup(self):
        c = json.loads(CONTRACT_PATH.read_text())
        client = _make_client(c)
        return client, c

    def test_zero_dow_flags_rejected(self, setup):
        """All DOW flags = 0 must return 400."""
        client, c = setup
        features = _valid_features(c)
        for name in ["dow_monday", "dow_tuesday", "dow_wednesday",
                      "dow_thursday", "dow_friday"]:
            features[c["features"].index(name)] = 0.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "exactly_one_dow" in resp.json()["detail"]["violations"][0]

    def test_two_dow_flags_rejected(self, setup):
        """Two DOW flags set must return 400."""
        client, c = setup
        features = _valid_features(c)
        features[c["features"].index("dow_monday")] = 1.0
        features[c["features"].index("dow_wednesday")] = 1.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "exactly_one_dow" in resp.json()["detail"]["violations"][0]

    def test_zero_session_flags_rejected(self, setup):
        """All session flags = 0 must return 400."""
        client, c = setup
        features = _valid_features(c)
        for name in ["session_asian", "session_european", "session_american"]:
            features[c["features"].index(name)] = 0.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "exactly_one_session" in resp.json()["detail"]["violations"][0]

    def test_two_sessions_rejected(self, setup):
        """Two session flags set must return 400."""
        client, c = setup
        features = _valid_features(c)
        features[c["features"].index("session_asian")] = 1.0
        features[c["features"].index("session_european")] = 1.0
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 400
        assert "exactly_one_session" in resp.json()["detail"]["violations"][0]

    def test_valid_vector_passes_all_checks(self, setup):
        """A valid feature vector should pass domain + plausibility."""
        client, c = setup
        features = _valid_features(c)
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────
# MQL5 structural fixes (file-level assertions)
# ──────────────────────────────────────────────────────────────────

class TestMQL5Fixes:

    @pytest.fixture(scope="class")
    def mq5_content(self):
        mq5_path = Path("AdvanceEA.mq5")
        assert mq5_path.exists(), "AdvanceEA.mq5 not found"
        return mq5_path.read_text()

    def test_g_useml_declared(self, mq5_content):
        """Runtime ML toggle must use g_UseML (not input variable)."""
        assert "bool g_UseML" in mq5_content

    def test_no_input_var_assignment(self, mq5_content):
        """No runtime assignment to input UseMLPredictions."""
        lines = mq5_content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip the input declaration itself
            if stripped.startswith("input"):
                continue
            # Skip comments
            if stripped.startswith("//"):
                continue
            # Check for assignment to UseMLPredictions
            if "UseMLPredictions" in stripped and "=" in stripped:
                # Allow: g_UseML = UseMLPredictions (reading from input)
                if "g_UseML" in stripped:
                    continue
                assert False, f"Found runtime assignment to input var at line {i+1}: {stripped}"

    def test_daily_limit_before_newbar(self, mq5_content):
        """CheckDailyLimits must appear before the isNewBar guard in OnTick."""
        ontick_start = mq5_content.find("void OnTick()")
        assert ontick_start > 0
        ontick_block = mq5_content[ontick_start:ontick_start + 800]
        daily_pos = ontick_block.find("CheckDailyLimits()")
        newbar_pos = ontick_block.find("if(isNewBar)")
        assert daily_pos < newbar_pos, (
            f"CheckDailyLimits ({daily_pos}) must come before isNewBar guard ({newbar_pos})"
        )

    def test_manage_positions_before_trailing(self, mq5_content):
        """ManagePositions (breakeven) must execute before ManageTrailingStops."""
        ontick_start = mq5_content.find("void OnTick()")
        ontick_block = mq5_content[ontick_start:ontick_start + 1200]
        bp_pos = ontick_block.find("ManagePositions()")
        ts_pos = ontick_block.find("ManageTrailingStops()")
        assert bp_pos < ts_pos, (
            f"ManagePositions ({bp_pos}) must come before ManageTrailingStops ({ts_pos})"
        )

    def test_partial_close_tracking_exists(self, mq5_content):
        """Partial close must track already-closed tickets."""
        assert "partialClosedTickets" in mq5_content
        assert "alreadyClosed" in mq5_content

    def test_api_timeout_reduced(self, mq5_content):
        """API timeout must be 2000ms (not 5000ms)."""
        assert "int timeout = 2000" in mq5_content

    def test_max_retries_reduced(self, mq5_content):
        """Max retries must be 2 (not 3)."""
        assert "int maxRetries = 2" in mq5_content

    def test_stale_values_cleared_before_api(self, mq5_content):
        """lastMLPrediction must be cleared at start of GetMLPrediction."""
        fn_start = mq5_content.find("int GetMLPrediction(")
        fn_block = mq5_content[fn_start:fn_start + 400]
        clear_pos = fn_block.find('lastMLPrediction = ""')
        assert clear_pos > 0, "lastMLPrediction not cleared at start of GetMLPrediction"

    def test_validated_values_stored_after_checks(self, mq5_content):
        """lastMLPrediction assignment must come AFTER validation checks."""
        fn_start = mq5_content.find("int GetMLPrediction(")
        fn_block = mq5_content[fn_start:]
        # Find the validation comment
        validation_end = fn_block.find("// Store VALIDATED values")
        confidence_check = fn_block.find("confidence < 0.0 || confidence > 1.0")
        assert validation_end > confidence_check, (
            "Validated value storage must come after confidence range check"
        )

    def test_confidence_threshold_080(self, mq5_content):
        """Default ML confidence threshold must be 0.80."""
        assert "ML_Confidence_Threshold = 0.80" in mq5_content
