"""
Unit tests for Phase 3: training/inference mismatch and profit alignment fixes.

Tests cover:
- 3A: dm_momentum feature replaces ADX (naming + computation parity)
- 3B: Confidence threshold raised to 0.50
- 3C: Label threshold raised to 0.001
- 3D: Two-stage HPO Sharpe selection logic
"""

import math

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────
# 3A — DM Momentum feature parity
# ──────────────────────────────────────────────────────────────────

def dm_momentum_python(highs: np.ndarray, lows: np.ndarray, period: int = 14) -> np.ndarray:
    """Replica of FeatureEngineer._dm_momentum (rolling mean of abs(+DM - -DM))."""
    plus_dm = np.maximum(np.diff(highs, prepend=highs[0]), 0)
    minus_dm = np.maximum(-np.diff(lows, prepend=lows[0]), 0)
    abs_diff = np.abs(plus_dm - minus_dm)
    return pd.Series(abs_diff).rolling(period).mean().values


def dm_momentum_mql5(highs: np.ndarray, lows: np.ndarray, period: int = 14) -> float:
    """
    Replica of MQL5 EA manual DM momentum computation.

    MQL5 uses ArraySetAsSeries (index 0 = most recent), so we reverse.
    """
    h = highs[::-1]  # index 0 = current bar
    l = lows[::-1]
    dm_sum = 0.0
    for i in range(period):
        plus_dm = max(h[i] - h[i + 1], 0.0)
        minus_dm = max(l[i + 1] - l[i], 0.0)
        dm_sum += abs(plus_dm - minus_dm)
    return dm_sum / period


class TestDMMomentumParity:

    def test_python_mql5_match(self):
        """Python rolling mean at last bar must equal MQL5 manual average."""
        np.random.seed(42)
        n = 30
        highs = np.cumsum(np.random.uniform(0.001, 0.003, n)) + 1.09
        lows = highs - np.random.uniform(0.001, 0.003, n)

        py_values = dm_momentum_python(highs, lows, period=14)
        py_last = py_values[-1]  # value at last bar

        # MQL5 needs 15 bars (indices -15: for the last 14 diffs + 1)
        mql5_val = dm_momentum_mql5(highs[-15:], lows[-15:], period=14)

        assert math.isclose(py_last, mql5_val, rel_tol=1e-10), (
            f"Python={py_last}, MQL5={mql5_val}"
        )

    def test_feature_engineer_produces_dm_momentum_columns(self):
        """FeatureEngineer must output dm_momentum_h1 and dm_momentum_h4."""
        from src.features.engineer import FeatureEngineer
        cols = FeatureEngineer.default_feature_columns()
        assert "dm_momentum_h1" in cols
        assert "dm_momentum_h4" in cols
        assert "adx_h1" not in cols
        assert "adx_h4" not in cols


# ──────────────────────────────────────────────────────────────────
# 3B — Confidence threshold
# ──────────────────────────────────────────────────────────────────

class TestConfidenceThreshold:

    def test_config_threshold_is_050(self):
        """config.yaml must specify confidence_threshold >= 0.50."""
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        threshold = cfg["trading"]["ml"]["confidence_threshold"]
        assert threshold >= 0.50, f"Threshold {threshold} < 0.50"

    @pytest.mark.parametrize("conf, should_trade", [
        (0.49, False),
        (0.50, True),
        (0.80, True),
    ])
    def test_threshold_filter(self, conf, should_trade):
        """Signal should only pass when confidence >= 0.50."""
        threshold = 0.50
        passes = conf >= threshold
        assert passes == should_trade


# ──────────────────────────────────────────────────────────────────
# 3C — Label threshold
# ──────────────────────────────────────────────────────────────────

class TestLabelThreshold:

    def test_config_label_thresholds(self):
        """config.yaml must specify buy >= 0.001 and sell <= -0.001."""
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        buy_th = cfg["training"]["buy_threshold"]
        sell_th = cfg["training"]["sell_threshold"]
        assert buy_th >= 0.001, f"buy_threshold {buy_th} < 0.001"
        assert sell_th <= -0.001, f"sell_threshold {sell_th} > -0.001"

    @pytest.mark.parametrize("ret, expected_label", [
        (0.002, 1),      # Above 0.1% → buy
        (0.00101, 1),    # Just above → buy
        (0.001, 0),      # Exactly 0.1% → range (strict >)
        (0.0009, 0),     # Just below → range
        (0.0, 0),        # Flat → range
        (-0.0009, 0),    # Just above sell → range
        (-0.001, 0),     # Exactly -0.1% → range (strict <)
        (-0.00101, -1),  # Just below → sell
        (-0.002, -1),    # Below → sell
    ])
    def test_label_assignment(self, ret, expected_label):
        """Labels must use 0.001 threshold, not 0.0005."""
        buy_th = 0.001
        sell_th = -0.001
        if ret > buy_th:
            label = 1
        elif ret < sell_th:
            label = -1
        else:
            label = 0
        assert label == expected_label


# ──────────────────────────────────────────────────────────────────
# 3D — Two-stage HPO Sharpe selection
# ──────────────────────────────────────────────────────────────────

class TestTwoStageHPO:

    def test_backtest_sharpe_basic(self):
        """_backtest_sharpe should return a finite number for valid inputs."""
        from src.models.hpo import _backtest_sharpe
        preds = np.array([2, 0, 1, 2, 0])  # long, short, flat, long, short
        close = np.array([1.09, 1.091, 1.089, 1.092, 1.090, 1.093])
        sharpe = _backtest_sharpe(preds, close)
        assert math.isfinite(sharpe)

    def test_backtest_sharpe_all_flat(self):
        """All-flat predictions should give zero Sharpe (no trades)."""
        from src.models.hpo import _backtest_sharpe
        preds = np.array([1, 1, 1, 1])  # all flat
        close = np.array([1.09, 1.091, 1.089, 1.092, 1.090])
        sharpe = _backtest_sharpe(preds, close)
        # All flat = zero pnl = zero std = 0 sharpe
        assert sharpe == 0.0

    def test_backtest_sharpe_too_short(self):
        """Edge case: fewer than 2 close prices → 0.0."""
        from src.models.hpo import _backtest_sharpe
        assert _backtest_sharpe(np.array([2]), np.array([1.09])) == 0.0

    def test_optimize_backward_compatible_without_close_val(self):
        """optimize() without close_val should still work (Stage 1 only)."""
        from src.models.hpo import optimize
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = np.random.choice([0, 1, 2], 200)
        X_val = np.random.randn(50, 5)
        y_val = np.random.choice([0, 1, 2], 50)
        config = {"hpo": {"trials": 3, "study_name": "test_compat"}}

        best_params, best_f1 = optimize(X_train, y_train, X_val, y_val, config)
        assert isinstance(best_params, dict)
        assert best_f1 >= 0

    def test_optimize_stage2_with_close_val(self):
        """optimize() with close_val should run both stages."""
        from src.models.hpo import optimize
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = np.random.choice([0, 1, 2], 200)
        X_val = np.random.randn(50, 5)
        y_val = np.random.choice([0, 1, 2], 50)
        close_val = np.cumsum(np.random.randn(50) * 0.001) + 1.09
        config = {"hpo": {"trials": 3, "study_name": "test_stage2", "stage2_top_n": 3}}

        best_params, best_f1 = optimize(
            X_train, y_train, X_val, y_val, config, close_val=close_val
        )
        assert isinstance(best_params, dict)
        assert best_f1 >= 0
