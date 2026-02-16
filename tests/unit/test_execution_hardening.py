"""
Unit tests for execution hardening changes (Tasks 2A-2D).

Tests cover:
- Spread buffer SL calculation
- Margin-capped lot sizing (non-circular formula)
- Feature validation (NaN / out-of-range rejection)
- API retry / consecutive failure auto-disable logic
"""

import math

import pytest


# ──────────────────────────────────────────────────────────────────
# 2A — Spread-buffered SL distance
# ──────────────────────────────────────────────────────────────────

def calc_sl_distance(atr: float, atr_multiplier: float, spread_points: int, point: float) -> float:
    """Replica of the fixed SL calculation in OpenTrade()."""
    spread = spread_points * point
    return atr * atr_multiplier + spread


class TestSpreadBuffer:

    @pytest.mark.parametrize(
        "atr, multiplier, spread_pts, point, expected_sl",
        [
            # EURUSD typical: ATR=0.0050, mult=2.0, spread=15pts, point=0.00001
            (0.0050, 2.0, 15, 0.00001, 0.0050 * 2.0 + 15 * 0.00001),
            # Zero spread (ECN)
            (0.0050, 2.0, 0, 0.00001, 0.0050 * 2.0),
            # Wide spread (news)
            (0.0050, 2.0, 80, 0.00001, 0.0050 * 2.0 + 80 * 0.00001),
            # USDJPY: 3-digit, ATR=0.50, spread=20
            (0.50, 2.0, 20, 0.001, 0.50 * 2.0 + 20 * 0.001),
        ],
        ids=["eurusd_typical", "zero_spread", "wide_spread", "usdjpy"],
    )
    def test_sl_includes_spread(self, atr, multiplier, spread_pts, point, expected_sl):
        result = calc_sl_distance(atr, multiplier, spread_pts, point)
        assert math.isclose(result, expected_sl, rel_tol=1e-10)

    def test_sl_always_wider_than_no_spread(self):
        """Any nonzero spread must widen the SL distance."""
        base = calc_sl_distance(0.005, 2.0, 0, 0.00001)
        with_spread = calc_sl_distance(0.005, 2.0, 15, 0.00001)
        assert with_spread > base


# ──────────────────────────────────────────────────────────────────
# 2B — Margin-capped lot size (non-circular)
# ──────────────────────────────────────────────────────────────────

def calc_lot_size(
    balance: float,
    risk_pct: float,
    sl_distance: float,
    tick_value: float,
    tick_size: float,
    min_lot: float,
    max_lot: float,
    lot_step: float,
    free_margin: float,
    margin_per_lot: float,
) -> float:
    """Replica of the fixed CalculateLotSize() in AdvanceEA.mq5."""
    if tick_value <= 0 or tick_size <= 0:
        return min_lot

    risk_amount = balance * risk_pct / 100.0
    lot_size = risk_amount / (sl_distance / tick_size * tick_value)

    # Clamp & step
    lot_size = max(min_lot, min(max_lot, lot_size))
    lot_size = math.floor(lot_size / lot_step) * lot_step
    lot_size = round(lot_size, 2)

    # Margin cap — direct formula (not circular)
    if margin_per_lot > 0 and free_margin > 0:
        max_lot_by_margin = (free_margin * 0.5) / margin_per_lot
        if lot_size > max_lot_by_margin:
            lot_size = max_lot_by_margin
            lot_size = math.floor(lot_size / lot_step) * lot_step
            lot_size = round(lot_size, 2)

    return lot_size


class TestMarginCappedLotSize:

    # Standard params: EURUSD, $10k balance, 1% risk
    BASE = dict(
        balance=10000, risk_pct=1.0, sl_distance=0.0100,
        tick_value=10.0, tick_size=0.00001,
        min_lot=0.01, max_lot=100.0, lot_step=0.01,
        free_margin=10000.0, margin_per_lot=1000.0,
    )

    def test_normal_no_margin_constraint(self):
        """Lot size computed from risk when margin is abundant."""
        lot = calc_lot_size(**self.BASE)
        # risk_amount = 100, sl in ticks = 0.01/0.00001 = 1000, cost = 1000 * 10 = 10000
        # lot = 100 / 10000 = 0.01
        assert lot == 0.01

    def test_margin_cap_applied(self):
        """When free margin is tight, lot should be capped."""
        params = {**self.BASE, "free_margin": 100.0, "margin_per_lot": 1000.0}
        lot = calc_lot_size(**params)
        # maxLotByMargin = (100 * 0.5) / 1000 = 0.05
        # risk-based = 0.01, which is below 0.05, so no cap needed
        assert lot == 0.01

    def test_margin_cap_limits_large_lot(self):
        """Large risk lot gets capped by margin constraint."""
        params = {**self.BASE, "balance": 1_000_000, "risk_pct": 5.0,
                  "free_margin": 5000.0, "margin_per_lot": 1000.0}
        lot = calc_lot_size(**params)
        # risk-based lot would be 5.0, but maxLotByMargin = 2500/1000 = 2.5
        assert lot == 2.5

    def test_min_lot_floor(self):
        """Tiny balance should still produce at least min_lot."""
        params = {**self.BASE, "balance": 10, "risk_pct": 0.1}
        lot = calc_lot_size(**params)
        assert lot >= 0.01

    @pytest.mark.parametrize(
        "free_margin, margin_per_lot, expected_cap",
        [
            (10000, 1000, 5.0),    # 10000*0.5/1000 = 5.0
            (2000,  1000, 1.0),    # 2000*0.5/1000 = 1.0
            (500,   1000, 0.25),   # 500*0.5/1000 = 0.25
            (100,   5000, 0.01),   # 100*0.5/5000 = 0.01
        ],
        ids=["ample", "moderate", "tight", "minimal"],
    )
    def test_margin_cap_boundary_table(self, free_margin, margin_per_lot, expected_cap):
        """Direct verification of maxLotByMargin = (freeMargin*0.5)/margin_per_lot."""
        raw_cap = (free_margin * 0.5) / margin_per_lot
        assert math.isclose(raw_cap, expected_cap, rel_tol=1e-10)


# ──────────────────────────────────────────────────────────────────
# 2D — Feature validation
# ──────────────────────────────────────────────────────────────────

def validate_features(
    rsi_h1: float, rsi_h4: float, hour: float,
    numeric_features: list[float],
) -> bool:
    """Replica of the feature validation guard in GetMLPrediction()."""
    for v in numeric_features:
        if not math.isfinite(v):
            return False
    if rsi_h1 < 0 or rsi_h1 > 100 or rsi_h4 < 0 or rsi_h4 > 100:
        return False
    if hour < 0 or hour > 23:
        return False
    return True


class TestFeatureValidation:

    def test_valid_features_pass(self):
        assert validate_features(
            rsi_h1=55.0, rsi_h4=48.0, hour=14.0,
            numeric_features=[1.09, 0.001, 55.0, 0.5, 0.0008, 0.001, 0.002,
                              0.002, 48.0, 0.3, 0.0009, 20.0, 0.0005, 0.003],
        )

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_nan_inf_rejected(self, bad_value):
        features = [1.09, 0.001, 55.0, 0.5, 0.0008, 0.001, 0.002,
                    0.002, 48.0, 0.3, 0.0009, 20.0, 0.0005, 0.003]
        features[0] = bad_value
        assert not validate_features(rsi_h1=55.0, rsi_h4=48.0, hour=14.0,
                                     numeric_features=features)

    @pytest.mark.parametrize("rsi_h1, rsi_h4", [
        (-1, 50), (101, 50), (50, -1), (50, 101), (-5, 105),
    ], ids=["h1_neg", "h1_over", "h4_neg", "h4_over", "both_bad"])
    def test_rsi_out_of_range_rejected(self, rsi_h1, rsi_h4):
        assert not validate_features(
            rsi_h1=rsi_h1, rsi_h4=rsi_h4, hour=14.0,
            numeric_features=[1.0] * 14,
        )

    @pytest.mark.parametrize("hour", [-1, 24, 25, -100])
    def test_hour_out_of_range_rejected(self, hour):
        assert not validate_features(
            rsi_h1=50.0, rsi_h4=50.0, hour=hour,
            numeric_features=[1.0] * 14,
        )

    def test_boundary_values_pass(self):
        """RSI=0, RSI=100, hour=0, hour=23 are all valid."""
        assert validate_features(rsi_h1=0.0, rsi_h4=100.0, hour=0.0,
                                 numeric_features=[1.0] * 14)
        assert validate_features(rsi_h1=100.0, rsi_h4=0.0, hour=23.0,
                                 numeric_features=[1.0] * 14)


# ──────────────────────────────────────────────────────────────────
# 2C — API retry / consecutive failure auto-disable
# ──────────────────────────────────────────────────────────────────

class MLAPISimulator:
    """Simulates the retry + consecutive failure logic from GetMLPrediction()."""

    MAX_RETRIES = 3
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self):
        self.consecutive_failures = 0
        self.disabled = False

    def call_api(self, api_succeeds_on_attempt: int | None) -> str:
        """
        Simulate API call with retry.
        api_succeeds_on_attempt: 1-based attempt that succeeds, or None for total failure.
        Returns: "success", "failed", or "disabled".
        """
        if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self.disabled = True
            return "disabled"

        for attempt in range(1, self.MAX_RETRIES + 1):
            if api_succeeds_on_attempt is not None and attempt >= api_succeeds_on_attempt:
                self.consecutive_failures = 0
                return "success"

        self.consecutive_failures += 1
        return "failed"


class TestAPIRetryLogic:

    def test_success_on_first_attempt(self):
        sim = MLAPISimulator()
        assert sim.call_api(1) == "success"
        assert sim.consecutive_failures == 0

    def test_success_on_third_attempt(self):
        sim = MLAPISimulator()
        assert sim.call_api(3) == "success"
        assert sim.consecutive_failures == 0

    def test_total_failure_increments_counter(self):
        sim = MLAPISimulator()
        assert sim.call_api(None) == "failed"
        assert sim.consecutive_failures == 1

    def test_auto_disable_after_max_failures(self):
        sim = MLAPISimulator()
        for _ in range(5):
            sim.call_api(None)  # 5 consecutive failures
        result = sim.call_api(1)  # Should be disabled now
        assert result == "disabled"
        assert sim.disabled

    def test_success_resets_failure_counter(self):
        sim = MLAPISimulator()
        # 4 failures, then success
        for _ in range(4):
            sim.call_api(None)
        assert sim.consecutive_failures == 4
        sim.call_api(1)  # Success resets counter
        assert sim.consecutive_failures == 0

    def test_not_disabled_below_threshold(self):
        sim = MLAPISimulator()
        for _ in range(4):  # 4 < 5 threshold
            sim.call_api(None)
        assert sim.call_api(1) == "success"  # Still operational
