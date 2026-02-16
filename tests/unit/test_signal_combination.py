"""
Truth-table unit test for the strict signal combination contract.

Contract (CombineWithTechnical=true):
    Trade ONLY if ml_signal == tech_signal AND ml_signal != 0 AND confidence >= threshold.
    Otherwise return 0.

This test exhaustively covers:
    ml_signal  ∈ {-1, 0, 1}
    tech_signal ∈ {-1, 0, 1}
    confidence  ∈ {low (below threshold), high (at/above threshold)}
"""

import pytest

# Threshold used in the EA
ML_CONFIDENCE_THRESHOLD = 0.40


def combine_signals(ml_signal: int, tech_signal: int, confidence: float) -> int:
    """
    Pure-Python replica of the fixed CombineWithTechnical logic in AdvanceEA.mq5.

    Strict agreement contract — NO fallback to technical signal.
    """
    if ml_signal == tech_signal and ml_signal != 0 and confidence >= ML_CONFIDENCE_THRESHOLD:
        return ml_signal
    return 0


class TestSignalCombinationTruthTable:
    """Exhaustive truth table for strict agreement contract."""

    # ── Cases that SHOULD produce a trade ──────────────────────────
    @pytest.mark.parametrize(
        "ml, tech, conf, expected",
        [
            ( 1,  1, 0.80,  1),   # Both BUY, high confidence → BUY
            (-1, -1, 0.80, -1),   # Both SELL, high confidence → SELL
            ( 1,  1, 0.40,  1),   # Both BUY, exactly at threshold → BUY
            (-1, -1, 0.40, -1),   # Both SELL, exactly at threshold → SELL
        ],
        ids=[
            "agree_buy_high",
            "agree_sell_high",
            "agree_buy_threshold",
            "agree_sell_threshold",
        ],
    )
    def test_trade_executed(self, ml, tech, conf, expected):
        assert combine_signals(ml, tech, conf) == expected

    # ── Cases that must return 0 (no trade) ────────────────────────
    @pytest.mark.parametrize(
        "ml, tech, conf",
        [
            # Disagreement — signals differ
            ( 1, -1, 0.80),   # BUY vs SELL, high conf
            (-1,  1, 0.80),   # SELL vs BUY, high conf
            ( 1,  0, 0.80),   # BUY vs NONE, high conf
            (-1,  0, 0.80),   # SELL vs NONE, high conf
            ( 0,  1, 0.80),   # NONE vs BUY, high conf
            ( 0, -1, 0.80),   # NONE vs SELL, high conf
            ( 0,  0, 0.80),   # Both NONE, high conf
            # Low confidence — even when signals agree
            ( 1,  1, 0.30),   # Agree BUY, low conf
            (-1, -1, 0.30),   # Agree SELL, low conf
            ( 1,  1, 0.39),   # Agree BUY, just under threshold
            (-1, -1, 0.39),   # Agree SELL, just under threshold
            # Disagreement AND low confidence
            ( 1, -1, 0.30),
            (-1,  1, 0.30),
            ( 1,  0, 0.30),
            ( 0,  1, 0.30),
            ( 0,  0, 0.30),
        ],
        ids=[
            "disagree_buy_sell_high",
            "disagree_sell_buy_high",
            "buy_vs_none_high",
            "sell_vs_none_high",
            "none_vs_buy_high",
            "none_vs_sell_high",
            "both_none_high",
            "agree_buy_low",
            "agree_sell_low",
            "agree_buy_just_under",
            "agree_sell_just_under",
            "disagree_buy_sell_low",
            "disagree_sell_buy_low",
            "buy_vs_none_low",
            "none_vs_buy_low",
            "both_none_low",
        ],
    )
    def test_no_trade(self, ml, tech, conf):
        assert combine_signals(ml, tech, conf) == 0

    def test_no_fallback_to_technical(self):
        """
        Regression: old code fell back to tech_signal when ML confidence was low.
        This must never happen under the strict contract.
        """
        # tech_signal=1 (BUY), ml_signal=-1 (SELL), confidence below threshold
        assert combine_signals(-1, 1, 0.30) == 0
        # tech_signal=1 (BUY), ml_signal=0 (NONE), confidence below threshold
        assert combine_signals(0, 1, 0.30) == 0
        # tech_signal=-1 (SELL), ml_signal=0 (NONE), confidence below threshold
        assert combine_signals(0, -1, 0.30) == 0
