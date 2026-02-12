"""
Unit tests verifying day-of-week feature parity between Python and MQL5.

Python convention:  pd.Timestamp.dayofweek → 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
MQL5 convention:    MqlDateTime.day_of_week → 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
MQL5 remapping:     python_dow = (mql5_dow + 6) % 7
"""

import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer


def _make_h1_data(date_str: str, periods: int = 250) -> pd.DataFrame:
    """Build minimal H1 OHLCV frame starting on the given date."""
    idx = pd.date_range(date_str, periods=periods, freq="h")
    return pd.DataFrame(
        {
            "time": idx,
            "open": 1.0900,
            "high": 1.0920,
            "low": 1.0880,
            "close": 1.0910,
        }
    )


def _mql5_remap(mql5_dow: int) -> int:
    """Replicate the MQL5 remapping formula: python_dow = (mql5_dow + 6) % 7."""
    return (mql5_dow + 6) % 7


class TestDayOfWeekParity:
    """Verify Python FeatureEngineer DOW encoding matches remapped MQL5 convention."""

    @pytest.mark.parametrize(
        "date_str, expected_day, mql5_dow",
        [
            ("2024-01-01 10:00", "monday", 1),      # Mon → MQL5=1
            ("2024-01-02 10:00", "tuesday", 2),      # Tue → MQL5=2
            ("2024-01-03 10:00", "wednesday", 3),     # Wed → MQL5=3
            ("2024-01-04 10:00", "thursday", 4),      # Thu → MQL5=4
            ("2024-01-05 10:00", "friday", 5),        # Fri → MQL5=5
        ],
    )
    def test_dow_onehot_matches_mql5_remap(self, date_str, expected_day, mql5_dow):
        """Each trading day's one-hot flag must be 1 in exactly the right column."""
        fe = FeatureEngineer(config={})
        df = _make_h1_data(date_str, periods=250)
        result = fe.engineer(df)

        # Grab the row whose timestamp matches the target hour
        ts = pd.Timestamp(date_str)
        row = result.loc[result["time"] == ts]
        assert len(row) == 1, f"Expected exactly 1 row for {date_str}"
        row = row.iloc[0]

        # Python side: the target column must be 1, all others 0
        days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        for day in days:
            col = f"dow_{day}"
            expected = 1 if day == expected_day else 0
            assert row[col] == expected, (
                f"dow_{day} should be {expected} on {date_str}, got {row[col]}"
            )

        # MQL5 side: remapped value must equal Python dayofweek for the same day
        python_dow = _mql5_remap(mql5_dow)
        assert python_dow == ts.dayofweek, (
            f"MQL5 remap({mql5_dow}) = {python_dow}, but Python dayofweek = {ts.dayofweek}"
        )

    @pytest.mark.parametrize("mql5_dow", [0, 6])
    def test_weekend_days_produce_no_flag(self, mql5_dow):
        """MQL5 Sun(0) and Sat(6) should not set any trading-day flag."""
        python_dow = _mql5_remap(mql5_dow)
        # python_dow 5 = Sat, 6 = Sun — neither is in 0..4
        assert python_dow not in range(5), (
            f"Weekend MQL5 dow={mql5_dow} mapped to python_dow={python_dow}, which is a weekday"
        )

    def test_remap_formula_full_week(self):
        """Verify the remapping formula for all 7 MQL5 day_of_week values."""
        # MQL5 → Python expected mapping
        expected = {
            0: 6,  # Sun → 6
            1: 0,  # Mon → 0
            2: 1,  # Tue → 1
            3: 2,  # Wed → 2
            4: 3,  # Thu → 3
            5: 4,  # Fri → 4
            6: 5,  # Sat → 5
        }
        for mql5_val, py_val in expected.items():
            assert _mql5_remap(mql5_val) == py_val, (
                f"Remap({mql5_val}) expected {py_val}, got {_mql5_remap(mql5_val)}"
            )
