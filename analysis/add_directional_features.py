#!/usr/bin/env python3
"""
Directional Features: Add momentum and direction-predicting features
to the existing feature set.

These features complement the existing regime/volatility features
by adding actual directional signal.

Integration: Add these feature calculations to your trading_bot/features.py
after the existing features, then include the new feature names in the
feature list.
"""

import pandas as pd
import numpy as np


def add_directional_features(df: pd.DataFrame, suffix: str = "") -> list:
    """
    Add directional/momentum features to a DataFrame that has OHLCV columns.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        suffix: timeframe suffix like '_h1' or '_h4'

    Returns:
        List of new feature column names added
    """
    new_features = []
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]

    # =========================================================================
    # 1. MULTI-PERIOD MOMENTUM (Rate of Change)
    #    These directly measure "where is price going"
    # =========================================================================

    # Short-term momentum: 4-bar and 8-bar ROC
    for period in [4, 8]:
        col = f"roc_{period}{suffix}"
        df[col] = c.pct_change(period)
        new_features.append(col)

    # Medium-term momentum: 24-bar ROC (1 day on H1)
    col = f"roc_24{suffix}"
    df[col] = c.pct_change(24)
    new_features.append(col)

    # =========================================================================
    # 2. DIRECTIONAL MOVEMENT (Plus DI - Minus DI)
    #    ADX tells you IF market is trending, this tells you WHICH DIRECTION
    # =========================================================================
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)

    # Where both are positive, keep only the larger
    both_pos = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_adj = plus_dm.copy()
    minus_dm_adj = minus_dm.copy()
    plus_dm_adj[both_pos & (minus_dm >= plus_dm)] = 0
    minus_dm_adj[both_pos & (plus_dm > minus_dm)] = 0

    # Smoothed directional indicators
    smooth_plus = plus_dm_adj.rolling(14).mean()
    smooth_minus = minus_dm_adj.rolling(14).mean()

    col = f"di_diff{suffix}"
    df[col] = smooth_plus - smooth_minus  # Positive = bullish, negative = bearish
    new_features.append(col)

    # =========================================================================
    # 3. PRICE POSITION IN RANGE (Williams %R style)
    #    Where is price relative to recent high/low?
    #    Near top of range = bearish pressure, near bottom = bullish pressure
    # =========================================================================
    for period in [14, 48]:
        highest = h.rolling(period).max()
        lowest = l.rolling(period).min()
        col = f"price_position_{period}{suffix}"
        df[col] = (c - lowest) / (highest - lowest + 1e-10)
        new_features.append(col)

    # =========================================================================
    # 4. CLOSE-TO-CLOSE MOMENTUM SLOPE
    #    Linear regression slope of close prices over N bars
    #    More robust than simple ROC, captures trend direction
    # =========================================================================
    for period in [10, 20]:
        col = f"momentum_slope_{period}{suffix}"
        # Normalized slope: fit line to normalized prices
        df[col] = c.rolling(period).apply(
            lambda x: np.polyfit(range(len(x)), x / x.iloc[0] - 1, 1)[0] if len(x) == period else 0,
            raw=False
        )
        new_features.append(col)

    # =========================================================================
    # 5. BUYING/SELLING PRESSURE
    #    Close position within the bar's range indicates intra-bar direction
    #    Averaged over N bars for a pressure reading
    # =========================================================================
    bar_position = (c - l) / (h - l + 1e-10)  # 0=closed at low, 1=closed at high

    col = f"buy_pressure_5{suffix}"
    df[col] = bar_position.rolling(5).mean()
    new_features.append(col)

    col = f"buy_pressure_14{suffix}"
    df[col] = bar_position.rolling(14).mean()
    new_features.append(col)

    # =========================================================================
    # 6. HIGHER-LOW / LOWER-HIGH SEQUENCE
    #    Count of consecutive higher lows (bullish) or lower highs (bearish)
    #    This is the essence of price action direction
    # =========================================================================
    higher_low = (l > l.shift(1)).astype(int)
    lower_high = (h < h.shift(1)).astype(int)

    col = f"higher_lows_5{suffix}"
    df[col] = higher_low.rolling(5).sum()  # 0-5 scale, 5 = strong uptrend
    new_features.append(col)

    col = f"lower_highs_5{suffix}"
    df[col] = lower_high.rolling(5).sum()  # 0-5 scale, 5 = strong downtrend
    new_features.append(col)

    # Net direction: higher_lows - lower_highs
    col = f"structure_direction_5{suffix}"
    df[col] = df[f"higher_lows_5{suffix}"] - df[f"lower_highs_5{suffix}"]
    new_features.append(col)

    # =========================================================================
    # 7. VOLUME MOMENTUM (if tick_volume available)
    # =========================================================================
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float)

        # Volume-weighted direction: high volume on up bars = bullish
        up_bar = (c > o).astype(float)
        down_bar = (c < o).astype(float)

        col = f"vol_direction_10{suffix}"
        vol_up = (tv * up_bar).rolling(10).sum()
        vol_down = (tv * down_bar).rolling(10).sum()
        df[col] = (vol_up - vol_down) / (vol_up + vol_down + 1e-10)
        new_features.append(col)

    return new_features


def add_cross_timeframe_features(df_h1: pd.DataFrame, df_h4: pd.DataFrame) -> list:
    """
    Add features that capture alignment between H1 and H4 timeframes.

    This requires both DataFrames to be aligned (H4 values forward-filled to H1 rows).
    Assumes H4 features already exist in the main DataFrame with _h4 suffix.

    Call this on the merged DataFrame that already has both H1 and H4 columns.
    """
    new_features = []

    # Trend alignment: are H1 and H4 moving in the same direction?
    if "ema50_ema200_h1" in df_h1.columns and "ema50_ema200_h4" in df_h1.columns:
        h1_trend = np.sign(df_h1["ema50_ema200_h1"])
        h4_trend = np.sign(df_h1["ema50_ema200_h4"])

        col = "trend_alignment"
        df_h1[col] = h1_trend * h4_trend  # +1 = aligned, -1 = conflicting
        new_features.append(col)

    # Momentum alignment
    if "roc_4_h1" in df_h1.columns and "roc_4_h4" in df_h1.columns:
        col = "momentum_alignment"
        df_h1[col] = np.sign(df_h1["roc_4_h1"]) * np.sign(df_h1["roc_4_h4"])
        new_features.append(col)

    return new_features


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================
"""
To integrate into your trading_bot/features.py:

1. Import this module or copy the functions

2. In your engineer_features() function, after existing feature calculation:

    # Add directional features for H1
    h1_cols = ['open', 'high', 'low', 'close']
    # (assuming df already has H1 OHLC)
    dir_features_h1 = add_directional_features(df, suffix='_h1')

    # Add directional features for H4
    # (on the H4 resampled data before merging, or on merged df)
    dir_features_h4 = add_directional_features(df_h4, suffix='_h4')

    # Add cross-timeframe alignment
    cross_features = add_cross_timeframe_features(df)

3. Add the new feature names to your feature list:
    all_features = existing_features + dir_features_h1 + dir_features_h4 + cross_features

4. The key new features and what they measure:
    - roc_4, roc_8, roc_24: "Which direction is price moving?"
    - di_diff: "Is buying or selling pressure stronger?"
    - price_position_14/48: "Is price near the top or bottom of its range?"
    - momentum_slope_10/20: "What is the trend direction and strength?"
    - buy_pressure_5/14: "Are bars closing near highs or lows?"
    - structure_direction_5: "Is price making higher lows or lower highs?"
    - trend_alignment: "Do H1 and H4 agree on direction?"
"""


if __name__ == "__main__":
    # Quick test with sample data
    print("Directional Features Module")
    print("This module adds direction-predicting features to complement")
    print("the existing regime/volatility features.")
    print()
    print("New features per timeframe:")
    print("  - roc_4, roc_8, roc_24        (3) Multi-period momentum")
    print("  - di_diff                      (1) Directional movement bias")
    print("  - price_position_14/48         (2) Price location in range")
    print("  - momentum_slope_10/20         (2) Trend slope")
    print("  - buy_pressure_5/14            (2) Close position pressure")
    print("  - higher_lows_5, lower_highs_5 (2) Price structure")
    print("  - structure_direction_5         (1) Net structure direction")
    print("  - vol_direction_10             (1) Volume-weighted direction (if volume available)")
    print("  Total: 14 per timeframe (+ 2 cross-timeframe)")
    print()
    print("See integration instructions in the module docstring.")
