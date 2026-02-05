#!/usr/bin/env python3
"""
Validate: Do the bot's bad windows correspond to bullish EUR/USD periods?

If sell-only bot loses money during bullish trends, that's a structural flaw.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.config import get_config
from src.data_collection.loader import DataLoader
from src.features.engineer import FeatureEngineer

project_root = Path(__file__).parent


def main():
    config = get_config()

    loader = DataLoader(config._config)
    raw_path = project_root / config.get("paths.data_raw_file", "EURUSD_H1_clean.csv")
    df = loader.load(raw_path)

    fe = FeatureEngineer(config._config)
    df = fe.engineer(df)

    dt_col = "datetime" if "datetime" in df.columns else ("time" if "time" in df.columns else None)
    df = df.dropna(subset=["close"]).iloc[:-1].copy()

    n = len(df)
    n_windows = 5
    test_size = n // (n_windows + 1)
    min_train = test_size

    print("=" * 90)
    print("TREND BIAS ANALYSIS: Is the sell bot fighting bullish trends?")
    print("=" * 90)
    print()

    # Walk-forward results for reference
    wf_pips = [0, 130, -202, -142, 293]

    print(f"{'Window':>8} | {'Period':>25} | {'Open':>8} | {'Close':>8} | "
          f"{'Move':>8} | {'Pips':>7} | {'Trend':>10} | {'Bot Pips':>8} | {'Match':>6}")
    print("-" * 110)

    for w in range(n_windows):
        test_start = min_train + w * test_size
        test_end = min(test_start + test_size, n)

        if test_start >= n or test_end <= test_start:
            break

        test_df = df.iloc[test_start:test_end]

        if dt_col and dt_col in test_df.columns:
            start_date = str(test_df.iloc[0][dt_col])[:10]
            end_date = str(test_df.iloc[-1][dt_col])[:10]
            period = f"{start_date} → {end_date}"
        else:
            period = f"bar {test_start} → {test_end}"

        open_price = test_df["close"].iloc[0]
        close_price = test_df["close"].iloc[-1]
        move_pips = (close_price - open_price) / 0.0001

        if move_pips > 50:
            trend = "BULLISH"
        elif move_pips < -50:
            trend = "BEARISH"
        else:
            trend = "SIDEWAYS"

        bot_pips = wf_pips[w]

        # Does the bot struggle when market is bullish?
        if trend == "BULLISH" and bot_pips < 0:
            match = "⚠ BAD"
        elif trend == "BEARISH" and bot_pips > 0:
            match = "✓ GOOD"
        elif trend == "SIDEWAYS":
            match = "~ FLAT"
        elif bot_pips > 0:
            match = "✓ OK"
        else:
            match = "✗ LOSS"

        print(f"  W {w+1:>3}   | {period:>25} | {open_price:>7.5f} | {close_price:>7.5f} | "
              f"{move_pips:>+7.0f} | {move_pips:>+6.0f}p | {trend:>10} | {bot_pips:>+7} | {match}")

    print()
    print("=" * 90)
    print("MONTHLY BREAKDOWN — Price direction vs bot performance")
    print("=" * 90)
    print()

    # Monthly breakdown across entire dataset
    if dt_col and dt_col in df.columns:
        df["_dt"] = pd.to_datetime(df[dt_col])
        df["_month"] = df["_dt"].dt.to_period("M")

        months = df.groupby("_month").agg(
            open_price=("close", "first"),
            close_price=("close", "last"),
            high=("high", "max"),
            low=("low", "min"),
            bars=("close", "count")
        )

        bullish_months = 0
        bearish_months = 0
        sideways_months = 0

        print(f"{'Month':>10} | {'Open':>8} | {'Close':>8} | {'Move Pips':>9} | {'Trend':>10}")
        print("-" * 60)

        for period, row in months.iterrows():
            move = (row["close_price"] - row["open_price"]) / 0.0001
            if move > 50:
                trend = "BULLISH"
                bullish_months += 1
            elif move < -50:
                trend = "BEARISH"
                bearish_months += 1
            else:
                trend = "SIDEWAYS"
                sideways_months += 1

            print(f"{str(period):>10} | {row['open_price']:>7.5f} | {row['close_price']:>7.5f} | "
                  f"{move:>+8.0f}p | {trend}")

        print()
        total = bullish_months + bearish_months + sideways_months
        print(f"Summary: {bullish_months} bullish ({bullish_months/total:.0%}) | "
              f"{bearish_months} bearish ({bearish_months/total:.0%}) | "
              f"{sideways_months} sideways ({sideways_months/total:.0%})")
        print()

        if bullish_months > bearish_months:
            print("⚠ WARNING: EUR/USD has been predominantly BULLISH in this dataset.")
            print("  A sell-only bot has a structural disadvantage during bullish periods.")
            print("  Consider: adding a Buy model, or pausing during strong uptrends.")
        elif bearish_months > bullish_months:
            print("✓ EUR/USD has been predominantly BEARISH — favorable for sell-only.")
        else:
            print("~ EUR/USD is roughly balanced between bullish and bearish months.")


if __name__ == "__main__":
    main()
