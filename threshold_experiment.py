#!/usr/bin/env python3
"""
Threshold Experiment: 3-class model with Sell-only trading.
Analyzes Sell prediction precision at various confidence thresholds.
Output saved to logs/threshold_experiment.log
"""

import sys
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.config import get_config

# --- Dedicated log file: logs/threshold_experiment.log ---
LOG_FILE = Path(__file__).parent / "logs" / "threshold_experiment.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logger = logging.getLogger("threshold_experiment")


class TeeWriter:
    """Write to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


THRESHOLDS = [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.45, 0.50, 0.55, 0.60]


def load_data():
    config = get_config()
    tp_pips = float(config.get("training.tp_pips", 20.0))
    sl_pips = float(config.get("training.sl_pips", 15.0))

    data_file = config.get("paths.data_processed_file", "EURUSD_H1_clean.csv")
    model_path = config.get("model.path", "xgb_eurusd_h1.pkl")

    df = pd.read_csv(data_file)
    model = joblib.load(model_path)

    with open("features_used.json") as f:
        features = json.load(f)

    # Holdout split (same as pipeline)
    train_ratio = float(config.get("training.train_ratio", 0.6))
    val_ratio = float(config.get("training.val_ratio", 0.2))
    n = len(df)
    val_end = int(n * (train_ratio + val_ratio))
    holdout = df.iloc[val_end:].copy()

    X = holdout[features].values
    # 3-class labels: Sell(-1)=0, Range(0)=1, Buy(1)=2
    y_true = pd.Series(holdout["label"]).map({-1: 0, 0: 1, 1: 2}).values

    # Get probabilities (3-class: Sell=0, Range=1, Buy=2)
    probas = model.predict_proba(X)
    preds = np.argmax(probas, axis=1)
    max_probs = np.max(probas, axis=1)

    # Regime filter
    use_regime = bool(config.get("trading.regime_filter.enabled", False))
    regime_mask = np.ones(len(holdout), dtype=bool)
    if use_regime:
        chop_h1_pct = float(config.get("trading.regime_filter.chop_h1_percentile", 50))
        chop_h4_pct = float(config.get("trading.regime_filter.chop_h4_percentile", 50))
        chop_h1_thresh = np.percentile(holdout["choppiness_h1"].dropna(), chop_h1_pct)
        chop_h4_thresh = np.percentile(holdout["choppiness_h4"].dropna(), chop_h4_pct)
        regime_mask = (holdout["choppiness_h1"].values < chop_h1_thresh) & \
                      (holdout["choppiness_h4"].values < chop_h4_thresh)

    return holdout, y_true, probas, preds, max_probs, tp_pips, sl_pips, regime_mask, use_regime


def main():
    # Tee all print output to log file
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 80)
    print("THRESHOLD EXPERIMENT — 3-class model, Sell-only trading")
    print("=" * 80)

    holdout, y_true, probas, preds, max_probs, tp_pips, sl_pips, regime_mask, use_regime = load_data()

    be_rate = sl_pips / (tp_pips + sl_pips) * 100
    sell_base_rate = (y_true == 0).mean() * 100

    # Sell probability stats (class 0)
    sell_probs = probas[:, 0]
    sell_pred_mask = preds == 0

    print(f"TP={tp_pips} pips, SL={sl_pips} pips")
    print(f"Break-even win rate: {be_rate:.1f}%")
    print(f"Holdout samples: {len(holdout)}")
    print(f"Actual Sell rate: {sell_base_rate:.1f}%")
    print(f"Overall accuracy: {(preds == y_true).mean():.1%}")
    if use_regime:
        print(f"Regime filter: ON (chop_h1+h4 below median, {regime_mask.sum()}/{len(holdout)} bars pass)")
    print(f"Sell probability range: [{sell_probs.min():.4f}, {sell_probs.max():.4f}]")

    if sell_pred_mask.sum() > 0:
        sell_pred_probs = max_probs[sell_pred_mask]
        print(f"Sell prediction confidence range: [{sell_pred_probs.min():.4f}, {sell_pred_probs.max():.4f}]")
        print(f"Sell prediction count: {sell_pred_mask.sum()} ({sell_pred_mask.mean():.1%} of holdout)")

    # --- Threshold sweep (Sell-only) ---
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP — Sell-only signals at each confidence level")
    print("=" * 80)
    print(f"{'Thresh':>7} | {'Trades':>7} | {'TruePos':>8} | {'Precision':>9} | "
          f"{'Recall':>7} | {'ExpPips':>8} | {'TotalPips':>10}")
    print("-" * 80)

    total_sell = (y_true == 0).sum()

    for thresh in THRESHOLDS:
        # Only trade when model predicts Sell AND confidence >= threshold AND regime allows
        signals = (preds == 0) & (max_probs >= thresh) & regime_mask
        n_trades = signals.sum()

        if n_trades == 0:
            print(f"  {thresh:.2f}  | {'--- no trades ---':^65}")
            continue

        true_pos = (signals & (y_true == 0)).sum()
        precision = true_pos / n_trades * 100
        recall = true_pos / total_sell * 100 if total_sell > 0 else 0

        # Expected pips: correct Sell = +TP, wrong Sell = -SL
        exp_pips = (precision / 100) * tp_pips - (1 - precision / 100) * sl_pips
        total_pips = exp_pips * n_trades

        above_be = " ***" if precision > be_rate else ""

        print(f"  {thresh:.2f}  | {n_trades:6d}  | {true_pos:7d}  | {precision:7.1f}%  | "
              f"{recall:5.1f}%  | {exp_pips:+7.2f}  | {total_pips:+9.1f}{above_be}")

    # --- Class-level analysis ---
    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY")
    print("=" * 80)

    for cls, name in [(0, "Sell"), (1, "Range"), (2, "Buy")]:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == cls).sum()
        acc = correct / mask.sum() * 100
        print(f"  {name:>6}: {correct}/{mask.sum()} = {acc:.1f}%")

    # --- Sell probability distribution ---
    print("\n" + "=" * 80)
    print("SELL PREDICTION CONFIDENCE DISTRIBUTION")
    print("=" * 80)

    regime_sell_mask = sell_pred_mask & regime_mask
    if regime_sell_mask.sum() > 0:
        sell_confidences = max_probs[regime_sell_mask]
        sell_actual = y_true[regime_sell_mask]

        bins = [(0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50),
                (0.50, 0.60), (0.60, 0.70), (0.70, 1.00)]

        print(f"{'Confidence':>12} | {'Count':>6} | {'Actual Sell%':>12} | {'Above BE?':>10}")
        print("-" * 55)

        for lo, hi in bins:
            mask = (sell_confidences >= lo) & (sell_confidences < hi)
            if mask.sum() == 0:
                continue
            actual_sell_pct = (sell_actual[mask] == 0).mean() * 100
            above = "YES" if actual_sell_pct > be_rate else "no"
            print(f"  [{lo:.2f}-{hi:.2f})  | {mask.sum():5d}  | {actual_sell_pct:10.1f}%  | {above:>8}")
    else:
        print("  No Sell predictions in holdout.")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"  Break-even precision: {be_rate:.1f}%")
    print(f"  Sell base rate: {sell_base_rate:.1f}%")
    print(f"  *** = precision above break-even (profitable Sell signals)")
    print(f"  Look for threshold where precision > {be_rate:.1f}% with decent trade count")


if __name__ == "__main__":
    main()
    # Restore stdout
    if isinstance(sys.stdout, TeeWriter):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")
