#!/usr/bin/env python3
"""
Threshold Experiment: Buy Model Optimization
Analyzes Buy prediction precision at various confidence thresholds.

The Buy model uses inverse labeling:
  - Class 0: Buy wins (price UP 20 pips before DOWN 15 pips)
  - Class 1: Range (neither barrier hit)
  - Class 2: Buy loses (price DOWN 15 pips before UP 20 pips)

This finds the optimal threshold that maximizes Buy signal profitability.
Output saved to logs/threshold_experiment_buy.log
"""

import sys
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import get_config

# --- Dedicated log file: logs/threshold_experiment_buy.log ---
LOG_FILE = project_root / "logs" / "threshold_experiment_buy.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logger = logging.getLogger("threshold_experiment_buy")


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
    def isatty(self):
        return self.stdout.isatty() if hasattr(self.stdout, 'isatty') else False


# Test more granular range for Buy model
THRESHOLDS = [0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34,
              0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.42, 0.45, 0.50]


def load_data():
    config = get_config()
    tp_pips = float(config.get("training.tp_pips", 20.0))
    sl_pips = float(config.get("training.sl_pips", 15.0))

    data_file = config.get("paths.data_processed_file", "data/EURUSD_H1_clean.csv")
    buy_model_path = config.get("model.buy_path", "models/xgb_eurusd_h1_buy.pkl")

    df = pd.read_csv(data_file)
    model = joblib.load(buy_model_path)

    # Load Buy model features (should be same 14 features)
    features_file = project_root / "features_used_buy.json"
    if not features_file.exists():
        features_file = project_root / "features_used.json"

    with open(features_file) as f:
        features = json.load(f)

    # Holdout split (same as pipeline)
    train_ratio = float(config.get("training.train_ratio", 0.6))
    val_ratio = float(config.get("training.val_ratio", 0.2))
    n = len(df)
    val_end = int(n * (train_ratio + val_ratio))
    holdout = df.iloc[val_end:].copy()

    X = holdout[features].values

    # Buy model was trained on buy_label if it exists, otherwise create it
    if "buy_label" not in holdout.columns:
        # Create buy labels on the fly (TP up, SL down)
        from src.features.engineer import FeatureEngineer
        fe = FeatureEngineer(config._config)
        holdout = fe.engineer(holdout)

    # Buy model labels: 0=buy wins, 1=range, 2=buy loses
    y_true = holdout["buy_label"].values if "buy_label" in holdout.columns else None

    if y_true is None:
        # Fallback: infer from regular label (invert)
        # Sell label: -1=sell wins, 0=range, 1=buy wins
        # Convert to buy label: 0=buy wins, 1=range, 2=buy loses
        y_true = holdout["label"].map({1: 0, 0: 1, -1: 2}).values

    # Get probabilities
    # Buy model output: [buy_wins_prob, range_prob, buy_loses_prob]
    probas = model.predict_proba(X)
    preds = np.argmax(probas, axis=1)
    max_probs = np.max(probas, axis=1)

    # Regime filter - DISABLE for debugging
    use_regime = False  # Temporarily disabled to see raw model performance
    regime_mask = np.ones(len(holdout), dtype=bool)
    if use_regime:
        chop_h1_pct = float(config.get("trading.regime_filter.chop_h1_percentile", 50))
        chop_h4_pct = float(config.get("trading.regime_filter.chop_h4_percentile", 50))

        if "choppiness_h1" in holdout.columns and "choppiness_h4" in holdout.columns:
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
    print("THRESHOLD EXPERIMENT — Buy Model Optimization")
    print("=" * 80)

    holdout, y_true, probas, preds, max_probs, tp_pips, sl_pips, regime_mask, use_regime = load_data()

    be_rate = sl_pips / (tp_pips + sl_pips) * 100
    buy_base_rate = (y_true == 0).mean() * 100

    # Buy probability stats (class 0 = buy wins)
    buy_probs = probas[:, 0]
    buy_pred_mask = preds == 0

    print(f"\nBuy Model Labeling:")
    print(f"  Class 0: Buy wins (price UP {tp_pips} before DOWN {sl_pips})")
    print(f"  Class 1: Range (neither barrier)")
    print(f"  Class 2: Buy loses (price DOWN {sl_pips} first)")
    print(f"\nTP={tp_pips} pips, SL={sl_pips} pips")
    print(f"Break-even win rate: {be_rate:.1f}%")
    print(f"Holdout samples: {len(holdout)}")
    print(f"Actual Buy win rate: {buy_base_rate:.1f}%")
    print(f"Overall accuracy: {(preds == y_true).mean():.1%}")
    if use_regime:
        print(f"Regime filter: ON (chop_h1+h4 below median, {regime_mask.sum()}/{len(holdout)} bars pass)")
    print(f"Buy probability range: [{buy_probs.min():.4f}, {buy_probs.max():.4f}]")

    if buy_pred_mask.sum() > 0:
        buy_pred_probs = max_probs[buy_pred_mask]
        print(f"Buy prediction confidence range: [{buy_pred_probs.min():.4f}, {buy_pred_probs.max():.4f}]")
        print(f"Buy prediction count: {buy_pred_mask.sum()} ({buy_pred_mask.mean():.1%} of holdout)")

    # --- Threshold sweep (Buy-only) ---
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP — Buy-only signals at each confidence level")
    print("=" * 80)
    print(f"{'Thresh':>7} | {'Trades':>7} | {'Wins':>8} | {'WinRate':>9} | "
          f"{'Recall':>7} | {'ExpPips':>8} | {'TotalPips':>10}")
    print("-" * 80)

    total_buy = (y_true == 0).sum()
    best_thresh = None
    best_total_pips = -float('inf')

    for thresh in THRESHOLDS:
        # Only trade when model predicts Buy wins AND confidence >= threshold AND regime allows
        signals = (preds == 0) & (max_probs >= thresh) & regime_mask
        n_trades = signals.sum()

        if n_trades == 0:
            print(f"  {thresh:.2f}  | {'--- no trades ---':^65}")
            continue

        wins = (signals & (y_true == 0)).sum()
        win_rate = wins / n_trades * 100
        recall = wins / total_buy * 100 if total_buy > 0 else 0

        # Expected pips: correct Buy = +TP, wrong Buy = -SL
        exp_pips = (win_rate / 100) * tp_pips - (1 - win_rate / 100) * sl_pips
        total_pips = exp_pips * n_trades

        above_be = " ***" if win_rate > be_rate else ""

        if total_pips > best_total_pips:
            best_total_pips = total_pips
            best_thresh = thresh

        print(f"  {thresh:.2f}  | {n_trades:6d}  | {wins:7d}  | {win_rate:7.1f}%  | "
              f"{recall:5.1f}%  | {exp_pips:+7.2f}  | {total_pips:+9.1f}{above_be}")

    # --- Class-level analysis ---
    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY")
    print("=" * 80)

    for cls, name in [(0, "Buy wins"), (1, "Range"), (2, "Buy loses")]:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == cls).sum()
        acc = correct / mask.sum() * 100
        print(f"  {name:>10}: {correct}/{mask.sum()} = {acc:.1f}%")

    # --- Buy probability distribution ---
    print("\n" + "=" * 80)
    print("BUY PREDICTION CONFIDENCE DISTRIBUTION")
    print("=" * 80)

    regime_buy_mask = buy_pred_mask & regime_mask
    if regime_buy_mask.sum() > 0:
        buy_confidences = max_probs[regime_buy_mask]
        buy_actual = y_true[regime_buy_mask]

        bins = [(0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45),
                (0.45, 0.50), (0.50, 0.60), (0.60, 1.00)]

        print(f"{'Confidence':>12} | {'Count':>6} | {'Actual Win%':>12} | {'Above BE?':>10}")
        print("-" * 55)

        for lo, hi in bins:
            mask = (buy_confidences >= lo) & (buy_confidences < hi)
            if mask.sum() == 0:
                continue
            actual_win_pct = (buy_actual[mask] == 0).mean() * 100
            above = "YES" if actual_win_pct > be_rate else "no"
            print(f"  [{lo:.2f}-{hi:.2f})  | {mask.sum():5d}  | {actual_win_pct:10.1f}%  | {above:>8}")
    else:
        print("  No Buy predictions in holdout.")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLD RECOMMENDATION")
    print("=" * 80)
    print(f"  Break-even win rate: {be_rate:.1f}%")
    print(f"  Buy base rate: {buy_base_rate:.1f}%")
    print(f"  *** = win rate above break-even (profitable Buy signals)")

    if best_thresh is not None:
        print(f"\n  BEST THRESHOLD: {best_thresh:.2f}")
        print(f"  Max total pips: {best_total_pips:+.1f}")
        print(f"\n  Compare to current (0.34): {'BETTER' if best_thresh != 0.34 else 'SAME'}")
    else:
        print(f"\n  No profitable threshold found!")


if __name__ == "__main__":
    main()
    # Restore stdout
    if isinstance(sys.stdout, TeeWriter):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    print(f"\nResults saved to {LOG_FILE}")
