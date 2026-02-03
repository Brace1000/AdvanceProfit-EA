#!/usr/bin/env python3
"""
Barrier Sweep: Find optimal TP/SL levels for strongest Sell signal.

Tests multiple TP/SL combinations, retrains a quick model for each,
and evaluates Sell precision across thresholds.

Output saved to logs/barrier_sweep.log
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.config import get_config
from src.data_collection.loader import DataLoader
from src.data_collection.validator import DataValidator
from src.features.engineer import FeatureEngineer

LOG_FILE = Path(__file__).parent / "logs" / "barrier_sweep.log"
LOG_FILE.parent.mkdir(exist_ok=True)


class TeeWriter:
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


def triple_barrier_label(df, tp_pips, sl_pips, max_horizon=24):
    """Label using triple barrier method. Returns label array."""
    pip_value = 0.0001
    tp_price = tp_pips * pip_value
    sl_price = sl_pips * pip_value
    labels = np.zeros(len(df), dtype=int)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(len(df) - 1):
        entry = closes[i]
        end = min(i + 1 + max_horizon, len(df))
        future_highs = highs[i+1:end]
        future_lows = lows[i+1:end]

        tp_th = tp_price / entry
        sl_th = sl_price / entry

        ret_high = (future_highs - entry) / entry
        ret_low = (future_lows - entry) / entry

        # Buy: price goes up tp_th before down sl_th
        tp_up = np.where(ret_high >= tp_th)[0]
        sl_up = np.where(ret_low <= -sl_th)[0]
        first_up_tp = tp_up[0] if len(tp_up) > 0 else max_horizon
        first_up_sl = sl_up[0] if len(sl_up) > 0 else max_horizon

        # Sell: price goes down tp_th before up sl_th
        tp_dn = np.where(ret_low <= -tp_th)[0]
        sl_dn = np.where(ret_high >= sl_th)[0]
        first_dn_tp = tp_dn[0] if len(tp_dn) > 0 else max_horizon
        first_dn_sl = sl_dn[0] if len(sl_dn) > 0 else max_horizon

        if first_up_tp < first_up_sl and first_up_tp < max_horizon:
            labels[i] = 1   # Buy
        elif first_dn_tp < first_dn_sl and first_dn_tp < max_horizon:
            labels[i] = -1  # Sell
        else:
            labels[i] = 0   # Range

    return labels


def evaluate_combo(df_featured, feature_cols, tp_pips, sl_pips, max_horizon=24):
    """
    Re-label data with given TP/SL, train 3-class model, evaluate Sell signal.
    Uses fixed params (no HPO) for speed.
    """
    df = df_featured.copy()

    # Need datetime index for labeling if close/high/low are available
    labels = triple_barrier_label(df, tp_pips, sl_pips, max_horizon)
    df["label"] = labels

    # Drop NaN and last row
    df = df.dropna(subset=feature_cols).iloc[:-1].copy()

    X = df[feature_cols].values
    y = pd.Series(df["label"]).map({-1: 0, 0: 1, 1: 2}).values

    # Class distribution
    sell_pct = (y == 0).mean()
    range_pct = (y == 1).mean()
    buy_pct = (y == 2).mean()

    # 3-way split (60/20/20)
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, X_holdout = X[:val_end], X[val_end:]
    y_train, y_holdout = y[:val_end], y[val_end:]

    # Fixed conservative params (no HPO for speed)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 3,
        "n_estimators": 80,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 40,
        "gamma": 0.8,
        "reg_alpha": 50.0,
        "reg_lambda": 50.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    # Class-balanced weights
    try:
        classes = np.array([0, 1, 2])
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        sw = cw[y_train]
    except Exception:
        sw = None

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False, sample_weight=sw)

    # Predictions on holdout
    probas = model.predict_proba(X_holdout)
    preds = np.argmax(probas, axis=1)
    max_probs = np.max(probas, axis=1)

    # Overall accuracy
    holdout_acc = (preds == y_holdout).mean()

    # Sell-only analysis at various thresholds
    # Sell = class 0 in the model
    sell_results = {}
    for thresh in [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]:
        sell_mask = (preds == 0) & (max_probs >= thresh)
        n_trades = sell_mask.sum()
        if n_trades == 0:
            continue
        sell_correct = (y_holdout[sell_mask] == 0).sum()
        precision = sell_correct / n_trades
        be_rate = sl_pips / (tp_pips + sl_pips)
        exp_pips = precision * tp_pips - (1 - precision) * sl_pips
        sell_results[thresh] = {
            'trades': n_trades,
            'precision': precision,
            'exp_pips': exp_pips,
            'total_pips': exp_pips * n_trades,
            'above_be': precision > be_rate,
        }

    # Find best threshold (most total pips with positive expectancy)
    best_thresh = None
    best_total = -999999
    for thresh, res in sell_results.items():
        if res['exp_pips'] > 0 and res['total_pips'] > best_total:
            best_total = res['total_pips']
            best_thresh = thresh

    # Prob range for sell predictions
    sell_pred_mask = preds == 0
    sell_prob_range = ""
    if sell_pred_mask.sum() > 0:
        sell_probs = probas[sell_pred_mask, 0]
        sell_prob_range = f"[{sell_probs.min():.3f}, {sell_probs.max():.3f}]"

    return {
        'tp': tp_pips,
        'sl': sl_pips,
        'rr': tp_pips / sl_pips,
        'sell_pct': sell_pct,
        'range_pct': range_pct,
        'buy_pct': buy_pct,
        'holdout_acc': holdout_acc,
        'sell_prob_range': sell_prob_range,
        'sell_results': sell_results,
        'best_thresh': best_thresh,
        'best_total': best_total,
        'be_rate': sl_pips / (tp_pips + sl_pips),
    }


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 90)
    print("BARRIER SWEEP — Finding optimal TP/SL for Sell signal")
    print("=" * 90)

    # Load and feature-engineer data once
    config = get_config()
    loader = DataLoader(config._config)
    validator = DataValidator(config._config)
    fe = FeatureEngineer(config._config)

    raw_path = config.get("paths.data_raw_file", "EURUSD_H1_raw.csv")
    df = loader.load(raw_path)
    validator.validate_ohlc(df)
    df = fe.engineer(df)

    feature_cols = FeatureEngineer.default_feature_columns()
    # Load pruned features if available
    features_path = Path("features_used.json")
    if features_path.exists():
        with open(features_path) as f:
            saved_features = json.load(f)
        # Use saved features if they're all present
        if all(f in df.columns for f in saved_features):
            feature_cols = saved_features

    print(f"Data: {len(df)} rows, {len(feature_cols)} features")

    # TP/SL combinations to test
    combos = [
        # Symmetric
        (10, 10), (15, 15), (20, 20),
        # Current
        (25, 15),
        # Tighter SL (higher R:R, fewer Sell labels)
        (20, 10), (15, 10), (25, 10), (30, 10),
        # Wider SL (lower R:R, more Sell labels, higher precision needed)
        (20, 15), (15, 15), (20, 20),
        # Small moves (more labels, lower R:R requirement)
        (10, 8), (12, 8), (15, 8),
        # Large moves (fewer labels, need very high precision)
        (30, 15), (30, 20),
    ]
    # Deduplicate
    combos = list(dict.fromkeys(combos))

    print(f"Testing {len(combos)} TP/SL combinations...\n")

    results = []
    for tp, sl in combos:
        print(f"  Testing TP={tp}, SL={sl} (R:R={tp/sl:.1f}:1)...", end=" ", flush=True)
        res = evaluate_combo(df, feature_cols, tp, sl)
        results.append(res)

        n_profitable = sum(1 for r in res['sell_results'].values() if r['above_be'])
        if res['best_thresh']:
            best = res['sell_results'][res['best_thresh']]
            print(f"acc={res['holdout_acc']:.1%}, best_sell: {best['trades']}trades "
                  f"@ {best['precision']:.1%} prec, {best['total_pips']:+.0f} pips")
        else:
            print(f"acc={res['holdout_acc']:.1%}, no profitable Sell threshold")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 90)
    print("SUMMARY — All TP/SL combinations ranked by best total pips")
    print("=" * 90)
    print(f"{'TP':>4} {'SL':>4} {'R:R':>5} | {'Sell%':>5} {'Rng%':>5} {'Buy%':>5} | "
          f"{'Acc':>5} | {'BE%':>5} | {'BestTh':>6} {'Trades':>6} {'Prec':>6} {'ExpPip':>7} {'TotPip':>8} | {'SellProbRange':>16}")
    print("-" * 110)

    # Sort by best_total descending
    results.sort(key=lambda r: r['best_total'], reverse=True)

    for res in results:
        if res['best_thresh'] and res['best_total'] > 0:
            best = res['sell_results'][res['best_thresh']]
            marker = " <<<"
            print(f"{res['tp']:4.0f} {res['sl']:4.0f} {res['rr']:5.1f} | "
                  f"{res['sell_pct']:5.1%} {res['range_pct']:5.1%} {res['buy_pct']:5.1%} | "
                  f"{res['holdout_acc']:5.1%} | {res['be_rate']:5.1%} | "
                  f"{res['best_thresh']:6.2f} {best['trades']:5d}  {best['precision']:5.1%} "
                  f"{best['exp_pips']:+6.2f}  {best['total_pips']:+7.0f}  | "
                  f"{res['sell_prob_range']:>16}{marker}")
        else:
            print(f"{res['tp']:4.0f} {res['sl']:4.0f} {res['rr']:5.1f} | "
                  f"{res['sell_pct']:5.1%} {res['range_pct']:5.1%} {res['buy_pct']:5.1%} | "
                  f"{res['holdout_acc']:5.1%} | {res['be_rate']:5.1%} | "
                  f"{'---':>6} {'---':>6} {'---':>6} {'---':>7} {'---':>8} | "
                  f"{res['sell_prob_range']:>16}")

    # === DETAILED SELL ANALYSIS for top 3 ===
    print("\n" + "=" * 90)
    print("DETAILED SELL THRESHOLD ANALYSIS — Top 3 combos")
    print("=" * 90)

    for i, res in enumerate(results[:3]):
        if not res['sell_results']:
            continue
        be = res['be_rate'] * 100
        print(f"\n  TP={res['tp']}, SL={res['sl']} (R:R={res['rr']:.1f}:1, BE={be:.1f}%)")
        print(f"  {'Thresh':>7} | {'Trades':>6} | {'Precision':>9} | {'ExpPips':>7} | {'TotalPips':>9}")
        print(f"  " + "-" * 50)
        for thresh in sorted(res['sell_results'].keys()):
            r = res['sell_results'][thresh]
            marker = " ***" if r['above_be'] else ""
            print(f"  {thresh:7.2f} | {r['trades']:5d}  | {r['precision']:8.1%}  | "
                  f"{r['exp_pips']:+6.2f}  | {r['total_pips']:+8.0f}{marker}")

    print("\n" + "=" * 90)
    print(f"Results saved to {LOG_FILE}")
    print("=" * 90)

    tee.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
