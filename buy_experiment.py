#!/usr/bin/env python3
"""
Buy Model Experiment: Does a dedicated binary Buy-vs-Not-Buy model have an edge?

Sweeps both TP/SL barriers AND model params to give Buy a fair shot.
The Sell model found its edge through barrier + param tuning, so Buy
deserves the same treatment before we conclude anything.

Output saved to logs/buy_experiment.log
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from itertools import product
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.config import get_config
from src.data_collection.loader import DataLoader
from src.data_collection.validator import DataValidator
from src.features.engineer import FeatureEngineer

LOG_FILE = Path(__file__).parent / "logs" / "buy_experiment.log"
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

        tp_up = np.where(ret_high >= tp_th)[0]
        sl_up = np.where(ret_low <= -sl_th)[0]
        first_up_tp = tp_up[0] if len(tp_up) > 0 else max_horizon
        first_up_sl = sl_up[0] if len(sl_up) > 0 else max_horizon

        tp_dn = np.where(ret_low <= -tp_th)[0]
        sl_dn = np.where(ret_high >= sl_th)[0]
        first_dn_tp = tp_dn[0] if len(tp_dn) > 0 else max_horizon
        first_dn_sl = sl_dn[0] if len(sl_dn) > 0 else max_horizon

        if first_up_tp < first_up_sl and first_up_tp < max_horizon:
            labels[i] = 1
        elif first_dn_tp < first_dn_sl and first_dn_tp < max_horizon:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels


def evaluate_combo(df, feature_cols, tp_pips, sl_pips, params, max_horizon=24):
    """
    Train binary Buy model with given TP/SL and params, evaluate on holdout.
    Returns dict with results or None if no edge found.
    """
    labels = triple_barrier_label(df, tp_pips, sl_pips, max_horizon)
    df_work = df.copy()
    df_work["label"] = labels
    df_work = df_work.dropna(subset=feature_cols).iloc[:-1].copy()

    X = df_work[feature_cols].values
    y = (df_work["label"] == 1).astype(int).values

    buy_rate = y.mean()
    be_rate = sl_pips / (tp_pips + sl_pips)

    # 3-way split
    n = len(X)
    val_end = int(n * 0.8)
    X_train_val = X[:val_end]
    y_train_val = y[:val_end]
    X_holdout = X[val_end:]
    y_holdout = y[val_end:]

    # Class-balanced weights
    try:
        classes = np.array([0, 1])
        cw = compute_class_weight("balanced", classes=classes, y=y_train_val)
        sw = np.array([cw[label] for label in y_train_val])
    except Exception:
        sw = None

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_val, y_train_val, verbose=False, sample_weight=sw)

    probas = model.predict_proba(X_holdout)
    buy_probs = probas[:, 1]
    preds = (buy_probs >= 0.5).astype(int)

    holdout_acc = (preds == y_holdout).mean()

    # Threshold sweep
    total_buy = y_holdout.sum()
    best_thresh = None
    best_total = -999999
    thresh_results = {}

    for thresh in [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.45, 0.50, 0.55, 0.60]:
        signals = buy_probs >= thresh
        n_trades = signals.sum()
        if n_trades == 0:
            continue

        true_pos = (signals & (y_holdout == 1)).sum()
        precision = true_pos / n_trades
        recall = true_pos / total_buy if total_buy > 0 else 0
        exp_pips = precision * tp_pips - (1 - precision) * sl_pips
        total_pips = exp_pips * n_trades

        thresh_results[thresh] = {
            'trades': n_trades, 'precision': precision, 'recall': recall,
            'exp_pips': exp_pips, 'total_pips': total_pips,
            'above_be': precision > be_rate,
        }

        if exp_pips > 0 and total_pips > best_total:
            best_total = total_pips
            best_thresh = thresh

    return {
        'tp': tp_pips, 'sl': sl_pips, 'rr': tp_pips / sl_pips,
        'buy_rate': buy_rate, 'be_rate': be_rate, 'holdout_acc': holdout_acc,
        'prob_range': f"[{buy_probs.min():.4f}, {buy_probs.max():.4f}]",
        'prob_spread': buy_probs.max() - buy_probs.min(),
        'best_thresh': best_thresh, 'best_total': best_total,
        'thresh_results': thresh_results, 'params': params,
    }


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 90)
    print("BUY MODEL EXPERIMENT — Sweep barriers AND params")
    print("=" * 90)

    config = get_config()
    loader = DataLoader(config._config)
    validator = DataValidator(config._config)
    fe = FeatureEngineer(config._config)

    raw_path = config.get("paths.data_raw_file", "EURUSD_H1_raw.csv")
    df = loader.load(raw_path)
    validator.validate_ohlc(df)
    df = fe.engineer(df)

    features_path = Path("features_used.json")
    if features_path.exists():
        with open(features_path) as f:
            feature_cols = json.load(f)
        if not all(f in df.columns for f in feature_cols):
            feature_cols = FeatureEngineer.default_feature_columns()
    else:
        feature_cols = FeatureEngineer.default_feature_columns()

    print(f"Data: {len(df)} rows, {len(feature_cols)} features\n")

    # TP/SL combos
    barrier_combos = [
        (20, 15), (15, 10), (20, 10), (15, 15),
        (25, 15), (10, 8), (12, 8),
    ]

    # Param configs to sweep — from heavily regularized to more expressive
    param_configs = {
        "heavy_reg": {
            "max_depth": 3, "n_estimators": 80, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 40,
            "gamma": 0.8, "reg_alpha": 50.0, "reg_lambda": 50.0,
        },
        "moderate_reg": {
            "max_depth": 3, "n_estimators": 100, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 30,
            "gamma": 0.5, "reg_alpha": 20.0, "reg_lambda": 20.0,
        },
        "light_reg": {
            "max_depth": 3, "n_estimators": 120, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 20,
            "gamma": 0.3, "reg_alpha": 5.0, "reg_lambda": 5.0,
        },
        "shallow_wide": {
            "max_depth": 2, "n_estimators": 150, "learning_rate": 0.03,
            "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 50,
            "gamma": 1.0, "reg_alpha": 30.0, "reg_lambda": 30.0,
        },
        "deeper": {
            "max_depth": 4, "n_estimators": 60, "learning_rate": 0.02,
            "subsample": 0.6, "colsample_bytree": 0.5, "min_child_weight": 40,
            "gamma": 1.0, "reg_alpha": 80.0, "reg_lambda": 80.0,
        },
    }

    total_combos = len(barrier_combos) * len(param_configs)
    print(f"Testing {len(barrier_combos)} barriers x {len(param_configs)} param configs = {total_combos} combinations\n")

    all_results = []
    count = 0

    for tp, sl in barrier_combos:
        for pname, pconfig in param_configs.items():
            count += 1
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
                **pconfig,
            }

            sys.stdout.flush()
            print(f"  [{count}/{total_combos}] TP={tp}, SL={sl}, params={pname}...", end=" ", flush=True)

            res = evaluate_combo(df, feature_cols, tp, sl, params)
            res['param_name'] = pname
            all_results.append(res)

            if res['best_thresh']:
                best = res['thresh_results'][res['best_thresh']]
                print(f"acc={res['holdout_acc']:.1%}, prob={res['prob_range']}, "
                      f"best: {best['trades']}tr @ {best['precision']:.1%} = {best['total_pips']:+.0f}pip")
            else:
                print(f"acc={res['holdout_acc']:.1%}, prob={res['prob_range']}, no edge")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 90)
    print("SUMMARY — All combinations ranked by best total pips")
    print("=" * 90)
    print(f"{'TP':>4} {'SL':>4} {'R:R':>5} {'Params':>14} | {'Buy%':>5} {'BE%':>5} {'Acc':>5} | "
          f"{'Spread':>7} | {'BestTh':>6} {'Trades':>6} {'Prec':>6} {'TotPip':>8}")
    print("-" * 95)

    all_results.sort(key=lambda r: r['best_total'], reverse=True)

    profitable = []
    for res in all_results:
        if res['best_thresh'] and res['best_total'] > 0:
            best = res['thresh_results'][res['best_thresh']]
            marker = " <<<"
            profitable.append(res)
            print(f"{res['tp']:4.0f} {res['sl']:4.0f} {res['rr']:5.1f} {res['param_name']:>14} | "
                  f"{res['buy_rate']:5.1%} {res['be_rate']:5.1%} {res['holdout_acc']:5.1%} | "
                  f"{res['prob_spread']:7.4f} | "
                  f"{res['best_thresh']:6.2f} {best['trades']:5d}  {best['precision']:5.1%} "
                  f"{best['total_pips']:+7.0f}{marker}")
        else:
            print(f"{res['tp']:4.0f} {res['sl']:4.0f} {res['rr']:5.1f} {res['param_name']:>14} | "
                  f"{res['buy_rate']:5.1%} {res['be_rate']:5.1%} {res['holdout_acc']:5.1%} | "
                  f"{res['prob_spread']:7.4f} | "
                  f"{'---':>6} {'---':>6} {'---':>6} {'---':>8}")

    # === DETAILED ANALYSIS for profitable combos ===
    if profitable:
        print("\n" + "=" * 90)
        print(f"DETAILED THRESHOLD ANALYSIS — {len(profitable)} profitable combos")
        print("=" * 90)

        for res in profitable[:5]:
            be = res['be_rate'] * 100
            print(f"\n  TP={res['tp']}, SL={res['sl']}, params={res['param_name']} "
                  f"(R:R={res['rr']:.1f}:1, BE={be:.1f}%)")
            print(f"  {'Thresh':>7} | {'Trades':>6} | {'Precision':>9} | {'ExpPips':>7} | {'TotalPips':>9}")
            print(f"  " + "-" * 55)
            for thresh in sorted(res['thresh_results'].keys()):
                r = res['thresh_results'][thresh]
                marker = " ***" if r['above_be'] else ""
                print(f"  {thresh:7.2f} | {r['trades']:5d}  | {r['precision']:8.1%}  | "
                      f"{r['exp_pips']:+6.2f}  | {r['total_pips']:+8.0f}{marker}")

    # === VERDICT ===
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    if profitable:
        best = profitable[0]
        best_t = best['thresh_results'][best['best_thresh']]
        print(f"  Best Buy combo: TP={best['tp']}, SL={best['sl']}, params={best['param_name']}")
        print(f"    Threshold={best['best_thresh']}, Trades={best_t['trades']}, "
              f"Precision={best_t['precision']:.1%}, Total={best_t['total_pips']:+.0f} pips")
        print(f"  Sell model comparison: TP=20/SL=15, thresh=0.34, +222 pips (TP/SL sim)")

        if best['best_total'] > 200:
            print(f"\n  Buy has meaningful edge — dual model architecture is viable")
        elif best['best_total'] > 0:
            print(f"\n  Buy edge is marginal ({best['best_total']:+.0f} pips) — may not survive real trading costs")
        else:
            print(f"\n  Buy edge is not reliable")
    else:
        print(f"  No profitable Buy configuration found across {total_combos} combinations.")
        print(f"  Buy signal does not exist with current features regardless of params.")
        print(f"  Recommendation: Stay Sell-only.")

    print("\n" + "=" * 90)
    print(f"Results saved to {LOG_FILE}")
    print("=" * 90)

    tee.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
