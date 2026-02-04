#!/usr/bin/env python3
"""
Regime Analysis: Does Sell precision improve in specific market regimes?

Uses the existing trained model and holdout data — no retraining needed.
Splits holdout predictions by regime features (ADX, choppiness, squeeze,
ATR) and checks if Sell precision concentrates in certain conditions.

Output saved to logs/regime_analysis.log
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.config import get_config

LOG_FILE = Path(__file__).parent / "logs" / "regime_analysis.log"
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


def analyze_regime(holdout, y_true, preds, max_probs, feature_name, feature_values,
                   tp_pips, sl_pips, n_bins=4, sell_thresh=0.34):
    """
    Split holdout by a regime feature into quantile bins.
    For each bin, report Sell precision and trade count.
    """
    be_rate = sl_pips / (tp_pips + sl_pips)

    # Use quantile bins for even distribution
    try:
        bins = pd.qcut(feature_values, q=n_bins, duplicates='drop')
    except ValueError:
        bins = pd.cut(feature_values, bins=n_bins, duplicates='drop')

    bin_labels = bins.cat.categories

    print(f"\n  {'Regime Bin':>25} | {'Samples':>7} | {'SellSig':>7} | {'TruePos':>7} | "
          f"{'Prec':>6} | {'ExpPip':>7} | {'TotPip':>8} | {'vs BE':>6}")
    print(f"  {'-' * 90}")

    bin_results = []

    for b in bin_labels:
        mask = bins == b
        n_samples = mask.sum()

        # Sell signals in this regime
        sell_mask = mask & (preds == 0) & (max_probs >= sell_thresh)
        n_sell = sell_mask.sum()

        if n_sell == 0:
            print(f"  {str(b):>25} | {n_samples:6d}  | {0:6d}  | {'---':>7} | "
                  f"{'---':>6} | {'---':>7} | {'---':>8} | {'---':>6}")
            continue

        true_pos = (sell_mask & (y_true == 0)).sum()
        precision = true_pos / n_sell
        exp_pips = precision * tp_pips - (1 - precision) * sl_pips
        total_pips = exp_pips * n_sell
        vs_be = precision - be_rate

        marker = "+" if precision > be_rate else "-"

        print(f"  {str(b):>25} | {n_samples:6d}  | {n_sell:6d}  | {true_pos:6d}  | "
              f"{precision:5.1%} | {exp_pips:+6.2f} | {total_pips:+7.0f}  | {vs_be:+5.1%} {marker}")

        bin_results.append({
            'bin': str(b),
            'samples': n_samples,
            'sell_signals': n_sell,
            'precision': precision,
            'exp_pips': exp_pips,
            'total_pips': total_pips,
            'above_be': precision > be_rate,
        })

    return bin_results


def analyze_combined_regimes(holdout, y_true, preds, max_probs,
                             tp_pips, sl_pips, sell_thresh=0.34):
    """
    Test combined regime filters: pairs and triples of conditions.
    """
    be_rate = sl_pips / (tp_pips + sl_pips)

    # Define regime conditions using median splits
    conditions = {}

    # ADX: high = trending
    if 'adx_h4' in holdout.columns:
        med = holdout['adx_h4'].median()
        conditions['adx_h4_high'] = holdout['adx_h4'] >= med
        conditions['adx_h4_low'] = holdout['adx_h4'] < med

    # Choppiness: low = trending
    if 'choppiness_h1' in holdout.columns:
        med = holdout['choppiness_h1'].median()
        conditions['chop_h1_low'] = holdout['choppiness_h1'] < med
        conditions['chop_h1_high'] = holdout['choppiness_h1'] >= med

    if 'choppiness_h4' in holdout.columns:
        med = holdout['choppiness_h4'].median()
        conditions['chop_h4_low'] = holdout['choppiness_h4'] < med
        conditions['chop_h4_high'] = holdout['choppiness_h4'] >= med

    # Squeeze: low = trending (BB inside KC)
    if 'squeeze_ratio_h1' in holdout.columns:
        med = holdout['squeeze_ratio_h1'].median()
        conditions['squeeze_h1_low'] = holdout['squeeze_ratio_h1'] < med
        conditions['squeeze_h1_high'] = holdout['squeeze_ratio_h1'] >= med

    # ATR: high = volatile
    if 'atr_ratio_h1' in holdout.columns:
        med = holdout['atr_ratio_h1'].median()
        conditions['atr_h1_high'] = holdout['atr_ratio_h1'] >= med
        conditions['atr_h1_low'] = holdout['atr_ratio_h1'] < med

    if 'atr_ratio_h4' in holdout.columns:
        med = holdout['atr_ratio_h4'].median()
        conditions['atr_h4_high'] = holdout['atr_ratio_h4'] >= med
        conditions['atr_h4_low'] = holdout['atr_ratio_h4'] < med

    # RSI: extreme zones
    if 'rsi_h1' in holdout.columns:
        conditions['rsi_h1_low'] = holdout['rsi_h1'] < 40
        conditions['rsi_h1_mid'] = (holdout['rsi_h1'] >= 40) & (holdout['rsi_h1'] <= 60)
        conditions['rsi_h1_high'] = holdout['rsi_h1'] > 60

    # EMA trend: bearish for Sell
    if 'ema50_ema200_h1' in holdout.columns:
        conditions['ema_h1_bearish'] = holdout['ema50_ema200_h1'] < 0
        conditions['ema_h1_bullish'] = holdout['ema50_ema200_h1'] >= 0

    if 'ema50_ema200_h4' in holdout.columns:
        conditions['ema_h4_bearish'] = holdout['ema50_ema200_h4'] < 0
        conditions['ema_h4_bullish'] = holdout['ema50_ema200_h4'] >= 0

    sell_base = (preds == 0) & (max_probs >= sell_thresh)

    # === Single conditions ===
    print(f"\n  {'Condition':>25} | {'Sells':>6} | {'TruePos':>7} | {'Prec':>6} | "
          f"{'ExpPip':>7} | {'TotPip':>8} | {'vs BE':>6}")
    print(f"  {'-' * 80}")

    # Baseline
    n_sell = sell_base.sum()
    tp = (sell_base & (y_true == 0)).sum()
    prec = tp / n_sell if n_sell > 0 else 0
    exp = prec * tp_pips - (1 - prec) * sl_pips
    print(f"  {'BASELINE (no filter)':>25} | {n_sell:5d}  | {tp:6d}  | {prec:5.1%} | "
          f"{exp:+6.2f} | {exp * n_sell:+7.0f}  | {'---':>6}")

    single_results = []
    for name, mask in conditions.items():
        filtered = sell_base & mask
        n_sell = filtered.sum()
        if n_sell < 20:
            continue

        tp = (filtered & (y_true == 0)).sum()
        prec = tp / n_sell
        exp = prec * tp_pips - (1 - prec) * sl_pips
        total = exp * n_sell
        vs_be = prec - be_rate

        marker = "+" if prec > be_rate else "-"
        print(f"  {name:>25} | {n_sell:5d}  | {tp:6d}  | {prec:5.1%} | "
              f"{exp:+6.2f} | {total:+7.0f}  | {vs_be:+5.1%} {marker}")

        single_results.append({
            'name': name, 'sells': n_sell, 'precision': prec,
            'exp_pips': exp, 'total_pips': total, 'above_be': prec > be_rate,
        })

    # === Best pairs ===
    print(f"\n  Top condition pairs (combining two filters):")
    print(f"  {'Condition A + B':>40} | {'Sells':>6} | {'Prec':>6} | "
          f"{'ExpPip':>7} | {'TotPip':>8}")
    print(f"  {'-' * 80}")

    pair_results = []
    cond_names = list(conditions.keys())

    for i in range(len(cond_names)):
        for j in range(i + 1, len(cond_names)):
            name_a, name_b = cond_names[i], cond_names[j]
            combined = sell_base & conditions[name_a] & conditions[name_b]
            n_sell = combined.sum()
            if n_sell < 20:
                continue

            tp = (combined & (y_true == 0)).sum()
            prec = tp / n_sell
            exp = prec * tp_pips - (1 - prec) * sl_pips
            total = exp * n_sell

            pair_results.append({
                'name': f"{name_a} + {name_b}",
                'sells': n_sell, 'precision': prec,
                'exp_pips': exp, 'total_pips': total,
            })

    # Sort by total pips and show top 15
    pair_results.sort(key=lambda r: r['total_pips'], reverse=True)
    for res in pair_results[:15]:
        marker = " ***" if res['precision'] > be_rate else ""
        print(f"  {res['name']:>40} | {res['sells']:5d}  | {res['precision']:5.1%} | "
              f"{res['exp_pips']:+6.2f} | {res['total_pips']:+7.0f}{marker}")

    return single_results, pair_results


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 90)
    print("REGIME ANALYSIS — Where does the Sell edge concentrate?")
    print("=" * 90)

    config = get_config()
    tp_pips = float(config.get("training.tp_pips", 20.0))
    sl_pips = float(config.get("training.sl_pips", 15.0))
    be_rate = sl_pips / (tp_pips + sl_pips)
    sell_thresh = 0.34

    data_file = config.get("paths.data_processed_file", "EURUSD_H1_clean.csv")
    model_path = config.get("model.path", "xgb_eurusd_h1.pkl")

    df = pd.read_csv(data_file)
    model = joblib.load(model_path)

    with open("features_used.json") as f:
        features = json.load(f)

    # Holdout split
    train_ratio = float(config.get("training.train_ratio", 0.6))
    val_ratio = float(config.get("training.val_ratio", 0.2))
    n = len(df)
    val_end = int(n * (train_ratio + val_ratio))
    holdout = df.iloc[val_end:].copy()

    X = holdout[features].values
    y_true = pd.Series(holdout["label"]).map({-1: 0, 0: 1, 1: 2}).values

    probas = model.predict_proba(X)
    preds = np.argmax(probas, axis=1)
    max_probs = np.max(probas, axis=1)

    sell_mask = (preds == 0) & (max_probs >= sell_thresh)
    total_sell = sell_mask.sum()
    total_tp = (sell_mask & (y_true == 0)).sum()
    baseline_prec = total_tp / total_sell if total_sell > 0 else 0

    print(f"TP={tp_pips}, SL={sl_pips}, BE={be_rate:.1%}")
    print(f"Holdout: {len(holdout)} bars, {total_sell} Sell signals, "
          f"baseline precision={baseline_prec:.1%}")

    # === PER-FEATURE REGIME ANALYSIS ===
    regime_features = [
        ('adx_h4', 'ADX H4 (trend strength)'),
        ('choppiness_h1', 'Choppiness H1 (low=trending)'),
        ('choppiness_h4', 'Choppiness H4 (low=trending)'),
        ('squeeze_ratio_h1', 'Squeeze H1 (low=tight BB/KC)'),
        ('squeeze_ratio_h4', 'Squeeze H4 (low=tight BB/KC)'),
        ('atr_ratio_h1', 'ATR Ratio H1 (volatility)'),
        ('atr_ratio_h4', 'ATR Ratio H4 (volatility)'),
        ('rsi_h1', 'RSI H1'),
        ('ema50_ema200_h1', 'EMA Cross H1 (neg=bearish)'),
        ('ema50_ema200_h4', 'EMA Cross H4 (neg=bearish)'),
        ('body_ratio_h1', 'Body Ratio H1'),
        ('body_ratio_h4', 'Body Ratio H4'),
    ]

    print("\n" + "=" * 90)
    print("PART 1: PER-FEATURE REGIME SPLITS (quartile bins)")
    print("=" * 90)

    for feat_col, feat_name in regime_features:
        if feat_col not in holdout.columns:
            continue
        print(f"\n--- {feat_name} ---")
        analyze_regime(holdout, y_true, preds, max_probs,
                       feat_col, holdout[feat_col], tp_pips, sl_pips,
                       n_bins=4, sell_thresh=sell_thresh)

    # === COMBINED REGIME FILTERS ===
    print("\n" + "=" * 90)
    print("PART 2: COMBINED REGIME FILTERS")
    print("=" * 90)

    single_results, pair_results = analyze_combined_regimes(
        holdout, y_true, preds, max_probs, tp_pips, sl_pips, sell_thresh
    )

    # === BEST REGIME FILTER SUMMARY ===
    print("\n" + "=" * 90)
    print("SUMMARY — Best regime filters")
    print("=" * 90)

    profitable_singles = [r for r in single_results if r['above_be']]
    profitable_singles.sort(key=lambda r: r['total_pips'], reverse=True)

    print(f"\n  Baseline: {baseline_prec:.1%} precision, {total_sell} trades")
    print(f"  Break-even: {be_rate:.1%}\n")

    if profitable_singles:
        print(f"  Best single filters (above BE):")
        for r in profitable_singles[:5]:
            print(f"    {r['name']:>25}: {r['precision']:.1%} prec, "
                  f"{r['sells']} trades, {r['total_pips']:+.0f} pips")

    profitable_pairs = [r for r in pair_results if r['precision'] > be_rate]
    if profitable_pairs:
        print(f"\n  Best combined filters (above BE):")
        for r in profitable_pairs[:5]:
            print(f"    {r['name']:>40}: {r['precision']:.1%} prec, "
                  f"{r['sells']} trades, {r['total_pips']:+.0f} pips")

    # Check if any filter meaningfully improves over baseline
    best_single = profitable_singles[0] if profitable_singles else None
    best_pair = profitable_pairs[0] if profitable_pairs else None

    print(f"\n  VERDICT:")
    if best_pair and best_pair['precision'] > baseline_prec + 0.05:
        print(f"    Regime filtering adds significant edge!")
        print(f"    Best filter: {best_pair['name']}")
        print(f"    Precision: {baseline_prec:.1%} → {best_pair['precision']:.1%} "
              f"(+{best_pair['precision'] - baseline_prec:.1%})")
        print(f"    Trade-off: {total_sell} → {best_pair['sells']} trades "
              f"({best_pair['sells']/total_sell:.0%} retained)")
    elif best_single and best_single['precision'] > baseline_prec + 0.03:
        print(f"    Moderate improvement with regime filter.")
        print(f"    Best filter: {best_single['name']}")
        print(f"    Precision: {baseline_prec:.1%} → {best_single['precision']:.1%}")
    else:
        print(f"    No regime filter significantly improves Sell precision.")
        print(f"    The edge is diffuse across regimes, not concentrated.")

    print("\n" + "=" * 90)
    print(f"Results saved to {LOG_FILE}")
    print("=" * 90)

    tee.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
