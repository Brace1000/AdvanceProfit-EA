#!/usr/bin/env python3
"""
Train Buy Model for EUR/USD H1

Learning from Sell model experience:
1. Same 14 features work well for direction prediction
2. Triple barrier labeling (inverse: price UP before DOWN)
3. Same hyperparameters as starting point
4. Walk-forward validation to ensure real edge
5. Test specifically during bullish periods (where Sell failed)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import get_config
from src.data_collection.loader import DataLoader
from src.features.engineer import FeatureEngineer

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# Log file setup
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "train_buy_model.log"

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

sys.stdout = Tee(log_file)


def create_buy_labels(df, tp_pips=20, sl_pips=15, max_holding=50):
    """
    Create triple barrier labels for BUY direction.

    Label 0: Buy wins (price goes UP tp_pips before DOWN sl_pips)
    Label 1: Range (neither barrier hit in max_holding bars)
    Label 2: Buy loses / Sell wins (price goes DOWN sl_pips first)
    """
    labels = []
    close = df['close'].values

    for i in range(len(df)):
        if i + max_holding >= len(df):
            labels.append(1)  # Not enough data, mark as range
            continue

        entry_price = close[i]
        tp_price = entry_price + tp_pips * 0.0001  # BUY: TP is above
        sl_price = entry_price - sl_pips * 0.0001  # BUY: SL is below

        label = 1  # Default: range

        for j in range(1, max_holding + 1):
            future_price = close[i + j]

            # Check TP first (price went UP)
            if future_price >= tp_price:
                label = 0  # Buy wins
                break
            # Check SL (price went DOWN)
            elif future_price <= sl_price:
                label = 2  # Buy loses
                break

        labels.append(label)

    return np.array(labels)


def analyze_label_distribution(labels, df, dt_col=None):
    """Analyze label distribution and timing."""
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 60)

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f"\nTotal samples: {total}")
    for u, c in zip(unique, counts):
        label_name = {0: "Buy wins", 1: "Range", 2: "Buy loses"}[u]
        print(f"  Label {u} ({label_name}): {c} ({c/total*100:.1f}%)")

    # Buy win rate (what we need for profitable trading)
    buy_wins = counts[0] if 0 in unique else 0
    buy_loses = counts[2] if 2 in unique else 0
    buy_trades = buy_wins + buy_loses

    if buy_trades > 0:
        base_win_rate = buy_wins / buy_trades * 100
        print(f"\nBase Buy win rate (excluding range): {base_win_rate:.1f}%")
        print(f"  Breakeven needed for TP=20/SL=15: 42.9%")
        if base_win_rate > 42.9:
            print(f"  --> ABOVE breakeven by {base_win_rate - 42.9:.1f}%")
        else:
            print(f"  --> BELOW breakeven by {42.9 - base_win_rate:.1f}%")


def walk_forward_validation(df, feature_cols, n_windows=5):
    """
    Walk-forward validation for Buy model.
    Train on past, test on future, measure real edge.
    """
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    n = len(df)
    test_size = n // (n_windows + 1)
    min_train = test_size

    dt_col = "datetime" if "datetime" in df.columns else ("time" if "time" in df.columns else None)

    all_results = []

    print(f"\nData: {n} rows")
    print(f"Windows: {n_windows}, ~{test_size} bars per test")
    print()

    print(f"{'Win':>3} | {'Period':>25} | {'Trades':>7} | {'Wins':>6} | {'WinR':>7} | {'Pips':>9}")
    print("-" * 75)

    for w in range(n_windows):
        train_end = min_train + w * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_start >= n or test_end <= test_start:
            break

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if len(train_df) < 100 or len(test_df) < 100:
            continue

        # Period string
        if dt_col:
            start_date = str(test_df.iloc[0][dt_col])[:10]
            end_date = str(test_df.iloc[-1][dt_col])[:10]
            period = f"{start_date} to {end_date}"
        else:
            period = f"bars {test_start}-{test_end}"

        # Create labels for training data
        train_labels = create_buy_labels(train_df)
        test_labels = create_buy_labels(test_df)

        # Prepare features
        X_train = train_df[feature_cols].values
        y_train = train_labels
        X_test = test_df[feature_cols].values
        y_test = test_labels

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        # Predict
        probas = model.predict_proba(X_test)
        buy_probs = probas[:, 0]  # Probability of Buy winning

        # Simple threshold strategy (same as Sell model)
        buy_thresh = 0.34
        buy_signals = buy_probs >= buy_thresh

        # Calculate results
        n_signals = buy_signals.sum()

        if n_signals > 0:
            signal_labels = y_test[buy_signals]
            wins = (signal_labels == 0).sum()  # Buy wins
            losses = (signal_labels == 2).sum()  # Buy loses
            trades = wins + losses

            if trades > 0:
                win_rate = wins / trades * 100
                # Pips: wins * 20 - losses * 15
                pips = wins * 20 - losses * 15
            else:
                win_rate = 0
                pips = 0
        else:
            trades = 0
            wins = 0
            win_rate = 0
            pips = 0

        print(f"W{w+1:>2} | {period:>25} | {trades:>7} | {wins:>6} | {win_rate:>6.1f}% | {pips:>+8}")

        all_results.append({
            'window': w + 1,
            'period': period,
            'trades': trades,
            'wins': wins,
            'win_rate': win_rate,
            'pips': pips
        })

    # Aggregate results
    print("-" * 75)
    total_trades = sum(r['trades'] for r in all_results)
    total_wins = sum(r['wins'] for r in all_results)
    total_pips = sum(r['pips'] for r in all_results)

    if total_trades > 0:
        overall_wr = total_wins / total_trades * 100
    else:
        overall_wr = 0

    print(f"{'TOTAL':>3} | {'-':>25} | {total_trades:>7} | {total_wins:>6} | {overall_wr:>6.1f}% | {total_pips:>+8}")

    return all_results, total_pips


def train_final_model(df, feature_cols, labels):
    """Train final model on all data."""
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    X = df[feature_cols].values
    y = labels

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        verbosity=0
    )

    model.fit(X, y)

    # Feature importance
    print("\nFeature Importance:")
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  {i+1}. {feature_cols[idx]}: {importance[idx]:.4f}")

    return model


def main():
    config = get_config()

    print("=" * 70)
    print("BUY MODEL TRAINING PIPELINE")
    print("=" * 70)
    print("\nLearning from Sell model:")
    print("  - Same 14 features (proven for direction)")
    print("  - Triple barrier labeling (inverse direction)")
    print("  - Walk-forward validation (no look-ahead bias)")
    print("  - 34% confidence threshold (tested)")
    print()

    # Load data
    print("Loading data...")
    loader = DataLoader(config._config)
    raw_path = project_root / config.get("paths.data_raw_file", "EURUSD_H1_clean.csv")
    df = loader.load(raw_path)

    fe = FeatureEngineer(config._config)
    df = fe.engineer(df)

    # Load feature columns (same as Sell model)
    features_path = project_root / "features_used.json"
    with open(features_path) as f:
        feature_cols = json.load(f)

    print(f"Features: {len(feature_cols)}")
    print(f"Data rows: {len(df)}")

    # Drop NaN
    df = df.dropna(subset=feature_cols).copy()
    print(f"After dropna: {len(df)} rows")

    # Create Buy labels
    print("\nCreating Buy labels (TP=20 up, SL=15 down)...")
    labels = create_buy_labels(df, tp_pips=20, sl_pips=15, max_holding=50)
    df['buy_label'] = labels

    # Analyze distribution
    dt_col = "datetime" if "datetime" in df.columns else None
    analyze_label_distribution(labels, df, dt_col)

    # Walk-forward validation
    results, total_pips = walk_forward_validation(df, feature_cols, n_windows=5)

    # Decision: proceed with training?
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if total_pips > 0:
        print(f"\n✓ Walk-forward shows POSITIVE edge: {total_pips:+} pips")
        print("  Proceeding with final model training...")

        # Train final model
        model = train_final_model(df, feature_cols, labels)

        # Save model
        model_path = project_root / "models" / "xgb_eurusd_h1_buy.pkl"
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")

        # Save features used (same as sell, but confirm)
        buy_features_path = project_root / "features_used_buy.json"
        with open(buy_features_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"Features saved to: {buy_features_path}")

    else:
        print(f"\n✗ Walk-forward shows NEGATIVE edge: {total_pips:+} pips")
        print("  Model may not be profitable. Review before using.")

        # Still train for analysis
        model = train_final_model(df, feature_cols, labels)
        model_path = project_root / "models" / "xgb_eurusd_h1_buy.pkl"
        joblib.dump(model, model_path)
        print(f"\nModel saved (for analysis): {model_path}")

    print()
    print("=" * 70)
    print(f"Results saved to {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
