#!/usr/bin/env python3
"""
Walk-Forward Validation: Buy Model Threshold Sweep

Tests multiple confidence thresholds across walk-forward windows to find
the optimal threshold that maximizes real-world edge.

This is the definitive test - combines:
1. Walk-forward retraining (no look-ahead bias)
2. Threshold optimization (finding best signal filter)
3. No regime filter (testing raw model edge)

Output saved to logs/walk_forward_buy_threshold_sweep.log
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import get_config
from src.data_collection.loader import DataLoader
from src.features.engineer import FeatureEngineer

LOG_FILE = project_root / "logs" / "walk_forward_buy_threshold_sweep.log"
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
    def isatty(self):
        return self.stdout.isatty() if hasattr(self.stdout, 'isatty') else False


def create_buy_labels(df, tp_pips=20, sl_pips=15, max_holding=50):
    """Create triple barrier labels for BUY direction."""
    labels = []
    close = df['close'].values

    for i in range(len(df)):
        if i + max_holding >= len(df):
            labels.append(1)
            continue

        entry_price = close[i]
        tp_price = entry_price + tp_pips * 0.0001
        sl_price = entry_price - sl_pips * 0.0001
        label = 1

        for j in range(1, max_holding + 1):
            future_price = close[i + j]
            if future_price >= tp_price:
                label = 0  # Buy wins
                break
            elif future_price <= sl_price:
                label = 2  # Buy loses
                break

        labels.append(label)

    return np.array(labels)


def simulate_buy_trades(df, buy_mask, tp_pips, sl_pips, max_horizon, commission,
                        cb_enabled=False, cb_max_losses=5, cb_max_dd_pips=100,
                        cb_cooldown=48, cb_reset_on_win=True):
    """Run TP/SL simulation on buy signals with optional circuit breaker."""
    pip_value = 0.0001
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    buy_indices = np.where(buy_mask)[0]

    trade_results = []
    in_trade_until = -1
    consecutive_losses = 0
    peak_pnl = 0.0
    running_pnl = 0.0
    paused_until = -1
    cb_triggers = 0

    for idx in buy_indices:
        if idx <= in_trade_until:
            continue

        if cb_enabled and idx < paused_until:
            continue

        entry = closes[idx]
        tp_price = tp_pips * pip_value
        sl_price = sl_pips * pip_value
        horizon_end = min(idx + 1 + max_horizon, len(df))

        if idx + 1 >= len(df):
            continue

        outcome = None
        exit_bar = idx
        for j in range(idx + 1, horizon_end):
            if (highs[j] - entry) >= tp_price:
                outcome = tp_pips * pip_value - commission
                exit_bar = j
                break
            if (entry - lows[j]) >= sl_price:
                outcome = -(sl_pips * pip_value) - commission
                exit_bar = j
                break

        if outcome is None:
            exit_price = closes[horizon_end - 1]
            outcome = (exit_price - entry) - commission
            exit_bar = horizon_end - 1

        trade_results.append(outcome)
        in_trade_until = exit_bar

        if cb_enabled:
            running_pnl += outcome
            peak_pnl = max(peak_pnl, running_pnl)
            dd_pips = (peak_pnl - running_pnl) / pip_value

            if outcome <= 0:
                consecutive_losses += 1
            else:
                if cb_reset_on_win:
                    consecutive_losses = 0

            triggered = False
            if consecutive_losses >= cb_max_losses:
                triggered = True
            if dd_pips >= cb_max_dd_pips:
                triggered = True

            if triggered:
                paused_until = idx + cb_cooldown
                cb_triggers += 1
                consecutive_losses = 0

    return trade_results, cb_triggers


def test_threshold(df, feature_cols, labels, threshold, tp_pips, sl_pips, max_horizon,
                   commission, cb_settings, n_windows):
    """Run walk-forward for a single threshold."""
    n = len(df)
    test_size = n // (n_windows + 1)
    min_train = test_size

    window_results = []

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

        # Train model
        train_labels = create_buy_labels(train_df, tp_pips, sl_pips, max_horizon)
        X_train = train_df[feature_cols].values
        y_train = train_labels

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
        X_test = test_df[feature_cols].values
        probas = model.predict_proba(X_test)
        buy_probs = probas[:, 0]

        # Apply threshold
        buy_signals = buy_probs >= threshold

        # Simulate
        results, cb_count = simulate_buy_trades(
            test_df, buy_signals, tp_pips, sl_pips, max_horizon, commission,
            **cb_settings
        )

        if len(results) > 0:
            wins = sum(1 for r in results if r > 0)
            trades = len(results)
            win_rate = wins / trades * 100
            pips = sum(results) / 0.0001
        else:
            wins = 0
            trades = 0
            win_rate = 0
            pips = 0

        window_results.append({
            'trades': trades,
            'wins': wins,
            'win_rate': win_rate,
            'pips': pips,
            'cb_triggers': cb_count
        })

    # Aggregate
    total_trades = sum(r['trades'] for r in window_results)
    total_wins = sum(r['wins'] for r in window_results)
    total_pips = sum(r['pips'] for r in window_results)
    total_cb = sum(r['cb_triggers'] for r in window_results)
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'threshold': threshold,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': overall_wr,
        'total_pips': total_pips,
        'cb_triggers': total_cb
    }


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 90)
    print("WALK-FORWARD THRESHOLD SWEEP — Buy Model")
    print("=" * 90)

    config = get_config()

    # Parameters
    tp_pips = 20.0
    sl_pips = 15.0
    max_horizon = 50
    commission = 0.00001
    n_windows = 5

    # Circuit breaker
    cb_settings = {
        'cb_enabled': True,
        'cb_max_losses': 5,
        'cb_max_dd_pips': 100,
        'cb_cooldown': 48,
        'cb_reset_on_win': True
    }

    # Thresholds to test
    THRESHOLDS = [0.25, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.45, 0.50]

    print(f"\nParameters:")
    print(f"  TP: {tp_pips} pips | SL: {sl_pips} pips | Max holding: {max_horizon} bars")
    print(f"  Commission: {commission:.5f}")
    print(f"  Circuit breaker: ON (5 losses OR 100 pip DD -> 48 bar pause)")
    print(f"  Regime filter: OFF")
    print(f"  Walk-forward windows: {n_windows}")
    print(f"  Thresholds to test: {len(THRESHOLDS)}")

    # Load data
    print(f"\nLoading data...")
    loader = DataLoader(config._config)
    raw_path = project_root / config.get("paths.data_raw_file", "data/EURUSD_H1_clean.csv")
    df = loader.load(raw_path)

    fe = FeatureEngineer(config._config)
    df = fe.engineer(df)

    features_file = project_root / "features_used_buy.json"
    if not features_file.exists():
        features_file = project_root / "features_used.json"
    with open(features_file) as f:
        feature_cols = json.load(f)

    df = df.dropna(subset=feature_cols).copy()
    print(f"Data: {len(df)} rows")

    # Create labels once
    print(f"Creating Buy labels...")
    labels = create_buy_labels(df, tp_pips, sl_pips, max_horizon)
    df['buy_label'] = labels

    # Test each threshold
    print(f"\nTesting {len(THRESHOLDS)} thresholds across {n_windows} walk-forward windows...")
    print("(This will take a few minutes...)")
    print()

    be_rate = sl_pips / (tp_pips + sl_pips) * 100
    all_results = []

    print(f"{'Thresh':>7} | {'Trades':>7} | {'Wins':>6} | {'WinRate':>8} | {'Pips':>9} | {'CB':>4} | {'vs BE':>8}")
    print("-" * 90)

    for thresh in THRESHOLDS:
        result = test_threshold(
            df, feature_cols, labels, thresh, tp_pips, sl_pips, max_horizon,
            commission, cb_settings, n_windows
        )

        all_results.append(result)

        vs_be = result['win_rate'] - be_rate
        above_be = " ***" if result['win_rate'] > be_rate else ""

        print(f"  {result['threshold']:.2f}  | {result['total_trades']:6d}  | "
              f"{result['total_wins']:5d}  | {result['win_rate']:6.1f}%  | "
              f"{result['total_pips']:+8.0f}  | {result['cb_triggers']:3d}  | "
              f"{vs_be:+6.1f}%{above_be}")

    # Find best
    best_pips = max(all_results, key=lambda x: x['total_pips'])
    best_wr = max(all_results, key=lambda x: x['win_rate'])
    best_trades = max(all_results, key=lambda x: x['total_trades'])

    print()
    print("=" * 90)
    print("OPTIMIZATION RESULTS")
    print("=" * 90)
    print(f"\n  Break-even win rate: {be_rate:.1f}%")
    print(f"\n  Best by PIPS: {best_pips['threshold']:.2f} → {best_pips['total_pips']:+.0f} pips "
          f"({best_pips['total_trades']} trades, {best_pips['win_rate']:.1f}% WR)")
    print(f"  Best by WIN RATE: {best_wr['threshold']:.2f} → {best_wr['win_rate']:.1f}% "
          f"({best_wr['total_trades']} trades, {best_wr['total_pips']:+.0f} pips)")
    print(f"  Most TRADES: {best_trades['threshold']:.2f} → {best_trades['total_trades']} trades "
          f"({best_trades['win_rate']:.1f}% WR, {best_trades['total_pips']:+.0f} pips)")

    print(f"\n  Current threshold (0.34): {[r for r in all_results if r['threshold'] == 0.34][0]['total_pips']:+.0f} pips")
    print(f"  Potential improvement: {best_pips['total_pips'] - [r for r in all_results if r['threshold'] == 0.34][0]['total_pips']:+.0f} pips")

    if best_pips['threshold'] == 0.34:
        print(f"\n  ✓ Current threshold (0.34) is already optimal!")
    else:
        print(f"\n  → Recommend changing threshold to {best_pips['threshold']:.2f}")

    print()


if __name__ == "__main__":
    main()
    if isinstance(sys.stdout, TeeWriter):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")
