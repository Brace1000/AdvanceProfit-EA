#!/usr/bin/env python3
"""
Walk-Forward Validation: Buy Model (No Regime Filter)

Tests Buy model edge across time periods without regime filtering.
This reveals the true raw edge of the Buy model's predictions.

Output saved to logs/walk_forward_buy.log
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

LOG_FILE = project_root / "logs" / "walk_forward_buy.log"
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
    """
    Create triple barrier labels for BUY direction.

    Label 0: Buy wins (price goes UP tp_pips before DOWN sl_pips)
    Label 1: Range (neither barrier hit in max_holding bars)
    Label 2: Buy loses (price goes DOWN sl_pips first)
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
    skipped = 0

    for idx in buy_indices:
        if idx <= in_trade_until:
            continue

        if cb_enabled and idx < paused_until:
            skipped += 1
            continue

        entry = closes[idx]
        tp_price = tp_pips * pip_value
        sl_price = sl_pips * pip_value
        horizon_end = min(idx + 1 + max_horizon, len(df))

        if idx + 1 >= len(df):
            continue

        outcome = None
        exit_bar = idx
        # For BUY: profit when price goes UP, loss when price goes DOWN
        for j in range(idx + 1, horizon_end):
            # TP hit: price went up by tp_pips
            if (highs[j] - entry) >= tp_price:
                outcome = tp_pips * pip_value - commission
                exit_bar = j
                break
            # SL hit: price went down by sl_pips
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
                reason = f"{consecutive_losses} consecutive losses"
            if dd_pips >= cb_max_dd_pips:
                triggered = True
                reason = f"{dd_pips:.0f} pip drawdown"

            if triggered:
                paused_until = idx + cb_cooldown
                cb_triggers += 1
                consecutive_losses = 0

    return trade_results, cb_triggers, skipped


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 80)
    print("WALK-FORWARD VALIDATION — Buy Model (No Regime Filter)")
    print("=" * 80)

    config = get_config()

    # Parameters
    tp_pips = 20.0
    sl_pips = 15.0
    max_horizon = 50
    commission = 0.00001
    confidence_threshold = 0.34  # Will test if this is optimal

    # Circuit breaker settings
    cb_enabled = True
    cb_max_losses = 5
    cb_max_dd_pips = 100
    cb_cooldown = 48
    cb_reset_on_win = True

    print(f"\nParameters:")
    print(f"  TP: {tp_pips} pips | SL: {sl_pips} pips | Max holding: {max_horizon} bars")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Commission: {commission:.5f}")
    print(f"  Circuit breaker: {'ON' if cb_enabled else 'OFF'}")
    if cb_enabled:
        print(f"    - Max consecutive losses: {cb_max_losses}")
        print(f"    - Max drawdown: {cb_max_dd_pips} pips")
        print(f"    - Cooldown: {cb_cooldown} bars")
    print(f"  Regime filter: OFF (testing raw model edge)")

    # Load data
    print(f"\nLoading data...")
    loader = DataLoader(config._config)
    raw_path = project_root / config.get("paths.data_raw_file", "data/EURUSD_H1_clean.csv")
    df = loader.load(raw_path)

    fe = FeatureEngineer(config._config)
    df = fe.engineer(df)

    # Load features
    features_file = project_root / "features_used_buy.json"
    if not features_file.exists():
        features_file = project_root / "features_used.json"
    with open(features_file) as f:
        feature_cols = json.load(f)

    df = df.dropna(subset=feature_cols).copy()
    print(f"Data: {len(df)} rows after dropna")

    # Create Buy labels
    print(f"Creating Buy labels...")
    labels = create_buy_labels(df, tp_pips=tp_pips, sl_pips=sl_pips, max_holding=max_horizon)
    df['buy_label'] = labels

    # Walk-forward windows
    n_windows = 5
    n = len(df)
    test_size = n // (n_windows + 1)
    min_train = test_size

    print(f"\nWalk-forward: {n_windows} windows, ~{test_size} bars per test")
    print()

    # Results table
    print(f"{'Win':>3} | {'Period':>25} | {'Trades':>7} | {'Wins':>6} | {'WinR':>7} | {'Pips':>9} | {'CB':>3}")
    print("-" * 80)

    all_results = []

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
        if "datetime" in test_df.columns:
            start_date = str(test_df.iloc[0]["datetime"])[:10]
            end_date = str(test_df.iloc[-1]["datetime"])[:10]
            period = f"{start_date} to {end_date}"
        else:
            period = f"bars {test_start}-{test_end}"

        # Train model
        train_labels = create_buy_labels(train_df, tp_pips, sl_pips, max_horizon)
        X_train = train_df[feature_cols].values
        y_train = train_labels

        # Train XGBoost
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

        # Predict on test
        X_test = test_df[feature_cols].values
        probas = model.predict_proba(X_test)
        buy_probs = probas[:, 0]  # Probability of Buy winning

        # Buy signals: confidence >= threshold
        buy_signals = buy_probs >= confidence_threshold

        # Simulate trades
        results, cb_count, skipped = simulate_buy_trades(
            test_df, buy_signals, tp_pips, sl_pips, max_horizon, commission,
            cb_enabled, cb_max_losses, cb_max_dd_pips, cb_cooldown, cb_reset_on_win
        )

        if len(results) == 0:
            print(f"W{w+1:>2} | {period:>25} | {'---':>7} | {'---':>6} | {'---':>7} | {'---':>9} | {'-':>3}")
            continue

        wins = sum(1 for r in results if r > 0)
        losses = sum(1 for r in results if r <= 0)
        trades = wins + losses
        win_rate = wins / trades * 100 if trades > 0 else 0

        pips = sum(results) / 0.0001

        print(f"W{w+1:>2} | {period:>25} | {trades:>7} | {wins:>6} | {win_rate:>6.1f}% | {pips:>+8.0f} | {cb_count:>3}")

        all_results.append({
            'window': w + 1,
            'period': period,
            'trades': trades,
            'wins': wins,
            'win_rate': win_rate,
            'pips': pips,
            'cb_triggers': cb_count
        })

    # Aggregate
    print("-" * 80)
    total_trades = sum(r['trades'] for r in all_results)
    total_wins = sum(r['wins'] for r in all_results)
    total_pips = sum(r['pips'] for r in all_results)
    total_cb = sum(r['cb_triggers'] for r in all_results)

    if total_trades > 0:
        overall_wr = total_wins / total_trades * 100
    else:
        overall_wr = 0

    print(f"{'TOTAL':>3} | {'-':>25} | {total_trades:>7} | {total_wins:>6} | {overall_wr:>6.1f}% | {total_pips:>+8.0f} | {total_cb:>3}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total trades: {total_trades}")
    print(f"  Total wins: {total_wins}")
    print(f"  Overall win rate: {overall_wr:.1f}%")
    print(f"  Breakeven win rate: {sl_pips / (tp_pips + sl_pips) * 100:.1f}%")
    print(f"  Total pips: {total_pips:+.0f}")
    print(f"  Circuit breaker triggers: {total_cb}")

    be_rate = sl_pips / (tp_pips + sl_pips) * 100
    if overall_wr > be_rate:
        print(f"\n  ✓ Win rate ABOVE breakeven by {overall_wr - be_rate:.1f}%")
    else:
        print(f"\n  ✗ Win rate BELOW breakeven by {be_rate - overall_wr:.1f}%")

    if total_pips > 0:
        print(f"  ✓ Buy model shows POSITIVE edge: {total_pips:+.0f} pips")
    else:
        print(f"  ✗ Buy model shows NEGATIVE edge: {total_pips:+.0f} pips")

    print()


if __name__ == "__main__":
    main()
    if isinstance(sys.stdout, TeeWriter):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")
