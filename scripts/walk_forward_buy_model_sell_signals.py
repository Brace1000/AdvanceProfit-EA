#!/usr/bin/env python3
"""
Walk-Forward Test: Can the Buy model's SELL signals be trusted?

Hypothesis: When the Buy model outputs very high sell_prob (label 2 = "buy loses"),
it's identifying strong bearish conditions. At high enough confidence, these
could be tradeable sell signals — cherry-picking no-brainer sells.

This tests sell_prob from the Buy model (NOT the dedicated Sell model)
across multiple confidence thresholds in walk-forward validation.

Trade setup: Standard SELL with TP=20 below, SL=15 above (same as EA).
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import get_config
from src.data_collection.loader import DataLoader
from src.features.engineer import FeatureEngineer

LOG_FILE = project_root / "logs" / "walk_forward_buy_model_sell_signals.log"
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
    """Create triple barrier labels for BUY direction (used for training)."""
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


def simulate_sell_trades(df, sell_mask, tp_pips, sl_pips, max_horizon, commission,
                         cb_enabled=False, cb_max_losses=5, cb_max_dd_pips=100,
                         cb_cooldown=48, cb_reset_on_win=True):
    """
    Simulate SELL trades on bars where sell_mask is True.
    TP = price drops tp_pips (profit), SL = price rises sl_pips (loss).
    """
    pip_value = 0.0001
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    sell_indices = np.where(sell_mask)[0]

    trade_results = []
    in_trade_until = -1
    consecutive_losses = 0
    peak_pnl = 0.0
    running_pnl = 0.0
    paused_until = -1
    cb_triggers = 0
    skipped = 0

    for idx in sell_indices:
        if idx <= in_trade_until:
            continue

        if cb_enabled and idx < paused_until:
            skipped += 1
            continue

        entry = closes[idx]
        horizon_end = min(idx + 1 + max_horizon, len(df))

        if idx + 1 >= len(df):
            continue

        outcome = None
        exit_bar = idx
        # For SELL: profit when price goes DOWN, loss when price goes UP
        for j in range(idx + 1, horizon_end):
            # TP hit: price dropped tp_pips (low reached target)
            if (entry - lows[j]) >= tp_pips * pip_value:
                outcome = tp_pips * pip_value - commission
                exit_bar = j
                break
            # SL hit: price rose sl_pips (high reached stop)
            if (highs[j] - entry) >= sl_pips * pip_value:
                outcome = -(sl_pips * pip_value) - commission
                exit_bar = j
                break

        if outcome is None:
            exit_price = closes[horizon_end - 1]
            outcome = (entry - exit_price) - commission  # Sell profit = entry - exit
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

    return trade_results, cb_triggers, skipped


def test_threshold(df, feature_cols, threshold, tp_pips, sl_pips, max_horizon,
                   commission, cb_settings, n_windows, spread_threshold=0.0):
    """Run walk-forward for a single sell_prob threshold from the Buy model."""
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

        # Train Buy model (same as production)
        train_labels = create_buy_labels(train_df, tp_pips=20, sl_pips=15, max_holding=50)
        X_train = train_df[feature_cols].values

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, train_labels)

        # Predict on test
        X_test = test_df[feature_cols].values
        probas = model.predict_proba(X_test)

        # KEY DIFFERENCE: Use sell_prob (label 2 = "buy loses") as sell signal
        sell_probs = probas[:, 2]
        buy_probs = probas[:, 0]
        range_probs = probas[:, 1]

        # Sell signals: sell_prob >= threshold AND sell dominates
        if spread_threshold > 0:
            max_other = np.maximum(buy_probs, range_probs)
            sell_signals = (sell_probs >= threshold) & ((sell_probs - max_other) >= spread_threshold)
        else:
            sell_signals = sell_probs >= threshold

        # Simulate SELL trades with standard TP=20/SL=15
        results, cb_count, skipped = simulate_sell_trades(
            test_df, sell_signals, tp_pips, sl_pips, max_horizon, commission,
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
            'window': w + 1,
            'trades': trades,
            'wins': wins,
            'win_rate': win_rate,
            'pips': pips,
            'cb_triggers': cb_count,
            'signals': int(sell_signals.sum()),
            'skipped': skipped
        })

    # Aggregate
    total_trades = sum(r['trades'] for r in window_results)
    total_wins = sum(r['wins'] for r in window_results)
    total_pips = sum(r['pips'] for r in window_results)
    total_cb = sum(r['cb_triggers'] for r in window_results)
    total_signals = sum(r['signals'] for r in window_results)
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    return {
        'threshold': threshold,
        'total_signals': total_signals,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': overall_wr,
        'total_pips': total_pips,
        'cb_triggers': total_cb,
        'windows': window_results
    }


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 95)
    print("WALK-FORWARD TEST: Buy Model's SELL Signals")
    print("Can the Buy model identify no-brainer sells at high confidence?")
    print("=" * 95)

    config = get_config()

    # Standard sell trade parameters (same as EA)
    tp_pips = 20.0
    sl_pips = 15.0
    max_horizon = 50
    commission = 0.00001
    n_windows = 5

    cb_settings = {
        'cb_enabled': True,
        'cb_max_losses': 5,
        'cb_max_dd_pips': 100,
        'cb_cooldown': 48,
        'cb_reset_on_win': True
    }

    # Test high thresholds — we want cherry-picked, high-conviction sells
    THRESHOLDS = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

    be_rate = sl_pips / (tp_pips + sl_pips) * 100

    print(f"\nParameters:")
    print(f"  Trade: SELL (TP={tp_pips} below, SL={sl_pips} above)")
    print(f"  Break-even win rate: {be_rate:.1f}%")
    print(f"  Signal source: Buy model's sell_prob (label 2 = 'buy loses')")
    print(f"  Commission: {commission:.5f}")
    print(f"  Circuit breaker: ON (5 losses OR 100 pip DD -> 48 bar pause)")
    print(f"  Walk-forward windows: {n_windows}")
    print(f"  Thresholds: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f}")

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

    # =========================================================================
    # TEST 1: Pure threshold (sell_prob >= X)
    # =========================================================================
    print(f"\n{'=' * 95}")
    print("TEST 1: Pure Threshold (sell_prob >= X)")
    print(f"{'=' * 95}")
    print()
    print(f"{'Thresh':>7} | {'Signals':>8} | {'Trades':>7} | {'Wins':>6} | {'WinRate':>8} | {'Pips':>9} | {'CB':>4} | {'vs BE':>8} | {'Pips/Trade':>10}")
    print("-" * 100)

    all_results_1 = []
    for thresh in THRESHOLDS:
        result = test_threshold(
            df, feature_cols, thresh, tp_pips, sl_pips, max_horizon,
            commission, cb_settings, n_windows, spread_threshold=0.0
        )
        all_results_1.append(result)

        vs_be = result['win_rate'] - be_rate
        ppt = result['total_pips'] / result['total_trades'] if result['total_trades'] > 0 else 0
        above_be = " ***" if result['win_rate'] > be_rate else ""

        print(f"  {result['threshold']:.2f}  | {result['total_signals']:7d}  | "
              f"{result['total_trades']:6d}  | {result['total_wins']:5d}  | "
              f"{result['win_rate']:6.1f}%  | {result['total_pips']:+8.0f}  | "
              f"{result['cb_triggers']:3d}  | {vs_be:+6.1f}%{above_be:4s} | {ppt:+9.1f}")

    # =========================================================================
    # TEST 2: Threshold + Spread (sell must dominate buy and range)
    # =========================================================================
    print(f"\n{'=' * 95}")
    print("TEST 2: Threshold + Confidence Spread (sell_prob - max(buy,range) >= 1.5%)")
    print(f"{'=' * 95}")
    print()
    print(f"{'Thresh':>7} | {'Signals':>8} | {'Trades':>7} | {'Wins':>6} | {'WinRate':>8} | {'Pips':>9} | {'CB':>4} | {'vs BE':>8} | {'Pips/Trade':>10}")
    print("-" * 100)

    all_results_2 = []
    for thresh in THRESHOLDS:
        result = test_threshold(
            df, feature_cols, thresh, tp_pips, sl_pips, max_horizon,
            commission, cb_settings, n_windows, spread_threshold=0.015
        )
        all_results_2.append(result)

        vs_be = result['win_rate'] - be_rate
        ppt = result['total_pips'] / result['total_trades'] if result['total_trades'] > 0 else 0
        above_be = " ***" if result['win_rate'] > be_rate else ""

        print(f"  {result['threshold']:.2f}  | {result['total_signals']:7d}  | "
              f"{result['total_trades']:6d}  | {result['total_wins']:5d}  | "
              f"{result['win_rate']:6.1f}%  | {result['total_pips']:+8.0f}  | "
              f"{result['cb_triggers']:3d}  | {vs_be:+6.1f}%{above_be:4s} | {ppt:+9.1f}")

    # =========================================================================
    # TEST 3: Higher spread requirement (sell must CLEARLY dominate)
    # =========================================================================
    print(f"\n{'=' * 95}")
    print("TEST 3: Threshold + Large Spread (sell_prob - max(buy,range) >= 5%)")
    print(f"{'=' * 95}")
    print()
    print(f"{'Thresh':>7} | {'Signals':>8} | {'Trades':>7} | {'Wins':>6} | {'WinRate':>8} | {'Pips':>9} | {'CB':>4} | {'vs BE':>8} | {'Pips/Trade':>10}")
    print("-" * 100)

    all_results_3 = []
    for thresh in THRESHOLDS:
        result = test_threshold(
            df, feature_cols, thresh, tp_pips, sl_pips, max_horizon,
            commission, cb_settings, n_windows, spread_threshold=0.05
        )
        all_results_3.append(result)

        vs_be = result['win_rate'] - be_rate
        ppt = result['total_pips'] / result['total_trades'] if result['total_trades'] > 0 else 0
        above_be = " ***" if result['win_rate'] > be_rate else ""

        print(f"  {result['threshold']:.2f}  | {result['total_signals']:7d}  | "
              f"{result['total_trades']:6d}  | {result['total_wins']:5d}  | "
              f"{result['win_rate']:6.1f}%  | {result['total_pips']:+8.0f}  | "
              f"{result['cb_triggers']:3d}  | {vs_be:+6.1f}%{above_be:4s} | {ppt:+9.1f}")

    # =========================================================================
    # Per-window detail for best threshold
    # =========================================================================
    # Find best by pips from all three tests
    all_combined = all_results_1 + all_results_2 + all_results_3
    profitable = [r for r in all_combined if r['total_pips'] > 0 and r['total_trades'] >= 10]

    if profitable:
        best = max(profitable, key=lambda x: x['total_pips'])
        print(f"\n{'=' * 95}")
        print(f"BEST RESULT — Per-Window Detail")
        print(f"{'=' * 95}")
        print(f"  Threshold: {best['threshold']:.2f}")
        print(f"  Total: {best['total_trades']} trades, {best['win_rate']:.1f}% WR, {best['total_pips']:+.0f} pips")
        print()

        dt_col = "datetime" if "datetime" in df.columns else None

        print(f"  {'Win':>3} | {'Trades':>7} | {'Wins':>6} | {'WinR':>7} | {'Pips':>9} | {'CB':>3}")
        print(f"  {'-' * 55}")
        for wr in best['windows']:
            print(f"  W{wr['window']:>2} | {wr['trades']:>7} | {wr['wins']:>6} | "
                  f"{wr['win_rate']:>6.1f}% | {wr['pips']:>+8.0f} | {wr['cb_triggers']:>3}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 95}")
    print("SUMMARY")
    print(f"{'=' * 95}")
    print(f"  Break-even win rate: {be_rate:.1f}%")
    print(f"  Trade: SELL TP={tp_pips}/SL={sl_pips}")
    print()

    for label, results in [("Pure threshold", all_results_1),
                           ("+ 1.5% spread", all_results_2),
                           ("+ 5.0% spread", all_results_3)]:
        profitable_r = [r for r in results if r['total_pips'] > 0 and r['total_trades'] >= 5]
        if profitable_r:
            best_r = max(profitable_r, key=lambda x: x['total_pips'])
            print(f"  {label:20s} | Best: {best_r['threshold']:.2f} → "
                  f"{best_r['total_pips']:+.0f} pips, {best_r['win_rate']:.1f}% WR, "
                  f"{best_r['total_trades']} trades")
        else:
            print(f"  {label:20s} | No profitable threshold found")

    # Compare with Buy model's buy signals and dedicated Sell model
    print(f"\n  Context:")
    print(f"    Buy model BUY signals (0.40):  +1177 pips, 47.2% WR, 836 trades")
    print(f"    Dedicated Sell model (0.34):    +74 pips, 47.0% WR, 217 trades")
    print(f"    Dedicated Sell model (no CB):   -185 pips, 44.7% WR, 459 trades")

    print()


if __name__ == "__main__":
    main()
    if isinstance(sys.stdout, TeeWriter):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")
