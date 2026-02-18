"""
Walk-Forward Analysis: Directional Accuracy & Max Favorable Excursion

Tracks how well the model predicts direction, even when TP isn't reached.
For each trade, records:
- Final outcome (TP/SL/timeout)
- Max favorable excursion (MFE) - highest profit before close
- % of TP achieved
- Distribution across TP% buckets

Runs without circuit breaker to see pure signal quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Parameters
TP_PIPS = 20.0
SL_PIPS = 15.0
MAX_HOLDING_BARS = 50
COMMISSION_PIPS = 0.1  # 0.00001 * 10000 = 0.1 pip per trade

BUY_THRESHOLD = 0.40
BUY_SPREAD = 0.015
SELL_THRESHOLD = 0.56

NUM_WINDOWS = 5

# Paths
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "EURUSD_H1_clean.csv"
features_path = project_root / "features_used_buy.json"
model_path = project_root / "models" / "xgb_eurusd_h1_buy.pkl"

print("="*90)
print("WALK-FORWARD: Directional Accuracy & Max Favorable Excursion")
print("="*90)
print("\nParameters:")
print(f"  TP: {TP_PIPS} pips | SL: {SL_PIPS} pips | Max holding: {MAX_HOLDING_BARS} bars")
print(f"  Commission: {COMMISSION_PIPS} pips per trade")
print(f"  Circuit breaker: OFF (tracking pure signal quality)")
print(f"  Buy threshold: {BUY_THRESHOLD} | Spread: {BUY_SPREAD*100}%")
print(f"  Sell threshold: {SELL_THRESHOLD}")
print(f"  Walk-forward windows: {NUM_WINDOWS}")

# Load data
print(f"\nLoading data from {data_path}...")
df = pd.read_csv(data_path)
print(f"Data: {len(df)} rows")

# Load features and model
import json
import joblib

with open(features_path) as f:
    feature_names = json.load(f)

model = joblib.load(model_path)
print(f"Model loaded: {len(feature_names)} features")

# Create labels for validation (same as training)
def create_buy_labels(df, tp_pips=TP_PIPS, sl_pips=SL_PIPS, max_bars=MAX_HOLDING_BARS):
    """Triple barrier labeling for BUY trades"""
    labels = []

    for i in range(len(df) - max_bars):
        entry = df.iloc[i]['close']
        tp_price = entry + tp_pips * 0.0001
        sl_price = entry - sl_pips * 0.0001

        hit_tp = False
        hit_sl = False

        for j in range(i + 1, min(i + 1 + max_bars, len(df))):
            high = df.iloc[j]['high']
            low = df.iloc[j]['low']

            if high >= tp_price:
                hit_tp = True
                break
            if low <= sl_price:
                hit_sl = True
                break

        if hit_tp:
            labels.append(0)  # Buy wins
        elif hit_sl:
            labels.append(2)  # Buy loses
        else:
            labels.append(1)  # Range

    # Pad remaining
    labels.extend([1] * (len(df) - len(labels)))
    return labels

print("\nCreating labels...")
df['label'] = create_buy_labels(df)

# Walk-forward setup
total_bars = len(df)
window_size = total_bars // NUM_WINDOWS

def simulate_trade(entry_idx, direction, df, tp_pips, sl_pips, max_bars):
    """
    Simulate a single trade and track MFE

    Returns dict with:
    - outcome: 'TP', 'SL', 'TIMEOUT'
    - pips: final pips
    - mfe_pips: max favorable excursion in pips
    - tp_pct: % of TP achieved (based on MFE)
    """
    entry_price = df.iloc[entry_idx]['close']

    if direction == 'BUY':
        tp_price = entry_price + tp_pips * 0.0001
        sl_price = entry_price - sl_pips * 0.0001
        pip_multiplier = 1
    else:  # SELL
        tp_price = entry_price - tp_pips * 0.0001
        sl_price = entry_price + sl_pips * 0.0001
        pip_multiplier = -1

    max_favorable_pips = 0
    final_pips = 0
    outcome = 'TIMEOUT'

    for j in range(entry_idx + 1, min(entry_idx + 1 + max_bars, len(df))):
        high = df.iloc[j]['high']
        low = df.iloc[j]['low']
        close = df.iloc[j]['close']

        # Calculate favorable excursion this bar
        if direction == 'BUY':
            bar_best = high - entry_price
            bar_worst = low - entry_price
        else:
            bar_best = entry_price - low
            bar_worst = entry_price - high

        bar_best_pips = bar_best * 10000
        bar_worst_pips = bar_worst * 10000

        # Update MFE
        if bar_best_pips > max_favorable_pips:
            max_favorable_pips = bar_best_pips

        # Check TP hit first
        if direction == 'BUY' and high >= tp_price:
            outcome = 'TP'
            final_pips = tp_pips - COMMISSION_PIPS
            max_favorable_pips = max(max_favorable_pips, tp_pips)
            break
        elif direction == 'SELL' and low <= tp_price:
            outcome = 'TP'
            final_pips = tp_pips - COMMISSION_PIPS
            max_favorable_pips = max(max_favorable_pips, tp_pips)
            break

        # Check SL hit
        if direction == 'BUY' and low <= sl_price:
            outcome = 'SL'
            final_pips = -sl_pips - COMMISSION_PIPS
            break
        elif direction == 'SELL' and high >= sl_price:
            outcome = 'SL'
            final_pips = -sl_pips - COMMISSION_PIPS
            break

    # Timeout
    if outcome == 'TIMEOUT':
        close_price = df.iloc[min(entry_idx + max_bars, len(df) - 1)]['close']
        if direction == 'BUY':
            final_pips = (close_price - entry_price) * 10000 - COMMISSION_PIPS
        else:
            final_pips = (entry_price - close_price) * 10000 - COMMISSION_PIPS

    # Calculate % of TP achieved
    tp_pct = (max_favorable_pips / tp_pips) * 100 if tp_pips > 0 else 0

    # Calculate bars held
    if outcome == 'TIMEOUT':
        bars_held = max_bars
    else:
        bars_held = j - entry_idx

    return {
        'outcome': outcome,
        'pips': final_pips,
        'mfe_pips': max_favorable_pips,
        'tp_pct': tp_pct,
        'bars_held': bars_held
    }

def run_walkforward(mode='BUY_ONLY'):
    """Run walk-forward with MFE tracking"""
    from xgboost import XGBClassifier

    results = []

    for window in range(1, NUM_WINDOWS + 1):
        train_end = window * window_size
        test_start = train_end
        test_end = min((window + 1) * window_size, total_bars)

        if test_start >= total_bars:
            break

        # Train fresh model each window
        X_train = df.iloc[:train_end][feature_names].values
        y_train = df.iloc[:train_end]['label'].values

        # Ensure all 3 classes present (range class is very rare at 0.5%)
        # Without this, XGBoost trains a 2-class model with fundamentally
        # different probability distributions
        for needed_class in [0, 1, 2]:
            if needed_class not in y_train:
                X_train = np.vstack([X_train, X_train[:1]])
                y_train = np.append(y_train, needed_class)

        wf_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            verbosity=0
        )
        wf_model.fit(X_train, y_train)

        # Test
        X_test = df.iloc[test_start:test_end][feature_names].values
        probs = wf_model.predict_proba(X_test)

        window_trades = []

        i = 0
        while i < len(X_test):
            buy_prob = probs[i][0]
            range_prob = probs[i][1]
            sell_prob = probs[i][2]

            signal = None
            direction = None

            if mode in ['BUY_ONLY', 'BOTH']:
                buy_spread = buy_prob - max(sell_prob, range_prob)
                if buy_prob >= BUY_THRESHOLD and buy_spread >= BUY_SPREAD:
                    signal = 1
                    direction = 'BUY'

            if signal is None and mode in ['SELL_ONLY', 'BOTH']:
                if sell_prob >= SELL_THRESHOLD:
                    signal = -1
                    direction = 'SELL'

            if signal is not None:
                actual_idx = test_start + i
                trade_result = simulate_trade(actual_idx, direction, df, TP_PIPS, SL_PIPS, MAX_HOLDING_BARS)

                window_trades.append({
                    'window': window,
                    'direction': direction,
                    'buy_prob': buy_prob,
                    'sell_prob': sell_prob,
                    'range_prob': range_prob,
                    **trade_result
                })

                # Skip forward to after trade actually closes (not always 50 bars)
                i += trade_result['bars_held']
            else:
                i += 1

        results.extend(window_trades)

    return pd.DataFrame(results)

# Run analysis for BUY_ONLY
print("\n" + "="*90)
print("TEST 1: BUY_ONLY Mode")
print("="*90)

buy_results = run_walkforward('BUY_ONLY')

if len(buy_results) > 0:
    print(f"\nTotal trades: {len(buy_results)}")
    print(f"Outcomes: TP={len(buy_results[buy_results.outcome=='TP'])} | " +
          f"SL={len(buy_results[buy_results.outcome=='SL'])} | " +
          f"TIMEOUT={len(buy_results[buy_results.outcome=='TIMEOUT'])}")
    print(f"Win rate: {len(buy_results[buy_results.outcome=='TP']) / len(buy_results) * 100:.1f}%")
    print(f"Total pips: {buy_results.pips.sum():.0f}")

    # MFE Analysis
    print(f"\nMax Favorable Excursion (MFE) Analysis:")
    print(f"  Average MFE: {buy_results.mfe_pips.mean():.1f} pips")
    print(f"  Average TP%: {buy_results.tp_pct.mean():.1f}%")

    # TP% buckets
    print(f"\nDistribution by % of TP Achieved:")
    print(f"  {'Bucket':<15} {'Count':>8} {'Pct':>8} {'Avg Pips':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10}")

    buckets = [
        ("0-20%", 0, 20),
        ("20-40%", 20, 40),
        ("40-60%", 40, 60),
        ("60-80%", 60, 80),
        ("80-100%", 80, 100),
        ("100%+ (TP)", 100, float('inf'))
    ]

    for label, low, high in buckets:
        mask = (buy_results.tp_pct >= low) & (buy_results.tp_pct < high)
        count = mask.sum()
        pct = count / len(buy_results) * 100
        avg_pips = buy_results[mask].pips.mean() if count > 0 else 0
        print(f"  {label:<15} {count:>8} {pct:>7.1f}% {avg_pips:>10.1f}")

    # Breakdown by outcome
    print(f"\nMFE by Outcome:")
    for outcome in ['TP', 'SL', 'TIMEOUT']:
        mask = buy_results.outcome == outcome
        if mask.sum() > 0:
            avg_mfe = buy_results[mask].mfe_pips.mean()
            avg_tp_pct = buy_results[mask].tp_pct.mean()
            print(f"  {outcome:<10} Avg MFE: {avg_mfe:>6.1f} pips ({avg_tp_pct:>5.1f}% of TP)")

# Run analysis for BOTH mode
print("\n" + "="*90)
print("TEST 2: BOTH Mode (Buy + Sell)")
print("="*90)

both_results = run_walkforward('BOTH')

if len(both_results) > 0:
    print(f"\nTotal trades: {len(both_results)}")

    # Split by direction
    buy_trades = both_results[both_results.direction == 'BUY']
    sell_trades = both_results[both_results.direction == 'SELL']

    print(f"\nBuy trades: {len(buy_trades)}")
    if len(buy_trades) > 0:
        print(f"  Win rate: {len(buy_trades[buy_trades.outcome=='TP']) / len(buy_trades) * 100:.1f}%")
        print(f"  Pips: {buy_trades.pips.sum():.0f}")
        print(f"  Avg MFE: {buy_trades.mfe_pips.mean():.1f} pips ({buy_trades.tp_pct.mean():.1f}% of TP)")

    print(f"\nSell trades: {len(sell_trades)}")
    if len(sell_trades) > 0:
        print(f"  Win rate: {len(sell_trades[sell_trades.outcome=='TP']) / len(sell_trades) * 100:.1f}%")
        print(f"  Pips: {sell_trades.pips.sum():.0f}")
        print(f"  Avg MFE: {sell_trades.mfe_pips.mean():.1f} pips ({sell_trades.tp_pct.mean():.1f}% of TP)")

    print(f"\nCombined:")
    print(f"  Win rate: {len(both_results[both_results.outcome=='TP']) / len(both_results) * 100:.1f}%")
    print(f"  Total pips: {both_results.pips.sum():.0f}")

    # Combined TP% distribution
    print(f"\nCombined Distribution by % of TP Achieved:")
    print(f"  {'Bucket':<15} {'Count':>8} {'Pct':>8} {'Avg Pips':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10}")

    for label, low, high in buckets:
        mask = (both_results.tp_pct >= low) & (both_results.tp_pct < high)
        count = mask.sum()
        pct = count / len(both_results) * 100
        avg_pips = both_results[mask].pips.mean() if count > 0 else 0
        print(f"  {label:<15} {count:>8} {pct:>7.1f}% {avg_pips:>10.1f}")

    # MFE by outcome
    print(f"\nMFE by Outcome (Combined):")
    for outcome in ['TP', 'SL', 'TIMEOUT']:
        mask = both_results.outcome == outcome
        if mask.sum() > 0:
            avg_mfe = both_results[mask].mfe_pips.mean()
            avg_tp_pct = both_results[mask].tp_pct.mean()
            count = mask.sum()
            print(f"  {outcome:<10} {count:>4} trades | Avg MFE: {avg_mfe:>6.1f} pips ({avg_tp_pct:>5.1f}% of TP)")

# Directional accuracy analysis
print("\n" + "="*90)
print("DIRECTIONAL ACCURACY ANALYSIS")
print("="*90)
print("\n'Directionally correct' = MFE > 0 (trade moved in our favor at some point)")

for mode_name, mode_results in [("BUY_ONLY", buy_results), ("BOTH", both_results)]:
    if len(mode_results) > 0:
        print(f"\n{mode_name}:")

        # Overall directional accuracy
        dir_correct = (mode_results.mfe_pips > 0).sum()
        dir_accuracy = dir_correct / len(mode_results) * 100

        print(f"  Directionally correct: {dir_correct}/{len(mode_results)} ({dir_accuracy:.1f}%)")
        print(f"  TP win rate: {len(mode_results[mode_results.outcome=='TP']) / len(mode_results) * 100:.1f}%")
        print(f"  Gap: {dir_accuracy - (len(mode_results[mode_results.outcome=='TP']) / len(mode_results) * 100):.1f}% of trades were right direction but didn't reach TP")

        # Analyze "losing" trades
        losses = mode_results[mode_results.outcome == 'SL']
        if len(losses) > 0:
            losses_with_mfe = losses[losses.mfe_pips > 0]
            print(f"\n  SL trades that went in our favor first: {len(losses_with_mfe)}/{len(losses)} ({len(losses_with_mfe)/len(losses)*100:.1f}%)")
            if len(losses_with_mfe) > 0:
                print(f"    Average MFE before reversal: {losses_with_mfe.mfe_pips.mean():.1f} pips ({losses_with_mfe.tp_pct.mean():.1f}% of TP)")
                print(f"    Max MFE before reversal: {losses_with_mfe.mfe_pips.max():.1f} pips ({losses_with_mfe.tp_pct.max():.1f}% of TP)")

print("\n" + "="*90)
print("Analysis complete!")
print("="*90)
