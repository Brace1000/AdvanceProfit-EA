"""
Walk-Forward Strategy Comparison
Tests 7 exit management strategies against the same model signals.
All strategies use BOTH mode (buy + sell) without circuit breaker.
Each strategy manages its own trade timing independently.

Strategies:
1. Raw TP=20/SL=15 (control)
2. Twin Trade (current EA: Banker TP=10 + Runner TP=20 w/ BE+trail)
3. The Ladder (progressive BE ratchet)
4. No Ceiling (trail only, no fixed TP)
5. Time Decay (dynamic TP/SL based on bars elapsed)
6. The Free Ride (banker funds risk-free runner)
7. The Harvester (3-tier: 40% TP=8, 30% TP=14+BE, 30% trail)
"""

import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path
from xgboost import XGBClassifier

# ── Parameters ────────────────────────────────────────────────────────
TP_PIPS = 20.0
SL_PIPS = 15.0
MAX_HOLDING = 50
COMMISSION = 0.1  # pips per position open

BUY_THRESHOLD = 0.40
BUY_SPREAD = 0.015
SELL_THRESHOLD = 0.56
NUM_WINDOWS = 5

# ── Paths ─────────────────────────────────────────────────────────────
project = Path(__file__).parent.parent
data_path = project / "data" / "EURUSD_H1_clean.csv"
features_path = project / "features_used_buy.json"
log_path = project / "logs" / "walk_forward_strategy_comparison.log"

# ── Logging to file (avoids PowerShell encoding issues) ──────────────
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)

# ── Bar data helper ───────────────────────────────────────────────────
def get_bars(entry_idx, direction, df, max_bars=MAX_HOLDING):
    """Pre-compute bar-by-bar pips relative to entry.
    Returns list of dicts with:
      best  = max favorable movement this bar (positive = good)
      worst = max adverse movement this bar (negative = bad)
      close = P&L at bar close
    """
    entry_price = df.iloc[entry_idx]['close']
    bars = []
    end = min(entry_idx + 1 + max_bars, len(df))
    for j in range(entry_idx + 1, end):
        h = df.iloc[j]['high']
        l = df.iloc[j]['low']
        c = df.iloc[j]['close']
        if direction == 'BUY':
            bars.append({
                'best':  (h - entry_price) * 10000,
                'worst': (l - entry_price) * 10000,
                'close': (c - entry_price) * 10000,
            })
        else:
            bars.append({
                'best':  (entry_price - l) * 10000,
                'worst': (entry_price - h) * 10000,
                'close': (entry_price - c) * 10000,
            })
    return bars

# ══════════════════════════════════════════════════════════════════════
# STRATEGY FUNCTIONS
# Each returns: {pips, bars_held, outcome, mfe}
# pips is normalized to full-lot equivalent
# ══════════════════════════════════════════════════════════════════════

def strat_raw(bars):
    """Strategy 1: Raw single trade. TP=20, SL=15."""
    mfe = 0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        if b['best'] >= 20.0:
            return dict(pips=20.0 - COMMISSION, bars_held=i+1, outcome='TP', mfe=mfe)
        if b['worst'] <= -15.0:
            return dict(pips=-15.0 - COMMISSION, bars_held=i+1, outcome='SL', mfe=mfe)
    return dict(pips=bars[-1]['close'] - COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_twin(bars):
    """Strategy 2: Twin Trade. A=50% TP10/SL15, B=50% TP20/SL15 BE@10 trail@8."""
    a_open, b_open = True, True
    a_pips, b_pips = 0.0, 0.0
    a_bars, b_bars = 0, 0
    b_sl = -15.0
    b_peak = 0.0
    mfe = 0.0

    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])

        if a_open:
            if b['best'] >= 10.0:
                a_pips = 10.0 - COMMISSION
                a_open = False; a_bars = i+1
            elif b['worst'] <= -15.0:
                a_pips = -15.0 - COMMISSION
                a_open = False; a_bars = i+1

        if b_open:
            if b['best'] > b_peak:
                b_peak = b['best']
            # BE at +10
            if b_peak >= 10.0 and b_sl < 0:
                b_sl = 0.0
            # Trail at 8 pips once BE triggered
            if b_sl >= 0:
                trail = b_peak - 8.0
                if trail > b_sl:
                    b_sl = trail
            # Check TP
            if b['best'] >= 20.0:
                b_pips = 20.0 - COMMISSION
                b_open = False; b_bars = i+1
            elif b['worst'] <= b_sl:
                b_pips = b_sl - COMMISSION
                b_open = False; b_bars = i+1

        if not a_open and not b_open:
            break

    if a_open:
        a_pips = bars[-1]['close'] - COMMISSION; a_bars = len(bars)
    if b_open:
        b_pips = max(bars[-1]['close'], b_sl) - COMMISSION; b_bars = len(bars)

    return dict(pips=0.5*a_pips + 0.5*b_pips,
                bars_held=max(a_bars, b_bars), outcome='TWIN', mfe=mfe)


def strat_ladder(bars):
    """Strategy 3: The Ladder. TP=20, progressive SL ratchet."""
    tp = 20.0
    sl = -15.0
    mfe = 0.0
    ratchets = [(4.0, -10.0), (8.0, -3.0), (10.0, 0.0), (14.0, 8.0), (17.0, 12.0)]
    r_idx = 0

    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        # Advance ratchets
        while r_idx < len(ratchets) and mfe >= ratchets[r_idx][0]:
            sl = ratchets[r_idx][1]
            r_idx += 1
        if b['best'] >= tp:
            return dict(pips=tp - COMMISSION, bars_held=i+1, outcome='TP', mfe=mfe)
        if b['worst'] <= sl:
            return dict(pips=sl - COMMISSION, bars_held=i+1, outcome='SL_LADDER', mfe=mfe)

    final = max(bars[-1]['close'], sl)
    return dict(pips=final - COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_no_ceiling(bars):
    """Strategy 4: No Ceiling. No TP, trail@8 activates at +5, BE at +10."""
    sl = -15.0
    peak = 0.0
    trailing = False
    mfe = 0.0

    for i, b in enumerate(bars):
        if b['best'] > peak:
            peak = b['best']
        mfe = max(mfe, peak)
        # Activate trail at +5
        if peak >= 5.0:
            trailing = True
        # BE at +10
        if peak >= 10.0 and sl < 0:
            sl = 0.0
        # Trailing stop (8 pip distance)
        if trailing:
            trail = peak - 8.0
            if trail > sl:
                sl = trail
        # Check SL / trail stop
        if b['worst'] <= sl:
            return dict(pips=sl - COMMISSION, bars_held=i+1, outcome='TRAIL', mfe=mfe)

    final = max(bars[-1]['close'], sl)
    return dict(pips=final - COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_time_decay(bars):
    """Strategy 5: Time Decay. TP/SL tighten over time, force close after 30 bars."""
    mfe = 0.0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        n = i + 1  # bar number (1-indexed)

        if n <= 10:
            tp, sl = 20.0, -15.0
        elif n <= 20:
            tp, sl = 14.0, -12.0
        elif n <= 30:
            tp, sl = 14.0, -8.0
            # Take any profit >= 3 pips
            if b['best'] >= 3.0:
                return dict(pips=3.0 - COMMISSION, bars_held=n, outcome='TD_PROFIT', mfe=mfe)
        else:
            # Force close at market
            return dict(pips=b['close'] - COMMISSION, bars_held=n, outcome='TD_CLOSE', mfe=mfe)

        if b['best'] >= tp:
            return dict(pips=tp - COMMISSION, bars_held=n, outcome='TP', mfe=mfe)
        if b['worst'] <= sl:
            return dict(pips=sl - COMMISSION, bars_held=n, outcome='SL', mfe=mfe)

    final = bars[-1]['close']
    return dict(pips=final - COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_free_ride(bars):
    """Strategy 6: Free Ride. A=50% TP8/SL15, B=50% trail, BE when A hits TP."""
    a_open, b_open = True, True
    a_pips, b_pips = 0.0, 0.0
    a_bars, b_bars = 0, 0
    a_hit_tp = False
    b_sl = -15.0
    b_peak = 0.0
    mfe = 0.0

    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])

        # Position A
        if a_open:
            if b['best'] >= 8.0:
                a_pips = 8.0 - COMMISSION
                a_open = False; a_bars = i+1; a_hit_tp = True
                # Activate BE + trail on B
                b_sl = max(b_sl, 0.0)
            elif b['worst'] <= -15.0:
                a_pips = -15.0 - COMMISSION
                a_open = False; a_bars = i+1

        # Position B
        if b_open:
            if b['best'] > b_peak:
                b_peak = b['best']
            # Trail only after A hits TP
            if a_hit_tp:
                trail = b_peak - 8.0
                if trail > b_sl:
                    b_sl = trail
            # Check SL
            if b['worst'] <= b_sl:
                b_pips = b_sl - COMMISSION
                b_open = False; b_bars = i+1

        if not a_open and not b_open:
            break

    if a_open:
        a_pips = bars[-1]['close'] - COMMISSION; a_bars = len(bars)
    if b_open:
        final = max(bars[-1]['close'], b_sl)
        b_pips = final - COMMISSION; b_bars = len(bars)

    return dict(pips=0.5*a_pips + 0.5*b_pips,
                bars_held=max(a_bars, b_bars), outcome='FREE_RIDE', mfe=mfe)


def strat_harvester(bars):
    """Strategy 7: Harvester. A=40% TP8/SL15, B=30% TP14/SL15 BE@8, C=30% trail@10 BE@8."""
    a_open, b_open, c_open = True, True, True
    a_pips, b_pips, c_pips = 0.0, 0.0, 0.0
    a_bars, b_bars, c_bars = 0, 0, 0
    b_sl, c_sl = -15.0, -15.0
    c_peak = 0.0
    mfe = 0.0

    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])

        # Tier A: 40%, TP=8
        if a_open:
            if b['best'] >= 8.0:
                a_pips = 8.0 - COMMISSION
                a_open = False; a_bars = i+1
            elif b['worst'] <= -15.0:
                a_pips = -15.0 - COMMISSION
                a_open = False; a_bars = i+1

        # Tier B: 30%, TP=14, BE at +8
        if b_open:
            if mfe >= 8.0 and b_sl < 0:
                b_sl = 0.0
            if b['best'] >= 14.0:
                b_pips = 14.0 - COMMISSION
                b_open = False; b_bars = i+1
            elif b['worst'] <= b_sl:
                b_pips = b_sl - COMMISSION
                b_open = False; b_bars = i+1

        # Tier C: 30%, no TP, trail@10, BE at +8
        if c_open:
            if b['best'] > c_peak:
                c_peak = b['best']
            if c_peak >= 8.0 and c_sl < 0:
                c_sl = 0.0
            if c_sl >= 0:
                trail = c_peak - 10.0
                if trail > c_sl:
                    c_sl = trail
            if b['worst'] <= c_sl:
                c_pips = c_sl - COMMISSION
                c_open = False; c_bars = i+1

        if not a_open and not b_open and not c_open:
            break

    if a_open:
        a_pips = bars[-1]['close'] - COMMISSION; a_bars = len(bars)
    if b_open:
        b_pips = max(bars[-1]['close'], b_sl) - COMMISSION; b_bars = len(bars)
    if c_open:
        c_pips = max(bars[-1]['close'], c_sl) - COMMISSION; c_bars = len(bars)

    return dict(pips=0.4*a_pips + 0.3*b_pips + 0.3*c_pips,
                bars_held=max(a_bars, b_bars, c_bars), outcome='HARVESTER', mfe=mfe)


# ── All strategies ────────────────────────────────────────────────────
STRATEGIES = {
    'Raw TP20':    strat_raw,
    'Twin Trade':  strat_twin,
    'Ladder':      strat_ladder,
    'No Ceiling':  strat_no_ceiling,
    'Time Decay':  strat_time_decay,
    'Free Ride':   strat_free_ride,
    'Harvester':   strat_harvester,
}

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

print("="*90)
print("WALK-FORWARD STRATEGY COMPARISON")
print("="*90)
print(f"\nParameters:")
print(f"  Base TP/SL: {TP_PIPS}/{SL_PIPS} pips | Max hold: {MAX_HOLDING} bars")
print(f"  Buy threshold: {BUY_THRESHOLD} | Spread: {BUY_SPREAD*100}%")
print(f"  Sell threshold: {SELL_THRESHOLD}")
print(f"  Circuit breaker: OFF | Walk-forward windows: {NUM_WINDOWS}")
print(f"  Commission: {COMMISSION} pips per position")
print(f"  Strategies: {len(STRATEGIES)}")

# Load data
print(f"\nLoading data...")
df = pd.read_csv(data_path)
print(f"Data: {len(df)} rows")

with open(features_path) as f:
    feature_names = json.load(f)

# Create labels
def create_buy_labels(df):
    labels = []
    for i in range(len(df) - MAX_HOLDING):
        entry = df.iloc[i]['close']
        tp_price = entry + TP_PIPS * 0.0001
        sl_price = entry - SL_PIPS * 0.0001
        hit_tp, hit_sl = False, False
        for j in range(i + 1, min(i + 1 + MAX_HOLDING, len(df))):
            if df.iloc[j]['high'] >= tp_price:
                hit_tp = True; break
            if df.iloc[j]['low'] <= sl_price:
                hit_sl = True; break
        if hit_tp:
            labels.append(0)
        elif hit_sl:
            labels.append(2)
        else:
            labels.append(1)
    labels.extend([1] * (len(df) - len(labels)))
    return labels

print("Creating labels...")
df['label'] = create_buy_labels(df)

total_bars = len(df)
window_size = total_bars // NUM_WINDOWS

# ── Walk-forward loop ─────────────────────────────────────────────────
# For each strategy, maintain independent tracking
results = {name: [] for name in STRATEGIES}
window_results = {name: {w: [] for w in range(1, NUM_WINDOWS + 1)} for name in STRATEGIES}

print(f"\nRunning walk-forward across {NUM_WINDOWS} windows...")

for window in range(1, NUM_WINDOWS + 1):
    train_end = window * window_size
    test_start = train_end
    test_end = min((window + 1) * window_size, total_bars)

    if test_start >= total_bars:
        break

    # Train
    X_train = df.iloc[:train_end][feature_names].values
    y_train = df.iloc[:train_end]['label'].values

    for needed in [0, 1, 2]:
        if needed not in y_train:
            X_train = np.vstack([X_train, X_train[:1]])
            y_train = np.append(y_train, needed)

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        objective='multi:softprob', num_class=3,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    # Predict
    X_test = df.iloc[test_start:test_end][feature_names].values
    probs = model.predict_proba(X_test)

    # Generate signal list: (test_offset, direction)
    signals = []
    for i in range(len(X_test)):
        buy_prob = probs[i][0]
        range_prob = probs[i][1]
        sell_prob = probs[i][2]

        buy_spread = buy_prob - max(sell_prob, range_prob)
        if buy_prob >= BUY_THRESHOLD and buy_spread >= BUY_SPREAD:
            signals.append((i, 'BUY'))
        elif sell_prob >= SELL_THRESHOLD:
            signals.append((i, 'SELL'))

    # For each strategy, independently simulate trades
    for strat_name, strat_fn in STRATEGIES.items():
        next_avail = 0
        for sig_offset, direction in signals:
            if sig_offset < next_avail:
                continue  # still in a trade

            actual_idx = test_start + sig_offset
            if actual_idx + MAX_HOLDING >= len(df):
                break

            bar_data = get_bars(actual_idx, direction, df, MAX_HOLDING)
            if len(bar_data) == 0:
                continue

            result = strat_fn(bar_data)
            result['direction'] = direction
            result['window'] = window

            results[strat_name].append(result)
            window_results[strat_name][window].append(result)

            next_avail = sig_offset + result['bars_held']

    print(f"  Window {window}: {len(signals)} signals generated")

# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("RESULTS SUMMARY")
print("="*90)

print(f"\n {'Strategy':<16} | {'Trades':>7} | {'Wins':>6} | {'WinRate':>7} | {'Pips':>8} | {'Pips/Tr':>8} | {'Avg Bars':>8}")
print(f" {'-'*16}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

strategy_summaries = {}
for name in STRATEGIES:
    trades = results[name]
    n = len(trades)
    if n == 0:
        print(f" {name:<16} |       0 |      - |       - |        - |        - |        -")
        continue

    wins = sum(1 for t in trades if t['pips'] > 0)
    total_pips = sum(t['pips'] for t in trades)
    avg_pips = total_pips / n
    avg_bars = sum(t['bars_held'] for t in trades) / n
    wr = wins / n * 100

    strategy_summaries[name] = {
        'trades': n, 'wins': wins, 'wr': wr,
        'pips': total_pips, 'avg_pips': avg_pips, 'avg_bars': avg_bars
    }

    print(f" {name:<16} | {n:>7} | {wins:>6} | {wr:>6.1f}% | {total_pips:>+8.0f} | {avg_pips:>+8.2f} | {avg_bars:>8.1f}")

# ── Rank by total pips ────────────────────────────────────────────────
print(f"\n{'='*90}")
print("RANKING BY TOTAL PIPS")
print(f"{'='*90}")

ranked = sorted(strategy_summaries.items(), key=lambda x: x[1]['pips'], reverse=True)
for rank, (name, s) in enumerate(ranked, 1):
    baseline_diff = s['pips'] - strategy_summaries.get('Raw TP20', {}).get('pips', 0)
    print(f"  #{rank} {name:<16}  {s['pips']:>+8.0f} pips  ({s['wr']:.1f}% WR, {s['trades']} trades, {s['avg_pips']:>+.2f}/trade)")

# ── Rank by pips per trade ────────────────────────────────────────────
print(f"\n{'='*90}")
print("RANKING BY PIPS PER TRADE (EV)")
print(f"{'='*90}")

ranked_ev = sorted(strategy_summaries.items(), key=lambda x: x[1]['avg_pips'], reverse=True)
for rank, (name, s) in enumerate(ranked_ev, 1):
    print(f"  #{rank} {name:<16}  {s['avg_pips']:>+8.2f} pips/trade  ({s['trades']} trades × {s['avg_pips']:>+.2f} = {s['pips']:>+.0f} total)")

# ── Per-window detail for top 3 ──────────────────────────────────────
print(f"\n{'='*90}")
print("PER-WINDOW DETAIL (Top 3 by total pips)")
print(f"{'='*90}")

top3 = [name for name, _ in ranked[:3]]
for name in top3:
    print(f"\n  {name}:")
    print(f"  {'Win':>5} | {'Trades':>7} | {'Wins':>6} | {'WR':>6} | {'Pips':>8} | {'Pips/Tr':>8}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for w in range(1, NUM_WINDOWS + 1):
        wt = window_results[name][w]
        n = len(wt)
        if n == 0:
            print(f"  W {w:>2} |       0 |      - |      - |        - |        -")
            continue
        wins = sum(1 for t in wt if t['pips'] > 0)
        pips = sum(t['pips'] for t in wt)
        wr = wins / n * 100
        avg = pips / n
        print(f"  W {w:>2} | {n:>7} | {wins:>6} | {wr:>5.1f}% | {pips:>+8.0f} | {avg:>+8.2f}")

# ── Direction breakdown ───────────────────────────────────────────────
print(f"\n{'='*90}")
print("BUY vs SELL BREAKDOWN")
print(f"{'='*90}")

for name in STRATEGIES:
    trades = results[name]
    if not trades:
        continue
    buys = [t for t in trades if t['direction'] == 'BUY']
    sells = [t for t in trades if t['direction'] == 'SELL']

    buy_pips = sum(t['pips'] for t in buys) if buys else 0
    sell_pips = sum(t['pips'] for t in sells) if sells else 0
    buy_wr = (sum(1 for t in buys if t['pips'] > 0) / len(buys) * 100) if buys else 0
    sell_wr = (sum(1 for t in sells if t['pips'] > 0) / len(sells) * 100) if sells else 0

    print(f"  {name:<16}  BUY: {len(buys):>4} trades, {buy_wr:>5.1f}% WR, {buy_pips:>+7.0f} pips  |  "
          f"SELL: {len(sells):>4} trades, {sell_wr:>5.1f}% WR, {sell_pips:>+7.0f} pips")

# ── Estimated vs Actual EV comparison ─────────────────────────────────
print(f"\n{'='*90}")
print("ESTIMATED vs ACTUAL EV")
print(f"{'='*90}")

estimates = {
    'Raw TP20':    0.33,
    'Twin Trade':  2.08,
    'Ladder':      4.32,
    'No Ceiling':  5.45,  # midpoint of range
    'Time Decay':  3.25,  # midpoint of range
    'Free Ride':   2.57,
    'Harvester':   2.34,
}

print(f"\n {'Strategy':<16} | {'Est. EV':>8} | {'Actual EV':>9} | {'Diff':>8} | {'Accurate?':>10}")
print(f" {'-'*16}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")

for name in STRATEGIES:
    if name not in strategy_summaries:
        continue
    est = estimates.get(name, 0)
    actual = strategy_summaries[name]['avg_pips']
    diff = actual - est
    accurate = "YES" if abs(diff) < 1.0 else ("OVER" if est > actual else "UNDER")
    print(f" {name:<16} | {est:>+8.2f} | {actual:>+9.2f} | {diff:>+8.2f} | {accurate:>10}")

print(f"\n{'='*90}")
print("Analysis complete! Results saved to logs/walk_forward_strategy_comparison.log")
print(f"{'='*90}")
