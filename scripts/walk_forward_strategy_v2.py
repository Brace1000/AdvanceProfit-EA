"""
Walk-Forward Strategy Comparison v2

Tests 7 exit strategies + 1 hybrid across 5 configurations:
  A: BUY_ONLY + Circuit Breaker
  B: BOTH + Circuit Breaker
  C: BOTH + Circuit Breaker + Sell Spread Gate
  D: HYBRID (Raw TP20 Buy + Free Ride Sell) + CB
  E: HYBRID + CB + Sell Spread Gate

Circuit breaker: 5 consecutive losses OR 100-pip drawdown -> 48-bar pause
Sell spread gate: sell_prob - max(buy_prob, range_prob) >= threshold
  (matches buy's spread gate: buy_prob - max(sell_prob, range_prob) >= 1.5%)
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from xgboost import XGBClassifier

# ── Parameters ────────────────────────────────────────────────────────
TP_PIPS = 20.0
SL_PIPS = 15.0
MAX_HOLDING = 50
COMMISSION = 0.1  # pips per position

BUY_THRESHOLD = 0.40
BUY_SPREAD = 0.015
SELL_THRESHOLD = 0.56
SELL_SPREAD = 0.015   # same spread gate as buy, applied when sell range filter is ON
NUM_WINDOWS = 5

# Circuit breaker
CB_MAX_LOSSES = 5
CB_MAX_DD_PIPS = 100
CB_COOLDOWN = 48
CB_RESET_ON_WIN = True

# ── Paths ─────────────────────────────────────────────────────────────
project = Path(__file__).parent.parent
data_path = project / "data" / "EURUSD_H1_clean.csv"
features_path = project / "features_used_buy.json"
log_path = project / "logs" / "walk_forward_strategy_v2.log"

# ── Logging ───────────────────────────────────────────────────────────
class Logger:
    def __init__(self, filepath):
        filepath.parent.mkdir(exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)


# ══════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Tracks consecutive losses and drawdown. Pauses trading when triggered."""

    def __init__(self):
        self.consecutive_losses = 0
        self.peak_pnl = 0.0
        self.running_pnl = 0.0
        self.paused_until = -1
        self.triggers = 0
        self.skipped = 0

    def is_paused(self, bar_idx):
        return bar_idx < self.paused_until

    def update(self, pips, bar_idx):
        self.running_pnl += pips
        self.peak_pnl = max(self.peak_pnl, self.running_pnl)
        dd = self.peak_pnl - self.running_pnl

        if pips <= 0:
            self.consecutive_losses += 1
        elif CB_RESET_ON_WIN:
            self.consecutive_losses = 0

        if self.consecutive_losses >= CB_MAX_LOSSES or dd >= CB_MAX_DD_PIPS:
            self.paused_until = bar_idx + CB_COOLDOWN
            self.triggers += 1
            self.consecutive_losses = 0


# ══════════════════════════════════════════════════════════════════════
# BAR DATA HELPER
# ══════════════════════════════════════════════════════════════════════

def get_bars(entry_idx, direction, df, max_bars=MAX_HOLDING):
    """Bar-by-bar pips relative to entry. best=favorable, worst=adverse."""
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
# ══════════════════════════════════════════════════════════════════════

def strat_raw(bars):
    """Raw single trade. TP=20, SL=15."""
    mfe = 0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        if b['best'] >= 20.0:
            return dict(pips=20.0 - COMMISSION, bars_held=i+1, outcome='TP', mfe=mfe)
        if b['worst'] <= -15.0:
            return dict(pips=-15.0 - COMMISSION, bars_held=i+1, outcome='SL', mfe=mfe)
    return dict(pips=bars[-1]['close'] - COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_twin(bars):
    """Twin Trade. A=50% TP10/SL15, B=50% TP20/SL15 BE@10 trail@8."""
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
                a_pips = 10.0 - COMMISSION; a_open = False; a_bars = i+1
            elif b['worst'] <= -15.0:
                a_pips = -15.0 - COMMISSION; a_open = False; a_bars = i+1
        if b_open:
            if b['best'] > b_peak:
                b_peak = b['best']
            if b_peak >= 10.0 and b_sl < 0:
                b_sl = 0.0
            if b_sl >= 0:
                trail = b_peak - 8.0
                if trail > b_sl:
                    b_sl = trail
            if b['best'] >= 20.0:
                b_pips = 20.0 - COMMISSION; b_open = False; b_bars = i+1
            elif b['worst'] <= b_sl:
                b_pips = b_sl - COMMISSION; b_open = False; b_bars = i+1
        if not a_open and not b_open:
            break

    if a_open:
        a_pips = bars[-1]['close'] - COMMISSION; a_bars = len(bars)
    if b_open:
        b_pips = max(bars[-1]['close'], b_sl) - COMMISSION; b_bars = len(bars)

    return dict(pips=0.5*a_pips + 0.5*b_pips,
                bars_held=max(a_bars, b_bars), outcome='TWIN', mfe=mfe)


def strat_ladder(bars):
    """The Ladder. TP=20, progressive SL ratchet."""
    tp = 20.0
    sl = -15.0
    mfe = 0.0
    ratchets = [(4.0, -10.0), (8.0, -3.0), (10.0, 0.0), (14.0, 8.0), (17.0, 12.0)]
    r_idx = 0

    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
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
    """No Ceiling. No TP, trail@8 activates at +5, BE at +10."""
    sl = -15.0
    peak = 0.0
    trailing = False
    mfe = 0.0

    for i, b in enumerate(bars):
        if b['best'] > peak:
            peak = b['best']
        mfe = max(mfe, peak)
        if peak >= 5.0:
            trailing = True
        if peak >= 10.0 and sl < 0:
            sl = 0.0
        if trailing:
            trail = peak - 8.0
            if trail > sl:
                sl = trail
        if b['worst'] <= sl:
            return dict(pips=sl - COMMISSION, bars_held=i+1, outcome='TRAIL', mfe=mfe)

    final = max(bars[-1]['close'], sl)
    return dict(pips=final - COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_time_decay(bars):
    """Time Decay. TP/SL tighten over time, force close after 30 bars."""
    mfe = 0.0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        n = i + 1
        if n <= 10:
            tp, sl = 20.0, -15.0
        elif n <= 20:
            tp, sl = 14.0, -12.0
        elif n <= 30:
            tp, sl = 14.0, -8.0
            if b['best'] >= 3.0:
                return dict(pips=3.0 - COMMISSION, bars_held=n, outcome='TD_PROFIT', mfe=mfe)
        else:
            return dict(pips=b['close'] - COMMISSION, bars_held=n, outcome='TD_CLOSE', mfe=mfe)
        if b['best'] >= tp:
            return dict(pips=tp - COMMISSION, bars_held=n, outcome='TP', mfe=mfe)
        if b['worst'] <= sl:
            return dict(pips=sl - COMMISSION, bars_held=n, outcome='SL', mfe=mfe)

    return dict(pips=bars[-1]['close'] - COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_free_ride(bars):
    """Free Ride. A=50% TP8/SL15, B=50% trail, BE when A hits TP."""
    a_open, b_open = True, True
    a_pips, b_pips = 0.0, 0.0
    a_bars, b_bars = 0, 0
    a_hit_tp = False
    b_sl = -15.0
    b_peak = 0.0
    mfe = 0.0

    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        if a_open:
            if b['best'] >= 8.0:
                a_pips = 8.0 - COMMISSION; a_open = False; a_bars = i+1; a_hit_tp = True
                b_sl = max(b_sl, 0.0)
            elif b['worst'] <= -15.0:
                a_pips = -15.0 - COMMISSION; a_open = False; a_bars = i+1
        if b_open:
            if b['best'] > b_peak:
                b_peak = b['best']
            if a_hit_tp:
                trail = b_peak - 8.0
                if trail > b_sl:
                    b_sl = trail
            if b['worst'] <= b_sl:
                b_pips = b_sl - COMMISSION; b_open = False; b_bars = i+1
        if not a_open and not b_open:
            break

    if a_open:
        a_pips = bars[-1]['close'] - COMMISSION; a_bars = len(bars)
    if b_open:
        b_pips = max(bars[-1]['close'], b_sl) - COMMISSION; b_bars = len(bars)

    return dict(pips=0.5*a_pips + 0.5*b_pips,
                bars_held=max(a_bars, b_bars), outcome='FREE_RIDE', mfe=mfe)


def strat_harvester(bars):
    """Harvester. A=40% TP8/SL15, B=30% TP14/SL15 BE@8, C=30% trail@10 BE@8."""
    a_open, b_open, c_open = True, True, True
    a_pips, b_pips, c_pips = 0.0, 0.0, 0.0
    a_bars, b_bars, c_bars = 0, 0, 0
    b_sl, c_sl = -15.0, -15.0
    c_peak = 0.0
    mfe = 0.0

    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        if a_open:
            if b['best'] >= 8.0:
                a_pips = 8.0 - COMMISSION; a_open = False; a_bars = i+1
            elif b['worst'] <= -15.0:
                a_pips = -15.0 - COMMISSION; a_open = False; a_bars = i+1
        if b_open:
            if mfe >= 8.0 and b_sl < 0:
                b_sl = 0.0
            if b['best'] >= 14.0:
                b_pips = 14.0 - COMMISSION; b_open = False; b_bars = i+1
            elif b['worst'] <= b_sl:
                b_pips = b_sl - COMMISSION; b_open = False; b_bars = i+1
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
                c_pips = c_sl - COMMISSION; c_open = False; c_bars = i+1
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


# ── Strategy registry ────────────────────────────────────────────────
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
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_signals(probs, mode, sell_spread_gate=False):
    """Generate signal list from model probabilities.

    Args:
        probs: model.predict_proba output (n_samples, 3)
        mode: 'BUY_ONLY', 'BOTH', or 'HYBRID'
        sell_spread_gate: if True, sell signals also require spread check
            sell_prob - max(buy_prob, range_prob) >= SELL_SPREAD

    Returns list of (test_offset, direction)
    """
    signals = []
    for i in range(len(probs)):
        buy_prob = probs[i][0]
        range_prob = probs[i][1]
        sell_prob = probs[i][2]

        buy_spread = buy_prob - max(sell_prob, range_prob)

        if mode in ('BUY_ONLY',):
            # Buy signals only
            if buy_prob >= BUY_THRESHOLD and buy_spread >= BUY_SPREAD:
                signals.append((i, 'BUY'))

        elif mode in ('BOTH', 'HYBRID'):
            # Buy priority, then sell
            if buy_prob >= BUY_THRESHOLD and buy_spread >= BUY_SPREAD:
                signals.append((i, 'BUY'))
            elif sell_prob >= SELL_THRESHOLD:
                # Optional: require sell spread gate (same logic as buy spread)
                if sell_spread_gate:
                    sell_spread = sell_prob - max(buy_prob, range_prob)
                    if sell_spread >= SELL_SPREAD:
                        signals.append((i, 'SELL'))
                else:
                    signals.append((i, 'SELL'))

    return signals


# ══════════════════════════════════════════════════════════════════════
# RUN ONE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

def run_config(config_name, mode, use_cb, sell_spread_gate, df, feature_names):
    """Run all strategies for a single configuration.

    Returns dict: {strategy_name: [list of trade dicts]}
    """
    total_bars = len(df)
    window_size = total_bars // NUM_WINDOWS

    # For HYBRID mode, we only run the hybrid dispatch (not all 7)
    if mode == 'HYBRID':
        strat_names = ['HYBRID']
    else:
        strat_names = list(STRATEGIES.keys())

    results = {name: [] for name in strat_names}
    cb_info = {name: {'triggers': 0, 'skipped': 0} for name in strat_names}

    for window in range(1, NUM_WINDOWS + 1):
        train_end = window * window_size
        test_start = train_end
        test_end = min((window + 1) * window_size, total_bars)

        if test_start >= total_bars:
            break

        # ── Train ──
        X_train = df.iloc[:train_end][feature_names].values
        y_train = df.iloc[:train_end]['label'].values

        # Ensure all 3 classes present
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

        # ── Predict ──
        test_df = df.iloc[test_start:test_end]
        X_test = test_df[feature_names].values
        probs = model.predict_proba(X_test)

        # ── Generate signals ──
        signals = generate_signals(probs, mode, sell_spread_gate)

        # ── Run each strategy with independent CB ──
        for strat_name in strat_names:
            cb = CircuitBreaker() if use_cb else None
            next_avail = 0

            for sig_offset, direction in signals:
                if sig_offset < next_avail:
                    continue

                actual_idx = test_start + sig_offset
                if actual_idx + MAX_HOLDING >= len(df):
                    break

                # Circuit breaker check
                if cb and cb.is_paused(actual_idx):
                    cb.skipped += 1
                    continue

                bar_data = get_bars(actual_idx, direction, df, MAX_HOLDING)
                if len(bar_data) == 0:
                    continue

                # Dispatch strategy
                if strat_name == 'HYBRID':
                    if direction == 'BUY':
                        result = strat_raw(bar_data)
                    else:
                        result = strat_free_ride(bar_data)
                else:
                    result = STRATEGIES[strat_name](bar_data)

                result['direction'] = direction
                result['window'] = window

                results[strat_name].append(result)

                # Update circuit breaker with net pips
                if cb:
                    cb.update(result['pips'], actual_idx)

                next_avail = sig_offset + result['bars_held']

            # Track CB stats across windows
            if cb:
                cb_info[strat_name]['triggers'] += cb.triggers
                cb_info[strat_name]['skipped'] += cb.skipped

    return results, cb_info


# ══════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════

def summarize(trades):
    """Compute summary stats for a list of trades."""
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'wins': 0, 'wr': 0, 'pips': 0, 'avg_pips': 0, 'avg_bars': 0}
    wins = sum(1 for t in trades if t['pips'] > 0)
    total_pips = sum(t['pips'] for t in trades)
    avg_bars = sum(t['bars_held'] for t in trades) / n
    return {
        'trades': n, 'wins': wins, 'wr': wins / n * 100,
        'pips': total_pips, 'avg_pips': total_pips / n, 'avg_bars': avg_bars
    }


def print_config_results(config_name, results, cb_info):
    """Print full results for one configuration."""
    print(f"\n{'='*95}")
    print(f"  {config_name}")
    print(f"{'='*95}")

    print(f"\n {'Strategy':<16} | {'Trades':>7} | {'Wins':>6} | {'WR':>7} | "
          f"{'Pips':>8} | {'Pips/Tr':>8} | {'CB':>4} | {'Skip':>5}")
    print(f" {'-'*16}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-"
          f"{'-'*8}-+-{'-'*8}-+-{'-'*4}-+-{'-'*5}")

    summaries = {}
    for name in results:
        s = summarize(results[name])
        summaries[name] = s
        cbi = cb_info.get(name, {'triggers': 0, 'skipped': 0})

        if s['trades'] == 0:
            print(f" {name:<16} |       0 |      - |       - |"
                  f"        - |        - |    - |     -")
        else:
            print(f" {name:<16} | {s['trades']:>7} | {s['wins']:>6} | {s['wr']:>6.1f}% | "
                  f"{s['pips']:>+8.0f} | {s['avg_pips']:>+8.2f} | {cbi['triggers']:>4} | {cbi['skipped']:>5}")

    # Ranking
    ranked = sorted(summaries.items(), key=lambda x: x[1]['pips'], reverse=True)
    print(f"\n  Ranking:")
    for rank, (name, s) in enumerate(ranked, 1):
        if s['trades'] > 0:
            print(f"    #{rank} {name:<16} {s['pips']:>+8.0f} pips  "
                  f"({s['wr']:.1f}% WR, {s['trades']} trades, {s['avg_pips']:>+.2f}/trade)")

    # Buy vs Sell breakdown
    has_sell = any(any(t['direction'] == 'SELL' for t in results[name]) for name in results)
    if has_sell:
        print(f"\n  Buy vs Sell:")
        for name in results:
            trades = results[name]
            if not trades:
                continue
            buys = [t for t in trades if t['direction'] == 'BUY']
            sells = [t for t in trades if t['direction'] == 'SELL']
            bp = sum(t['pips'] for t in buys) if buys else 0
            sp = sum(t['pips'] for t in sells) if sells else 0
            bwr = (sum(1 for t in buys if t['pips'] > 0) / len(buys) * 100) if buys else 0
            swr = (sum(1 for t in sells if t['pips'] > 0) / len(sells) * 100) if sells else 0
            print(f"    {name:<16} BUY: {len(buys):>4}tr {bwr:>5.1f}%WR {bp:>+7.0f}p | "
                  f"SELL: {len(sells):>4}tr {swr:>5.1f}%WR {sp:>+7.0f}p")

    return summaries


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

print("=" * 95)
print("WALK-FORWARD STRATEGY COMPARISON v2")
print("=" * 95)
print(f"\nParameters:")
print(f"  TP/SL: {TP_PIPS}/{SL_PIPS} pips | Max hold: {MAX_HOLDING} bars | Commission: {COMMISSION} pip/pos")
print(f"  Buy threshold: {BUY_THRESHOLD} | Spread gate: {BUY_SPREAD*100}%")
print(f"  Sell threshold: {SELL_THRESHOLD}")
print(f"  Circuit breaker: {CB_MAX_LOSSES} losses OR {CB_MAX_DD_PIPS}-pip DD -> {CB_COOLDOWN}-bar pause")
print(f"  Sell spread gate: sell_prob - max(buy_prob, range_prob) >= {SELL_SPREAD*100}%")
print(f"  Walk-forward windows: {NUM_WINDOWS}")
print(f"  Strategies: 7 + HYBRID (Raw TP20 Buy + Free Ride Sell)")

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

# ── Run all configurations ────────────────────────────────────────────
CONFIGS = [
    ('A: BUY_ONLY + CB',              'BUY_ONLY', True,  False),
    ('B: BOTH + CB',                  'BOTH',     True,  False),
    ('C: BOTH + CB + Sell Spread',    'BOTH',     True,  True),
    ('D: HYBRID + CB',                'HYBRID',   True,  False),
    ('E: HYBRID + CB + Sell Spread',  'HYBRID',   True,  True),
]

all_summaries = {}

for config_name, mode, use_cb, use_regime in CONFIGS:
    print(f"\n>>> Running: {config_name} ...")
    results, cb_info = run_config(config_name, mode, use_cb, use_regime, df, feature_names)
    summaries = print_config_results(config_name, results, cb_info)
    all_summaries[config_name] = summaries


# ══════════════════════════════════════════════════════════════════════
# CROSS-CONFIGURATION COMPARISON
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*95}")
print("CROSS-CONFIGURATION COMPARISON")
print(f"{'='*95}")

# Best strategy per config
print(f"\n  Best strategy per configuration:")
print(f"  {'Config':<28} | {'Best Strategy':<16} | {'Pips':>8} | {'Trades':>7} | {'WR':>7} | {'Pips/Tr':>8}")
print(f"  {'-'*28}-+-{'-'*16}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

for config_name, _, _, _ in CONFIGS:
    sums = all_summaries[config_name]
    if not sums:
        continue
    best_name, best_s = max(sums.items(), key=lambda x: x[1]['pips'])
    if best_s['trades'] > 0:
        print(f"  {config_name:<28} | {best_name:<16} | {best_s['pips']:>+8.0f} | "
              f"{best_s['trades']:>7} | {best_s['wr']:>6.1f}% | {best_s['avg_pips']:>+8.2f}")

# Raw TP20 across configs (apples-to-apples)
print(f"\n  Raw TP20 across configurations:")
print(f"  {'Config':<28} | {'Trades':>7} | {'WR':>7} | {'Pips':>8} | {'Pips/Tr':>8}")
print(f"  {'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

for config_name, mode, _, _ in CONFIGS:
    sums = all_summaries[config_name]
    key = 'Raw TP20' if 'Raw TP20' in sums else 'HYBRID'
    s = sums.get(key, {})
    if s and s.get('trades', 0) > 0:
        label = key if key == 'HYBRID' else 'Raw TP20'
        print(f"  {config_name:<28} | {s['trades']:>7} | {s['wr']:>6.1f}% | "
              f"{s['pips']:>+8.0f} | {s['avg_pips']:>+8.2f}")

# HYBRID vs best individual strategy
print(f"\n  HYBRID vs best alternatives:")
for config_suffix, mode_label in [('B: BOTH + CB', 'BOTH+CB'), ('C: BOTH + CB + Sell Spread', 'BOTH+CB+SellSpread')]:
    sums_both = all_summaries.get(config_suffix, {})
    hybrid_key = config_suffix.replace('B: BOTH', 'D: HYBRID').replace('C: BOTH', 'E: HYBRID')
    sums_hybrid = all_summaries.get(hybrid_key, {})

    if sums_both and sums_hybrid:
        best_both_name, best_both = max(sums_both.items(), key=lambda x: x[1]['pips'])
        hybrid_s = sums_hybrid.get('HYBRID', {})
        if best_both.get('trades', 0) > 0 and hybrid_s.get('trades', 0) > 0:
            print(f"    {mode_label}:")
            print(f"      Best individual: {best_both_name:<16} {best_both['pips']:>+8.0f} pips ({best_both['trades']} trades)")
            print(f"      HYBRID:          {'Raw+FreeRide':<16} {hybrid_s['pips']:>+8.0f} pips ({hybrid_s['trades']} trades)")
            diff = hybrid_s['pips'] - best_both['pips']
            print(f"      Difference:      {diff:>+8.0f} pips ({'HYBRID wins' if diff > 0 else 'Individual wins'})")


# ══════════════════════════════════════════════════════════════════════
# REFERENCE: Previous results without CB
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*95}")
print("REFERENCE: Impact of Circuit Breaker")
print(f"{'='*95}")
print(f"\n  Previous results (BOTH mode, NO circuit breaker, from v1):")
print(f"    Raw TP20:    -104 pips  (1539 trades, 42.9% WR)")
print(f"    Free Ride:   +305 pips  (2457 trades, 66.5% WR)")
print(f"    Twin Trade:  -194 pips  (2180 trades, 60.6% WR)")

print(f"\n  Original walk-forward (BUY_ONLY + CB, no spread gate, from threshold sweep):")
print(f"    Raw TP20:   +1177 pips  (836 trades, 47.2% WR)")
print(f"    Note: Used close-only labels & no spread gate. Different from this test.")


print(f"\n{'='*95}")
print(f"Analysis complete! Results saved to {log_path}")
print(f"{'='*95}")
