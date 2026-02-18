"""
Walk-Forward: Hybrid TP Test

Tests direction-aware TP:
  BUY  signals → TP = 20.0 pips  (optimal from dollar_targets test)
  SELL signals → TP =  7.5 pips  (optimal from dollar_targets test)

Compared against four reference configs from the same walk-forward:
  BOTH @ 10.5p  — best single-TP BOTH result
  BOTH @ 20.0p  — current EA behaviour
  BUY_ONLY  @ 20.0p
  SELL_ONLY @  7.5p

All configs: no CB, no spread gate, 1% equity risk / trade, compounding.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from xgboost import XGBClassifier

# ── Parameters ────────────────────────────────────────────────────────
STARTING_EQUITY = 10_000.0
RISK_PCT        = 0.01
SL_PIPS         = 15.0
MAX_HOLDING     = 50
COMMISSION      = 0.1
PIP_VALUE       = 10.0

BUY_THRESHOLD   = 0.40
SELL_THRESHOLD  = 0.56
NUM_WINDOWS     = 5

BUY_TP  = 20.0   # optimal for buy channel
SELL_TP =  7.5   # optimal for sell channel

# ── Paths ─────────────────────────────────────────────────────────────
project       = Path(__file__).parent.parent
data_path     = project / "data" / "EURUSD_H1_clean.csv"
features_path = project / "features_used_buy.json"
log_path      = project / "logs" / "walk_forward_hybrid_tp.log"

# ── Logger ────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, fp):
        fp.parent.mkdir(exist_ok=True)
        self.t = sys.stdout
        self.f = open(fp, 'w', encoding='utf-8')
    def write(self, m): self.t.write(m); self.f.write(m)
    def flush(self):    self.t.flush();  self.f.flush()

sys.stdout = Logger(log_path)

# ── Helpers ────────────────────────────────────────────────────────────
def get_bars(entry_idx, direction, df):
    ep = df.iloc[entry_idx]['close']
    bars = []
    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLDING, len(df))):
        h, l, c = df.iloc[j]['high'], df.iloc[j]['low'], df.iloc[j]['close']
        if direction == 'BUY':
            bars.append(((h - ep)*1e4, (l - ep)*1e4, (c - ep)*1e4))
        else:
            bars.append(((ep - l)*1e4, (ep - h)*1e4, (ep - c)*1e4))
    return bars

def simulate(bars, tp_pips):
    for i, (best, worst, close) in enumerate(bars):
        if best  >= tp_pips:   return tp_pips  - COMMISSION, i + 1
        if worst <= -SL_PIPS:  return -SL_PIPS - COMMISSION, i + 1
    return max(bars[-1][2], -SL_PIPS) - COMMISSION, len(bars)

def compound_run(signals, buy_tp, sell_tp=None):
    """
    signals: list of (bar_idx, direction, bars)
    If sell_tp is None, buy_tp is used for all directions.
    Returns (final_equity, n_trades, win_rate, buy_trades, sell_trades,
             buy_wins, sell_wins, buy_pips_total, sell_pips_total)
    """
    equity    = STARTING_EQUITY
    nxt       = 0
    b_trades = s_trades = 0
    b_wins   = s_wins   = 0
    b_pips   = s_pips   = 0.0

    for bar_idx, direction, bars in signals:
        if bar_idx < nxt:
            continue
        tp          = (sell_tp if (sell_tp and direction == 'SELL') else buy_tp)
        pip_result, bars_held = simulate(bars, tp)
        lot         = equity * RISK_PCT / (SL_PIPS * PIP_VALUE)
        equity      = max(equity + pip_result * lot * PIP_VALUE, 1.0)
        nxt         = bar_idx + bars_held

        if direction == 'BUY':
            b_trades += 1
            b_pips   += pip_result
            if pip_result > 0: b_wins += 1
        else:
            s_trades += 1
            s_pips   += pip_result
            if pip_result > 0: s_wins += 1

    total = b_trades + s_trades
    wr    = (b_wins + s_wins) / total * 100 if total > 0 else 0.0
    return equity, total, wr, b_trades, s_trades, b_wins, s_wins, b_pips, s_pips


# ── Label creation ─────────────────────────────────────────────────────
def create_labels(df):
    labels = []
    for i in range(len(df) - MAX_HOLDING):
        e    = df.iloc[i]['close']
        tp_p = e + 20.0 * 0.0001
        sl_p = e - SL_PIPS * 0.0001
        ht = hs = False
        for j in range(i + 1, min(i + 1 + MAX_HOLDING, len(df))):
            if df.iloc[j]['high'] >= tp_p: ht = True; break
            if df.iloc[j]['low']  <= sl_p: hs = True; break
        labels.append(0 if ht else (2 if hs else 1))
    labels.extend([1] * (len(df) - len(labels)))
    return labels


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

SEP = "=" * 75

print(SEP)
print("WALK-FORWARD: HYBRID TP TEST")
print(SEP)
print(f"\nHybrid: BUY @ {BUY_TP}p  |  SELL @ {SELL_TP}p")
print(f"Starting equity: ${STARTING_EQUITY:,.0f}  |  Risk: 1%/trade  |  No CB  |  No spread gate")

print("\nLoading data...", flush=True)
df = pd.read_csv(data_path)
n  = len(df)
years = n / 6240
print(f"Data: {n} rows  (~{years:.2f} trading years)")

with open(features_path) as f:
    feature_names = json.load(f)

print("Creating labels...", flush=True)
df['label'] = create_labels(df)
window_size = n // NUM_WINDOWS

# ── Signal modes to collect ────────────────────────────────────────────
# BOTH and BUY_ONLY/SELL_ONLY can share one model training run per window.
# We collect BOTH signals (with direction preserved) and derive subsets.

print("\nRunning walk-forward (5 windows)...", flush=True)
all_signals = []   # (bar_idx, direction, bars) — BOTH mode, no gate

for w in range(1, NUM_WINDOWS + 1):
    train_end  = w * window_size
    test_start = train_end
    test_end   = min((w + 1) * window_size, n)
    if test_start >= n:
        break

    X_tr = df.iloc[:train_end][feature_names].values
    y_tr = df.iloc[:train_end]['label'].values
    for cls in [0, 1, 2]:
        if cls not in y_tr:
            X_tr = np.vstack([X_tr, X_tr[:1]])
            y_tr = np.append(y_tr, cls)

    mdl = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03,
                        objective='multi:softprob', num_class=3,
                        random_state=42, verbosity=0)
    mdl.fit(X_tr, y_tr)
    probs = mdl.predict_proba(df.iloc[test_start:test_end][feature_names].values)

    for i, p in enumerate(probs):
        ai = test_start + i
        if ai + MAX_HOLDING >= n:
            break
        bp, sp = p[0], p[2]
        if bp >= BUY_THRESHOLD:
            bars = get_bars(ai, 'BUY', df)
            if bars: all_signals.append((ai, 'BUY', bars))
        elif sp >= SELL_THRESHOLD:
            bars = get_bars(ai, 'SELL', df)
            if bars: all_signals.append((ai, 'SELL', bars))

    print(f"  Window {w}: {test_end - test_start} bars tested", flush=True)

buy_signals  = [(i, d, b) for i, d, b in all_signals if d == 'BUY']
sell_signals = [(i, d, b) for i, d, b in all_signals if d == 'SELL']

print(f"\nTotal signals: {len(all_signals)}  "
      f"(BUY: {len(buy_signals)}, SELL: {len(sell_signals)})")

# ── Run all configs ────────────────────────────────────────────────────
configs = [
    ("HYBRID  BUY@20 + SELL@7.5", all_signals,  BUY_TP, SELL_TP),
    ("BOTH    @ 10.5p",            all_signals,  10.5,   None),
    ("BOTH    @ 20.0p (EA)",       all_signals,  20.0,   None),
    ("BUY_ONLY  @ 20.0p",          buy_signals,  20.0,   None),
    ("SELL_ONLY @  7.5p",          sell_signals,  7.5,   None),
]

print(f"\n{SEP}")
print("RESULTS")
print(SEP)
print(f"\n  {'Config':<30} | {'Trades':>7} | {'WR':>7} | {'Final Eq':>12} | {'Gain':>9} | {'Ann ROI':>8}")
print(f"  {'-'*30}-+-{'-'*7}-+-{'-'*7}-+-{'-'*12}-+-{'-'*9}-+-{'-'*8}")

results = []
for label, signals, buy_tp, sell_tp in configs:
    eq, total, wr, bt, st, bw, sw, bp, sp = compound_run(signals, buy_tp, sell_tp)
    gain    = eq - STARTING_EQUITY
    ann_roi = gain / STARTING_EQUITY / years * 100
    results.append((label, eq, total, wr, bt, st, bw, sw, bp, sp, gain, ann_roi))
    print(f"  {label:<30} | {total:>7,} | {wr:>6.1f}% | ${eq:>11,.0f} | {gain:>+9,.0f} | {ann_roi:>+7.1f}%")

# ── Buy vs Sell breakdown ─────────────────────────────────────────────
print(f"\n{SEP}")
print("BUY vs SELL BREAKDOWN")
print(SEP)
print(f"\n  {'Config':<30} | {'Buy Tr':>6} | {'Buy WR':>7} | {'Buy Pips':>9} | "
      f"{'Sell Tr':>7} | {'Sell WR':>7} | {'Sell Pips':>9}")
print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}")

for label, eq, total, wr, bt, st, bw, sw, bp_sum, sp_sum, gain, ann_roi in results:
    b_wr = bw / bt * 100 if bt > 0 else 0.0
    s_wr = sw / st * 100 if st > 0 else 0.0
    print(f"  {label:<30} | {bt:>6,} | {b_wr:>6.1f}% | {bp_sum:>+9.0f}p | "
          f"{st:>7,} | {s_wr:>6.1f}% | {sp_sum:>+9.0f}p")

# ── Monthly breakdown ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("DOLLAR SUMMARY")
print(SEP)
print(f"\n  Period: ~{years:.2f} years  |  Starting: ${STARTING_EQUITY:,.0f}")
print()
for label, eq, total, wr, bt, st, bw, sw, bp_sum, sp_sum, gain, ann_roi in results:
    monthly = gain / years / 12
    print(f"  {label}")
    print(f"    Final equity : ${eq:>10,.0f}")
    print(f"    Total gain   : ${gain:>+10,.0f}  ({ann_roi:>+.1f}%/yr)")
    print(f"    Per month    : ${monthly:>+10,.0f}/month  (non-compounding after 1st year: ${gain/years/12:,.0f})")
    print()

print(SEP)
print(f"Analysis complete!  Log: {log_path}")
print(SEP)
