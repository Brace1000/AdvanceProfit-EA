"""
Walk-Forward: Dollar Target Compounding Test

Tests fixed pip TPs from 1.5 → 15 pips ($10 → $100 at $10k initial equity, 1% risk)
plus the reference TP=20 pips (current EA), with full equity compounding.

  Lot size each trade = equity × 0.01 / (15 × $10/pip)
  Win/Loss in $ = pip_result × lot_size × $10/pip

At $10,000 initial equity, 1% risk → lot = 0.6667, $/pip = $6.67:
  $10 target  →  1.5 pip TP  (need >90.9% WR to be +EV)
  $20 target  →  3.0 pip TP  (need >83.3% WR)
  $50 target  →  7.5 pip TP  (need >66.7% WR)
  $100 target → 15.0 pip TP  (need >50.0% WR — 1:1 RR)
  Current EA  → 20.0 pip TP  (need >42.9% WR — our actual WR)

Configs tested (no CB, no spread gate):
  A: BOTH      — buy_prob >= 0.40 OR sell_prob >= 0.56
  B: BUY_ONLY  — buy_prob >= 0.40
  C: SELL_ONLY — sell_prob >= 0.56
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
COMMISSION      = 0.1       # pips per position
PIP_VALUE       = 10.0      # USD/pip/lot (EUR/USD, USD account)

BUY_THRESHOLD   = 0.40
SELL_THRESHOLD  = 0.56
NUM_WINDOWS     = 5

# Dollar targets → pip TPs
# pip_TP = dollar_target × SL_PIPS / (STARTING_EQUITY × RISK_PCT)
#        = dollar_target × 15 / 100 = dollar_target × 0.15
DOLLAR_TARGETS  = list(range(10, 101, 10))      # [10, 20, 30, ..., 100]
PIP_TPS         = [d * SL_PIPS / (STARTING_EQUITY * RISK_PCT) for d in DOLLAR_TARGETS]
# [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0]

REFERENCE_TP_PIPS   = 20.0  # current EA
REFERENCE_DOLLAR    = REFERENCE_TP_PIPS / SL_PIPS * (STARTING_EQUITY * RISK_PCT)
# = 20/15 * $100 = $133.33

ALL_TPS    = PIP_TPS + [REFERENCE_TP_PIPS]
ALL_LABELS = [f"${d}" for d in DOLLAR_TARGETS] + ["$133\n(TP20)"]

# ── Paths ─────────────────────────────────────────────────────────────
project       = Path(__file__).parent.parent
data_path     = project / "data" / "EURUSD_H1_clean.csv"
features_path = project / "features_used_buy.json"
log_path      = project / "logs" / "walk_forward_dollar_targets.log"

# ── Logger ────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, fp):
        fp.parent.mkdir(exist_ok=True)
        self.t = sys.stdout
        self.f = open(fp, 'w', encoding='utf-8')
    def write(self, m): self.t.write(m); self.f.write(m)
    def flush(self):    self.t.flush();  self.f.flush()

sys.stdout = Logger(log_path)

# ── Bar data ──────────────────────────────────────────────────────────
def get_bars(entry_idx, direction, df):
    """Returns list of (best_pips, worst_pips, close_pips) from perspective of direction."""
    ep = df.iloc[entry_idx]['close']
    bars = []
    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLDING, len(df))):
        h, l, c = df.iloc[j]['high'], df.iloc[j]['low'], df.iloc[j]['close']
        if direction == 'BUY':
            bars.append(((h - ep) * 1e4, (l - ep) * 1e4, (c - ep) * 1e4))
        else:
            bars.append(((ep - l) * 1e4, (ep - h) * 1e4, (ep - c) * 1e4))
    return bars

# ── Trade simulation ───────────────────────────────────────────────────
def simulate(bars, tp_pips):
    """Returns (pip_result, bars_held)."""
    for i, (best, worst, close) in enumerate(bars):
        if best  >= tp_pips:   return tp_pips  - COMMISSION, i + 1
        if worst <= -SL_PIPS:  return -SL_PIPS - COMMISSION, i + 1
    return max(bars[-1][2], -SL_PIPS) - COMMISSION, len(bars)

# ── Compounding engine ─────────────────────────────────────────────────
def compound_run(signals_bars, tp_pips):
    """
    Process signals chronologically with non-overlap constraint.
    Lot size scales with current equity (compounding).
    Returns (final_equity, n_trades, win_rate_pct).
    """
    equity = STARTING_EQUITY
    nxt    = 0
    trades = 0
    wins   = 0
    for bar_idx, bars in signals_bars:
        if bar_idx < nxt:
            continue
        pip_result, bars_held = simulate(bars, tp_pips)
        lot    = equity * RISK_PCT / (SL_PIPS * PIP_VALUE)
        dollar = pip_result * lot * PIP_VALUE
        equity = max(equity + dollar, 1.0)
        nxt    = bar_idx + bars_held
        trades += 1
        if pip_result > 0:
            wins += 1
    wr = wins / trades * 100 if trades > 0 else 0.0
    return equity, trades, wr

# ── Label creation ─────────────────────────────────────────────────────
def create_labels(df):
    labels = []
    for i in range(len(df) - MAX_HOLDING):
        e    = df.iloc[i]['close']
        tp_p = e + 20.0 * 0.0001      # model trained on 20-pip TP labels
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

SEP = "=" * 85

print(SEP)
print("WALK-FORWARD: DOLLAR TARGET COMPOUNDING TEST")
print(SEP)
print(f"\nStarting equity : ${STARTING_EQUITY:,.0f}")
print(f"Risk per trade  : {RISK_PCT*100:.0f}%  →  $100 initial SL = 15 pips")
print(f"TP targets      : {', '.join(f'${d}({p:.1f}p)' for d,p in zip(DOLLAR_TARGETS, PIP_TPS))}")
print(f"Reference (EA)  : TP=20 pips = ${REFERENCE_DOLLAR:.0f} equiv at initial equity")
print(f"No CB | No spread gate | Buy≥{BUY_THRESHOLD} | Sell≥{SELL_THRESHOLD}")

print("\nLoading data...")
df = pd.read_csv(data_path)
n  = len(df)
years = n / 6240
print(f"Data: {n} rows  (~{years:.2f} trading years)")

with open(features_path) as f:
    feature_names = json.load(f)

print("Creating labels...", flush=True)
df['label'] = create_labels(df)
window_size = n // NUM_WINDOWS

CONFIGS = [
    ('A: BOTH (no gate)',  'BOTH'),
    ('B: BUY_ONLY',        'BUY_ONLY'),
    ('C: SELL_ONLY',       'SELL_ONLY'),
]

# ── Column layout ──────────────────────────────────────────────────────
CW      = 9   # column width
col_hdr = [f"${d}" for d in DOLLAR_TARGETS] + ["Ref $133"]

all_results = {}   # config -> list of (equity, trades, wr) per TP level

for config_name, mode in CONFIGS:
    print(f"\n{'='*85}")
    print(f"  {config_name}")
    print(f"{'='*85}")
    print(f"  Training walk-forward models...", flush=True)

    # ── Phase 1: signal collection ─────────────────────────────────
    raw_signals = []
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
            if mode == 'BUY_ONLY' and bp >= BUY_THRESHOLD:
                raw_signals.append((ai, 'BUY'))
            elif mode == 'SELL_ONLY' and sp >= SELL_THRESHOLD:
                raw_signals.append((ai, 'SELL'))
            elif mode == 'BOTH':
                if bp >= BUY_THRESHOLD:
                    raw_signals.append((ai, 'BUY'))
                elif sp >= SELL_THRESHOLD:
                    raw_signals.append((ai, 'SELL'))

    # ── Phase 2: pre-compute bar data ──────────────────────────────
    signals_bars = []
    for bar_idx, direction in raw_signals:
        bars = get_bars(bar_idx, direction, df)
        if bars:
            signals_bars.append((bar_idx, bars))
    print(f"  Raw signals: {len(raw_signals)}  |  Bars pre-computed: {len(signals_bars)}")

    # ── Phase 3: simulate each TP level ───────────────────────────
    col_results = []
    for tp_pips in ALL_TPS:
        eq, tr, wr = compound_run(signals_bars, tp_pips)
        col_results.append((eq, tr, wr))

    all_results[config_name] = col_results

    # ── Print per-config detail ────────────────────────────────────
    print(f"\n  {'TP (pips)':<12} | {'$ target':>8} | {'Trades':>7} | {'WR':>7} | {'Final Equity':>13} | {'Gain':>9} | {'Ann. ROI':>8}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*13}-+-{'-'*9}-+-{'-'*8}")
    for (d_tgt, pip_tp), (eq, tr, wr) in zip(
            list(zip(DOLLAR_TARGETS, PIP_TPS)) + [(int(REFERENCE_DOLLAR), REFERENCE_TP_PIPS)],
            col_results):
        gain    = eq - STARTING_EQUITY
        ann_roi = gain / STARTING_EQUITY / years * 100
        print(f"  {pip_tp:<12.1f} | ${d_tgt:>7,.0f} | {tr:>7} | {wr:>6.1f}% | ${eq:>12,.0f} | {gain:>+9,.0f} | {ann_roi:>+7.1f}%")


# ══════════════════════════════════════════════════════════════════════
# CROSS-CONFIG SUMMARY
# ══════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("CROSS-CONFIG FINAL EQUITY SUMMARY")
print(SEP)
print(f"\n  Period: ~{years:.2f} trading years  |  Starting: ${STARTING_EQUITY:,.0f}  |  1% risk/trade")
print()

# Header
print(f"  {'Config':<22} | " + " | ".join(f"{h:>{CW}}" for h in col_hdr))
print(f"  {'-'*22}-+-" + "-+-".join("-"*CW for _ in col_hdr))

for config_name, col_results in all_results.items():
    eq_str = " | ".join(f"${r[0]:>7,.0f}" for r in col_results)
    print(f"  {config_name:<22} | {eq_str}")

print()
print(f"  WR at each TP level (need WR > breakeven to be profitable):")
print(f"  {'Breakeven WR':<22} | " + " | ".join(
    f"{tp/(tp+SL_PIPS)*100:>{CW}.1f}%" for tp in ALL_TPS))
print()

# WR rows
for config_name, col_results in all_results.items():
    wr_str = " | ".join(f"{r[2]:>{CW}.1f}%" for r in col_results)
    print(f"  {config_name:<22} | {wr_str}")

print()
print(f"  Trades executed (overlap-adjusted) per TP level:")
for config_name, col_results in all_results.items():
    tr_str = " | ".join(f"{r[1]:>{CW},d}" for r in col_results)
    print(f"  {config_name:<22} | {tr_str}")

# ── Best TP per config ─────────────────────────────────────────────
print(f"\n{SEP}")
print("OPTIMAL TP PER CONFIG")
print(SEP)
for config_name, col_results in all_results.items():
    best_idx = max(range(len(col_results)), key=lambda i: col_results[i][0])
    best_eq, best_tr, best_wr = col_results[best_idx]
    if best_idx < len(DOLLAR_TARGETS):
        best_label = f"${DOLLAR_TARGETS[best_idx]} ({ALL_TPS[best_idx]:.1f} pips)"
    else:
        best_label = f"$133 (20 pips — current EA)"
    gain    = best_eq - STARTING_EQUITY
    ann_roi = gain / STARTING_EQUITY / years * 100
    print(f"  {config_name:<22}  Best TP: {best_label:<28}  "
          f"Final: ${best_eq:>10,.0f}  ({ann_roi:>+.1f}%/yr,  {best_wr:.1f}% WR,  {best_tr} trades)")

print(f"\n{SEP}")
print(f"Analysis complete!  Log: {log_path}")
print(SEP)
