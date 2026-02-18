"""
Walk-Forward: SL Comparison Test

Compares SL=15.0 pips vs SL=10.5 pips for the two best configs:
  1. HYBRID   — BUY @ 20p TP, SELL @ 7.5p TP
  2. BOTH     — uniform TP = 10.5p

At 1% risk, lot sizing scales inversely with SL:
  SL=15.0  → lot = equity × 0.01 / (15.0 × $10) = equity / 1500  → $6.67/pip at $10k
  SL=10.5  → lot = equity × 0.01 / (10.5 × $10) = equity / 1050  → $9.52/pip at $10k  (43% larger)

Breakeven WR shifts with SL:
  HYBRID BUY  TP=20, SL=15.0 → need >42.9% WR
  HYBRID BUY  TP=20, SL=10.5 → need >34.4% WR  (easier to clear)
  HYBRID SELL TP=7.5, SL=15.0 → need >66.7% WR
  HYBRID SELL TP=7.5, SL=10.5 → need >58.3% WR  (easier to clear, but tighter stop = more SL hits)

No CB | No spread gate | Compounding 1% risk/trade
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
MAX_HOLDING     = 50
COMMISSION      = 0.1
PIP_VALUE       = 10.0

BUY_THRESHOLD   = 0.40
SELL_THRESHOLD  = 0.56
NUM_WINDOWS     = 5

SL_VALUES  = [15.0, 10.5]       # the two SLs to compare
LABEL_SL   = 15.0               # SL used for model training labels (unchanged)

# ── Paths ─────────────────────────────────────────────────────────────
project       = Path(__file__).parent.parent
data_path     = project / "data" / "EURUSD_H1_clean.csv"
features_path = project / "features_used_buy.json"
log_path      = project / "logs" / "walk_forward_sl_test.log"

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
    """Pre-compute full MAX_HOLDING bars relative to entry (direction-aware)."""
    ep = df.iloc[entry_idx]['close']
    bars = []
    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLDING, len(df))):
        h, l, c = df.iloc[j]['high'], df.iloc[j]['low'], df.iloc[j]['close']
        if direction == 'BUY':
            bars.append(((h - ep)*1e4, (l - ep)*1e4, (c - ep)*1e4))
        else:
            bars.append(((ep - l)*1e4, (ep - h)*1e4, (ep - c)*1e4))
    return bars

def simulate(bars, tp_pips, sl_pips):
    """Returns (pip_result, bars_held) for given TP and SL."""
    for i, (best, worst, close) in enumerate(bars):
        if best  >= tp_pips:  return tp_pips  - COMMISSION, i + 1
        if worst <= -sl_pips: return -sl_pips - COMMISSION, i + 1
    return max(bars[-1][2], -sl_pips) - COMMISSION, len(bars)

def compound_run(signals, buy_tp, sl_pips, sell_tp=None):
    """
    signals : list of (bar_idx, direction, bars)
    buy_tp  : TP for BUY trades (and for SELL if sell_tp is None)
    sell_tp : TP for SELL trades (None = use buy_tp)
    sl_pips : stop-loss for ALL trades
    """
    equity   = STARTING_EQUITY
    nxt      = 0
    b_tr = s_tr = b_win = s_win = 0
    b_pip = s_pip = 0.0

    for bar_idx, direction, bars in signals:
        if bar_idx < nxt:
            continue
        tp          = sell_tp if (sell_tp and direction == 'SELL') else buy_tp
        pip_result, bars_held = simulate(bars, tp, sl_pips)
        lot         = equity * RISK_PCT / (sl_pips * PIP_VALUE)
        equity      = max(equity + pip_result * lot * PIP_VALUE, 1.0)
        nxt         = bar_idx + bars_held

        if direction == 'BUY':
            b_tr += 1; b_pip += pip_result
            if pip_result > 0: b_win += 1
        else:
            s_tr += 1; s_pip += pip_result
            if pip_result > 0: s_win += 1

    total = b_tr + s_tr
    wr    = (b_win + s_win) / total * 100 if total > 0 else 0.0
    return equity, total, wr, b_tr, s_tr, b_win, s_win, b_pip, s_pip

# ── Label creation ─────────────────────────────────────────────────────
def create_labels(df):
    labels = []
    for i in range(len(df) - MAX_HOLDING):
        e    = df.iloc[i]['close']
        tp_p = e + 20.0 * 0.0001
        sl_p = e - LABEL_SL * 0.0001
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

SEP = "=" * 80

print(SEP)
print("WALK-FORWARD: SL COMPARISON TEST  (SL=15.0 vs SL=10.5)")
print(SEP)
print(f"\nComparing:")
print(f"  Config 1: HYBRID — BUY @ 20p TP  |  SELL @ 7.5p TP")
print(f"  Config 2: BOTH   — uniform TP = 10.5p")
print(f"  SL values: {SL_VALUES}")
print(f"\nLot size at $10k, 1% risk:")
for sl in SL_VALUES:
    lot = STARTING_EQUITY * RISK_PCT / (sl * PIP_VALUE)
    print(f"  SL={sl:4.1f}p  →  {lot:.4f} lots  →  ${lot*PIP_VALUE:.2f}/pip")
print(f"\nNo CB | No spread gate | Compounding")

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

# ── Walk-forward: collect signals (one run, both SLs share same model) ─
print("\nRunning walk-forward (5 windows, model trains once)...", flush=True)
all_signals = []

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

    print(f"  Window {w}: done", flush=True)

buy_sigs  = [(i, d, b) for i, d, b in all_signals if d == 'BUY']
sell_sigs = [(i, d, b) for i, d, b in all_signals if d == 'SELL']
print(f"\nSignals: {len(all_signals)} total  (BUY: {len(buy_sigs)}, SELL: {len(sell_sigs)})")

# ── Simulate all combos ────────────────────────────────────────────────
# Each combo: (label, signals, buy_tp, sell_tp_or_None, sl_pips)
combos = []
for sl in SL_VALUES:
    combos.append((f"HYBRID  BUY@20+SELL@7.5  SL={sl:.1f}", all_signals, 20.0, 7.5,  sl))
    combos.append((f"BOTH    @ 10.5p          SL={sl:.1f}", all_signals, 10.5, None, sl))

print(f"\n{SEP}")
print("RESULTS")
print(SEP)
print(f"\n  {'Config':<42} | {'Trades':>7} | {'WR':>7} | {'Final Eq':>12} | {'Gain':>9} | {'Ann ROI':>8}")
print(f"  {'-'*42}-+-{'-'*7}-+-{'-'*7}-+-{'-'*12}-+-{'-'*9}-+-{'-'*8}")

all_res = []
for label, signals, buy_tp, sell_tp, sl in combos:
    eq, total, wr, bt, st, bw, sw, bp, sp = compound_run(signals, buy_tp, sl, sell_tp)
    gain    = eq - STARTING_EQUITY
    ann_roi = gain / STARTING_EQUITY / years * 100
    monthly = gain / years / 12
    all_res.append((label, eq, total, wr, bt, st, bw, sw, bp, sp, gain, ann_roi, monthly, sl))
    print(f"  {label:<42} | {total:>7,} | {wr:>6.1f}% | ${eq:>11,.0f} | {gain:>+9,.0f} | {ann_roi:>+7.1f}%")

# ── Buy vs Sell breakdown ──────────────────────────────────────────────
print(f"\n{SEP}")
print("BUY vs SELL BREAKDOWN")
print(SEP)
print(f"\n  {'Config':<42} | {'BuyTr':>5} | {'BuyWR':>6} | {'BuyPips':>8} | "
      f"{'SellTr':>6} | {'SellWR':>6} | {'SellPips':>8}")
print(f"  {'-'*42}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

for label, eq, total, wr, bt, st, bw, sw, bp, sp, gain, ann_roi, monthly, sl in all_res:
    bwr = bw / bt * 100 if bt > 0 else 0.0
    swr = sw / st * 100 if st > 0 else 0.0
    print(f"  {label:<42} | {bt:>5,} | {bwr:>5.1f}% | {bp:>+8.0f}p | "
          f"{st:>6,} | {swr:>5.1f}% | {sp:>+8.0f}p")

# ── Side-by-side SL comparison ─────────────────────────────────────────
print(f"\n{SEP}")
print("SIDE-BY-SIDE: SL IMPACT")
print(SEP)
print(f"\n  Period: ~{years:.2f} years  |  Starting: ${STARTING_EQUITY:,.0f}  |  1% risk/trade")

# Group by config type
for cfg_type, buy_tp, sell_tp_label in [("HYBRID  BUY@20+SELL@7.5", 20.0, "7.5"), ("BOTH    @ 10.5p        ", 10.5, "10.5")]:
    print(f"\n  {'─'*60}")
    print(f"  {cfg_type.strip()}")
    print(f"  {'─'*60}")
    print(f"  {'Metric':<22} | {'SL = 15.0p':>14} | {'SL = 10.5p':>14} | {'Difference':>12}")
    print(f"  {'-'*22}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")

    row15 = next(r for r in all_res if cfg_type.strip()[:6] in r[0] and r[13] == 15.0)
    row10 = next(r for r in all_res if cfg_type.strip()[:6] in r[0] and r[13] == 10.5)

    label15, eq15, tot15, wr15, bt15, st15, bw15, sw15, bp15, sp15, g15, roi15, mo15, _ = row15
    label10, eq10, tot10, wr10, bt10, st10, bw10, sw10, bp10, sp10, g10, roi10, mo10, _ = row10

    lot15 = STARTING_EQUITY * RISK_PCT / (15.0 * PIP_VALUE)
    lot10 = STARTING_EQUITY * RISK_PCT / (10.5 * PIP_VALUE)

    print(f"  {'Lot size (initial)':<22} | {lot15:>13.4f}L | {lot10:>13.4f}L | {lot10-lot15:>+11.4f}L")
    print(f"  {'$/pip (initial)':<22} | ${lot15*PIP_VALUE:>12.2f} | ${lot10*PIP_VALUE:>12.2f} | {'+' if lot10>lot15 else ''}{(lot10-lot15)*PIP_VALUE:>+11.2f}")
    print(f"  {'Trades':<22} | {tot15:>14,} | {tot10:>14,} | {tot10-tot15:>+12,}")
    print(f"  {'Win rate':<22} | {wr15:>13.1f}% | {wr10:>13.1f}% | {wr10-wr15:>+11.1f}%")
    bwr15 = bw15/bt15*100 if bt15>0 else 0
    bwr10 = bw10/bt10*100 if bt10>0 else 0
    swr15 = sw15/st15*100 if st15>0 else 0
    swr10 = sw10/st10*100 if st10>0 else 0
    print(f"  {'  Buy WR':<22} | {bwr15:>13.1f}% | {bwr10:>13.1f}% | {bwr10-bwr15:>+11.1f}%")
    print(f"  {'  Sell WR':<22} | {swr15:>13.1f}% | {swr10:>13.1f}% | {swr10-swr15:>+11.1f}%")
    print(f"  {'Final equity':<22} | ${eq15:>12,.0f} | ${eq10:>12,.0f} | ${eq10-eq15:>+11,.0f}")
    print(f"  {'Annual ROI':<22} | {roi15:>13.1f}% | {roi10:>13.1f}% | {roi10-roi15:>+11.1f}%")
    print(f"  {'$/month':<22} | ${mo15:>12,.0f} | ${mo10:>12,.0f} | ${mo10-mo15:>+11,.0f}")

print(f"\n{SEP}")
print(f"Analysis complete!  Log: {log_path}")
print(SEP)
