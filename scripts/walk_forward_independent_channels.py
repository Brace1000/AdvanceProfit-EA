"""
Walk-Forward: Independent Channel Test

Tests BUY and SELL channels independently, each with and without circuit breaker.
NO spread gate — buy and sell operate on their raw thresholds only.

Configurations:
  A: BUY_ONLY  | No CB   (baseline — should approach original +1177 conditions)
  B: BUY_ONLY  | CB ON   (production BUY_ONLY setup)
  C: SELL_ONLY | No CB   (baseline sell)
  D: SELL_ONLY | CB ON   (production SELL_ONLY setup)

Key difference from v2: NO spread gate on buy signals.
  Buy trigger:  buy_prob  >= 0.40  (no margin check against sell/range)
  Sell trigger: sell_prob >= 0.56  (unchanged — was never spread-gated)
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

BUY_THRESHOLD  = 0.40   # no spread gate — raw threshold only
SELL_THRESHOLD = 0.56
NUM_WINDOWS = 5

# Circuit breaker
CB_MAX_LOSSES   = 5
CB_MAX_DD_PIPS  = 100
CB_COOLDOWN     = 48
CB_RESET_ON_WIN = True

# ── Paths ─────────────────────────────────────────────────────────────
project       = Path(__file__).parent.parent
data_path     = project / "data" / "EURUSD_H1_clean.csv"
features_path = project / "features_used_buy.json"
log_path      = project / "logs" / "walk_forward_independent_channels.log"

# ── Logger ────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, fp):
        fp.parent.mkdir(exist_ok=True)
        self.t = sys.stdout
        self.f = open(fp, 'w', encoding='utf-8')
    def write(self, m):  self.t.write(m);  self.f.write(m)
    def flush(self):     self.t.flush();   self.f.flush()

sys.stdout = Logger(log_path)


# ══════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    def __init__(self):
        self.consecutive_losses = 0
        self.peak_pnl   = 0.0
        self.running_pnl = 0.0
        self.paused_until = -1
        self.triggers = 0
        self.skipped  = 0

    def is_paused(self, idx):
        return idx < self.paused_until

    def update(self, pips, idx):
        self.running_pnl += pips
        self.peak_pnl = max(self.peak_pnl, self.running_pnl)
        dd = self.peak_pnl - self.running_pnl

        if pips <= 0:
            self.consecutive_losses += 1
        elif CB_RESET_ON_WIN:
            self.consecutive_losses = 0

        if self.consecutive_losses >= CB_MAX_LOSSES or dd >= CB_MAX_DD_PIPS:
            self.paused_until = idx + CB_COOLDOWN
            self.triggers     += 1
            self.consecutive_losses = 0


# ══════════════════════════════════════════════════════════════════════
# BAR DATA
# ══════════════════════════════════════════════════════════════════════

def get_bars(entry_idx, direction, df):
    ep = df.iloc[entry_idx]['close']
    bars = []
    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_HOLDING, len(df))):
        h, l, c = df.iloc[j]['high'], df.iloc[j]['low'], df.iloc[j]['close']
        if direction == 'BUY':
            bars.append({'best': (h-ep)*1e4, 'worst': (l-ep)*1e4, 'close': (c-ep)*1e4})
        else:
            bars.append({'best': (ep-l)*1e4, 'worst': (ep-h)*1e4, 'close': (ep-c)*1e4})
    return bars


# ══════════════════════════════════════════════════════════════════════
# STRATEGY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def strat_raw(bars):
    mfe = 0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        if b['best']  >= 20.0: return dict(pips=20.0-COMMISSION, bars_held=i+1, outcome='TP',  mfe=mfe)
        if b['worst'] <= -15.0: return dict(pips=-15.0-COMMISSION,bars_held=i+1, outcome='SL',  mfe=mfe)
    return dict(pips=bars[-1]['close']-COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_twin(bars):
    ao, bo = True, True
    ap, bp = 0.0, 0.0
    ab, bb = 0, 0
    bsl = -15.0; bpk = 0.0; mfe = 0.0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        if ao:
            if   b['best']  >= 10.0: ap = 10.0-COMMISSION;  ao=False; ab=i+1
            elif b['worst'] <= -15.0: ap = -15.0-COMMISSION; ao=False; ab=i+1
        if bo:
            bpk = max(bpk, b['best'])
            if bpk >= 10.0 and bsl < 0: bsl = 0.0
            if bsl >= 0: bsl = max(bsl, bpk - 8.0)
            if   b['best']  >= 20.0: bp = 20.0-COMMISSION;  bo=False; bb=i+1
            elif b['worst'] <= bsl:   bp = bsl-COMMISSION;   bo=False; bb=i+1
        if not ao and not bo: break
    if ao: ap = bars[-1]['close']-COMMISSION;       ab=len(bars)
    if bo: bp = max(bars[-1]['close'],bsl)-COMMISSION; bb=len(bars)
    return dict(pips=0.5*ap+0.5*bp, bars_held=max(ab,bb), outcome='TWIN', mfe=mfe)


def strat_free_ride(bars):
    ao, bo = True, True
    ap, bp = 0.0, 0.0
    ab, bb = 0, 0
    atp = False; bsl = -15.0; bpk = 0.0; mfe = 0.0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best'])
        if ao:
            if   b['best']  >= 8.0:  ap=8.0-COMMISSION;  ao=False; ab=i+1; atp=True; bsl=max(bsl,0.0)
            elif b['worst'] <= -15.0: ap=-15.0-COMMISSION; ao=False; ab=i+1
        if bo:
            bpk = max(bpk, b['best'])
            if atp: bsl = max(bsl, bpk - 8.0)
            if b['worst'] <= bsl: bp=bsl-COMMISSION; bo=False; bb=i+1
        if not ao and not bo: break
    if ao: ap = bars[-1]['close']-COMMISSION;          ab=len(bars)
    if bo: bp = max(bars[-1]['close'],bsl)-COMMISSION; bb=len(bars)
    return dict(pips=0.5*ap+0.5*bp, bars_held=max(ab,bb), outcome='FREE_RIDE', mfe=mfe)


def strat_time_decay(bars):
    mfe = 0.0
    for i, b in enumerate(bars):
        mfe = max(mfe, b['best']); n = i+1
        if   n <= 10: tp, sl = 20.0, -15.0
        elif n <= 20: tp, sl = 14.0, -12.0
        elif n <= 30:
            if b['best'] >= 3.0: return dict(pips=3.0-COMMISSION, bars_held=n, outcome='TD_PROFIT', mfe=mfe)
            tp, sl = 14.0, -8.0
        else: return dict(pips=b['close']-COMMISSION, bars_held=n, outcome='TD_CLOSE', mfe=mfe)
        if b['best']  >= tp:  return dict(pips=tp-COMMISSION,  bars_held=n, outcome='TP', mfe=mfe)
        if b['worst'] <= sl:  return dict(pips=sl-COMMISSION,  bars_held=n, outcome='SL', mfe=mfe)
    return dict(pips=bars[-1]['close']-COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_no_ceiling(bars):
    sl=-15.0; pk=0.0; trailing=False; mfe=0.0
    for i, b in enumerate(bars):
        pk=max(pk,b['best']); mfe=max(mfe,pk)
        if pk>=5.0:  trailing=True
        if pk>=10.0 and sl<0: sl=0.0
        if trailing: sl=max(sl, pk-8.0)
        if b['worst']<=sl: return dict(pips=sl-COMMISSION, bars_held=i+1, outcome='TRAIL', mfe=mfe)
    return dict(pips=max(bars[-1]['close'],sl)-COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_ladder(bars):
    sl=-15.0; mfe=0.0; r=0
    ratchets=[(4.,-10.),(8.,-3.),(10.,0.),(14.,8.),(17.,12.)]
    for i, b in enumerate(bars):
        mfe=max(mfe,b['best'])
        while r<len(ratchets) and mfe>=ratchets[r][0]: sl=ratchets[r][1]; r+=1
        if b['best']  >= 20.0: return dict(pips=20.0-COMMISSION, bars_held=i+1, outcome='TP',        mfe=mfe)
        if b['worst'] <= sl:   return dict(pips=sl-COMMISSION,   bars_held=i+1, outcome='SL_LADDER', mfe=mfe)
    return dict(pips=max(bars[-1]['close'],sl)-COMMISSION, bars_held=len(bars), outcome='TIMEOUT', mfe=mfe)


def strat_harvester(bars):
    ao,bo,co=True,True,True; ap,bp,cp=0.,0.,0.; ab,bb,cb_=0,0,0
    bsl,csl=-15.,-15.; cpk=0.; mfe=0.
    for i,b in enumerate(bars):
        mfe=max(mfe,b['best'])
        if ao:
            if   b['best']  >= 8.: ap=8.-COMMISSION;   ao=False; ab=i+1
            elif b['worst'] <=-15.: ap=-15.-COMMISSION; ao=False; ab=i+1
        if bo:
            if mfe>=8. and bsl<0: bsl=0.
            if   b['best']  >=14.: bp=14.-COMMISSION;  bo=False; bb=i+1
            elif b['worst'] <=bsl: bp=bsl-COMMISSION;  bo=False; bb=i+1
        if co:
            cpk=max(cpk,b['best'])
            if cpk>=8. and csl<0: csl=0.
            if csl>=0: csl=max(csl,cpk-10.)
            if b['worst']<=csl: cp=csl-COMMISSION; co=False; cb_=i+1
        if not ao and not bo and not co: break
    if ao: ap=bars[-1]['close']-COMMISSION;           ab=len(bars)
    if bo: bp=max(bars[-1]['close'],bsl)-COMMISSION;  bb=len(bars)
    if co: cp=max(bars[-1]['close'],csl)-COMMISSION;  cb_=len(bars)
    return dict(pips=0.4*ap+0.3*bp+0.3*cp, bars_held=max(ab,bb,cb_), outcome='HARVESTER', mfe=mfe)


STRATEGIES = {
    'Raw TP20':   strat_raw,
    'Twin Trade': strat_twin,
    'Free Ride':  strat_free_ride,
    'Time Decay': strat_time_decay,
    'No Ceiling': strat_no_ceiling,
    'Ladder':     strat_ladder,
    'Harvester':  strat_harvester,
}


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

print("="*90)
print("WALK-FORWARD: INDEPENDENT CHANNEL TEST (No Spread Gate)")
print("="*90)
print(f"\nBuy  trigger: buy_prob  >= {BUY_THRESHOLD}  (NO spread gate)")
print(f"Sell trigger: sell_prob >= {SELL_THRESHOLD}")
print(f"CB: {CB_MAX_LOSSES} losses OR {CB_MAX_DD_PIPS}-pip DD -> {CB_COOLDOWN}-bar pause")
print(f"TP/SL: {TP_PIPS}/{SL_PIPS} | Max hold: {MAX_HOLDING} bars | Commission: {COMMISSION} pip/pos")

print("\nLoading data...")
df = pd.read_csv(data_path)
print(f"Data: {len(df)} rows")

with open(features_path) as f:
    feature_names = json.load(f)

# Labels (HIGH/LOW — same as v2)
def create_labels(df):
    labels = []
    for i in range(len(df) - MAX_HOLDING):
        e = df.iloc[i]['close']
        tp_p = e + TP_PIPS * 0.0001
        sl_p = e - SL_PIPS * 0.0001
        ht = hs = False
        for j in range(i+1, min(i+1+MAX_HOLDING, len(df))):
            if df.iloc[j]['high'] >= tp_p: ht=True; break
            if df.iloc[j]['low']  <= sl_p: hs=True; break
        labels.append(0 if ht else (2 if hs else 1))
    labels.extend([1]*(len(df)-len(labels)))
    return labels

print("Creating labels...")
df['label'] = create_labels(df)

total_bars  = len(df)
window_size = total_bars // NUM_WINDOWS

# ── Configurations ────────────────────────────────────────────────────
CONFIGS = [
    ('A: BUY_ONLY  | No CB', 'BUY_ONLY',  False),
    ('B: BUY_ONLY  | CB ON', 'BUY_ONLY',  True),
    ('C: SELL_ONLY | No CB', 'SELL_ONLY', False),
    ('D: SELL_ONLY | CB ON', 'SELL_ONLY', True),
]

all_summaries = {}

for config_name, mode, use_cb in CONFIGS:

    print(f"\n>>> {config_name}")
    print("="*90)
    print(f"  {'Strategy':<14} | {'Trades':>7} | {'Wins':>6} | {'WR':>7} | {'Pips':>8} | {'Pips/Tr':>8} | {'CB':>4} | {'Skip':>5}")
    print(f"  {'-'*14}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*4}-+-{'-'*5}")

    results  = {n: [] for n in STRATEGIES}
    cb_info  = {n: {'triggers':0,'skipped':0} for n in STRATEGIES}
    total_signals_per_window = []

    for window in range(1, NUM_WINDOWS+1):
        train_end  = window * window_size
        test_start = train_end
        test_end   = min((window+1)*window_size, total_bars)
        if test_start >= total_bars: break

        # Train
        X_tr = df.iloc[:train_end][feature_names].values
        y_tr = df.iloc[:train_end]['label'].values
        for needed in [0,1,2]:
            if needed not in y_tr:
                X_tr = np.vstack([X_tr, X_tr[:1]])
                y_tr = np.append(y_tr, needed)

        mdl = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.03,
                            objective='multi:softprob', num_class=3,
                            random_state=42, verbosity=0)
        mdl.fit(X_tr, y_tr)

        # Predict
        X_te   = df.iloc[test_start:test_end][feature_names].values
        probs  = mdl.predict_proba(X_te)

        # Signals — NO spread gate
        signals = []
        for i in range(len(probs)):
            bp = probs[i][0]; sp = probs[i][2]
            if mode == 'BUY_ONLY'  and bp >= BUY_THRESHOLD:
                signals.append((i, 'BUY'))
            elif mode == 'SELL_ONLY' and sp >= SELL_THRESHOLD:
                signals.append((i, 'SELL'))
        total_signals_per_window.append(len(signals))

        # Run strategies
        for sname, sfn in STRATEGIES.items():
            cb = CircuitBreaker() if use_cb else None
            nxt = 0

            for off, direction in signals:
                if off < nxt: continue
                aidx = test_start + off
                if aidx + MAX_HOLDING >= len(df): break
                if cb and cb.is_paused(aidx):
                    cb.skipped += 1; continue

                bdata = get_bars(aidx, direction, df)
                if not bdata: continue

                res = sfn(bdata)
                res['direction'] = direction
                res['window']    = window
                results[sname].append(res)

                if cb: cb.update(res['pips'], aidx)
                nxt = off + res['bars_held']

            if cb:
                cb_info[sname]['triggers'] += cb.triggers
                cb_info[sname]['skipped']  += cb.skipped

    print(f"  (Signals per window: {total_signals_per_window})")
    print()

    # Print results
    summaries = {}
    for sname in STRATEGIES:
        trades = results[sname]
        n = len(trades)
        if n == 0:
            print(f"  {sname:<14} |       0 |      - |       - |        - |        - |    - |     -")
            summaries[sname] = {'trades':0,'wins':0,'wr':0,'pips':0,'avg_pips':0,'avg_bars':0}
            continue
        wins  = sum(1 for t in trades if t['pips'] > 0)
        pips  = sum(t['pips'] for t in trades)
        abars = sum(t['bars_held'] for t in trades) / n
        wr    = wins / n * 100
        cbi   = cb_info[sname]
        print(f"  {sname:<14} | {n:>7} | {wins:>6} | {wr:>6.1f}% | {pips:>+8.0f} | {pips/n:>+8.2f} | {cbi['triggers']:>4} | {cbi['skipped']:>5}")
        summaries[sname] = {'trades':n,'wins':wins,'wr':wr,'pips':pips,'avg_pips':pips/n,'avg_bars':abars}

    ranked = sorted(summaries.items(), key=lambda x: x[1]['pips'], reverse=True)
    print(f"\n  Ranking:")
    for rank, (sname, s) in enumerate(ranked, 1):
        if s['trades'] > 0:
            print(f"    #{rank} {sname:<14} {s['pips']:>+8.0f} pips  ({s['wr']:.1f}% WR, {s['trades']} trades, {s['avg_pips']:>+.2f}/trade)")

    all_summaries[config_name] = summaries


# ══════════════════════════════════════════════════════════════════════
# CROSS-CHANNEL SUMMARY
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*90}")
print("CROSS-CHANNEL SUMMARY")
print(f"{'='*90}")

print(f"\n  Raw TP20 — the apples-to-apples baseline:")
print(f"  {'Config':<28} | {'Trades':>7} | {'WR':>7} | {'Pips':>8} | {'Pips/Tr':>8}")
print(f"  {'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
for cn, _, _ in CONFIGS:
    s = all_summaries[cn].get('Raw TP20', {})
    if s.get('trades',0) > 0:
        print(f"  {cn:<28} | {s['trades']:>7} | {s['wr']:>6.1f}% | {s['pips']:>+8.0f} | {s['avg_pips']:>+8.2f}")

print(f"\n  Free Ride — does it beat Raw TP20 per channel?")
print(f"  {'Config':<28} | {'Trades':>7} | {'WR':>7} | {'Pips':>8} | {'Pips/Tr':>8} | vs Raw")
print(f"  {'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for cn, _, _ in CONFIGS:
    fr = all_summaries[cn].get('Free Ride',{})
    rw = all_summaries[cn].get('Raw TP20',{})
    if fr.get('trades',0) > 0:
        diff = fr['pips'] - rw.get('pips',0)
        sign = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
        print(f"  {cn:<28} | {fr['trades']:>7} | {fr['wr']:>6.1f}% | {fr['pips']:>+8.0f} | {fr['avg_pips']:>+8.2f} | {sign:>8}")

print(f"\n  Best strategy per config:")
print(f"  {'Config':<28} | {'Best':<14} | {'Pips':>8} | {'WR':>7} | {'Pips/Tr':>8}")
print(f"  {'-'*28}-+-{'-'*14}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
for cn, _, _ in CONFIGS:
    sums = all_summaries[cn]
    best_n, best_s = max(sums.items(), key=lambda x: x[1]['pips'])
    if best_s.get('trades',0) > 0:
        print(f"  {cn:<28} | {best_n:<14} | {best_s['pips']:>+8.0f} | {best_s['wr']:>6.1f}% | {best_s['avg_pips']:>+8.2f}")

print(f"\n  CB impact (Raw TP20, buy vs sell):")
for no_cb, with_cb in [('A: BUY_ONLY  | No CB','B: BUY_ONLY  | CB ON'),
                        ('C: SELL_ONLY | No CB','D: SELL_ONLY | CB ON')]:
    a = all_summaries.get(no_cb,{}).get('Raw TP20',{})
    b = all_summaries.get(with_cb,{}).get('Raw TP20',{})
    if a.get('trades',0)>0 and b.get('trades',0)>0:
        diff = b['pips'] - a['pips']
        print(f"    {no_cb}: {a['pips']:>+7.0f} pips ({a['trades']} trades)  →  "
              f"{with_cb}: {b['pips']:>+7.0f} pips ({b['trades']} trades)  [CB effect: {diff:>+.0f}]")

print(f"\n  Reference: Original walk-forward (close-only labels, no spread gate):")
print(f"    BUY_ONLY + CB + Raw TP20: +1177 pips (836 trades, 47.2% WR) @ threshold 0.40")

print(f"\n{'='*90}")
print(f"Analysis complete! Log: {log_path}")
print(f"{'='*90}")
