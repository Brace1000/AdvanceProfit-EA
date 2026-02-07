#!/usr/bin/env python3
"""
Walk-Forward: Twin Trade vs Single Trade comparison.

Compares three strategies across walk-forward windows:
1. SINGLE: One trade, TP=20, SL=15 (current baseline)
2. TWIN: Two half-size trades per signal
   - Trade A "Banker": TP=10, SL=15 (quick grab)
   - Trade B "Runner": TP=20, SL=15 + trailing stop simulation
3. TWIN + TRAIL: Same as TWIN but Trade B uses breakeven + trailing stop

Output saved to logs/walk_forward_twin.log
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
from src.data_collection.validator import DataValidator
from src.features.engineer import FeatureEngineer

LOG_FILE = project_root / "logs" / "walk_forward_twin.log"
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


def triple_barrier_label(df, tp_pips, sl_pips, max_horizon=24):
    """Label using triple barrier method."""
    pip_value = 0.0001
    tp_price = tp_pips * pip_value
    sl_price = sl_pips * pip_value
    labels = np.zeros(len(df), dtype=int)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(len(df) - 1):
        entry = closes[i]
        end = min(i + 1 + max_horizon, len(df))
        future_highs = highs[i+1:end]
        future_lows = lows[i+1:end]

        tp_th = tp_price / entry
        sl_th = sl_price / entry

        ret_high = (future_highs - entry) / entry
        ret_low = (future_lows - entry) / entry

        tp_up = np.where(ret_high >= tp_th)[0]
        sl_up = np.where(ret_low <= -sl_th)[0]
        first_up_tp = tp_up[0] if len(tp_up) > 0 else max_horizon
        first_up_sl = sl_up[0] if len(sl_up) > 0 else max_horizon

        tp_dn = np.where(ret_low <= -tp_th)[0]
        sl_dn = np.where(ret_high >= sl_th)[0]
        first_dn_tp = tp_dn[0] if len(tp_dn) > 0 else max_horizon
        first_dn_sl = sl_dn[0] if len(sl_dn) > 0 else max_horizon

        if first_up_tp < first_up_sl and first_up_tp < max_horizon:
            labels[i] = 1
        elif first_dn_tp < first_dn_sl and first_dn_tp < max_horizon:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels


def simulate_single(df, sell_mask, tp_pips=20, sl_pips=15, max_horizon=24,
                    commission=0.0001, cb_enabled=True, cb_max_losses=5,
                    cb_max_dd_pips=100, cb_cooldown=48, cb_reset_on_win=True):
    """Standard single-trade simulation (baseline)."""
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

    for idx in sell_indices:
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
            if (entry - lows[j]) >= tp_price:
                outcome = tp_price - commission
                exit_bar = j
                break
            if (highs[j] - entry) >= sl_price:
                outcome = -sl_price - commission
                exit_bar = j
                break

        if outcome is None:
            exit_price = closes[horizon_end - 1]
            outcome = (entry - exit_price) - commission
            exit_bar = horizon_end - 1

        trade_results.append(outcome)
        in_trade_until = exit_bar

        if cb_enabled:
            running_pnl += outcome
            peak_pnl = max(peak_pnl, running_pnl)
            dd_pips = (peak_pnl - running_pnl) / pip_value
            if outcome <= 0:
                consecutive_losses += 1
            elif cb_reset_on_win:
                consecutive_losses = 0
            if consecutive_losses >= cb_max_losses or dd_pips >= cb_max_dd_pips:
                paused_until = exit_bar + cb_cooldown
                cb_triggers += 1
                consecutive_losses = 0

    return trade_results, cb_triggers


def simulate_twin(df, sell_mask, tp_a=10, tp_b=20, sl_pips=15,
                  be_trigger=10, be_offset=2, trail_pips=8, trail_step=2,
                  max_horizon=24, commission=0.0001,
                  cb_enabled=True, cb_max_losses=5, cb_max_dd_pips=100,
                  cb_cooldown=48, cb_reset_on_win=True):
    """
    Twin trade simulation.

    Trade A (Banker): TP=tp_a, SL=sl_pips, no trailing
    Trade B (Runner): TP=tp_b, SL=sl_pips, with breakeven + trailing stop

    Each trade is half-size, so results are halved to normalize against single trade.
    Returns combined pips per signal (Trade A result/2 + Trade B result/2).
    """
    pip_value = 0.0001
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    sell_indices = np.where(sell_mask)[0]

    trade_results = []  # Combined result per signal
    trade_a_results = []
    trade_b_results = []
    in_trade_until = -1
    consecutive_losses = 0
    peak_pnl = 0.0
    running_pnl = 0.0
    paused_until = -1
    cb_triggers = 0

    for idx in sell_indices:
        if idx <= in_trade_until:
            continue
        if cb_enabled and idx < paused_until:
            continue

        entry = closes[idx]
        tp_a_price = tp_a * pip_value
        tp_b_price = tp_b * pip_value
        sl_price = sl_pips * pip_value
        be_trigger_price = be_trigger * pip_value
        be_offset_price = be_offset * pip_value
        trail_price = trail_pips * pip_value
        trail_step_price = trail_step * pip_value
        horizon_end = min(idx + 1 + max_horizon, len(df))

        if idx + 1 >= len(df):
            continue

        # ─── Trade A: The Banker (simple TP/SL) ───
        outcome_a = None
        exit_bar_a = idx
        for j in range(idx + 1, horizon_end):
            if (entry - lows[j]) >= tp_a_price:
                outcome_a = tp_a_price - commission
                exit_bar_a = j
                break
            if (highs[j] - entry) >= sl_price:
                outcome_a = -sl_price - commission
                exit_bar_a = j
                break
        if outcome_a is None:
            exit_price = closes[horizon_end - 1]
            outcome_a = (entry - exit_price) - commission
            exit_bar_a = horizon_end - 1

        # ─── Trade B: The Runner (TP/SL + breakeven + trailing) ───
        outcome_b = None
        exit_bar_b = idx
        current_sl = entry + sl_price  # Initial SL (above entry for sell)
        best_price = entry  # Track best (lowest) price for trailing
        be_activated = False

        for j in range(idx + 1, horizon_end):
            low_j = lows[j]
            high_j = highs[j]
            close_j = closes[j]

            # Check SL hit first (current_sl may have been trailed)
            if high_j >= current_sl:
                outcome_b = -(current_sl - entry) - commission  # Negative if SL above entry
                exit_bar_b = j
                break

            # Check TP hit
            if (entry - low_j) >= tp_b_price:
                outcome_b = tp_b_price - commission
                exit_bar_b = j
                break

            # Update best price (lowest for sell trade)
            if low_j < best_price:
                best_price = low_j

            profit_from_entry = entry - best_price

            # Breakeven: once profit >= trigger, move SL to entry - offset
            if not be_activated and profit_from_entry >= be_trigger_price:
                new_sl = entry - be_offset_price  # Below entry = locked profit
                if new_sl < current_sl:
                    current_sl = new_sl
                    be_activated = True

            # Trailing: SL follows price, trail_pips behind best price
            if be_activated:
                trail_sl = best_price + trail_price  # trail_pips above best low
                if trail_sl < current_sl:
                    current_sl = trail_sl

        if outcome_b is None:
            exit_price = closes[horizon_end - 1]
            outcome_b = (entry - exit_price) - commission
            exit_bar_b = horizon_end - 1

        # Combined result: each trade is half size
        combined = (outcome_a / 2) + (outcome_b / 2)
        trade_results.append(combined)
        trade_a_results.append(outcome_a)
        trade_b_results.append(outcome_b)

        # Use the later exit bar for "in trade" blocking
        exit_bar = max(exit_bar_a, exit_bar_b)
        in_trade_until = exit_bar

        if cb_enabled:
            running_pnl += combined
            peak_pnl = max(peak_pnl, running_pnl)
            dd_pips = (peak_pnl - running_pnl) / pip_value
            if combined <= 0:
                consecutive_losses += 1
            elif cb_reset_on_win:
                consecutive_losses = 0
            if consecutive_losses >= cb_max_losses or dd_pips >= cb_max_dd_pips:
                paused_until = exit_bar + cb_cooldown
                cb_triggers += 1
                consecutive_losses = 0

    return trade_results, trade_a_results, trade_b_results, cb_triggers


def summarize(results, pip_value=0.0001):
    """Calculate summary stats for trade results."""
    if not results:
        return 0, 0, 0, 0, 0
    arr = np.array(results)
    total_pips = arr.sum() / pip_value
    wins = (arr > 0).sum()
    win_rate = wins / len(arr) if len(arr) > 0 else 0
    avg_pips = total_pips / len(arr)
    sharpe = (arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    return len(arr), total_pips, win_rate, avg_pips, sharpe


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 100)
    print("WALK-FORWARD: Single Trade vs Twin Trade Comparison")
    print("=" * 100)

    config = get_config()
    tp_pips = float(config.get("training.tp_pips", 20.0))
    sl_pips = float(config.get("training.sl_pips", 15.0))
    max_horizon = int(config.get("training.max_horizon_bars", 24))
    commission = float(config.get("backtesting.commission", 0.0001))
    sell_thresh = float(config.get("backtesting.confidence_threshold", 0.34))
    pip_value = 0.0001

    # Twin trade params
    tp_banker = 10  # Trade A: quick grab
    tp_runner = 20  # Trade B: full target
    be_trigger = 10  # Breakeven trigger
    be_offset = 2   # Lock in +2 pips
    trail_pips = 8   # Trail 8 pips behind
    trail_step = 2   # Update every 2 pips

    # Load and engineer features
    loader = DataLoader(config._config)
    validator = DataValidator(config._config)
    fe = FeatureEngineer(config._config)

    raw_path = project_root / config.get("paths.data_raw_file", "EURUSD_H1_clean.csv")
    df = loader.load(raw_path)
    validator.validate_ohlc(df)
    df = fe.engineer(df)

    features_path = project_root / "features_used.json"
    if features_path.exists():
        with open(features_path) as f:
            feature_cols = json.load(f)
        if not all(f in df.columns for f in feature_cols):
            feature_cols = FeatureEngineer.default_feature_columns()
    else:
        feature_cols = FeatureEngineer.default_feature_columns()

    # Label
    labels = triple_barrier_label(df, tp_pips, sl_pips, max_horizon)
    df["label"] = labels
    df = df.dropna(subset=feature_cols).iloc[:-1].copy()

    y_all = pd.Series(df["label"]).map({-1: 0, 0: 1, 1: 2}).values

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 3,
        "n_estimators": 80,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 40,
        "gamma": 0.8,
        "reg_alpha": 50.0,
        "reg_lambda": 50.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    has_datetime = "datetime" in df.columns or "time" in df.columns
    dt_col = "datetime" if "datetime" in df.columns else ("time" if "time" in df.columns else None)

    n = len(df)
    cb_max_losses = int(config.get("trading.circuit_breaker.max_consecutive_losses", 5))
    cb_max_dd = float(config.get("trading.circuit_breaker.max_drawdown_pips", 100))
    cb_cooldown = int(config.get("trading.circuit_breaker.cooldown_bars", 48))
    cb_reset = bool(config.get("trading.circuit_breaker.reset_on_win", True))

    # Regime filter columns
    chop_h1_col = "choppiness_h1"
    chop_h4_col = "choppiness_h4"

    print(f"Data: {n} rows, {len(feature_cols)} features")
    print(f"Single: TP={tp_pips}, SL={sl_pips}")
    print(f"Twin A (Banker): TP={tp_banker}, SL={sl_pips}")
    print(f"Twin B (Runner): TP={tp_runner}, SL={sl_pips}, BE@{be_trigger}→+{be_offset}, Trail={trail_pips}")
    print(f"Confidence threshold: {sell_thresh}")
    print(f"Circuit breaker: {cb_max_losses} losses OR {cb_max_dd} pip DD → {cb_cooldown} bar pause")

    n_windows = 5
    test_size = n // (n_windows + 1)
    min_train = test_size

    print(f"\nWalk-forward: {n_windows} windows, ~{test_size} bars per test window\n")

    # Run both strategies
    for strategy_name in ["SINGLE (baseline)", "TWIN (Banker + Runner)"]:
        is_twin = "TWIN" in strategy_name

        print(f"{'=' * 100}")
        print(f"  {strategy_name}")
        print(f"{'=' * 100}\n")

        header = (f"{'Window':>8} | {'Train':>8} | {'Test':>6} | {'Period':>25} | "
                  f"{'Trades':>6} | {'WinR':>5} | {'Pips':>7} | {'Sharpe':>6}")
        if is_twin:
            header += f" | {'A_WinR':>6} | {'A_Pips':>7} | {'B_WinR':>6} | {'B_Pips':>7}"
        print(header)
        print("-" * (120 if is_twin else 95))

        all_results = []
        all_a = []
        all_b = []
        window_pips = []

        for w in range(n_windows):
            test_start = min_train + w * test_size
            test_end = min(test_start + test_size, n)
            train_end = test_start

            if test_start >= n or test_end <= test_start:
                break

            X_train = df.iloc[:train_end][feature_cols].values
            y_train = y_all[:train_end]
            X_test = df.iloc[test_start:test_end][feature_cols].values
            test_df = df.iloc[test_start:test_end].copy()

            if dt_col and dt_col in df.columns:
                start_date = str(df.iloc[test_start][dt_col])[:10]
                end_date = str(df.iloc[test_end - 1][dt_col])[:10]
                period = f"{start_date} → {end_date}"
            else:
                period = f"bar {test_start} → {test_end}"

            # Train
            classes = np.unique(y_train)
            cw = compute_class_weight("balanced", classes=classes, y=y_train)
            weight_map = dict(zip(classes, cw))
            sample_weights = np.array([weight_map[y] for y in y_train])

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

            # Predict
            probs = model.predict_proba(X_test)
            sell_probs = probs[:, 0]

            # Sell mask with regime filter
            sell_mask_raw = sell_probs >= sell_thresh

            # Regime filter
            if chop_h1_col in test_df.columns and chop_h4_col in test_df.columns:
                train_df_regime = df.iloc[:train_end]
                median_h1 = train_df_regime[chop_h1_col].median()
                median_h4 = train_df_regime[chop_h4_col].median()
                regime_ok = ((test_df[chop_h1_col].values < median_h1) &
                            (test_df[chop_h4_col].values < median_h4))
                sell_mask = sell_mask_raw & regime_ok
            else:
                sell_mask = sell_mask_raw

            # Simulate
            if is_twin:
                results, a_results, b_results, cb_t = simulate_twin(
                    test_df, sell_mask, tp_a=tp_banker, tp_b=tp_runner,
                    sl_pips=sl_pips, be_trigger=be_trigger, be_offset=be_offset,
                    trail_pips=trail_pips, trail_step=trail_step,
                    max_horizon=max_horizon, commission=commission,
                    cb_enabled=True, cb_max_losses=cb_max_losses,
                    cb_max_dd_pips=cb_max_dd, cb_cooldown=cb_cooldown,
                    cb_reset_on_win=cb_reset)

                n_trades, total_pips, win_rate, avg_pip, sharpe = summarize(results)
                n_a, pips_a, wr_a, _, _ = summarize(a_results)
                n_b, pips_b, wr_b, _, _ = summarize(b_results)

                all_results.extend(results)
                all_a.extend(a_results)
                all_b.extend(b_results)
                window_pips.append(total_pips)

                cb_info = f" [CB:{cb_t}]" if cb_t > 0 else ""
                print(f"  W {w+1:>3}   | {train_end:>7}  | {test_end - test_start:>5} | "
                      f"{period:>25} | {n_trades:>5}  | {win_rate:>4.1%} | {total_pips:>+6.0f} | "
                      f"{sharpe:>+5.2f} | {wr_a:>5.1%} | {pips_a:>+6.0f} | "
                      f"{wr_b:>5.1%} | {pips_b:>+6.0f}{cb_info}")
            else:
                results, cb_t = simulate_single(
                    test_df, sell_mask, tp_pips=tp_pips, sl_pips=sl_pips,
                    max_horizon=max_horizon, commission=commission,
                    cb_enabled=True, cb_max_losses=cb_max_losses,
                    cb_max_dd_pips=cb_max_dd, cb_cooldown=cb_cooldown,
                    cb_reset_on_win=cb_reset)

                n_trades, total_pips, win_rate, avg_pip, sharpe = summarize(results)
                all_results.extend(results)
                window_pips.append(total_pips)

                cb_info = f" [CB:{cb_t}]" if cb_t > 0 else ""
                print(f"  W {w+1:>3}   | {train_end:>7}  | {test_end - test_start:>5} | "
                      f"{period:>25} | {n_trades:>5}  | {win_rate:>4.1%} | {total_pips:>+6.0f} | "
                      f"{sharpe:>+5.2f}{cb_info}")

        # Aggregate
        n_total, pips_total, wr_total, avg_total, sharpe_total = summarize(all_results)
        print(f"\n  --- {strategy_name} AGGREGATE ---")
        print(f"  Total trades:    {n_total}")
        print(f"  Total pips:      {pips_total:+.0f}")
        print(f"  Win rate:        {wr_total:.1%}")
        print(f"  Avg pips/trade:  {avg_total:+.1f}")
        print(f"  Sharpe (trade):  {sharpe_total:.2f}")
        pips_str = "  ".join([f"W{i+1}={p:+.0f}" for i, p in enumerate(window_pips)])
        print(f"  Per-window pips: {pips_str}")

        if is_twin and all_a:
            _, pips_a_tot, wr_a_tot, _, _ = summarize(all_a)
            _, pips_b_tot, wr_b_tot, _, _ = summarize(all_b)
            print(f"\n  Trade A (Banker) total: {pips_a_tot:+.0f} pips, {wr_a_tot:.1%} win rate")
            print(f"  Trade B (Runner) total: {pips_b_tot:+.0f} pips, {wr_b_tot:.1%} win rate")

        print()

    print("=" * 100)
    print(f"Results saved to {LOG_FILE}")
    print("=" * 100)

    sys.stdout = tee.stdout
    tee.close()


if __name__ == "__main__":
    main()
