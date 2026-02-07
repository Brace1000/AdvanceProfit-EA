#!/usr/bin/env python3
"""
Walk-Forward Validation: ML + Technical Confluence vs ML-Only

Compares:
- ML-only strategy (confidence threshold)
- ML + Technical (both must align, or technical fills gap)

Uses same walk-forward windows, regime filter, and circuit breaker as baseline.
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

LOG_FILE = project_root / "logs" / "walk_forward_combined.log"
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


def get_technical_signal(df, idx, ma_fast=10, ma_slow=30, rsi_period=14, trend_ma=50):
    """Get technical SELL signal if conditions met."""
    if idx < max(ma_fast, ma_slow, trend_ma, 3):
        return 0

    # MA crossover: fast < slow NOW, fast >= slow BEFORE
    ma_fast_now = df["close"].iloc[idx - ma_fast:idx + 1].mean()
    ma_slow_now = df["close"].iloc[idx - ma_slow:idx + 1].mean()
    ma_fast_prev = df["close"].iloc[idx - ma_fast - 1:idx].mean()
    ma_slow_prev = df["close"].iloc[idx - ma_slow - 1:idx].mean()

    bearish_cross = (ma_fast_now < ma_slow_now) and (ma_fast_prev >= ma_slow_prev)

    # RSI
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(rsi_period).mean()
    roll_down = down.rolling(rsi_period).mean()
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[idx]

    rsi_bearish = rsi_val < 50 and rsi_val > 30

    # Trend filter
    trend_ma_val = df["close"].iloc[idx - trend_ma:idx + 1].mean()
    downtrend = df["close"].iloc[idx] < trend_ma_val

    if bearish_cross and rsi_bearish and downtrend:
        return -1

    return 0


def simulate_trades(df, sell_mask_ml, sell_mask_tech, ml_confidences, combine=True,
                   tp_pips=20, sl_pips=15, max_horizon=24, commission=0.0001,
                   cb_enabled=False, cb_max_losses=5, cb_max_dd_pips=100,
                   cb_cooldown=48, cb_reset_on_win=True):
    """Run TP/SL simulation with ML + Technical confluence."""
    pip_value = 0.0001
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    trade_results = []
    in_trade_until = -1
    consecutive_losses = 0
    peak_pnl = 0.0
    running_pnl = 0.0
    paused_until = -1
    cb_triggers = 0
    skipped = 0

    for idx in range(len(df)):
        if idx <= in_trade_until:
            continue

        if cb_enabled and idx < paused_until:
            skipped += 1
            continue

        # Confluence logic
        tech_signal = sell_mask_tech[idx]
        ml_signal = sell_mask_ml[idx]
        ml_conf = ml_confidences[idx]

        if combine:
            # Technical + ML confluence
            if tech_signal and ml_signal:
                # Both agree
                pass
            elif tech_signal and not ml_signal and ml_conf < 0.34:
                # Technical SELL, ML weak - use technical
                pass
            elif ml_signal and not tech_signal:
                # ML SELL, no technical - still go
                pass
            else:
                # Conflict or no signal
                continue
        else:
            # ML-only
            if not ml_signal:
                continue

        # Execute trade
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
                outcome = -(sl_price) - commission
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

    return trade_results, cb_triggers, skipped


def main():
    tee = TeeWriter(LOG_FILE)
    sys.stdout = tee

    print("=" * 90)
    print("WALK-FORWARD COMBINED: ML + Technical Confluence vs ML-Only")
    print("=" * 90)

    config = get_config()
    tp_pips = float(config.get("training.tp_pips", 20.0))
    sl_pips = float(config.get("training.sl_pips", 15.0))
    max_horizon = int(config.get("training.max_horizon_bars", 24))
    commission = float(config.get("backtesting.commission", 0.0001))
    be_rate = sl_pips / (tp_pips + sl_pips)
    sell_thresh = float(config.get("backtesting.confidence_threshold", 0.34))
    pip_value = 0.0001

    # Technical params
    ma_fast = 10
    ma_slow = 30
    rsi_period = 14
    trend_ma = 50

    # Load and engineer features
    loader = DataLoader(config._config)
    validator = DataValidator(config._config)
    fe = FeatureEngineer(config._config)

    project_root = Path(__file__).parent
    raw_path = project_root / config.get("paths.data_raw_file", "EURUSD_H1_raw.csv")
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

    # 3-class mapping
    y_all = pd.Series(df["label"]).map({-1: 0, 0: 1, 1: 2}).values

    # Load model
    import joblib
    model = joblib.load(project_root / config.get("model.path", "xgb_eurusd_h1.pkl"))

    # Conservative params
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
    dt_col = "datetime" if "datetime" in df.columns else (
        "time" if "time" in df.columns else None)

    n = len(df)

    cb_max_losses = int(config.get("trading.circuit_breaker.max_consecutive_losses", 5))
    cb_max_dd = float(config.get("trading.circuit_breaker.max_drawdown_pips", 100))
    cb_cooldown = int(config.get("trading.circuit_breaker.cooldown_bars", 48))
    cb_reset = bool(config.get("trading.circuit_breaker.reset_on_win", True))

    print(f"Data: {n} rows, {len(feature_cols)} features")
    print(f"TP={tp_pips}, SL={sl_pips}, BE={be_rate:.1%}, sell_thresh={sell_thresh}")
    print(f"Technical: MA({ma_fast},{ma_slow}) + RSI({rsi_period}) + Trend({trend_ma})")
    print(f"Regime filter: chop_h1 + chop_h4 below median")
    print(f"Circuit breaker: {cb_max_losses} consecutive losses OR {cb_max_dd} pip drawdown → {cb_cooldown} bar cooldown")

    # Walk-forward windows
    n_windows = 5
    test_size = n // (n_windows + 1)
    min_train = test_size

    print(f"\nWalk-forward: {n_windows} windows, ~{test_size} bars per test window")
    print(f"Minimum training size: {min_train} bars")

    # Run both ML-only and Combined
    for run_label, combine_mode in [("ML-ONLY", False), ("ML + TECHNICAL", True)]:
        print(f"\n{'=' * 90}")
        print(f"  {run_label}")
        print(f"{'=' * 90}\n")

        print(f"{'Window':>8} | {'Train':>8} | {'Test':>6} | {'Period':>25} | "
              f"{'Sells':>5} | {'Tech':>4} | {'Trades':>6} | {'WinR':>5} | "
              f"{'Prec':>5} | {'Pips':>7} | {'Sharpe':>6}")
        print("-" * 125)

        all_trade_results = []
        window_summaries = []

        for w in range(n_windows):
            test_start = min_train + w * test_size
            test_end = min(test_start + test_size, n)
            train_end = test_start

            if test_start >= n or test_end <= test_start:
                break

            X_train = df.iloc[:train_end][feature_cols].values
            y_train = y_all[:train_end]
            X_test = df.iloc[test_start:test_end][feature_cols].values
            y_test = y_all[test_start:test_end]
            test_df = df.iloc[test_start:test_end].copy()

            if dt_col and dt_col in df.columns:
                start_date = str(df.iloc[test_start][dt_col])[:10]
                end_date = str(df.iloc[test_end - 1][dt_col])[:10]
                period = f"{start_date} → {end_date}"
            else:
                period = f"bars {test_start}-{test_end}"

            try:
                classes = np.array([0, 1, 2])
                cw = compute_class_weight("balanced", classes=classes, y=y_train)
                sw = np.array([cw[label] for label in y_train])
            except Exception:
                sw = None

            # Train model for this window
            model_w = xgb.XGBClassifier(**params)
            model_w.fit(X_train, y_train, verbose=False, sample_weight=sw)

            probas = model_w.predict_proba(X_test)
            preds = np.argmax(probas, axis=1)
            max_probs = np.max(probas, axis=1)

            # ML sell mask
            sell_mask_ml = (preds == 0) & (max_probs >= sell_thresh)

            # Technical sell mask
            sell_mask_tech = np.zeros(len(test_df), dtype=bool)
            for idx in range(len(test_df)):
                test_idx = test_start + idx
                if test_idx >= min_train:
                    tech_sig = get_technical_signal(
                        df.iloc[:test_idx+1], test_idx,
                        ma_fast=ma_fast, ma_slow=ma_slow,
                        rsi_period=rsi_period, trend_ma=trend_ma
                    )
                    sell_mask_tech[idx] = (tech_sig == -1)

            n_sells = int(sell_mask_ml.sum())
            n_tech = int(sell_mask_tech.sum())

            # Regime filter
            chop_h1_med = np.median(test_df["choppiness_h1"].dropna())
            chop_h4_med = np.median(test_df["choppiness_h4"].dropna())
            regime_mask = (test_df["choppiness_h1"].values < chop_h1_med) & \
                          (test_df["choppiness_h4"].values < chop_h4_med)

            sell_mask_ml_regime = sell_mask_ml & regime_mask
            sell_mask_tech_regime = sell_mask_tech & regime_mask

            if combine_mode:
                # Combined logic for regime
                final_mask = np.zeros(len(test_df), dtype=bool)
                for i in range(len(test_df)):
                    if not regime_mask[i]:
                        continue
                    if sell_mask_ml_regime[i] and sell_mask_tech_regime[i]:
                        final_mask[i] = True
                    elif sell_mask_tech_regime[i] and max_probs[i] < sell_thresh:
                        final_mask[i] = True
                    elif sell_mask_ml_regime[i] and not sell_mask_tech[i]:
                        final_mask[i] = True
                sell_mask_confluence = final_mask
            else:
                sell_mask_confluence = sell_mask_ml_regime

            n_regime = int(sell_mask_confluence.sum())

            if n_regime > 0:
                true_pos = (sell_mask_confluence & (y_test == 0)).sum()
                precision = true_pos / n_regime
            else:
                precision = 0.0

            trades, triggers, skipped = simulate_trades(
                test_df, sell_mask_confluence, sell_mask_tech_regime,
                max_probs, combine=combine_mode,
                tp_pips=tp_pips, sl_pips=sl_pips,
                max_horizon=max_horizon, commission=commission,
                cb_enabled=True, cb_max_losses=cb_max_losses,
                cb_max_dd_pips=cb_max_dd, cb_cooldown=cb_cooldown,
                cb_reset_on_win=cb_reset
            )

            if len(trades) > 0:
                trades_arr = np.array(trades)
                wins = (trades_arr > 0).sum()
                win_rate = wins / len(trades_arr)
                total_pips = float(trades_arr.sum() / pip_value)

                if len(trades_arr) > 1:
                    avg_t = trades_arr.mean()
                    std_t = trades_arr.std(ddof=1)
                    sharpe = (avg_t / std_t) * np.sqrt(252) if std_t > 0 else 0.0
                else:
                    sharpe = 0.0
            else:
                win_rate = 0.0
                total_pips = 0.0
                sharpe = 0.0

            print(f"  W{w+1:>4}   | {train_end:7d}  | {test_end - test_start:5d} | {period:>25} | "
                  f"{n_sells:4d}  | {n_tech:3d}  | {len(trades):5d}  | {win_rate:4.1%} | "
                  f"{precision:4.1%} | {total_pips:+6.0f} | {sharpe:+5.2f}")

            all_trade_results.extend(trades)
            window_summaries.append({
                'window': w + 1,
                'trades': len(trades),
                'win_rate': win_rate,
                'precision': precision,
                'total_pips': total_pips,
                'sharpe': sharpe,
            })

        # Aggregate
        print(f"\n  --- {run_label} AGGREGATE ---")

        if all_trade_results:
            all_arr = np.array(all_trade_results)
            total_trades = len(all_arr)
            total_wins = (all_arr > 0).sum()
            total_win_rate = total_wins / total_trades
            total_pips_all = float(all_arr.sum() / pip_value)
            avg_pips_per_trade = total_pips_all / total_trades

            avg_t = all_arr.mean()
            std_t = all_arr.std(ddof=1) if len(all_arr) > 1 else 0.0
            overall_sharpe = (avg_t / std_t) * np.sqrt(252) if std_t > 0 else 0.0

            profitable_windows = sum(1 for s in window_summaries if s['total_pips'] > 0)
            total_windows = len(window_summaries)

            print(f"  Total trades:    {total_trades}")
            print(f"  Total pips:      {total_pips_all:+.0f}")
            print(f"  Win rate:        {total_win_rate:.1%}")
            print(f"  Avg pips/trade:  {avg_pips_per_trade:+.1f}")
            print(f"  Sharpe (trade):  {overall_sharpe:.2f}")
            print(f"  Profitable windows: {profitable_windows}/{total_windows}")
            print(f"  Per-window pips: ", end="")
            for s in window_summaries:
                print(f"W{s['window']}={s['total_pips']:+.0f}  ", end="")
            print()
        else:
            print(f"  No trades generated")

    print("\n" + "=" * 90)
    print(f"Results saved to {LOG_FILE}")
    print("=" * 90)

    tee.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
