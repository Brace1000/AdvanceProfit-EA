#!/usr/bin/env python3
"""
Complete pipeline to prepare data, run HPO (Optuna), train model, and evaluate
Run this before starting the FastAPI server

Outputs:
- EURUSD_D1_clean.csv (cleaned data with features)
- xgb_eurusd_h1.pkl (trained model)
- best_params.json (best hyperparameters from Optuna)
- Prints evaluation metrics and a simple backtest summary
"""

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

PRINT_LINE = "=" * 60

print(PRINT_LINE)
print("EUR/USD Trading Model - Setup + HPO + Evaluation")
print(PRINT_LINE)

# ============================================================================
# STEP 1: Load and Clean Data
# ============================================================================
print("\n[1/4] Loading and cleaning data...")

RAW_PATH = Path("EURUSD_D1_raw.csv")
CLEAN_PATH = Path("EURUSD_D1_clean.csv")
MODEL_PATH = Path("xgb_eurusd_h1.pkl")
PARAMS_PATH = Path("best_params.json")

if not RAW_PATH.exists():
    raise FileNotFoundError(Data pipeline: run_all.py still expects EURUSD_D1_raw.csv. If you want automated data fetchers (e.g., yfinance, MT5, Binance/ccxt), I can add a data_fetch.py to build/update the raw CSV.
        f"Missing {RAW_PATH.name}. Please place your raw EUR/USD daily CSV in the project root."
    )

df = pd.read_csv(RAW_PATH, on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower()

print(f"  → Loaded {len(df)} rows")

# Convert price columns to numeric, coercing errors to NaN
price_cols = ["open", "high", "low", "close"]
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with any NaN values in price columns
initial_len = len(df)
df.dropna(subset=price_cols, inplace=True)
removed = initial_len - len(df)
if removed > 0:
    print(f"  → Removed {removed} rows with invalid price data")

# Ensure sorted by time if a date column exists
for time_col in ["time", "date", "datetime", "timestamp"]:
    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df.sort_values(time_col, inplace=True)
            df.reset_index(drop=True, inplace=True)
            break
        except Exception:
            pass

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[2/4] Building features...")

# EMA 50 and EMA 200
df["close_ema50"] = df["close"].ewm(span=50, adjust=False).mean()
df["close_ema200"] = df["close"].ewm(span=200, adjust=False).mean()
df["ema50_ema200"] = df["close_ema50"] - df["close_ema200"]

# RSI
delta = df["close"].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs = roll_up / (roll_down + 1e-10)
df["rsi"] = 100 - (100 / (1 + rs))
df["rsi_slope"] = df["rsi"].diff()

# ATR ratio
high_low = df["high"] - df["low"]
high_close = np.abs(df["high"] - df["close"].shift())
low_close = np.abs(df["low"] - df["close"].shift())
tr = high_low.combine(high_close, max).combine(low_close, max)
df["atr"] = tr.rolling(14).mean()
df["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)

# ADX approximation
df["plus_dm"] = df["high"].diff().clip(lower=0)
df["minus_dm"] = (-df["low"].diff()).clip(lower=0)
df["adx"] = (df["plus_dm"] - df["minus_dm"]).abs().rolling(14).mean()

# Body and range
df["body"] = df["close"] - df["open"]
df["range"] = df["high"] - df["low"]

# Hour and session (daily data)
df["hour"] = 0
df["session"] = 0

# Previous return
df["prev_return"] = df["close"].pct_change()

# Labels using realistic thresholds for daily
ahead_return = df["close"].shift(-1) / df["close"] - 1
buy_threshold = 0.002   # +0.2%
sell_threshold = -0.002 # -0.2%

labels = np.zeros(len(df), dtype=int)
labels[ahead_return > buy_threshold] = 1
labels[ahead_return < sell_threshold] = -1

df["label"] = labels

# Remove rows with NaN in features used
feature_cols = [
    "close_ema50", "ema50_ema200", "rsi", "rsi_slope",
    "atr_ratio", "adx", "body", "range",
    "hour", "session", "prev_return",
]

df.dropna(subset=feature_cols, inplace=True)

# Drop last row without forward label
df = df.iloc[:-1].copy()

# Save cleaned data
df.to_csv(CLEAN_PATH, index=False)
print(f"  → Created {len(df)} samples with features")
print(
    f"  → Label distribution: Sell={sum(df['label']==-1)}, "
    f"Range={sum(df['label']==0)}, Buy={sum(df['label']==1)}"
)

# ============================================================================
# STEP 3: HPO with Optuna + Time-Series CV
# ============================================================================
print("\n[3/4] Hyperparameter Optimization (Optuna) with TimeSeries CV...")

# Features and targets
FEATURES = feature_cols
X = df[FEATURES].values
# Map labels (-1,0,1) to (0,1,2)
y = pd.Series(df["label"]).map({-1: 0, 0: 1, 1: 2}).values

# Hold-out validation segment (time-aware): last 30% as validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

print(f"  → Training samples: {len(X_train)}")
print(f"  → Validation samples: {len(X_val)}")

# Define objective for Optuna

def objective(trial: optuna.trial.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    # TimeSeriesSplit inside training segment
    tscv = TimeSeriesSplit(n_splits=5)

    f1_scores = []

    for train_idx, test_idx in tscv.split(X_train):
        X_tr, X_te = X_train[train_idx], X_train[test_idx]
        y_tr, y_te = y_train[train_idx], y_train[test_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            verbose=False,
        )
        preds = model.predict(X_te)
        # Macro F1 across 3 classes to treat all classes equally
        f1 = f1_score(y_te, preds, average="macro")
        f1_scores.append(f1)

    return float(np.mean(f1_scores))

# Create or load an Optuna study
study = optuna.create_study(direction="maximize", study_name="xgb_eurusd_hpo")

# Reasonable number of trials for quick run. Increase for better results.
TRIALS = int(
    Path(".env_trials").read_text().strip()
    if Path(".env_trials").exists()
    else 30
)
print(f"  → Running {TRIALS} trials... (set .env_trials to change)")
study.optimize(objective, n_trials=TRIALS, show_progress_bar=False)

print("  → Best score (macro F1):", f"{study.best_value:.4f}")
print("  → Best params:")
for k, v in study.best_params.items():
    print(f"     {k}: {v}")

# Persist best params
with open(PARAMS_PATH, "w") as f:
    json.dump(study.best_params, f, indent=2)
print(f"  → Saved best params to {PARAMS_PATH}")

# ============================================================================
# STEP 4: Train Final Model + Evaluate + Simple Backtest
# ============================================================================
print("\n[4/4] Training final XGBoost model and evaluating...")

best_params = {
    **study.best_params,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,bobaigwa@LAP-051:~$ wine64 --version
wine64: command not found
bobaigwa@LAP-051:~$ wine64 ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/terminal64.exe
wine64: command not found
bobaigwa@LAP-051:~$ 

}

final_model = xgb.XGBClassifier(**best_params)

eval_set = [(X_train, y_train), (X_val, y_val)]
final_model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=False,
)

# Evaluate on train/val
train_acc = final_model.score(X_train, y_train)
val_acc = final_model.score(X_val, y_val)

print(f"  → Training accuracy: {train_acc:.2%}")
print(f"  → Validation accuracy: {val_acc:.2%}")

# Classification report on validation
y_pred_val = final_model.predict(X_val)
print("\n  Validation Classification Report:")
print(
    classification_report(
        y_val,
        y_pred_val,
        target_names=["Sell", "Range", "Buy"],
        zero_division=0,
        digits=4,
    )
)

# Save model
joblib.dump(final_model, MODEL_PATH)
print(f"  → Model saved as '{MODEL_PATH.name}'")

# Simple walk-forward backtest on the validation segment
# Assumptions:
# - Enter at today's close based on prediction from features at close
# - Exit next day close; position: +1 for Buy, -1 for Sell, 0 for Range
# - Deduct a small cost per trade to emulate spread/slippage
print("\n[Backtest] Simple validation-segment backtest (next-day close PnL)...")
val_df = df.iloc[len(X_train) :].copy()
val_df = val_df.reset_index(drop=True)
val_df["pred"] = y_pred_val  # 0=Sell,1=Range,2=Buy
val_df["position"] = val_df["pred"].map({0: -1, 1: 0, 2: 1})

# Next-day return from close to next close
val_df["next_ret"] = val_df["close"].pct_change(-1) * -1  # shift(-1) over close

# Trading cost per round-trip proxy (spread+slippage). Adjust as needed.
TRADING_COST = 0.0001  # 10 bps (example)

# Apply PnL: position * next_ret minus cost if position != 0
val_df["trade_cost"] = (val_df["position"] != 0).astype(float) * TRADING_COST
val_df["pnl"] = val_df["position"] * val_df["next_ret"] - val_df["trade_cost"]

# Drop last row with NaN next_ret
val_df = val_df.iloc[:-1]

cum_pnl = (1 + val_df["pnl"]).prod() - 1
avg_pnl = val_df["pnl"].mean()
std_pnl = val_df["pnl"].std(ddof=1)
sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl and std_pnl > 0 else 0.0

# Max drawdown
cum_curve = (1 + val_df["pnl"]).cumprod()
rolling_max = cum_curve.cummax()
drawdown = cum_curve / rolling_max - 1
max_dd = drawdown.min() if len(drawdown) else 0.0

wins = (val_df["pnl"] > 0).sum()
losses = (val_df["pnl"] < 0).sum()
win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

print(f"  → Trades: {(val_df['position'] != 0).sum()}")
print(f"  → Win rate: {win_rate:.2%}")
print(f"  → Cum. return: {cum_pnl:.2%}")
print(f"  → Sharpe (daily→annual): {sharpe:.2f}")
print(f"  → Max drawdown: {max_dd:.2%}")

# ============================================================================
# Summary + Hints
# ============================================================================
print("\n" + PRINT_LINE)
if val_acc > 0.35:  # Better than random for 3 classes
    print("✅ Setup Complete! Model trained with HPO and evaluated.")
else:
    print("⚠️  Setup Complete (but model accuracy is low)")
    print("   Consider: more data, different features, longer HPO, or simpler strategy")
print(PRINT_LINE)
print("\nTo start the FastAPI server:")
print("  1. First kill any existing server: kill -9 $(lsof -ti:8000)")
print("  2. Then run: uvicorn main:app --reload")
print("\nAPI will be available at: http://127.0.0.1:8000")
print("Documentation at: http://127.0.0.1:8000/docs")
print("\nTest prediction example:")
print(
    f"  Features needed: {FEATURES}"
)
print(PRINT_LINE)
