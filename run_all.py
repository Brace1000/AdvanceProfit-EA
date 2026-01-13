#!/usr/bin/env python3
"""
Complete pipeline to prepare data, train model, and verify setup
Run this before starting the FastAPI server
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path

print("=" * 60)
print("EUR/USD Trading Model - Complete Setup")
print("=" * 60)

# ============================================================================
# STEP 1: Load and Clean Data
# ============================================================================
print("\n[1/3] Loading and cleaning data...")
ort xgboost as xgb
df = pd.read_csv("EURUSD_D1_raw.csv", on_bad_lines='skip')
df.columns = df.columns.str.strip().str.lower()

print(f"  → Loaded {len(df)} rows")

# Convert price columns to numeric, coercing errors to NaN
price_cols = ["open", "high", "low", "close"]
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any NaN values in price columns
initial_len = len(df)
df.dropna(subset=price_cols, inplace=True)
removed = initial_len - len(df)
if removed > 0:
    print(f"  → Removed {removed} rows with invalid data")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[2/3] Building features...")

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

# Create labels with more realistic thresholds
future_return = df["close"].shift(-1) / df["close"] - 1
buy_threshold = 0.002   # 0.2% movement (more realistic for daily)
sell_threshold = -0.002

df["label"] = 0
df.loc[future_return > buy_threshold, "label"] = 1
df.loc[future_return < sell_threshold, "label"] = -1

# Remove rows with NaN
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("EURUSD_D1_clean.csv", index=False)
print(f"  → Created {len(df)} samples with features")
print(f"  → Label distribution: Sell={sum(df['label']==-1)}, "
      f"Range={sum(df['label']==0)}, Buy={sum(df['label']==1)}")

# ============================================================================
# STEP 3: Train Model (with regularization to prevent overfitting)
# ============================================================================
print("\n[3/3] Training XGBoost model...")

FEATURES = [
    "close_ema50", "ema50_ema200", "rsi", "rsi_slope",
    "atr_ratio", "adx", "body", "range",
    "hour", "session", "prev_return"
]

# Map labels (-1,0,1) to (0,1,2)
y = df["label"].map({-1: 0, 0: 1, 1: 2})
X = df[FEATURES]

# Use larger test set for better validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

print(f"  → Training samples: {len(X_train)}")
print(f"  → Validation samples: {len(X_val)}")

# Reduced complexity to prevent overfitting
model = xgb.XGBClassifier(
    n_estimators=100,      # Reduced from 400
    max_depth=3,           # Reduced from 5
    learning_rate=0.1,     # Increased from 0.04
    subsample=0.8,         # Reduced from 0.85
    colsample_bytree=0.8,  # Reduced from 0.85
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    reg_alpha=1.0,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    min_child_weight=3,    # Prevent overfitting
    gamma=0.1              # Minimum loss reduction
)

# Use early stopping
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

# Evaluate
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)

print(f"  → Training accuracy: {train_acc:.2%}")
print(f"  → Validation accuracy: {val_acc:.2%}")

# Calculate per-class metrics
from sklearn.metrics import classification_report
y_pred = model.predict(X_val)
print("\n  Validation Classification Report:")
print(classification_report(y_val, y_pred, target_names=['Sell', 'Range', 'Buy'], zero_division=0))

# Save model
joblib.dump(model, "xgb_eurusd_h1.pkl")
print(f"  → Model saved as 'xgb_eurusd_h1.pkl'")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
if val_acc > 0.35:  # Better than random for 3 classes
    print("✅ Setup Complete!")
else:
    print("⚠️  Setup Complete (but model accuracy is low)")
    print("   Consider: more data, different features, or simpler strategy")
print("=" * 60)
print("\nTo start the FastAPI server:")
print("  1. First kill any existing server: kill -9 $(lsof -ti:8000)")
print("  2. Then run: uvicorn main:app --reload")
print("\nAPI will be available at: http://127.0.0.1:8000")
print("Documentation at: http://127.0.0.1:8000/docs")
print("\nTest prediction example:")
print(f"  Features needed: {FEATURES}")
print("=" * 60)