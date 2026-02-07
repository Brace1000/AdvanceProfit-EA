"""Diagnostic script to understand model behavior."""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from src.config import get_config
from src.logger import get_logger

logger = get_logger("model_diagnostics")

config = get_config()

# Load data
data_file = Path(config.get("paths.data_processed_file", "EURUSD_H1_clean.csv"))
df = pd.read_csv(data_file)

# Load feature list from training (authoritative source)
features_path = Path("features_used.json")
if features_path.exists():
    with open(features_path) as f:
        feature_cols = json.load(f)
    print(f"Loaded {len(feature_cols)} features from {features_path}")
else:
    feature_cols = config.get("model.features", [])
    print("WARNING: features_used.json not found, using config.yaml")
train_ratio = float(config.get("training.train_ratio", 0.6))
val_ratio = float(config.get("training.val_ratio", 0.2))

# Split same way as training
n = len(df)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

df_holdout = df.iloc[val_end:].copy()
X_holdout = df_holdout[feature_cols].values

# Apply same label mapping as training: {-1: 0, 0: 1, 1: 2}
y_holdout = pd.Series(df_holdout['label']).map({-1: 0, 0: 1, 1: 2}).values

print("=" * 80)
print("CALIBRATED MODEL DIAGNOSTICS")
print("=" * 80)

# Load calibrated model
model_path = Path(config.get("model.path", "xgb_eurusd_h1.pkl"))
model = joblib.load(model_path)

print(f"\n1. MODEL TYPE")
print(f"   Type: {type(model)}")
print(f"   Is CalibratedClassifierCV: {hasattr(model, 'calibrated_classifiers_')}")

if hasattr(model, 'classes_'):
    print(f"   Classes: {model.classes_}")

if hasattr(model, 'calibrated_classifiers_'):
    print(f"   Number of calibrated classifiers: {len(model.calibrated_classifiers_)}")

    # Try to access base estimator
    if len(model.calibrated_classifiers_) > 0:
        cal_clf = model.calibrated_classifiers_[0]
        print(f"   Base estimator type: {type(cal_clf.estimator)}")

        if hasattr(cal_clf.estimator, 'classes_'):
            print(f"   Base estimator classes: {cal_clf.estimator.classes_}")

print(f"\n2. HOLDOUT SET")
print(f"   Samples: {len(X_holdout)}")
print(f"   Features: {len(feature_cols)}")
print(f"   True label distribution:")
for label in [0, 1, 2]:
    count = (y_holdout == label).sum()
    pct = count / len(y_holdout)
    print(f"      Class {label}: {count} ({pct:.1%})")

print(f"\n3. CALIBRATED MODEL PREDICTIONS")
probs = model.predict_proba(X_holdout)
preds = model.predict(X_holdout)

print(f"   Probabilities shape: {probs.shape}")
print(f"   Per-class mean probability:")
for i in range(probs.shape[1]):
    print(f"      Class {i}: {probs[:, i].mean():.4f}")

max_probs = probs.max(axis=1)
print(f"\n   Max probability percentiles:")
print(f"      Min:    {max_probs.min():.4f}")
print(f"      25th:   {np.percentile(max_probs, 25):.4f}")
print(f"      Median: {np.percentile(max_probs, 50):.4f}")
print(f"      75th:   {np.percentile(max_probs, 75):.4f}")
print(f"      95th:   {np.percentile(max_probs, 95):.4f}")
print(f"      Max:    {np.percentile(max_probs, 100):.4f}")

print(f"\n   Predicted class distribution:")
unique, counts = np.unique(preds, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"      Class {cls}: {cnt} ({cnt/len(preds):.1%})")

print(f"\n4. CONFUSION MATRIX (Calibrated)")
print(f"   True labels (rows) vs Predicted (cols):")
cm = confusion_matrix(y_holdout, preds)
print(cm)

print(f"\n5. CLASSIFICATION REPORT (Calibrated)")
print(classification_report(y_holdout, preds, zero_division=0))

# Try to get raw uncalibrated predictions
print(f"\n6. BASE (UNCALIBRATED) MODEL PREDICTIONS")
if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
    try:
        base_model = model.calibrated_classifiers_[0].estimator

        # Get raw predictions
        raw_probs = base_model.predict_proba(X_holdout)
        raw_preds = base_model.predict(X_holdout)

        print(f"   Raw probabilities shape: {raw_probs.shape}")
        print(f"   Per-class mean probability (RAW):")
        for i in range(raw_probs.shape[1]):
            print(f"      Class {i}: {raw_probs[:, i].mean():.4f}")

        raw_max_probs = raw_probs.max(axis=1)
        print(f"\n   Raw max probability percentiles:")
        print(f"      Min:    {raw_max_probs.min():.4f}")
        print(f"      25th:   {np.percentile(raw_max_probs, 25):.4f}")
        print(f"      Median: {np.percentile(raw_max_probs, 50):.4f}")
        print(f"      75th:   {np.percentile(raw_max_probs, 75):.4f}")
        print(f"      95th:   {np.percentile(raw_max_probs, 95):.4f}")
        print(f"      Max:    {np.percentile(raw_max_probs, 100):.4f}")

        print(f"\n   Raw predicted class distribution:")
        unique_raw, counts_raw = np.unique(raw_preds, return_counts=True)
        for cls, cnt in zip(unique_raw, counts_raw):
            print(f"      Class {cls}: {cnt} ({cnt/len(raw_preds):.1%})")

        print(f"\n   CONFUSION MATRIX (Raw/Uncalibrated)")
        cm_raw = confusion_matrix(y_holdout, raw_preds)
        print(cm_raw)

    except Exception as e:
        print(f"   Could not access base model predictions: {e}")
else:
    print(f"   Model is not calibrated wrapper - no base model to check")

# Check validation set used for calibration
print(f"\n7. VALIDATION SET (used for calibration)")
df_val = df.iloc[train_end:val_end].copy()

# Apply same label mapping as training
y_val = pd.Series(df_val['label']).map({-1: 0, 0: 1, 1: 2}).values

print(f"   Validation samples: {len(y_val)}")
print(f"   Validation label distribution:")
for label in [0, 1, 2]:
    count = (y_val == label).sum()
    pct = count / len(y_val)
    print(f"      Class {label}: {count} ({pct:.1%})")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
