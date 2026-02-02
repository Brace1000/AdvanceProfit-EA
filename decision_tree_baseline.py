"""
Decision Tree Baseline - See what rules the model actually finds.

Purpose: Before using XGBoost (black box), train an interpretable decision tree
so we can READ the rules and judge whether they make sense or are spurious.

A max_depth=3 tree has at most 8 leaf nodes, making rules easy to interpret.
If the rules look like "buy on Wednesdays when RSI > 52.3", the features are garbage.
If they look like "sell when RSI > 70 and trend is down", there might be real signal.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, f1_score

from src.config import get_config
from src.features.engineer import FeatureEngineer
from src.logger import get_logger

logger = get_logger("decision_tree_baseline")


def main():
    config = get_config()

    # Load processed data
    data_file = Path(config.get("paths.data_processed_file", "EURUSD_H1_clean.csv"))
    df = pd.read_csv(data_file)

    # Load features from training or default
    features_path = Path("features_used.json")
    if features_path.exists():
        with open(features_path) as f:
            feature_cols = json.load(f)
        print(f"Using {len(feature_cols)} features from features_used.json")
    else:
        feature_cols = FeatureEngineer.default_feature_columns()
        feature_cols = [f for f in feature_cols if f in df.columns]
        print(f"Using {len(feature_cols)} features from default list")

    # Verify features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
        return

    # Prepare data with same split as training
    train_ratio = float(config.get("training.train_ratio", 0.6))
    val_ratio = float(config.get("training.val_ratio", 0.2))

    X = df[feature_cols].values
    y = pd.Series(df["label"]).map({-1: 0, 0: 1, 1: 2}).values

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_holdout = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_holdout = y[val_end:]

    class_names = ["Sell", "Range", "Buy"]

    print("=" * 80)
    print("DECISION TREE BASELINE")
    print("=" * 80)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Holdout: {len(X_holdout)}")
    print(f"Features: {len(feature_cols)}")

    # Train trees at different depths to see complexity vs performance tradeoff
    for depth in [2, 3, 4, 5]:
        print(f"\n{'=' * 80}")
        print(f"DECISION TREE - max_depth={depth}")
        print(f"{'=' * 80}")

        tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=50,  # Require at least 50 samples per leaf
            class_weight="balanced",
            random_state=42,
        )
        tree.fit(X_train, y_train)

        # Accuracy on all splits
        train_acc = tree.score(X_train, y_train)
        val_acc = tree.score(X_val, y_val)
        holdout_acc = tree.score(X_holdout, y_holdout)

        train_f1 = f1_score(y_train, tree.predict(X_train), average="macro", zero_division=0)
        val_f1 = f1_score(y_val, tree.predict(X_val), average="macro", zero_division=0)
        holdout_f1 = f1_score(y_holdout, tree.predict(X_holdout), average="macro", zero_division=0)

        print(f"\n  Train accuracy:   {train_acc:.2%}  (F1: {train_f1:.3f})")
        print(f"  Val accuracy:     {val_acc:.2%}  (F1: {val_f1:.3f})")
        print(f"  Holdout accuracy: {holdout_acc:.2%}  (F1: {holdout_f1:.3f})")
        print(f"  Overfit gap:      {train_acc - holdout_acc:.2%}")

        # Show the rules (only for depth <= 3 to keep output readable)
        if depth <= 3:
            print(f"\n  TREE RULES:")
            rules = export_text(tree, feature_names=feature_cols, class_names=class_names)
            # Indent for readability
            for line in rules.split("\n"):
                print(f"    {line}")

        # Feature importance
        importances = tree.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n  TOP FEATURES:")
        for rank in range(min(10, len(feature_cols))):
            idx = indices[rank]
            if importances[idx] > 0:
                print(f"    {rank+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

        # Holdout classification report
        holdout_preds = tree.predict(X_holdout)
        print(f"\n  HOLDOUT CLASSIFICATION REPORT:")
        report = classification_report(
            y_holdout, holdout_preds,
            target_names=class_names,
            zero_division=0,
        )
        for line in report.split("\n"):
            print(f"    {line}")

    # Compare: what does "random" look like?
    print(f"\n{'=' * 80}")
    print("REFERENCE: RANDOM BASELINE")
    print(f"{'=' * 80}")
    print(f"  Random accuracy (3-class): 33.33%")
    print(f"  Holdout true distribution:")
    for cls, name in enumerate(class_names):
        count = (y_holdout == cls).sum()
        pct = count / len(y_holdout)
        print(f"    {name}: {count} ({pct:.1%})")

    majority_class = np.argmax(np.bincount(y_holdout))
    majority_acc = (y_holdout == majority_class).mean()
    print(f"  Always-predict-{class_names[majority_class]} accuracy: {majority_acc:.2%}")


if __name__ == "__main__":
    main()
