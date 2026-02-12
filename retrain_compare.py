"""Retrain model with all Phase 3 fixes and compare old vs new metrics."""
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, classification_report

from src.config import get_config
from src.pipelines.training import TrainingPipeline


def evaluate_model(model_path, X, y, close_prices, label="model"):
    """Evaluate a model on given data and return metrics dict."""
    model = joblib.load(model_path)
    preds = model.predict(X)

    accuracy = float(np.mean(preds == y))
    f1_macro = f1_score(y, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(y, preds, average=None, zero_division=0)

    # Backtest metrics
    position = np.where(preds == 2, 1.0, np.where(preds == 0, -1.0, 0.0))
    ret = np.diff(close_prices) / close_prices[:-1]
    n = min(len(position) - 1, len(ret))
    commission = 0.0001
    pnl = position[:n] * ret[:n] - (np.abs(position[:n]) > 0).astype(float) * commission

    trades = int(np.sum(np.abs(position[:n]) > 0))
    wins = int(np.sum(pnl > 0))
    losses = int(np.sum(pnl < 0))
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    avg_ret = float(np.mean(pnl)) if len(pnl) > 0 else 0.0
    std_ret = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    cum_curve = np.cumprod(1 + pnl)
    rolling_max = np.maximum.accumulate(cum_curve)
    drawdowns = cum_curve / rolling_max - 1
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "label": label,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_sell": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_range": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "f1_buy": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
        "total_trades": trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def main():
    print("=" * 60)
    print("PHASE 3: RETRAIN + COMPARISON")
    print("=" * 60)

    # Run training pipeline (new model with all fixes)
    print("\n[1/3] Running training pipeline with Phase 3 fixes...")
    pipeline = TrainingPipeline()
    result = pipeline.run()

    new_model_path = result["model_path"]
    old_model_path = "xgb_eurusd_h1_old.pkl"
    feature_cols = result["features_used"]

    print(f"\nNew model saved to: {new_model_path}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Class distribution: {result['class_distribution']}")

    # Load holdout data for comparison
    print("\n[2/3] Loading holdout data for comparison...")
    processed_path = result.get("processed_path", "EURUSD_H1_clean.csv")
    df = pd.read_csv(processed_path)

    # Recreate split
    config = get_config()
    train_ratio = float(config.get("training.train_ratio", 0.6))
    val_ratio = float(config.get("training.val_ratio", 0.2))
    n = len(df)
    val_end = int(n * (train_ratio + val_ratio))

    holdout_df = df.iloc[val_end:].copy()
    if len(holdout_df) < 10:
        print("WARNING: Holdout set too small for meaningful comparison")
        return

    # Check which features exist
    available_features = [f for f in feature_cols if f in holdout_df.columns]
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        print(f"WARNING: Missing features in holdout data: {missing}")
        print("Old model comparison skipped (feature mismatch)")

    X_holdout = holdout_df[available_features].values
    y_holdout = pd.Series(holdout_df["label"]).map({-1: 0, 0: 1, 1: 2}).values
    close_holdout = holdout_df["close"].values

    # Evaluate new model
    print("\n[3/3] Evaluating models on holdout...")
    new_metrics = evaluate_model(new_model_path, X_holdout, y_holdout, close_holdout, "NEW")

    # Try evaluating old model (may fail due to feature count mismatch)
    old_metrics = None
    try:
        old_model = joblib.load(old_model_path)
        n_features_old = old_model.n_features_in_
        if n_features_old == X_holdout.shape[1]:
            old_metrics = evaluate_model(old_model_path, X_holdout, y_holdout, close_holdout, "OLD")
        else:
            print(f"Old model expects {n_features_old} features, holdout has {X_holdout.shape[1]}. Skipping old comparison.")
    except Exception as e:
        print(f"Could not evaluate old model: {e}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)

    header = f"{'Metric':<25} {'NEW':>12}"
    if old_metrics:
        header += f" {'OLD':>12} {'Delta':>12}"
    print(header)
    print("-" * len(header))

    metrics_to_show = [
        ("Holdout Accuracy", "accuracy", ".2%"),
        ("F1 Macro", "f1_macro", ".4f"),
        ("F1 Sell", "f1_sell", ".4f"),
        ("F1 Range", "f1_range", ".4f"),
        ("F1 Buy", "f1_buy", ".4f"),
        ("Total Trades", "total_trades", "d"),
        ("Win Rate", "win_rate", ".2%"),
        ("Avg Trade Return", "avg_trade_return", ".6f"),
        ("Sharpe Ratio", "sharpe", ".2f"),
        ("Max Drawdown", "max_drawdown", ".2%"),
    ]

    for display_name, key, fmt in metrics_to_show:
        new_val = new_metrics[key]
        row = f"{display_name:<25} {new_val:>{12}{fmt}}"
        if old_metrics:
            old_val = old_metrics[key]
            delta = new_val - old_val
            row += f" {old_val:>{12}{fmt}} {delta:>+{12}{fmt}}"
        print(row)

    # Print selected hyperparameters
    print("\n" + "=" * 60)
    print("SELECTED HYPERPARAMETERS")
    print("=" * 60)
    params = config.get("model.params", {})
    if isinstance(params, dict):
        for k, v in sorted(params.items()):
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("FINAL THRESHOLDS")
    print("=" * 60)
    print(f"  Confidence threshold: {config.get('trading.ml.confidence_threshold')}")
    print(f"  Buy label threshold:  {config.get('training.buy_threshold')}")
    print(f"  Sell label threshold: {config.get('training.sell_threshold')}")

    # Save results
    results_out = {
        "new_metrics": new_metrics,
        "old_metrics": old_metrics,
        "features_used": feature_cols,
        "class_distribution": result["class_distribution"],
    }
    with open("retrain_results.json", "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\nResults saved to retrain_results.json")

    print(f"\nRetrained model confirmed saved: {new_model_path}")


if __name__ == "__main__":
    main()
