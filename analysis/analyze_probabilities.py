"""
Probability Bin Analysis - Analyzes model calibration and per-bin profitability.

This script answers the critical question: "Are higher model probabilities actually more profitable?"

Usage:
    python analyze_probabilities.py

Outputs:
    - Console table showing per-bin statistics
    - Identifies which probability ranges are profitable
    - Suggests optimal threshold adjustments
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.logger import get_logger

logger = get_logger("probability_analysis")


def analyze_probability_bins(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    trades_pnl: np.ndarray,
    bins: list = None,
    confidence_threshold: float = 0.55
) -> pd.DataFrame:
    """
    Analyze model predictions binned by probability.

    Args:
        y_true: True labels (0=Sell, 1=Range, 2=Buy)
        y_pred_proba: Model probability predictions (shape: [n_samples, 3])
        trades_pnl: PnL for each sample in pips (calculated from labels and TP/SL)
        bins: Probability bin edges (default: [0.55, 0.6, 0.65, ..., 1.0])
        confidence_threshold: Minimum threshold used for filtering

    Returns:
        DataFrame with columns: bin, count, win_rate, mean_pips, total_pips, avg_prob
    """
    if bins is None:
        bins = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # Get max probability and predicted class for each sample
    max_probs = y_pred_proba.max(axis=1)
    pred_classes = y_pred_proba.argmax(axis=1)

    # Filter to only trades that passed confidence threshold (not Range predictions)
    # Range = class 1, so we want class 0 (Sell) or class 2 (Buy)
    mask = (pred_classes != 1) & (max_probs >= confidence_threshold)

    filtered_max_probs = max_probs[mask]
    filtered_y_true = y_true[mask]
    filtered_pred_classes = pred_classes[mask]
    filtered_pnl = trades_pnl[mask]

    logger.info(f"Total samples: {len(y_true)}")
    logger.info(f"Samples passing threshold ({confidence_threshold}): {mask.sum()} ({mask.sum()/len(y_true):.1%})")

    # Bin the probabilities
    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    prob_bins = pd.cut(filtered_max_probs, bins=bins, labels=bin_labels, include_lowest=True)

    # Calculate statistics per bin
    results = []
    for bin_label in bin_labels:
        bin_mask = prob_bins == bin_label
        if bin_mask.sum() == 0:
            continue

        bin_pnl = filtered_pnl[bin_mask]
        bin_y_true = filtered_y_true[bin_mask]
        bin_y_pred = filtered_pred_classes[bin_mask]

        # Win = predicted direction matches true direction
        # Sell (0) wins if true is Sell (0), Buy (2) wins if true is Buy (2)
        wins = (bin_y_pred == bin_y_true).sum()
        win_rate = wins / len(bin_y_pred)

        results.append({
            'bin': bin_label,
            'count': len(bin_y_pred),
            'win_rate': win_rate,
            'mean_pips': bin_pnl.mean(),
            'total_pips': bin_pnl.sum(),
            'avg_prob': filtered_max_probs[bin_mask].mean()
        })

    return pd.DataFrame(results)


def calculate_pnl_from_labels(y_true: np.ndarray, y_pred: np.ndarray, tp_pips: float, sl_pips: float) -> np.ndarray:
    """
    Calculate PnL for each trade based on labels.

    Simplified backtest:
    - Correct directional prediction (Buy=2 predicted and true=2, or Sell=0 predicted and true=0): +TP
    - Wrong prediction: -SL
    - Range predictions (1) or true=Range: 0 pips

    Args:
        y_true: True labels
        y_pred: Predicted labels
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips

    Returns:
        Array of PnL in pips
    """
    pnl = np.zeros(len(y_true))

    # Win conditions
    wins = (y_pred == y_true) & (y_pred != 1)  # Correct and not Range
    losses = (y_pred != y_true) & (y_pred != 1) & (y_true != 1)  # Wrong and neither is Range

    pnl[wins] = tp_pips
    pnl[losses] = -sl_pips

    return pnl


def main():
    """Run probability bin analysis on holdout set."""
    logger.info("=" * 60)
    logger.info("Probability Bin Analysis")
    logger.info("=" * 60)

    # Load config
    config = get_config()

    # Get barrier settings
    tp_pips = float(config.get("training.tp_pips", 25.0))
    sl_pips = float(config.get("training.sl_pips", 15.0))
    confidence_threshold = float(config.get("backtesting.confidence_threshold", 0.55))

    logger.info(f"Barriers: TP={tp_pips} pips, SL={sl_pips} pips")
    logger.info(f"Confidence threshold: {confidence_threshold}")

    # Load processed data
    data_file = Path(config.get("paths.data_processed_file", "EURUSD_H1_clean.csv"))
    logger.info(f"Loading data from {data_file}")

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)

    # Load model
    model_path = Path(config.get("model.path", "xgb_eurusd_h1.pkl"))
    logger.info(f"Loading model from {model_path}")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    model = joblib.load(model_path)

    # Load feature list saved during training (authoritative source)
    features_path = Path("features_used.json")
    if features_path.exists():
        import json
        with open(features_path) as f:
            feature_cols = json.load(f)
        logger.info(f"Loaded {len(feature_cols)} features from {features_path}")
    else:
        feature_cols = config.get("model.features", [])
        logger.warning("features_used.json not found, falling back to config.yaml")

    if not feature_cols:
        logger.error("No features found - run training first")
        return

    # Verify features exist in data
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        logger.error(f"Features missing from data file: {missing}")
        return

    # Split data (use same split as training)
    train_ratio = float(config.get("training.train_ratio", 0.6))
    val_ratio = float(config.get("training.val_ratio", 0.2))

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Use holdout set only
    df_holdout = df.iloc[val_end:].copy()
    logger.info(f"Holdout set: {len(df_holdout)} samples")

    # Get features and labels
    X_holdout = df_holdout[feature_cols].values

    # Apply same label mapping as training: {-1: 0, 0: 1, 1: 2}
    y_holdout = pd.Series(df_holdout['label']).map({-1: 0, 0: 1, 1: 2}).values

    logger.info(f"Feature count: {len(feature_cols)}")
    logger.info(f"Holdout shape: {X_holdout.shape}")

    # Get probability predictions
    try:
        y_pred_proba = model.predict_proba(X_holdout)
        y_pred = model.predict(X_holdout)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(f"Model type: {type(model)}")
        logger.error(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")

        # If it's a calibrated model, check the base estimator
        if hasattr(model, 'calibrated_classifiers_'):
            logger.error(f"Calibrated classifiers: {len(model.calibrated_classifiers_)}")
            logger.error(f"Classes: {model.classes_}")

        raise

    # Calculate PnL for each sample
    trades_pnl = calculate_pnl_from_labels(y_holdout, y_pred, tp_pips, sl_pips)

    # First, show overall probability distribution (all predictions)
    max_probs = y_pred_proba.max(axis=1)
    pred_classes = y_pred_proba.argmax(axis=1)

    print("\n" + "=" * 80)
    print("OVERALL PROBABILITY DISTRIBUTION (ALL PREDICTIONS)")
    print("=" * 80)
    print(f"Total samples: {len(max_probs)}")
    print(f"Max probability stats:")
    print(f"  Min:    {max_probs.min():.3f}")
    print(f"  25th:   {np.percentile(max_probs, 25):.3f}")
    print(f"  Median: {np.percentile(max_probs, 50):.3f}")
    print(f"  75th:   {np.percentile(max_probs, 75):.3f}")
    print(f"  90th:   {np.percentile(max_probs, 90):.3f}")
    print(f"  95th:   {np.percentile(max_probs, 95):.3f}")
    print(f"  Max:    {max_probs.max():.3f}")
    print(f"\nPredictions by class:")
    print(f"  Sell (0):  {(pred_classes == 0).sum()} ({(pred_classes == 0).sum()/len(pred_classes):.1%})")
    print(f"  Range (1): {(pred_classes == 1).sum()} ({(pred_classes == 1).sum()/len(pred_classes):.1%})")
    print(f"  Buy (2):   {(pred_classes == 2).sum()} ({(pred_classes == 2).sum()/len(pred_classes):.1%})")
    print(f"\nAbove threshold {confidence_threshold}: {(max_probs >= confidence_threshold).sum()} ({(max_probs >= confidence_threshold).sum()/len(max_probs):.1%})")

    # Analyze bins
    logger.info("\n" + "=" * 60)
    logger.info("Analyzing Probability Bins")
    logger.info("=" * 60)

    bin_df = analyze_probability_bins(
        y_holdout,
        y_pred_proba,
        trades_pnl,
        confidence_threshold=confidence_threshold
    )

    # Display results
    print("\n" + "=" * 80)
    print("PROBABILITY BIN ANALYSIS")
    print("=" * 80)

    table_data = []
    for _, row in bin_df.iterrows():
        table_data.append([
            row['bin'],
            int(row['count']),
            f"{row['win_rate']:.1%}",
            f"{row['mean_pips']:.2f}",
            f"{row['total_pips']:.1f}",
            f"{row['avg_prob']:.3f}"
        ])

    headers = ['Probability Bin', 'Trades', 'Win Rate', 'Mean Pips/Trade', 'Total Pips', 'Avg Prob']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Check if any trades passed threshold
    if len(bin_df) == 0:
        print("\n" + "=" * 80)
        print("⚠ NO TRADES PASSED THRESHOLD")
        print("=" * 80)
        print(f"The model is extremely conservative - no predictions reached {confidence_threshold} confidence.")
        print("\nRECOMMENDATIONS:")
        print(f"1. Lower the confidence threshold to 0.45 or 0.50")
        print(f"2. Check uncalibrated probability distribution")
        print(f"3. Consider if calibration is too aggressive for this dataset")
        print("\n" + "=" * 80)
        return

    # Calculate overall statistics
    total_trades = bin_df['count'].sum()
    total_pips = bin_df['total_pips'].sum()
    weighted_win_rate = (bin_df['win_rate'] * bin_df['count']).sum() / total_trades
    mean_pips_overall = total_pips / total_trades

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total trades (after threshold): {int(total_trades)}")
    print(f"Overall win rate: {weighted_win_rate:.1%}")
    print(f"Mean pips per trade: {mean_pips_overall:.2f}")
    print(f"Total pips: {total_pips:.1f}")
    print(f"Expected R per trade: {mean_pips_overall / sl_pips:.3f}R")

    # Identify profitable bins
    profitable_bins = bin_df[bin_df['mean_pips'] > 0]
    if len(profitable_bins) > 0:
        print("\n" + "=" * 80)
        print("PROFITABLE BINS (mean_pips > 0)")
        print("=" * 80)
        for _, row in profitable_bins.iterrows():
            print(f"  {row['bin']}: {row['count']} trades, {row['mean_pips']:.2f} pips/trade, {row['win_rate']:.1%} win rate")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Check if higher probabilities correlate with better performance
    if len(bin_df) > 1:
        correlation = bin_df['avg_prob'].corr(bin_df['mean_pips'])
        print(f"Probability ↔ Performance correlation: {correlation:.3f}")

        if correlation > 0.3:
            print("✓ Higher probabilities correlate with better performance (good calibration)")
        elif correlation < -0.1:
            print("⚠ NEGATIVE correlation - model is miscalibrated! Consider probability calibration.")
        else:
            print("⚠ Weak correlation - probabilities don't strongly predict outcomes. Consider calibration.")

    # Suggest threshold adjustment
    best_bin = bin_df.loc[bin_df['mean_pips'].idxmax()]
    if best_bin['avg_prob'] > confidence_threshold + 0.05:
        print(f"\n💡 Consider raising threshold to ~{best_bin['avg_prob']:.2f} (best performing bin)")

    # Check for unprofitable low-prob bins
    low_prob_bins = bin_df[bin_df['avg_prob'] < confidence_threshold + 0.05]
    if len(low_prob_bins) > 0 and low_prob_bins['mean_pips'].mean() < 0:
        print(f"💡 Low probability bins ({confidence_threshold:.2f}-{confidence_threshold+0.05:.2f}) are losing - raise threshold")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
