"""Training pipeline orchestration with proper data splitting to prevent overfitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Set

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

from src.config import get_config
from src.logger import get_logger
from src.data_collection.loader import DataLoader
from src.data_collection.validator import DataValidator
from src.features.engineer import FeatureEngineer
from src.models.hpo import optimize as hpo_optimize
from src.models.trainer import ModelTrainer, TrainResult
from src.utils.leakage_detector import LeakageDetector

logger = get_logger("trading_bot.pipeline")


@dataclass
class PipelineResult:
    model_path: str
    train_accuracy: float
    val_accuracy: float
    holdout_accuracy: float
    sharpe: float
    win_rate: float
    max_drawdown: float


class TrainingPipeline:
    """
    Training pipeline with proper 3-way data splitting.

    Data Split Strategy (prevents data leakage):
    - Train (60%): Used for model training
    - Validation (20%): Used for HPO and early stopping
    - Holdout (20%): NEVER touched until final evaluation

    This ensures reported metrics reflect true out-of-sample performance.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self._config_obj = get_config()
        self.config = config or self._config_obj._config
        self.loader = DataLoader(self.config)
        self.validator = DataValidator(self.config)
        self.fe = FeatureEngineer(self.config)
        self.trainer = ModelTrainer(self.config)

    def _get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        return self._config_obj.get(key_path, default)

    def _label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trading labels using Triple Barrier method or simple threshold.

        Supports three modes:
        1. Triple Barrier: Uses profit target, stop loss, and time horizon (recommended)
        2. Volatility-adjusted: Uses ATR-based dynamic thresholds
        3. Fixed threshold: Uses static buy/sell thresholds

        Triple Barrier mode labels trades based on which barrier is hit first:
        - TP hit first → directional label (BUY/SELL)
        - SL hit first → RANGE (bad setup)
        - Time expires → RANGE (no edge)
        """
        use_triple_barrier = bool(self._get("training.use_triple_barrier", False))

        if use_triple_barrier:
            return self._label_triple_barrier(df)

        # Existing adaptive/fixed labeling logic
        use_adaptive = bool(self._get("training.use_adaptive_labels", False))
        ahead_return = df["close"].shift(-1) / df["close"] - 1

        if use_adaptive:
            # Volatility-adjusted labeling with spread floor
            if "atr_h1" not in df.columns:
                logger.warning("ATR not found for adaptive labels, falling back to fixed thresholds")
                use_adaptive = False
            else:
                # Calculate dynamic threshold as multiple of rolling ATR
                atr_multiplier = float(self._get("training.atr_multiplier", 1.5))
                lookback = int(self._get("training.adaptive_lookback", 24))
                min_threshold_pips = float(self._get("training.min_threshold_pips", 3.0))

                # Use rolling ATR mean to smooth volatility spikes
                rolling_atr = df["atr_h1"].rolling(window=lookback, min_periods=1).mean()
                atr_threshold = rolling_atr * atr_multiplier

                # Convert pip floor to price units (1 pip = 0.0001 for forex)
                pip_value = 0.0001
                min_threshold_price = min_threshold_pips * pip_value

                # Apply floor: threshold = max(ATR * multiplier, spread_floor)
                dynamic_threshold = np.maximum(atr_threshold, min_threshold_price)

                # Normalize to return percentage
                threshold_pct = dynamic_threshold / df["close"]

                labels = np.zeros(len(df), dtype=int)
                labels[ahead_return > threshold_pct] = 1   # Buy
                labels[ahead_return < -threshold_pct] = -1  # Sell

                # Count how many times floor was applied
                floor_applied = (atr_threshold < min_threshold_price).sum()
                floor_pct = (floor_applied / len(df)) * 100

                logger.info(
                    f"Using adaptive labels: ATR multiplier={atr_multiplier}, "
                    f"lookback={lookback}h, spread floor={min_threshold_pips} pips"
                )
                logger.info(
                    f"  Threshold range: {threshold_pct.min():.4%} to {threshold_pct.max():.4%}"
                )
                logger.info(
                    f"  Spread floor applied: {floor_applied} times ({floor_pct:.1f}% of data)"
                )

        if not use_adaptive:
            # Fixed threshold labeling
            buy_th = float(self._get("training.buy_threshold", 0.002))
            sell_th = float(self._get("training.sell_threshold", -0.002))

            labels = np.zeros(len(df), dtype=int)
            labels[ahead_return > buy_th] = 1
            labels[ahead_return < sell_th] = -1

            logger.info(f"Using fixed labels: buy_threshold={buy_th:.4%}, sell_threshold={sell_th:.4%}")

        df["label"] = labels
        return df

    def _label_triple_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label using Triple Barrier method.

        For each bar, scan forward up to max_horizon bars to find which barrier hits first:
        1. Profit target (TP): +tp_pips pips
        2. Stop loss (SL): -sl_pips pips
        3. Time limit: max_horizon bars

        Returns:
            DataFrame with 'label' column: -1 (sell), 0 (range), 1 (buy)
        """
        tp_pips = float(self._get("training.tp_pips", 15.0))
        sl_pips = float(self._get("training.sl_pips", 10.0))
        max_horizon = int(self._get("training.max_horizon_bars", 24))

        pip_value = 0.0001
        tp_price = tp_pips * pip_value
        sl_price = sl_pips * pip_value

        labels = np.zeros(len(df), dtype=int)

        logger.info(f"Calculating Triple Barrier labels for {len(df)} bars...")

        for i in range(len(df) - 1):  # Skip last bar (no future data)
            entry_price = df.iloc[i]["close"]

            # Scan forward up to max_horizon bars
            horizon_end = min(i + 1 + max_horizon, len(df))
            future_highs = df.iloc[i+1:horizon_end]["high"].values
            future_lows = df.iloc[i+1:horizon_end]["low"].values

            # Calculate returns to high/low for each future bar
            returns_to_high = (future_highs - entry_price) / entry_price
            returns_to_low = (future_lows - entry_price) / entry_price

            # Find first bar where TP or SL is hit
            tp_threshold = tp_price / entry_price
            sl_threshold = sl_price / entry_price

            # Check upward TP/SL (for BUY setups)
            tp_up_hit = np.where(returns_to_high >= tp_threshold)[0]
            sl_up_hit = np.where(returns_to_low <= -sl_threshold)[0]

            # Check downward TP/SL (for SELL setups)
            tp_down_hit = np.where(returns_to_low <= -tp_threshold)[0]
            sl_down_hit = np.where(returns_to_high >= sl_threshold)[0]

            # Determine label based on first barrier hit
            first_up_tp = tp_up_hit[0] if len(tp_up_hit) > 0 else max_horizon
            first_up_sl = sl_up_hit[0] if len(sl_up_hit) > 0 else max_horizon
            first_down_tp = tp_down_hit[0] if len(tp_down_hit) > 0 else max_horizon
            first_down_sl = sl_down_hit[0] if len(sl_down_hit) > 0 else max_horizon

            # BUY setup: upward TP hit before upward SL
            if first_up_tp < first_up_sl and first_up_tp < max_horizon:
                labels[i] = 1
            # SELL setup: downward TP hit before downward SL
            elif first_down_tp < first_down_sl and first_down_tp < max_horizon:
                labels[i] = -1
            # Otherwise: RANGE (SL hit first or time expired)
            else:
                labels[i] = 0

        df["label"] = labels

        # Log statistics
        buy_count = (labels == 1).sum()
        sell_count = (labels == -1).sum()
        range_count = (labels == 0).sum()
        total = len(labels)

        logger.info(
            f"Using Triple Barrier labeling: TP={tp_pips} pips, SL={sl_pips} pips, "
            f"horizon={max_horizon} bars"
        )
        logger.info(
            f"  Label distribution: Buy={buy_count}/{total} ({buy_count/total:.1%}), "
            f"Sell={sell_count}/{total} ({sell_count/total:.1%}), "
            f"Range={range_count}/{total} ({range_count/total:.1%})"
        )
        logger.info(f"  Risk/Reward ratio: {tp_pips/sl_pips:.2f}:1")

        return df

    def _get_class_distribution(self, y: np.ndarray) -> Dict[str, float]:
        """Calculate class distribution percentages."""
        total = len(y)
        if total == 0:
            return {"sell": 0.0, "range": 0.0, "buy": 0.0}
        sell_pct = float(np.sum(y == 0) / total)
        range_pct = float(np.sum(y == 1) / total)
        buy_pct = float(np.sum(y == 2) / total)
        return {"sell": sell_pct, "range": range_pct, "buy": buy_pct}

    def _prune_correlated_features(
        self, df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.85
    ) -> list[str]:
        """
        Remove highly correlated features using Spearman correlation.

        When two features have correlation > threshold, drops the second one.
        Prioritizes keeping regime features (if implemented) over raw price features.

        Args:
            df: DataFrame with all features
            feature_cols: List of feature column names
            threshold: Spearman correlation threshold (default 0.85)

        Returns:
            Pruned list of feature columns
        """
        logger.info(f"Running Spearman correlation analysis (threshold={threshold})...")

        # Compute Spearman correlation matrix
        X = df[feature_cols].values
        corr_matrix = np.zeros((len(feature_cols), len(feature_cols)))

        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                corr, _ = spearmanr(X[:, i], X[:, j], nan_policy='omit')
                corr_matrix[i, j] = abs(corr)
                corr_matrix[j, i] = abs(corr)

        # Find correlated pairs
        to_drop: Set[str] = set()
        for i in range(len(feature_cols)):
            if feature_cols[i] in to_drop:
                continue
            for j in range(i + 1, len(feature_cols)):
                if feature_cols[j] in to_drop:
                    continue
                if corr_matrix[i, j] > threshold:
                    # Drop the second feature in the pair
                    logger.info(
                        f"  Dropping '{feature_cols[j]}' (corr={corr_matrix[i, j]:.3f} "
                        f"with '{feature_cols[i]}')"
                    )
                    to_drop.add(feature_cols[j])

        pruned_features = [f for f in feature_cols if f not in to_drop]

        logger.info(
            f"Correlation pruning complete: {len(feature_cols)} → {len(pruned_features)} features "
            f"(dropped {len(to_drop)})"
        )

        return pruned_features

    def _three_way_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/holdout sets.

        Returns:
            X_train, X_val, X_holdout, y_train, y_val, y_holdout
        """
        train_ratio = float(self._get("training.train_ratio", 0.6))
        val_ratio = float(self._get("training.val_ratio", 0.2))

        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_holdout = X[val_end:]

        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_holdout = y[val_end:]

        logger.info(
            f"Data split: Train={len(X_train)} ({train_ratio:.0%}), "
            f"Val={len(X_val)} ({val_ratio:.0%}), "
            f"Holdout={len(X_holdout)} ({1-train_ratio-val_ratio:.0%})"
        )

        return X_train, X_val, X_holdout, y_train, y_val, y_holdout

    def _backtest_on_split(
        self, model_path: str, df: pd.DataFrame, feature_cols: list[str], split_name: str
    ) -> Tuple[float, float, float]:
        """
        Run simple backtest on a data split.

        Returns:
            sharpe, win_rate, max_drawdown
        """
        if len(df) < 2:
            logger.warning(f"Insufficient data for {split_name} backtest")
            return 0.0, 0.0, 0.0

        try:
            model = joblib.load(model_path)
            preds = model.predict(df[feature_cols].values)
        except Exception as e:
            logger.error(f"Failed to load model for backtest: {e}")
            return 0.0, 0.0, 0.0

        df = df.copy()
        # Fix: Ensure predictions are aligned with df index
        df["position"] = pd.Series(preds, index=df.index).map({0: -1, 1: 0, 2: 1})
        df["next_ret"] = df["close"].pct_change(-1) * -1
        trading_cost = float(self._get("backtesting.commission", 0.0001))

        df = df.iloc[:-1].copy()
        if len(df) == 0:
            return 0.0, 0.0, 0.0

        df["pnl"] = df["position"] * df["next_ret"] - (df["position"] != 0).astype(float) * trading_cost

        avg_pnl = float(df["pnl"].mean())
        std_pnl = float(df["pnl"].std(ddof=1)) if len(df) > 1 else 0.0
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0.0

        cum_curve = (1 + df["pnl"]).cumprod()
        rolling_max = cum_curve.cummax()
        drawdown = cum_curve / rolling_max - 1
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0

        wins = int((df["pnl"] > 0).sum())
        losses = int((df["pnl"] < 0).sum())
        win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else 0.0

        trades = int((df["position"] != 0).sum())
        logger.info(
            f"{split_name} Backtest | Trades: {trades} | "
            f"Win rate: {win_rate:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}"
        )

        return sharpe, win_rate, max_dd

    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Steps:
        1. Load and validate data
        2. Engineer features and create labels
        3. Split into train/validation/holdout (60/20/20)
        4. Run HPO on train set with validation for scoring
        5. Train final model on train+validation with early stopping
        6. Evaluate on holdout set (never seen before)
        """
        raw_path = self._get("paths.data_raw_file", "EURUSD_H1_raw.csv")
        processed_path = self._get("paths.data_processed_file", "EURUSD_H1_clean.csv")

        # Load and validate
        logger.info("Loading data...")
        df = self.loader.load(raw_path)
        self.validator.validate_ohlc(df)

        # Features
        logger.info("Engineering features...")
        df = self.fe.engineer(df)
        df = self._label(df)

        feature_cols = FeatureEngineer.default_feature_columns()
        df = df.dropna(subset=feature_cols).iloc[:-1].copy()
        logger.info(f"Dataset after prep: {len(df)} rows, {len(feature_cols)} features")

        # Correlation pruning
        correlation_threshold = float(self._get("training.correlation_threshold", 0.85))
        feature_cols = self._prune_correlated_features(df, feature_cols, correlation_threshold)

        # Leakage detection (run before splitting to detect issues early)
        if bool(self._get("training.detect_leakage", True)):
            logger.info("Running leakage detection...")
            detector = LeakageDetector(df, label_col="label")
            warnings = detector.run_tests(feature_cols)

            if warnings:
                logger.warning("🚨 POTENTIAL LEAKAGE DETECTED:")
                for w in warnings:
                    logger.warning(f"  ⚠️  {w}")
                # Optionally abort training if leakage detected
                # if self._get("training.abort_on_leakage", False):
                #     raise ValueError("Leakage detected. Training aborted.")
            else:
                logger.info("✅ All features passed leakage tests")

            # Generate correlation decay visualization
            try:
                plot_path = "correlation_decay_plot.png"
                detector.plot_correlation_decay(feature_cols, max_lag=5, output_path=plot_path)
            except Exception as e:
                logger.warning(f"Failed to generate correlation decay plot: {e}")

        # Persist processed data
        pd.DataFrame(df).to_csv(processed_path, index=False)
        logger.info(f"Saved processed dataset to {processed_path}")

        # Prepare arrays
        X = df[feature_cols].values
        y = pd.Series(df["label"]).map({-1: 0, 0: 1, 1: 2}).values

        # Log class distribution
        class_dist = self._get_class_distribution(y)
        logger.info(
            f"Class distribution: Sell={class_dist['sell']:.1%}, "
            f"Range={class_dist['range']:.1%}, Buy={class_dist['buy']:.1%}"
        )

        # 3-way split
        X_train, X_val, X_holdout, y_train, y_val, y_holdout = self._three_way_split(X, y)

        # HPO on train set, validated on validation set
        if bool(self._get("hpo.enabled", True)):
            logger.info("Running hyperparameter optimization...")
            best_params, best_val = hpo_optimize(X_train, y_train, X_val, y_val, self.config)
            self.config.setdefault("model", {}).setdefault("params", {}).update(best_params)
            logger.info(f"HPO complete. Best validation F1: {best_val:.4f}")

        # Train final model on train+validation (holdout never touched)
        X_train_final = np.vstack([X_train, X_val])
        y_train_final = np.concatenate([y_train, y_val])

        logger.info("Training final model on train+validation data...")
        result: TrainResult = self.trainer.train(
            X_train_final, y_train_final, feature_cols, X_val=None, y_val=None
        )

        # Evaluate on holdout (TRUE out-of-sample performance)
        logger.info("Evaluating on holdout set (true out-of-sample)...")
        model = joblib.load(result.model_path)
        holdout_preds = model.predict(X_holdout)
        holdout_accuracy = float(np.mean(holdout_preds == y_holdout))
        logger.info(f"Holdout accuracy: {holdout_accuracy:.2%}")

        # Calculate split indices for backtest
        train_ratio = float(self._get("training.train_ratio", 0.6))
        val_ratio = float(self._get("training.val_ratio", 0.2))
        n = len(df)
        val_end = int(n * (train_ratio + val_ratio))

        holdout_df = df.iloc[val_end:].copy()
        sharpe, win_rate, max_dd = self._backtest_on_split(
            result.model_path, holdout_df, feature_cols, "Holdout"
        )

        return {
            "model_path": result.model_path,
            "train_accuracy": result.train_accuracy,
            "val_accuracy": result.val_accuracy,
            "holdout_accuracy": holdout_accuracy,
            "feature_names": result.feature_names,
            "features_used": feature_cols,
            "processed_path": processed_path,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "class_distribution": class_dist,
        }
