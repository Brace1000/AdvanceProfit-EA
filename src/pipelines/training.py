"""Training pipeline orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib

from src.config import get_config
from src.logger import get_logger
from src.data_collection.loader import DataLoader
from src.data_collection.validator import DataValidator
from src.features.engineer import FeatureEngineer
from src.models.hpo import optimize as hpo_optimize
from src.models.trainer import ModelTrainer, TrainResult

logger = get_logger("trading_bot.pipeline")


@dataclass
class PipelineResult:
    model_path: str
    train_accuracy: float
    val_accuracy: float
    sharpe: float
    win_rate: float
    max_drawdown: float


class TrainingPipeline:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or get_config()._config
        self.loader = DataLoader(self.config)
        self.validator = DataValidator(self.config)
        self.fe = FeatureEngineer(self.config)
        self.trainer = ModelTrainer(self.config)

    def _label(self, df: pd.DataFrame) -> pd.DataFrame:
        buy_th = float(self.config.get("training.buy_threshold", 0.002))
        sell_th = float(self.config.get("training.sell_threshold", -0.002))
        ahead_return = df["close"].shift(-1) / df["close"] - 1
        labels = np.zeros(len(df), dtype=int)
        labels[ahead_return > buy_th] = 1
        labels[ahead_return < sell_th] = -1
        df["label"] = labels
        return df

    def run(self) -> Dict[str, Any]:
        raw_path = self.config.get("paths.data_raw_file", "EURUSD_D1_raw.csv")
        processed_path = self.config.get("paths.data_processed_file", "EURUSD_D1_clean.csv")

        # Load and validate
        df = self.loader.load(raw_path)
        self.validator.validate_ohlc(df)

        # Features
        df = self.fe.engineer(df)
        df = self._label(df)

        feature_cols = FeatureEngineer.default_feature_columns()
        df = df.dropna(subset=feature_cols).iloc[:-1].copy()
        logger.info(f"Dataset after feature and label prep: {len(df)} rows, {len(feature_cols)} features")

        # Persist processed
        pd.DataFrame(df).to_csv(processed_path, index=False)
        logger.info(f"Saved processed dataset to {processed_path}")

        # Prepare arrays
        X = df[feature_cols].values
        y = pd.Series(df["label"]).map({-1: 0, 0: 1, 1: 2}).values

        # HPO (optional)
        if bool(self.config.get("hpo.enabled", True)):
            X_train = X[: int(len(X) * (1 - float(self.config.get("training.test_size", 0.3))))]
            y_train = y[: len(X_train)]
            best_params, best_val = hpo_optimize(X_train, y_train, self.config)
            self.config.setdefault("model", {}).setdefault("params", {}).update(best_params)
            logger.info("Updated config with best HPO params for training stage")

        # Train
        result: TrainResult = self.trainer.train(X, y, feature_cols)

        # Simple backtest on validation segment
        test_size = float(self.config.get("training.test_size", 0.3))
        split_idx = int(len(X) * (1 - test_size))
        val_df = df.iloc[split_idx:].copy()
        y_val_true = pd.Series(val_df["label"]).map({-1: 0, 0: 1, 1: 2}).values
        try:
            preds_val = result and joblib.load(result.model_path).predict(val_df[feature_cols].values)
        except Exception:
            preds_val = None

        sharpe = 0.0
        win_rate = 0.0
        max_dd = 0.0
        if preds_val is not None and len(preds_val) == len(val_df):
            val_df["position"] = pd.Series(preds_val).map({0: -1, 1: 0, 2: 1}).values
            val_df["next_ret"] = val_df["close"].pct_change(-1) * -1
            trading_cost = float(self.config.get("backtest.trading_cost", 0.0001))
            val_df = val_df.iloc[:-1].copy()
            val_df["pnl"] = val_df["position"] * val_df["next_ret"] - (val_df["position"] != 0).astype(float) * trading_cost
            avg_pnl = float(val_df["pnl"].mean()) if len(val_df) else 0.0
            std_pnl = float(val_df["pnl"].std(ddof=1)) if len(val_df) > 1 else 0.0
            sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0.0
            cum_curve = (1 + val_df["pnl"]).cumprod()
            rolling_max = cum_curve.cummax()
            drawdown = cum_curve / rolling_max - 1
            max_dd = float(drawdown.min()) if len(drawdown) else 0.0
            wins = int((val_df["pnl"] > 0).sum())
            losses = int((val_df["pnl"] < 0).sum())
            win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else 0.0
            logger.info(f"Backtest | Trades: {(val_df['position'] != 0).sum()} | Win rate: {win_rate:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%}")

        return {
            "model_path": result.model_path,
            "train_accuracy": result.train_accuracy,
            "val_accuracy": result.val_accuracy,
            "feature_names": result.feature_names,
            "features_used": feature_cols,
            "processed_path": processed_path,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
        }
