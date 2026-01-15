"""Model training utilities for XGBoost."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.logger import get_logger

logger = get_logger("trading_bot.model")


@dataclass
class TrainResult:
    model_path: str
    train_accuracy: float
    val_accuracy: float
    feature_names: list[str]


class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        test_size = float(self.config.get("training.test_size", 0.3))
        return train_test_split(X, y, test_size=test_size, shuffle=False)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> TrainResult:
        X_train, X_val, y_train, y_val = self.split(X, y)
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(self.config.get("model.params", {}))
        # Early stopping configuration
        early_stopping_rounds = int(self.config.get("training.early_stopping_rounds", 20))
        model = xgb.XGBClassifier(**params)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        logger.info("Training XGBoost classifier")
        # Class-balanced sample weights to mitigate class imbalance
        classes = np.array([0, 1, 2], dtype=int)
        try:
            class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            sample_weight_train = class_weights[y_train]
        except Exception:
            sample_weight_train = None

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                sample_weight=sample_weight_train,
                verbose=False,
            )
        except TypeError:
            logger.warning("early_stopping_rounds not supported by installed xgboost. Training without early stopping.")
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                sample_weight=sample_weight_train,
                verbose=False,
            )

        train_acc = float(model.score(X_train, y_train))
        val_acc = float(model.score(X_val, y_val))
        logger.info(f"Train acc: {train_acc:.2%} | Val acc: {val_acc:.2%}")

        model_path = self.config.get("model.path", "xgb_eurusd_h1.pkl")
        joblib.dump(model, model_path)
        return TrainResult(model_path=model_path, train_accuracy=train_acc, val_accuracy=val_acc, feature_names=feature_names)
