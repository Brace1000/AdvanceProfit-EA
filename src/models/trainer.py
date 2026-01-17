"""Model training utilities for XGBoost."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight

from src.logger import get_logger

logger = get_logger("trading_bot.model")


def _get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested config value using dot notation."""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


@dataclass
class TrainResult:
    """Result of model training."""
    model_path: str
    train_accuracy: float
    val_accuracy: float
    feature_names: list[str]


class ModelTrainer:
    """
    XGBoost model trainer with proper validation handling.

    Supports two modes:
    1. With validation set: Uses early stopping on validation data
    2. Without validation set: Trains for fixed iterations (for final model)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        return _get_nested(self.config, key_path, default)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainResult:
        """
        Train XGBoost classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features
            X_val: Optional validation features (for early stopping)
            y_val: Optional validation labels

        Returns:
            TrainResult with model path and metrics
        """
        # Build parameters from config
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        model_params = self._get("model.params", {})
        if model_params:
            params.update(model_params)

        # Compute class-balanced sample weights
        sample_weight = self._compute_sample_weights(y_train)

        model = xgb.XGBClassifier(**params)

        logger.info(f"Training XGBoost with {len(X_train)} samples, {len(feature_names)} features")
        logger.info(f"Parameters: max_depth={params.get('max_depth')}, "
                    f"n_estimators={params.get('n_estimators')}, "
                    f"learning_rate={params.get('learning_rate')}")

        if X_val is not None and y_val is not None:
            # Training with early stopping on validation set
            early_stopping_rounds = int(self._get("training.early_stopping_rounds", 20))
            logger.info(f"Using early stopping with {len(X_val)} validation samples")

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
                sample_weight=sample_weight,
            )

            val_accuracy = float(model.score(X_val, y_val))
        else:
            # Final training without early stopping (all data is training data)
            logger.info("Training final model without early stopping")

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train)],
                verbose=False,
                sample_weight=sample_weight,
            )

            # No separate validation set, report train accuracy as placeholder
            val_accuracy = 0.0

        train_accuracy = float(model.score(X_train, y_train))

        logger.info(f"Train accuracy: {train_accuracy:.2%}")
        if val_accuracy > 0:
            logger.info(f"Validation accuracy: {val_accuracy:.2%}")

        # Check for overfitting
        if val_accuracy > 0 and (train_accuracy - val_accuracy) > 0.15:
            logger.warning(
                f"Potential overfitting detected! "
                f"Train: {train_accuracy:.2%}, Val: {val_accuracy:.2%}, "
                f"Gap: {train_accuracy - val_accuracy:.2%}"
            )

        # Save model
        model_path = self._get("model.path", "xgb_eurusd_h1.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Log feature importance
        self._log_feature_importance(model, feature_names)

        return TrainResult(
            model_path=model_path,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            feature_names=feature_names,
        )

    def _compute_sample_weights(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Compute class-balanced sample weights."""
        try:
            classes = np.array([0, 1, 2], dtype=int)
            class_weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=y
            )
            weights = class_weights[y]
            logger.debug(f"Class weights: {dict(zip(classes, class_weights))}")
            return weights
        except Exception as e:
            logger.warning(f"Could not compute sample weights: {e}")
            return None

    def _log_feature_importance(
        self, model: xgb.XGBClassifier, feature_names: list[str]
    ) -> None:
        """Log top feature importances."""
        try:
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]

            logger.info("Top 5 feature importances:")
            for i in sorted_idx[:5]:
                logger.info(f"  {feature_names[i]}: {importances[i]:.4f}")
        except Exception:
            pass
