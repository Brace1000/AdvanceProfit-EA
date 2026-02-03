"""Model training utilities for XGBoost."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

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
        # Build parameters from config — 3-class: Sell(0) / Range(1) / Buy(2)
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

        # Log feature importance (before calibration, as calibrated models don't expose feature_importances_)
        self._log_feature_importance(model, feature_names)

        # Optional probability calibration
        use_calibration = bool(self._get("training.use_probability_calibration", False))
        if use_calibration and X_val is not None and y_val is not None:
            logger.info("Calibrating probabilities using isotonic regression...")
            model = self._calibrate_probabilities(model, X_val, y_val)
            logger.info("✓ Probability calibration complete")

        # Save model (calibrated if calibration was enabled)
        model_path = self._get("model.path", "xgb_eurusd_h1.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        return TrainResult(
            model_path=model_path,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            feature_names=feature_names,
        )

    def _calibrate_probabilities(
        self,
        model: xgb.XGBClassifier,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> CalibratedClassifierCV:
        """
        Calibrate probability predictions using isotonic regression.

        Why calibrate:
        - XGBoost probabilities are "confidence scores", not true probabilities
        - Calibration makes them meaningful for decision-making
        - Isotonic regression works best for tree-based models

        Args:
            model: Trained XGBoost model
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Calibrated classifier wrapper
        """
        calibration_method = str(self._get("training.calibration_method", "isotonic"))

        # CalibratedClassifierCV with cv='prefit' uses the already-trained model
        calibrated_model = CalibratedClassifierCV(
            model,
            method=calibration_method,
            cv='prefit',
            n_jobs=-1
        )

        # Fit the calibrator on validation set
        calibrated_model.fit(X_val, y_val)

        logger.info(f"  Method: {calibration_method}")
        logger.info(f"  Calibration samples: {len(X_val)}")

        return calibrated_model

    def _compute_sample_weights(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Compute class-balanced sample weights (3-class: 0=Sell, 1=Range, 2=Buy)."""
        try:
            classes = np.array([0, 1, 2])
            class_weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=y
            )
            weight_map = dict(zip(classes, class_weights))
            weights = np.array([weight_map[label] for label in y])
            logger.debug(f"Class weights: {weight_map}")
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
